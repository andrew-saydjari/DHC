using Plots
using BenchmarkTools
using Profile
using FFTW
using Statistics
using Optim
using Images, FileIO, ImageIO
using Random, Distributions

Random.seed!(123)

push!(LOAD_PATH, pwd()*"/main")
push!(LOAD_PATH, pwd()*"/scratch_DF")
using DHC_2DUtils

##Copying over sandbox_df utilities here

function readdust(fname)

    RGBA_img = load(fname)
    img = reinterpret(UInt8,rawview(channelview(RGBA_img)))

    return img[1,:,:]
end

function DHC(image::Array{Float64,2}, filter_hash::Dict, filter_hash2::Dict=filter_hash;
    doS2::Bool=true, doS12::Bool=false, doS20::Bool=false)
    # image        - input for WST
    # filter_hash  - filter hash from fink_filter_hash
    # filter_hash2 - filters for second order.  Default to same as first order.
    # doS2         - compute S2 coeffs
    # doS12        - compute S2 coeffs
    # doS20        - compute S2 coeffs

    # Use 2 threads for FFT
    FFTW.set_num_threads(2)

    # array sizes
    (Nx, Ny)  = size(image)
    if Nx != Ny error("Input image must be square") end
    (Nf, )    = size(filter_hash["filt_index"])
    if Nf == 0 error("filter hash corrupted") end

    # allocate coeff arrays
    out_coeff = []
    S0  = zeros(Float64, 2)
    S1  = zeros(Float64, Nf)
    if doS2  S2  = zeros(Float64, Nf, Nf) end  # traditional 2nd order
    if doS12 S12 = zeros(Float64, Nf, Nf) end  # Fourier correlation
    if doS20 S20 = zeros(Float64, Nf, Nf) end  # real space correlation
    anyM2 = doS2 | doS12 | doS20
    anyrd = doS2 | doS20             # compute real domain with iFFT

    # allocate image arrays for internal use
    if doS12 im_fdf_0_1 = zeros(Float64,           Nx, Ny, Nf) end   # this must be zeroed!
    if anyrd im_rd_0_1  = Array{Float64, 3}(undef, Nx, Ny, Nf) end

    ## 0th Order
    S0[1]   = mean(image)
    norm_im = image.-S0[1]
    S0[2]   = sum(norm_im .* norm_im)/(Nx*Ny)
    # norm_im ./= sqrt(Nx*Ny*S0[2])
    # println("norm off")
    norm_im = copy(image)

    append!(out_coeff,S0[:])

    ## 1st Order
    im_fd_0 = fft(norm_im)  # total power=1.0

    # unpack filter_hash
    f_ind   = filter_hash["filt_index"]  # (J, L) array of filters represented as index value pairs
    f_val   = filter_hash["filt_value"]

    zarr = zeros(ComplexF64, Nx, Ny)  # temporary array to fill with zvals

    # make a FFTW "plan" for an array of the given size and type
    if anyrd
        P = plan_ifft(im_fd_0) end  # P is an operator, P*im is ifft(im)

    ## Main 1st Order and Precompute 2nd Order
    for f = 1:Nf
        S1tot = 0.0
        f_i = f_ind[f]  # CartesianIndex list for filter
        f_v = f_val[f]  # Values for f_i
        # for (ind, val) in zip(f_i, f_v)   # this is slower!
        if length(f_i) > 0
            for i = 1:length(f_i)
                ind       = f_i[i]
                zval      = f_v[i] * im_fd_0[ind]
                S1tot    += abs2(zval)
                zarr[ind] = zval        # filter*image in Fourier domain
                if doS12 im_fdf_0_1[ind,f] = abs(zval) end
            end
            S1[f] = S1tot/(Nx*Ny)  # image power
            if anyrd
                im_rd_0_1[:,:,f] .= abs2.(P*zarr) end
            zarr[f_i] .= 0
        end
    end
    append!(out_coeff, S1[:])

    # we stored the abs()^2, so take sqrt (this is faster to do all at once)
    if anyrd im_rd_0_1 .= sqrt.(im_rd_0_1) end


    if doS2
        f_ind2   = filter_hash2["filt_index"]  # (J, L) array of filters represented as index value pairs
        f_val2   = filter_hash2["filt_value"]

        ## Traditional second order
        for f1 = 1:Nf
            thisim = fft(im_rd_0_1[:,:,f1])  # Could try rfft here
            # println("  f1",f1,"  sum(fft):",sum(abs2.(thisim))/Nx^2, "  sum(im): ",sum(abs2.(im_rd_0_1[:,:,f1])))
            # Loop over f2 and do second-order convolution
            for f2 = 1:Nf
                f_i = f_ind2[f2]  # CartesianIndex list for filter
                f_v = f_val2[f2]  # Values for f_i
                # sum im^2 = sum(|fft|^2/npix)
                S2[f1,f2] = sum(abs2.(f_v .* thisim[f_i]))/(Nx*Ny)
            end
        end
        append!(out_coeff, S2)
    end


    # Fourier domain 2nd order
    if doS12
        Amat = reshape(im_fdf_0_1, Nx*Nx, Nf)
        S12  = Amat' * Amat
        append!(out_coeff, S12)
    end


    # Real domain 2nd order
    if doS20
        Amat = reshape(im_rd_0_1, Nx*Nx, Nf)
        S20  = Amat' * Amat
        append!(out_coeff, S20)
    end


    return out_coeff
end

function realspace_filter(Nx, f_i, f_v)

    zarr = zeros(ComplexF64, Nx, Nx)
    for i = 1:length(f_i)
        zarr[f_i[i]] = f_v[i] # filter*image in Fourier domain
    end
    filt = ifft(zarr)  # real space, complex
    return filt
end

#Deriv related funcs copied over from sandbox
function wst_S1_deriv(image::Array{Float64,2}, filter_hash::Dict)
    function conv(a,b)
        ifft(fft(a) .* fft(b))
    end

    # Use 2 threads for FFT
    FFTW.set_num_threads(2)

    # array sizes
    (Nx, Ny)  = size(image)
    (Nf, )    = size(filter_hash["filt_index"])

    # allocate output array
    dSdp  = zeros(Float64, Nx, Nx, Nf)

    # allocate image arrays for internal use
    #im_rdc_0_1 = Array{ComplexF64, 3}(undef, Nx, Ny, Nf)  # real domain, complex

    # Not sure what to do here -- for gradients I don't think we want these
    ## 0th Order
    #S0[1]   = mean(image)
    #norm_im = image.-S0[1]
    #S0[2]   = sum(norm_im .* norm_im)/(Nx*Ny)
    #norm_im ./= sqrt(Nx*Ny*S0[2])

    ## 1st Order
    im_fd_0 = fft(image)  # total power=1.0

    # unpack filter_hash
    f_ind   = filter_hash["filt_index"]  # (J, L) array of filters represented as index value pairs
    f_val   = filter_hash["filt_value"]

    zarr = zeros(ComplexF64, Nx, Nx)  # temporary array to fill with zvals

    # make a FFTW "plan" for an array of the given size and type
    P = plan_ifft(im_fd_0)  # P is an operator, P*im is ifft(im)

    # Loop over filters
    for f = 1:Nf
        f_i = f_ind[f]  # CartesianIndex list for filter
        f_v = f_val[f]  # Values for f_i

        zarr[f_i] = f_v .* im_fd_0[f_i]
        I_λ = P*zarr  # complex valued ifft of zarr
        #im_rdc_0_1[:,:,f] = I_lambda
        zarr[f_i] .= 0   # reset zarr for next loop

        #xfac = 2 .* real(I_λ)
        #yfac = 2 .* imag(I_λ)
        # it is clearly stupid to transform back and forth, but we will need these steps for the S20 part
        ψ_λ  = realspace_filter(Nx, f_i, f_v)
        # convolution requires flipping wavelet direction, equivalent to flipping sign of imaginary part.
        # dSdp[:,:,f] = real.(conv(xfac,real(ψ_λ))) - real.(conv(yfac,imag(ψ_λ)))
        dSdp[:,:,f] = 2 .* real.(conv(I_λ, ψ_λ))

    end
    return dSdp

end



function wst_S20_deriv(image::Array{Float64,2}, filter_hash::Dict)
    function conv(a,b)
        ifft(fft(a) .* fft(b))
    end

    # Use 2 threads for FFT
    FFTW.set_num_threads(2)

    # array sizes
    (Nx, Ny)  = size(image)
    (Nf, )    = size(filter_hash["filt_index"])

    # allocate output array
    dS20dp  = zeros(Float64, Nx, Nx, Nf, Nf)

    # allocate image arrays for internal use
    im_rdc_0_1 = Array{ComplexF64, 3}(undef, Nx, Ny, Nf)  # real domain, complex
    im_rd_0_1  = Array{Float64, 3}(undef, Nx, Ny, Nf)  # real domain, complex

    # Not sure what to do here -- for gradients I don't think we want these
    ## 0th Order
    #S0[1]   = mean(image)
    #norm_im = image.-S0[1]
    #S0[2]   = sum(norm_im .* norm_im)/(Nx*Ny)
    #norm_im ./= sqrt(Nx*Ny*S0[2])

    ## 1st Order
    im_fd_0 = fft(image)  # total power=1.0

    # unpack filter_hash
    f_ind   = filter_hash["filt_index"]  # (J, L) array of filters represented as index value pairs
    f_val   = filter_hash["filt_value"]

    zarr = zeros(ComplexF64, Nx, Nx)  # temporary array to fill with zvals

    # make a FFTW "plan" for an array of the given size and type
    P = plan_ifft(im_fd_0)  # P is an operator, P*im is ifft(im)

    # Loop over filters
    for f = 1:Nf
        f_i = f_ind[f]  # CartesianIndex list for filter
        f_v = f_val[f]  # Values for f_i

        zarr[f_i] = f_v .* im_fd_0[f_i]
        I_λ = P*zarr  # complex valued ifft of zarr
        zarr[f_i] .= 0   # reset zarr for next loop
        im_rdc_0_1[:,:,f] = I_λ
        im_rd_0_1[:,:,f]  = abs.(I_λ)
    end

    for f2 = 1:Nf
        ψ_λ  = realspace_filter(Nx, f_ind[f2], f_val[f2])
        uvec = im_rdc_0_1[:,:,f2] ./ im_rd_0_1[:,:,f2]
        for f1 = 1:Nf
            cfac = im_rd_0_1[:,:,f1].*uvec
            I1dI2 = real.(conv(cfac,ψ_λ))
            dS20dp[:,:,f1,f2] += I1dI2
            dS20dp[:,:,f2,f1] += I1dI2
        end
    end

    return dS20dp

end

function wst_synthS20(im_init, fixmask, S_targ, S20sig)
    # fixmask -  0=float in fit   1=fixed

    function wst_synth_chisq(vec_in) #Loss Function

        thisim = copy(im_init)
        thisim[indfloat] = vec_in

        S20arr  = DHC(thisim, fhash, doS2=false, doS20=true) #Check: Is this getting img_curr or the original thisim?
        i0 = 3+Nf #Why? Excludes S1, S2, incl only S20
        diff  = ((S20arr - S_targ)./S20sig)[i0:end]

        # should have some kind of weight here: to convert to redchisq?
        chisq = diff'*diff
        println(chisq)
        return chisq

    end

    function wst_synth_dchisq(storage, vec_in) #storage?

        thisim = copy(im_init)
        thisim[indfloat] = vec_in
        dS20dp = wst_S20_deriv(thisim, fhash)
        S20arr = DHC(thisim, fhash, doS2=false, doS20=true)
        i0 = 3+Nf
        # put both factors of S20sig in this array to weight
        diff   = ((S20arr - S_targ)./(S20sig.^2))[i0:end]

        # dSdp matrix * S1-S_targ is dchisq
        dchisq_im = (reshape(dS20dp, Nx*Nx, Nf*Nf) * diff).*2
        dchisq = reshape(dchisq_im, Nx, Nx)[indfloat]

        storage .= dchisq
    end

    (Nx, Ny)  = size(im_init)
    if Nx != Ny error("Input image must be square") end
    (Nf, )    = size(fhash["filt_index"])


    # index list of pixels to float in fit
    indfloat = findall(fixmask .== 0)

    # initial values for floating pixels
    vec_init = im_init[indfloat]
    println(length(vec_init), " pixels floating")

    eps = zeros(size(vec_init))
    eps[1] = 1e-4
    chisq1 = wst_synth_chisq(vec_init+eps./2)
    chisq0 = wst_synth_chisq(vec_init-eps./2)
    brute  = (chisq1-chisq0)/1e-4

    clever = zeros(size(vec_init))
    _bar = wst_synth_dchisq(clever, vec_init)
    println("Brute:  ",brute)
    println("Clever: ",clever[1])

    # call optimizer
    res = optimize(wst_synth_chisq, wst_synth_dchisq, vec_init, BFGS())

    # copy results into pixels of output image
    im_synth = copy(im_init)
    im_synth[indfloat] = Optim.minimizer(res)
    println(res)

    return im_synth
end




##SNIPPETS
#1) Visualizing filters in F space
filt, info = fink_filter_bank(1, 8, nx=256, wd=1, pc=1, shift=false, Omega=true)
#= Sums of intensities: sum(fftshift(filt[:, :, :]))/(256^2)
fink_filter_bank(1, 8, nx=256, wd=1, pc=1, shift=false, Omega=false) <~0.5
fink_filter_bank(1, 8, nx=256, wd=1, pc=1, shift=false, Omega=true) =1.15. 2 in Q1-2, 1 elsewhere, 0 in Q3-4. Main config.
fink_filter_bank(1, 8, nx=256, wd=1, pc=2, shift=false, Omega=false) = 0.70
fink_filter_bank(1, 8, nx=256, wd=1, pc=2, shift=false, Omega=true) = 1.35
fink_filter_bank(2, 8, nx=256, wd=1, pc=1, shift=false, Omega=false) = 0.65
fink_filter_bank(2, 8, nx=256, wd=1, pc=1, shift=false, Omega=false) = 1.14
fink_filter_bank(1, 8, nx=256, wd=2, pc=1, shift=false, Omega=true) = 1.36, blurrier
=#

#Vis: heatmap(sum(fftshift(filt[:, :, :]), dims=3)[:, :, 1])

#2) Calc WST coeff of WISE
Nx    = 64
gt_imgfull = Float64.(readdust(pwd()*"/scratch_DF/t115_clean_small.png"))
fhash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1, Omega=true)
gt_img    = imresize(gt_imgfull,(Nx,Nx)) #Check if we should be doing this? Warps the image?

coeffs_rs = DHC(gt_img, fhash, doS2=false, doS20=true)

#3) GD to coeffs_rs
mean_img = mean(gt_img)
img_curr = gt_img + rand(Normal(0.0, 0.5*mean_img), (Nx, Nx)) #noisy
im_synth = wst_synthS20(img_curr, fill(0.0, (Nx, Nx)), coeffs_rs, mean(coeffs_rs)*0.01)



function derivtestS1S2(Nx)
    eps = 1e-4
    fhash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1)
    im = rand(Float64, Nx, Nx).*0.1
    im[6,6]=1.0
    im1 = copy(im)
    im0 = copy(im)
    im1[2,3] += eps/2
    im0[2,3] -= eps/2
    dS1dp_existing = wst_S1_deriv(im, fhash)
    dS1dp, dS2dp = wst_S1S2_deriv(im, fhash)

    der0=DHC(im0,fhash,doS2=true,doS20=false)
    der1=DHC(im1,fhash,doS2=true,doS20=false)
    dSlim = (der1-der0) ./ eps
    Nf = length(fhash["filt_index"])
    lasts1ind = 2+Nf

    dS1lim23 = dSlim[3:lasts1ind]
    dS2lim23 = dSlim[lasts1ind+1:end]
    dS1dp_23 = dS1dp[2, 3, :]
    dS2dp_23 = dS2dp[2, 3, :, :]
    #=
    println("Checking dS1dp using existing deriv")
    Nf = length(fhash["filt_index"])
    lasts1ind = 2+Nf
    diff = dS1dp_existing[2, 3, :]-dSlim[3:lasts1ind]
    println(dSlim[3:lasts1ind])
    println("and")
    println(dS1dp_existing[2, 3, :])
    println("stdev: ",std(diff))
    println("--------------------------")=#
    println("Checking dS1dp using dS1S2")
    derdiff = dS1dp_23 - dS1lim23
    println(dS1lim23)
    println("and")
    println(dS1dp23)
    txt= @sprintf("Range of S1 der0, der1: Max= %.3E, %.3E Min=%.3E, %.3E ",maximum(der0[3:lasts1ind]),maximum(der1[3:lasts1ind]),minimum(der0[3:lasts1ind]),minimum(der1[3:lasts1ind]))
    print(txt)
    txt = @sprintf("Checking dS1dp using S1S2 deriv, mean(abs(diff/dS1lim)): %.3E", mean(abs.(diff./dS1lim23)))
    print(txt)
    println("stdev: ",std(derdiff))
    txt= @sprintf("Range of dS1lim: Max= ",maximum(abs.(dS1lim23))," Min= ",minimum(abs.(dS1lim23)))
    print(txt)
    println("--------------------------")
    println("Checking dS2dp using S1S2 deriv")
    Nf = length(fhash["filt_index"])
    println("Shape check",size(dS2dp23),size(dS2lim23))
    derdiff = reshape(dS2dp_23, Nf*Nf) - dS2lim23 #Column major issue here?
    println("Difference",derdiff)
    txt = @sprintf("Range of S2 der0, der1: Max= %.3E, %.3E Min=%.3E, %.3E \n",maximum(der0[lasts1ind+1:end]),maximum(der1[lasts1ind+1:end]),minimum(der0[lasts1ind+1:end]),minimum(der1[lasts1ind+1:end]))
    print(txt)
    txt = @sprintf("Range of dS2lim: Max=%.3E  Min=%.3E \n",maximum(abs.(dS2lim23)),minimum(abs.(dS2lim23)))
    print(txt)
    txt = @sprintf("Difference between dS2dp and dSlim Mean: %.3E stdev: %.3E mean(abs(derdiff/dS2lim)): %.3E \n",mean(derdiff),std(derdiff), mean(abs.(derdiff./dS2lim23)))
    print(txt)
    println("Examining only those S2 coeffs which are greater than eps=1e-3 for der0")
    eps_mask = findall(der0[lasts1ind+1:end] .> 1E-3)
    der0_s2good = der0[lasts1ind+1:end][eps_mask]
    der1_s2good = der1[lasts1ind+1:end][eps_mask]
    dSlim_s2good = dS2lim23[eps_mask]
    derdiff_good = derdiff[eps_mask]
    print(derdiff_good)
    txt = @sprintf("\nRange of S2 der0, der1: Max= %.3E, %.3E Min=%.3E, %.3E \n",maximum(der0_s2good),maximum(der1_s2good),minimum(der0_s2good),minimum(der1_s2good))
    print(txt)
    txt = @sprintf("Range of dS2lim: Max=%.3E  Min=%.3E \n",maximum(abs.(dSlim_s2good)),minimum(abs.(dSlim_s2good)))
    print(txt)
    txt = @sprintf("Mean: %.3E stdev: %.3E mean(abs(derdiff/dS2lim)): %.3E \n",mean(derdiff_good),std(derdiff_good), mean(abs.(derdiff_good ./dSlim_s2good)))
    print(txt)

    #plot(dS[3:end])
    #plot!(blarg[2,3,:])
    #plot(diff)
    return
end
