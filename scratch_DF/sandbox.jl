using Statistics
using Plots
using BenchmarkTools
using Profile
using FFTW
using Statistics
using Optim
using Images, FileIO

#using Profile
using LinearAlgebra

# put the cwd on the path so Julia can find the module
push!(LOAD_PATH, pwd()*"/main")
using DHC_2DUtils


@time fhash = fink_filter_hash(1, 8, nx=64, pc=1, wd=1)

function realspace_filter(Nx, f_i, f_v)

    zarr = zeros(ComplexF64, Nx, Nx)
    for i = 1:length(f_i)
        zarr[f_i[i]] = f_v[i] # filter*image in Fourier domain
    end
    filt = ifft(zarr)  # real space, complex
    return filt
end



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



function derivtest2(Nx)
    eps = 1e-4
    fhash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1)
    im = rand(Float64, Nx, Nx).*0.1
    im[6,6]=1.0
    im1 = copy(im)
    im0 = copy(im)
    im1[2,3] += eps/2
    im0[2,3] -= eps/2
    dS20dp = wst_S20_deriv(im, fhash)

    der0=DHC(im0,fhash,doS2=false,doS20=true)
    der1=DHC(im1,fhash,doS2=false,doS20=true)
    dS = (der1-der0) ./ eps

    Nf = length(fhash["filt_index"])
    i0 = 3+Nf
    blarg = dS20dp[2,3,:,:]
    diff = dS[i0:end]-reshape(blarg,Nf*Nf)
    println(dS[i0:end])
    println("and")
    println(blarg)
    println("stdev: ",std(diff))
    println()
    println(diff)

    #plot(dS[3:end])
    #plot!(blarg[2,3,:])
    #plot(diff)
    return
end

derivtest2(8)



Nx=128
fhash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1)
im = zeros(Float64, Nx, Nx)
im[6,6]=1.0
@benchmark blarg = wst_S1_deriv(im, fhash)


# S1 deriv time, Jan 30
# 32    17 ms
# 64    34 ms
# 128   115 ms
# 256   520 ms
# 512   2500 ms

Nx = 64
fhash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1)
im = zeros(Float64, Nx, Nx)
im[6,6]=1.0
@benchmark blarg = wst_S20_deriv(im, fhash)

Profile.clear()
@profile blarg = wst_S20_deriv(im, fhash)
Juno.profiler()


# S2 deriv time, Jan 30
# 8     28 ms
# 16    112
# 32    320
# 64    1000
# 128   5 sec
# 256   ---
# 512   ---



print(1)



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




# wst_S1_deriv agrees with brute force at 1e-10 level.
function derivtest(Nx)
    eps = 1e-4
    fhash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1)
    im = rand(Float64, Nx, Nx).*0.1
    im[6,6]=1.0
    im1 = copy(im)
    im0 = copy(im)
    im1[2,3] += eps/2
    im0[2,3] -= eps/2
    blarg = wst_S1_deriv(im, fhash)

    der0=DHC(im0,fhash,doS2=false)
    der1=DHC(im1,fhash,doS2=false)
    dS = (der1-der0) ./ eps


    diff = dS[3:end]-blarg[2,3,:]
    println(dS[3:end])
    println("and")
    println(blarg[2,3,:])
    println("stdev: ",std(diff))

    #plot(dS[3:end])
    #plot!(blarg[2,3,:])
    #plot(diff)
    return
end

derivtest(128)



Nx=32
im = rand(Nx,Nx)
@time fhash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1)
@benchmark Sarr = DHC(im, fhash, doS2=false)  # 0.36 ms
@benchmark Sarr = DHC(im, fhash, doS2=true)   # 6.4 ms
@benchmark Sarr = DHC(im, fhash, doS2=false, doS12=true)  # 0.7 ms
@benchmark Sarr = DHC(im, fhash, doS2=false, doS20=true)  # 2.1 ms


Nx=64
im = rand(Nx,Nx)
@time fhash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1)
@benchmark Sarr = DHC(im, fhash, doS2=false)  # 1.3 ms
@benchmark Sarr = DHC(im, fhash, doS2=true)   # 19 ms
@benchmark Sarr = DHC(im, fhash, doS2=false, doS12=true)  # 2.8 ms
@benchmark Sarr = DHC(im, fhash, doS2=false, doS20=true)  # 6.6 ms


Nx=256
im = rand(Nx,Nx)
@time fhash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1)
@benchmark Sarr = DHC(im, fhash, doS2=false)  # 25 ms
@benchmark Sarr = DHC(im, fhash, doS2=true)   # 375 ms
@benchmark Sarr = DHC(im, fhash, doS2=false, doS12=true)  # 55 ms
@benchmark Sarr = DHC(im, fhash, doS2=false, doS20=true)  # 130 ms


Nx = 64
fhash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1)
im = rand(Nx,Nx)
S_targ = DHC(im, fhash, doS2=false)

im = rand(Nx,Nx)







function wst_synth(im_init, fixmask)
    # fixmask -  0=float in fit   1=fixed

    function wst_synth_chisq(vec_in)

        thisim = copy(im_init)
        thisim[indfloat] = vec_in

        Sarr  = DHC(thisim, fhash, doS2=false)
        diff  = (Sarr .- S_targ)[3:end]

        # should have some kind of weight here
        chisq = diff'*diff
        return chisq

    end

    function wst_synth_dchisq(storage, vec_in)

        thisim = copy(im_init)
        thisim[indfloat] = vec_in
        dSdp   = wst_S1_deriv(thisim, fhash)
        S1arr  = DHC(thisim, fhash, doS2=false)
        diff  = (S1arr - S_targ)[3:end]

        # dSdp matrix * S1-S_targ is dchisq
        dchisq_im = (reshape(dSdp, Nx*Nx, Nf) * diff).*2
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

Nx    = 128
fhash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1)
im    = rand(Nx,Nx)
fixmask = im .> 0.5
S_targ = DHC(im, fhash, doS2=false)

init = copy(im)
init[findall(fixmask .==0)] .= 0

foo = wst_synth(init, fixmask)






#Optim.minimizer(optimize(FOM, im, LBFGS()))

# using dchisq function
# size   t(BFGS) t(LBFGS) [sec]
# 16x16    1
# 32x32    4        9
# 64x64    20      52
# 128x128         153 (fitting 50% of pix)


# without dchisq function
# size   t(BFGS) t(LBFGS) [sec]
# 16x16   27
# 32x32  355      110
# 64x64          1645
# 128x128        est 28,000  (8 hrs)


function readdust()

    RGBA_img = load(pwd()*"/scratch_DF/t115_clean_small.png")
    img = reinterpret(UInt8,rawview(channelview(RGBA_img)))

    return img[1,:,:]
end





function wst_synthS20(im_init, fixmask, S_targ, S20sig)
    # fixmask -  0=float in fit   1=fixed

    function wst_synth_chisq(vec_in)

        thisim = copy(im_init)
        thisim[indfloat] = vec_in

        S20arr  = DHC(thisim, fhash, doS2=false, doS20=true)
        i0 = 3+Nf
        diff  = ((S20arr - S_targ)./S20sig)[i0:end]

        # should have some kind of weight here
        chisq = diff'*diff
        println(chisq)
        return chisq

    end

    function wst_synth_dchisq(storage, vec_in)

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


function S20_weights(im, fhash, Nsam=10)

    (Nx, Ny)  = size(im)
    if Nx != Ny error("Input image must be square") end
    (Nf, )    = size(fhash["filt_index"])

    # fhash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1)

    S20   = DHC(im, fhash, doS2=false, doS20=true)
    Ns    = length(S20)
    S20arr = zeros(Float64, Ns, Nsam)
    for j=1:Nsam
        noise = rand(Nx,Nx)
        S20arr[:,j] = DHC(im+noise, fhash, doS2=false, doS20=true)
    end

    wt = zeros(Float64, Ns)
    for i=1:Ns
        wt[i] = std(S20arr[i,:])
    end

    return wt
end



# read dust map
dust = Float64.(readdust())
dust = dust[1:256,1:256]

Nx    = 64
fhash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1, Omega=true)
im    = imresize(dust,(Nx,Nx))
fixmask = rand(Nx,Nx) .< 0.1

S_targ = DHC(im, fhash, doS2=false, doS20=true)
S_targ[end] = 0
init = copy(im)
floatind = findall(fixmask .==0)
init[floatind] .+= rand(length(floatind)).*5 .-25

S20sig = S20_weights(im, fhash, 100)
foo = wst_synthS20(init, fixmask, S_targ, S20sig)
S_foo = DHC(foo, fhash, doS2=false, doS20=true)


# using dchisq function with S20
# size   t(BFGS) t(LBFGS) [sec]
# 8x8      33
# 16x16    -
# 32x32
