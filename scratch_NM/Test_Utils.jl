using Statistics
using Plots
using BenchmarkTools
using Profile
using FFTW
using Statistics
using Optim
using Images, FileIO, ImageIO
using Printf
using Revise
using Profile
using LinearAlgebra
using Distributions

push!(LOAD_PATH, pwd()*"/main")
using DHC_2DUtils
#include("Deriv_Utils_New.jl")
#import Deriv_Utils_New
push!(LOAD_PATH, pwd()*"/scratch_NM")
using Deriv_Utils_New
using Data_Utils
using Visualization

#Remove later
function readdust(Nx)

    RGBA_img = load(pwd()*"/scratch_DF/t115_clean_small.png")
    img = reinterpret(UInt8,rawview(channelview(RGBA_img)))

    return imresize(Float64.(img[1,:,:])[1:256, 1:256], (Nx, Nx))
end

function S2_uniweights(im, fhash; high=25, Nsam=10, iso=false, norm=true)
    #=
    Noise model: uniform[-high, high]
    =#

    (Nx, Ny)  = size(im)
    if Nx != Ny error("Input image must be square") end
    (N1iso, Nf)    = size(fhash["S1_iso_mat"])

    # fhash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1)

    S2   = DHC_compute(im, fhash, doS2=true, doS20=false, norm=norm, iso=iso)
    Ns    = length(S2)
    S2arr = zeros(Float64, Ns, Nsam)
    println("Ns", Ns)
    for j=1:Nsam
        noise = rand(Nx,Nx).*(2*high) .- high
        S2arr[:,j] = DHC_compute(im+noise, fhash, doS2=true, doS20=false, norm=norm, iso=iso)
    end
    wt = zeros(Float64, Ns)
    for i=1:Ns
        wt[i] = std(S2arr[i,:])
    end
    msub = S2arr .- mean(S2arr, dims=2)
    cov = (msub * msub')./(Nsam-1)

    return wt, cov
end

function DHC_old(image::Array{Float64,2}, filter_hash::Dict, filter_hash2::Dict=filter_hash;
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
        append!(out_coeff, S1[:]) #Why need the :?

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
            append!(out_coeff, S2) #Does this force S2 to flatten st f1 varies faster than f2?
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

function S2_whitenoiseweights(im, fhash; loc=0.0, sig=1.0, Nsam=10, iso=false, norm=true)
    #=
    Noise model: uniform[-high, high]
    =#

    (Nx, Ny)  = size(im)
    if Nx != Ny error("Input image must be square") end
    (N1iso, Nf)    = size(fhash["S1_iso_mat"])

    # fhash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1)

    S2   = DHC_compute(im, fhash, doS2=true, doS20=false, norm=norm, iso=iso)
    Ns    = length(S2)
    S2arr = zeros(Float64, Ns, Nsam)
    println("Ns", Ns)
    for j=1:Nsam
        noise = reshape(rand(Normal(loc, sig),Nx^2), (Nx, Nx))
        S2arr[:,j] = DHC_compute(im+noise, fhash, doS2=true, doS20=false, norm=norm, iso=iso)
    end
    wt = zeros(Float64, Ns)
    for i=1:Ns
        wt[i] = std(S2arr[i,:])
    end
    msub = S2arr .- mean(S2arr, dims=2)
    cov = (msub * msub')./(Nsam-1)

    return wt, cov
end


#Jacobian Test Funcs
function derivtestS1(Nx)
   eps = 1e-4
   fhash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1)
   im = rand(Float64, Nx, Nx).*0.1
   im[6,6]=1.0
   im1 = copy(im)
   im0 = copy(im)
   im1[2,3] += eps/2
   im0[2,3] -= eps/2
   blarg = Deriv_Utils_New.wst_S1_deriv(im, fhash)
   der0=DHC_compute(im0,fhash,doS2=false,norm=false)
   der1=DHC_compute(im1,fhash,doS2=false,norm=false)
   dS = (der1-der0) ./ eps

   diff_old = dS[3:end]-blarg[2,3,:]

   println(dS[3:end])
   println("and")
   println(blarg[2,3,:])
   println("Mean abs | Mean abs frac", mean(abs.(diff_old)), mean(abs.(diff_old./dS[3:end])))
   println("stdev: ",std(diff_old))
   return blarg[2, 3, :], dS[3:end]
end

function derivtestS1S2(Nx) #MUST RUN DHC_COMPUTE WITH NORM=FALSE.
    eps = 1e-4
    fhash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1)
    im = rand(Float64, Nx, Nx).*0.1
    im[6,6]=1.0
    im1 = copy(im)
    im0 = copy(im)
    im1[2,3] += eps/2
    im0[2,3] -= eps/2
    dS1dp_existing = Deriv_Utils_New.wst_S1_deriv(im, fhash)
    dS1S2dp = Deriv_Utils_New.wst_S1S2_deriv(im, fhash) #Deriv_Utils_New.wst_S1S2_deriv(im, fhash)

    der0=DHC_compute(im0,fhash,doS2=true,doS20=false, norm=false)
    der1=DHC_compute(im1,fhash,doS2=true,doS20=false, norm=false)
    dSlim = (der1-der0) ./ eps
    Nf = length(fhash["filt_index"])
    lasts1ind = 2+Nf

    dS1lim23 = dSlim[3:lasts1ind]
    dS2lim23 = dSlim[lasts1ind+1:end]
    dS1dp_23 = dS1S2dp[2, 3, 1:Nf]
    dS2dp_23 = dS1S2dp[2, 3, Nf+1:end]
    #=
    println("Checking dS1dp using existing deriv")
    Nf = length(fhash["filt_index"])
    lasts1ind = 2+Nf
    diff = dS1dp_existing[2, 3, :]-dSlim[3:lasts1ind]
    println(dSlim[3:lasts1ind])
    println("and")
    println(dS1dp_existing[2, 3, :])
    println("stdev: ",std(diff))
    println("--------------------------")
    =#
    println("Checking dS1dp using dS1S2")
    derdiff = dS1dp_23 - dS1lim23
    #println(dS1lim23)
    #println("and")
    #println(dS1dp_23)
    txt= @sprintf("Range of S1 der0, der1: Max= %.3E, %.3E Min=%.3E, %.3E ",maximum(der0[3:lasts1ind]),maximum(der1[3:lasts1ind]),minimum(der0[3:lasts1ind]),minimum(der1[3:lasts1ind]))
    print(txt)
    txt = @sprintf("Checking dS1dp using S1S2 deriv, mean(abs(diff/dS1lim)): %.3E", mean(abs.(derdiff./dS1lim23)))
    print(txt)
    println("stdev: ",std(derdiff))
    txt= @sprintf("Range of dS1lim: Max= %.3E Min= %.3E",maximum(abs.(dS1lim23)),minimum(abs.(dS1lim23)))
    print(txt)
    println("--------------------------")
    println("Checking dS2dp using S1S2 deriv")
    Nf = length(fhash["filt_index"])
    println("Shape check",size(dS2dp_23),size(dS2lim23))
    derdiff = dS2dp_23 - dS2lim23 #Column major issue here?
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
    return dS1dp_23, dS1lim23, reshape(dS2dp_23, (Nf, Nf)), reshape(dS2lim23, (Nf, Nf))
end

function derivtestS12(Nx)
    eps = 1e-4
    fhash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1)
    im = rand(Float64, Nx, Nx).*0.1
    im[6,6]=1.0
    im1 = copy(im)
    im0 = copy(im)
    im1[2,3] += eps/2
    im0[2,3] -= eps/2
    dS12dp = Deriv_Utils_New.wst_S12_deriv(im, fhash)

    der0=DHC_compute(im0, fhash, doS2=false, doS20=false, doS12=true, norm=false)
    der1=DHC_compute(im1, fhash, doS2=false, doS20=false, doS12=true, norm=false)
    dS = (der1-der0) ./ eps

    Nf = length(fhash["filt_index"])
    i0 = 3+Nf
    blarg = dS12dp[2, 3, :, :]
    diff = dS[i0:end]-reshape(blarg,Nf*Nf)
    println(dS[i0:end])
    println("and")
    println(blarg)
    println("stdev: ",std(diff))
    println(diff)
    println(size(diff), size(dS), size(dS12dp))
    txt = @sprintf("Mean: %.3E stdev: %.3E mean(abs(diff/dS12dp)): %.3E, mean(abs(dS12dp/dS)) : %.3E \n",mean(diff),std(diff), mean(abs.(diff ./dS12dp[2, 3, :, :][:])), mean(abs.(dS12dp[2, 3, :, :][:]/dS[i0:end])))
    print(txt)

    #plot(dS[3:end])
    #plot!(blarg[2,3,:])
    #plot(diff)
    return reshape(dS12dp[2, 3, :, :], (Nf, Nf)), reshape(dS[i0:end], (Nf, Nf))
end

#Deriv sum test funcs
function dS1S2sum_combtest(fhash) #Tested for non-iso
    (Nf, )    = size(fhash["filt_index"])
    Nx        = fhash["npix"]
    im = rand(Nx,Nx)
    mywts = rand(Nf + (Nf*Nf))
    println(size(mywts))

    # Use new faster code
    sum1 = Deriv_Utils_New.wst_S1S2_derivsum_comb(im, fhash, mywts)
    wts1 = reshape(mywts[1:Nf], (Nf, 1))
    dwS1 = reshape(Deriv_Utils_New.wst_S1_deriv(im, fhash), (Nx^2, Nf)) * wts1

    # Compare to established code
    dS1S2dp = Deriv_Utils_New.wst_S1S2_deriv(im, fhash)
    dS1dp = dS1S2dp[:, :, 1:Nf]
    dS2dp = dS1S2dp[:, :, Nf+1:end]
    term1 = reshape(dS1dp, (Nx^2, Nf)) * reshape(mywts[1:Nf], (Nf, 1))
    sum2 = zeros(Float64, Nx, Nx)
    for i=1:Nf*Nf sum2 += (dS2dp[:,:,i].*mywts[i+Nf]) end
    dScomb_brute = term1 + reshape(sum2, (Nx^2, 1))

    println("tot abs mean", mean(abs.(dScomb_brute -sum1)))
    println("tot Stdev: ",std(dScomb_brute-sum1))
    println("Ratio: ", mean(dScomb_brute./sum1))
    println("S1")
    println("Abs mean", mean(abs.(term1 - dwS1)))
    println("Abs mean", std(term1 - dwS1))
    println("Ratio: ", mean(term1./dwS1))
    println("S2 separate")
    sum1 = Deriv_Utils_New.wst_S2_deriv_sum(im, fhash, reshape(mywts[Nf+1:end], (Nf, Nf)))
    sum2 = zeros(Float64, Nx, Nx)
    for i=1:Nf*Nf sum2 += (dS2dp[:,:,i].*mywts[Nf+i]) end
    sum2 = reshape(sum2, (Nx^2, 1))
    println("abs mean", mean(abs.(sum2 -sum1)))
    println("Stdev: ",std(sum1-sum2))
    println("Ratio: ", mean(sum2./sum1))
    return
end

function dS20sum_test(fhash) #Adapted from sandbox.jl
    (Nf, )    = size(fhash["filt_index"])
    Nx        = fhash["npix"]
    im = rand(Nx,Nx)
    mywts = rand(Nf*Nf)

    # this is symmetrical, but not because S20 is symmetrical!
    wtgrid = reshape(mywts, Nf, Nf) + reshape(mywts, Nf, Nf)'
    wtvec  = reshape(wtgrid, Nf*Nf)

    # Use new faster code
    sum1 = Deriv_Utils_New.wst_S20_deriv_sum(im, fhash, wtgrid, FFTthreads=1)

    # Compare to established code
    dS20 = reshape(Deriv_Utils_New.wst_S20_deriv(im, fhash, FFTthreads=1),Nx,Nx,Nf*Nf)
    sum2 = zeros(Float64, Nx, Nx)
    for i=1:Nf*Nf sum2 += (dS20[:,:,i].*mywts[i]) end
    sum2 = reshape(sum2, (Nx^2, 1))
    println("Abs mean", mean(abs.(sum2 - sum1)))
    println("Abs mean", std(sum2 - sum1))
    println("Ratio: ", mean(sum2./sum1))
    return
end

function imgreconS2test(Nx, pixmask; norm=norm)
    epsilon=1e-5
    img = readdust(Nx)
    fhash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1, Omega=true)
    #img = (img .- mean(img))./std(img) #normalized
    high = quantile(img[:], 0.75) - quantile(img[:], 0.5)
    println("Added noise scale ", high)
    #Std Normal
    std_10pct = std(img)
    noise = reshape(rand(Normal(0.0, std_10pct), Nx^2), (Nx, Nx))
    init = img+noise
    #Uniform Added Noise
    #noise = rand(Nx, Nx).*(2*high) .- high
    #init = copy(img)#+noise
    #init[1, 2] += 10.0
    #init[23, 5] -= 25.0
    init = imfilter(init, Kernel.gaussian(0.8))

    s2w, s2icov = S2_whitenoiseweights(img, fhash, Nsam=10, loc=0.0, sig=std_10pct) #S2_uniweights(img, fhash, Nsam=10, high=high, iso=false, norm=norm)
    s2targ = DHC_compute(img, fhash, doS2=true, norm=norm, iso=false)
    mask = s2targ .>= epsilon

    mask[1]=false
    mask[2]=false
    println("NF=", size(fhash["filt_index"]), "Sel Coeffs", count((i->(i==true)), mask), size(mask), " Size s2targ, type ", typeof(s2targ[mask]), " Size s2w ", typeof(s2w[mask]))
    #recon_img = Deriv_Utils_New.image_recon_S2derivsum(init, fhash, Float64.(s2targ[mask]), s2icov[mask, mask], pixmask, coeff_mask=mask, optim_settings=Dict([("iterations", 1000), ("norm", norm)]))
    recon_img = Deriv_Utils_New.img_reconfunc_coeffmask(init, fhash, s2targ[mask], s2icov[mask, mask], pixmask, Dict([("iterations", 500)]), coeff_mask=mask)
    pl12 = plot(
        heatmap(img, title="Ground Truth"),
        heatmap(init, title="GT+N(0, std(I))"),
        heatmap(recon_img,title= "Reconstruction w S2"),
        heatmap(recon_img - img, title="Residual");
        layout=4,
    )
    return img, init, recon_img
end


#Calling functions############################################
ds1code, ds1lim = derivtestS1(16)
ds1code, ds1lim, ds2code, ds2lim = derivtestS1S2(16)
ds12code, ds12lim = derivtestS12(16)
#All above work correctly for norm=false

Nx=16
fhash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1)
dS1S2sum_combtest(fhash)

dS20sum_test(fhash)

#Image Recon tests############################################
#S2 with no pixmask, coeff_mask for large selection
img, init, recon = imgreconS2test(32, falses((32, 32)), norm=false)
Visualization.plot_synth_QA(img, init, recon, fink_filter_hash(1, 8, nx=32, pc=1, wd=1, Omega=true), fname="scratch_NM/TestPlots/WhiteNoiseS2tests/100pct_smooth.png")
mean(abs.(init - img)), mean(abs.(recon - img))
#Not working
#Compare all quantities with the equivalentimgrecon code that didnt use derivsum and only uses coeff_mask
#That fails with DHC_compute too. Try DHC_compute_old


#Examining S2w with new dhc-compute
Nx=32
img = readdust(Nx)
fhash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1, Omega=true)
(Nf,) = size(fhash["filt_index"])
s2w, s2icov = S2_weights(img, fhash, 10, iso=false, norm=true)
mask = s2w .>= 1e-5
mask[1]=false
mask[2]=false
println("Max/Min coeffmasked S2w", maximum(s2w[mask]), minimum(s2w[mask]))
println("Max/Min coeffmasked S2icov", maximum(s2icov[mask, mask]), minimum(s2icov[mask, mask]))


#imggt = readdust(32)
##starg = DHC_compute(imggt, fink_filter_hash(1, 8, nx=32, pc=1, wd=1, Omega=true), doS2=true, iso=false, norm=false)
#s2w, s2icov = S2_weights(imggt, fink_filter_hash(1, 8, nx=32, pc=1, wd=1, Omega=true), 10, iso=false, norm=false)
