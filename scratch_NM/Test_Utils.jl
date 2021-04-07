using Statistics
using Plots
using BenchmarkTools
using FFTW
using Statistics
using Optim
using Images, FileIO, ImageIO
using Printf
using Revise
using Profile
using LinearAlgebra
using Distributions
using FITSIO

push!(LOAD_PATH, pwd()*"/main")
using DHC_2DUtils
#include("Deriv_Utils_New.jl")
#import Deriv_Utils_New
push!(LOAD_PATH, pwd()*"/scratch_NM")
using Deriv_Utils_New
using Data_Utils
using Visualization
using LossFuncs
#Remove later

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

function dS20sum_test(fhash) #MODIFIED from sandbox.jl
    (Nf, )    = size(fhash["filt_index"])
    Nx        = fhash["npix"]
    im = rand(Nx,Nx)
    mywts = rand(Nf*Nf)

    # this is symmetrical, but not because S20 is symmetrical!
    wtgrid = reshape(mywts, Nf, Nf)
    wtvec  = reshape(wtgrid, Nf*Nf)

    # Use new faster code
    sum1 = Deriv_Utils_New.wst_S20_deriv_sum(im, fhash, wtgrid, FFTthreads=1)

    # Compare to established code
    dS20 = reshape(Deriv_Utils_New.wst_S20_deriv(im, fhash, FFTthreads=1),Nx,Nx,Nf*Nf)
    sum2 = zeros(Float64, Nx, Nx)
    for i=1:Nf*Nf sum2 += (dS20[:,:,i].*wtvec[i]) end
    sum2 = reshape(sum2, (Nx^2, 1))
    println("Abs mean", mean(abs.(sum2 - sum1)))
    println("Abs mean", std(sum2 - sum1))
    println("Ratio: ", mean(sum2./sum1))
    return
end

function imgreconS2test(Nx, pixmask; norm=norm, bias=0.0, std_added=nothing, std_sim=nothing, sim_smoothed=false, sim_smoothedval=0.8, input_smoothed=false, coeff_choice="S2", invcov="Diagonal", mask=nothing, lambda=nothing)
    #=
    Experiments handled by this wrapper:
    S2 | S12 | S20
    Create covariance using simulated noise + img true (Choice of smoothing or not smoothing)
    Init that is smoothed or not smoothed
    Choice of Diagonal | Diagonal+Eps | Full | Full+Eps:
        +Eps adds an epsilon to the diagonal of the covariance matrix to reduce Cond No
        Diagonal considers just the standard deviation of the simulated weights and Full adds an epsilon
    Pixmask: Keeps certain pixels fixed to true values in the image (Not tested using this wrapper yet). False=> Floating pixels.
    Coeffmask: Of length 2+Nf+Nf^2. Choice of coefficients to be optimized. True=>Optimized
    =#

    img = readdust(Nx)
    if std_added==nothing
        std_added=std(img)
    else

    end
    if std_sim==nothing
        std_sim =std(img)
    else

    end
    noise = reshape(rand(Normal(bias, std_added), Nx^2), (Nx, Nx))
    init = img+noise
    if input_smoothed init = imfilter(init, Kernel.gaussian(0.8)) end

    if coeff_choice=="S2"
        #=
        Default assumes coefficients to be optimized are: All S1 and S2 coefficients larger than 1e-5
        =#
        epsilon=1e-5
        fhash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1, Omega=true)
        (Nf,) = size(fhash["filt_index"])

        #Uniform Added Noise
        #noise = rand(Nx, Nx).*(2*high) .- high
        #init = copy(img)+noise
        #init[1, 2] += 10.0
        #init[23, 5] -= 25.0
        s2targ = DHC_compute(img, fhash, doS2=true, norm=norm, iso=false)
        #Mask
        if mask==nothing
            mask = (s2targ .>= epsilon)
            mask[1]=false
            mask[2]=false
        else
            if size(mask)[1]!=(2+Nf+Nf^2) error("Wrong length mask") end
        end
        sig2, s2cov = Data_Utils.S2_whitenoiseweights(img, fhash, Nsam=10, loc=bias, sig=std_sim, smooth=sim_smoothed, smoothval = sim_smoothedval, coeff_mask=mask) #S2_uniweights(img, fhash, Nsam=10, high=high, iso=false, norm=norm)
        sig2 = sig2[:]
        println("NF=", size(fhash["filt_index"]), "Sel Coeffs", count((i->(i==true)), mask), size(mask), " Size s2targ, type ", typeof(s2targ[mask]), " Size s2w ", typeof(sig2))

        if invcov=="Diagonal"
            s2icov = invert_covmat(sig2)
        elseif invcov=="Diagonal+Eps"
            s2icov = invert_covmat(sig2, 1e-5)
        elseif invcov=="Full+Eps"
            s2icov = invert_covmat(s2cov, 1e-5)
        else#s2icov
            s2icov = invert_covmat(s2cov)
        end
        recon_img = Deriv_Utils_New.image_recon_derivsum(init, fhash, Float64.(s2targ[mask]), s2icov, pixmask, "S2", coeff_mask=mask, optim_settings=Dict([("iterations", 1000), ("norm", norm)]))
        #recon_img = Deriv_Utils_New.img_reconfunc_coeffmask(init, fhash, s2targ[mask], s2icov[mask, mask], pixmask, Dict([("iterations", 500)]), coeff_mask=mask)
    elseif coeff_choice=="S20"
        #=
        Default assumes coefficients to be optimized are: All S20 coefficients
        =#
        fhash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1, Omega=true)
        (Nf,) = size(fhash["filt_index"])
        #img = (img .- mean(img))./std(img) #normalized

        #Uniform Added Noise
        #noise = rand(Nx, Nx).*(2*high) .- high
        #init = copy(img)+noise
        #init[1, 2] += 10.0
        #init[23, 5] -= 25.0
        #Mask
        if mask==nothing
            mask = trues(2+Nf+Nf^2)
            mask[1:2+Nf] .= false
        else
            if size(mask)[1]!=(2+Nf+Nf^2) error("Wrong length mask") end
        end

        s2targ = DHC_compute(img, fhash, doS2=false, doS20=true, norm=norm, iso=false)
        println("NF=", size(fhash["filt_index"]))

        sig2, s2cov = Data_Utils.S20_whitenoiseweights(img, fhash, Nsam=10, loc=bias, sig=std_sim, smooth=sim_smoothed, smoothval = sim_smoothedval, coeff_mask=mask, coeff_choice="S20") #S2_uniweights(img, fhash, Nsam=10, high=high, iso=false, norm=norm)
        sig2 = sig2[:]
        println("NF=", size(fhash["filt_index"]), "Sel Coeffs", count((i->(i==true)), mask), size(mask), " Size s2targ, type ", typeof(s2targ[mask]), " Size s2w ", typeof(sig2))

        if invcov=="Diagonal"
            s2icov = invert_covmat(sig2)
        elseif invcov=="Diagonal+Eps"
            s2icov = invert_covmat(sig2, 1e-10)
        elseif invcov=="Full+Eps"
            s2icov = invert_covmat(s2cov, 1e-10)
        else#s2icov
            s2icov = invert_covmat(s2cov)
        end
        if lambda==nothing
            recon_img = Deriv_Utils_New.image_recon_derivsum(init, fhash, Float64.(s2targ[mask]), s2icov, pixmask, "S20", coeff_mask=mask, optim_settings=Dict([("iterations", 1000), ("norm", norm)]))
        else
            recon_img = Deriv_Utils_New.image_recon_derivsum_regularized(init, fhash, Float64.(s2targ[mask]), s2icov, pixmask, "S20", coeff_mask=mask, optim_settings=Dict([("iterations", 1000), ("norm", norm)]), lambda=0.001)
        end

    else
        if coeff_choice!="S12" error("Not implemented") end
        img = readdust(Nx)
        fhash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1, Omega=true)
        (Nf,) = size(fhash["filt_index"])

        #Uniform Added Noise
        #noise = rand(Nx, Nx).*(2*high) .- high
        #init = copy(img)+noise
        #init[1, 2] += 10.0
        #init[23, 5] -= 25.0
        s2targ = DHC_compute(img, fhash, doS2=false, doS20=false, doS12=true, norm=norm, iso=false)
        #Mask
        if mask==nothing
            mask = (s2targ .>= 1e-5)
            mask[1:Nf+2] .= false
        else

        end
        sig2, s2cov = Data_Utils.S20_whitenoiseweights(img, fhash, Nsam=10, loc=bias, sig=std_sim, smooth=sim_smoothed, smoothval = sim_smoothedval, coeff_choice="S12", coeff_mask=mask) #S2_uniweights(img, fhash, Nsam=10, high=high, iso=false, norm=norm)
        sig2 = sig2[:]
        println("NF=", size(fhash["filt_index"]), "Sel Coeffs", count((i->(i==true)), mask), size(mask), " Size s2targ, type ", typeof(s2targ[mask]), " Size s2w ", typeof(sig2))

        if invcov=="Diagonal"
            s2icov = invert_covmat(sig2)
        elseif invcov=="Diagonal+Eps"
            s2icov = invert_covmat(sig2, 1e-10)
        elseif invcov=="Full+Eps"
            s2icov = invert_covmat(s2cov, 1e-10)
        else#s2icov
            s2icov = invert_covmat(s2cov)
        end
        recon_img = Deriv_Utils_New.image_recon_derivsum(init, fhash, Float64.(s2targ[mask]), s2icov, pixmask, "S12", coeff_mask=mask, optim_settings=Dict([("iterations", 1000), ("norm", norm)]))

    end
    return img, init, recon_img
end


function logimgreconS2test(Nx, pixmask; norm=norm, std_added=nothing, std_sim=nothing, sim_smoothed=false, sim_smoothedval=0.8, input_smoothed=false, coeff_choice="S2", invcov="Diagonal", mask=nothing)
    #=
    Uses the coefficients of the LOG of the image
    Experiments handled by this wrapper:
    S2 | S12 | S20
    Create covariance using simulated noise + img true (Choice of smoothing or not smoothing)
    Init that is smoothed or not smoothed
    Choice of Diagonal | Diagonal+Eps | Full | Full+Eps:
        +Eps adds an epsilon to the diagonal of the covariance matrix to reduce Cond No
        Diagonal considers just the standard deviation of the simulated weights and Full adds an epsilon
    Pixmask: Keeps certain pixels fixed to true values in the image (Not tested using this wrapper yet). False=> Floating pixels.
    Coeffmask: Of length 2+Nf+Nf^2. Choice of coefficients to be optimized. True=>Optimized
    =#
    img = readdust(Nx)
    oriimg = copy(img)
    logimg = log.(img)
    if std_added==nothing
        std_added = std(oriimg)
    else

    end

    epsilon=1e-5
    fhash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1, Omega=true)
    (Nf,) = size(fhash["filt_index"])

    #Std Normal
    noise = reshape(rand(Normal(0.0, std_added), Nx^2), (Nx, Nx))
    init = oriimg+noise

    if std_sim==nothing
        std_sim = std(oriimg)
    else

    end

    if input_smoothed init = imfilter(init, Kernel.gaussian(0.8)) end

    if coeff_choice=="S2"
        s2targ = DHC_compute(logimg, fhash, doS2=true, norm=norm, iso=false)
        #Mask: Default assumes coefficients to be optimized are: All S1 and S2 coefficients larger than 1e-5
        if mask==nothing
            mask = (s2targ .>= epsilon)
            mask[1]=false
            mask[2]=false
        else
            if size(mask)[1]!=(2+Nf+Nf^2) error("Wrong length mask") end
        end
    elseif coeff_choice=="S20"
        s2targ = DHC_compute(logimg, fhash, doS20=true, doS2=false, norm=norm, iso=false)
        #Mask: Default assumes only S20
        if mask==nothing
            mask = trues(2+Nf+Nf^2)
            mask[1:2+Nf] .= false
        else
            if size(mask)[1]!=(2+Nf+Nf^2) error("Wrong length mask") end
        end
    else
        if coeff_choice=="S12"
            s2targ = DHC_compute(logimg, fhash, doS20=false, doS12=true, doS2=false, norm=norm, iso=false)
            #Mask
            if mask==nothing
                mask = (s2targ .>= 1e-5)
                mask[1:Nf+2] .= false
            else
                if size(mask)[1]!=(2+Nf+Nf^2) error("Wrong length mask") end
            end
        end
    end


    sig2, s2cov = whitenoiseweights_forlog(oriimg, fhash, coeff_choice, Nsam=10, loc=0.0, sig=std_sim, smooth=sim_smoothed, smoothval = sim_smoothedval, coeff_mask=mask) #S2_uniweights(img, fhash, Nsam=10, high=high, iso=false, norm=norm)
    sig2 = sig2[:]
    println("NF=", size(fhash["filt_index"]), "Sel Coeffs", count((i->(i==true)), mask), size(mask), " Size s2targ, type ", typeof(s2targ[mask]), " Size s2w ", typeof(sig2))

    if invcov=="Diagonal"
        s2icov = invert_covmat(sig2)
    elseif invcov=="Diagonal+Eps"
        s2icov = invert_covmat(sig2, 1e-5)
    elseif invcov=="Full+Eps"
        s2icov = invert_covmat(s2cov, 1e-5)
    else#s2icov
        s2icov = invert_covmat(s2cov)
    end
    logrecon_img = Deriv_Utils_New.image_recon_derivsum(log.(init), fhash, Float64.(s2targ[mask]), s2icov, pixmask, coeff_choice, coeff_mask=mask, optim_settings=Dict([("iterations", 1000), ("norm", norm)]))
    recon_img = exp.(logrecon_img)
    return oriimg, init, recon_img
end#redundant?

function dbn_experiment(true_img, noisy_init, dbn_covar, pixmask; norm=norm, bias=0.0, std_sim=nothing, sim_smoothed=false, sim_smoothedval=0.8, input_smoothed=false, coeff_choice="S2", invcov="Diagonal", mask=nothing, lambda=nothing)
    #=
    Experiments handled by this wrapper:
    S20
    Regularized vs Non regularized
    Use externally calculated covariance
    Choice of Diagonal | Diagonal+Eps | Full | Full+Eps:
        +Eps adds an epsilon to the diagonal of the covariance matrix to reduce Cond No
        Diagonal considers just the standard deviation of the simulated weights and Full adds an epsilon
    Pixmask: Keeps certain pixels fixed to true values in the image (Not tested using this wrapper yet). False=> Floating pixels.
    Coeffmask: Of length 2+Nf+Nf^2. Choice of coefficients to be optimized. True=>Optimized
    =#

    (Nx, ) = size(true_img)

    #=
    Default assumes coefficients to be optimized are: All S20 coefficients
    =#
    fhash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1, Omega=true)
    (Nf,) = size(fhash["filt_index"])
    #img = (img .- mean(img))./std(img) #normalized

    #Uniform Added Noise
    #noise = rand(Nx, Nx).*(2*high) .- high
    #init = copy(img)+noise
    #init[1, 2] += 10.0
    #init[23, 5] -= 25.0
    #Mask
    if mask==nothing
        mask = trues(2+Nf+Nf^2)
        mask[1:2+Nf] .= false
    else
        if size(mask)[1]!=(2+Nf+Nf^2) error("Wrong length mask") end
    end

    s2targ = DHC_compute(img, fhash, doS2=false, doS20=true, norm=false, iso=false)
    println("NF=", size(fhash["filt_index"]))
    println("NF=", size(fhash["filt_index"]), "Sel Coeffs", count((i->(i==true)), mask), size(mask), " Size s2targ, type ", typeof(s2targ[mask]), " Size s2w ", typeof(sig2))

    if invcov=="Diagonal"
        s2icov = invert_covmat(Diagonal(dbn_covar))
    elseif invcov=="Diagonal+Eps"
        s2icov = invert_covmat(Diagonal(dbn_covar), 1e-10)
    elseif invcov=="Full+Eps"
        s2icov = invert_covmat(dbn_covar, 1e-10)
    else#s2icov
        s2icov = invert_covmat(dbn_covar)
    end
    if lambda==nothing
        recon_img = Deriv_Utils_New.image_recon_derivsum(init, fhash, Float64.(s2targ[mask]), s2icov, pixmask, "S20", coeff_mask=mask, optim_settings=Dict([("iterations", 1000), ("norm", norm)]))
    else
        recon_img = Deriv_Utils_New.image_recon_derivsum_regularized(init, fhash, Float64.(s2targ[mask]), s2icov, pixmask, "S20", coeff_mask=mask, optim_settings=Dict([("iterations", 1000), ("norm", norm)]), lambda=0.001)
    end
    return img, init, recon_img
end#redundant?





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
img, init, recon = imgreconS2test(64, falses((64, 64)), norm=false, sim_smoothed=false, input_smoothed=false, coeff_choice="S2", invcov="Diagonal")
Visualization.plot_synth_QA(img, init, recon, fink_filter_hash(1, 8, nx=32, pc=1, wd=1, Omega=true), fname="scratch_NM/TestPlots/WhiteNoiseS2tests/S20_passed_chisq.png")
mean(abs.(init - img)), mean(abs.(recon - img))
#Not working
#Compare all quantities with the equivalentimgrecon code that didnt use derivsum and only uses coeff_mask
#That fails with DHC_compute too. Try DHC_compute_old


#Examining S2w with new dhc-compute
Nx=32
img = readdust(Nx)
fhash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1, Omega=true)
(Nf,) = size(fhash["filt_index"])
s2w, s2cov = S2_weights(img, fhash, 10, iso=false, norm=true)
mask = s2w .>= 1e-5
mask[1]=false
mask[2]=false
println("Max/Min coeffmasked S2w", maximum(s2w[mask]), minimum(s2w[mask]))
println("Max/Min coeffmasked S2icov", maximum(s2icov[mask, mask]), minimum(s2icov[mask, mask]))

Nx=64
img = readdust(Nx)
fhash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1, Omega=true)
(Nf,) = size(fhash["filt_index"])
s2targ = DHC_compute(img, fhash, doS2=false, doS20=false, doS12=true, norm=false, iso=false)
std_sim=std(img)
s2w, s2icov = Data_Utils.S20_whitenoiseweights(img, fhash, Nsam=10, loc=0.0, sig=std_sim, smooth=false, smoothval =0.8, coeff_choice="S12") #S2_uniweights(img, fhash, Nsam=10, high=high, iso=false, norm=norm)
mask = s2targ .>= 1e-5
mask[1:Nf+2] .= false
println("Nf=", size(fhash["filt_index"]))
println("NumSelCoeffs=", count((i->(i==1)), mask))


#NOTE: #1 Choosing what the covariance matrix should be for different coefficients
#S2
Nx=64
img = readdust(Nx)
fhash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1, Omega=true)
(Nf,) = size(fhash["filt_index"])
s2targ = DHC_compute(img, fhash, doS2=true, doS20=false, doS12=false, norm=false, iso=false)
#Incl only S1, S2 that are larger than 1e-5
mask = s2targ .>= 1e-5
mask[1:2] .= false
std_sim=std(img)
s2w, s2cov = Data_Utils.S2_whitenoiseweights(img, fhash, Nsam=10, loc=0.0, sig=std_sim, smooth=false, smoothval =0.8, coeff_mask=mask) #S2_uniweights(img, fhash, Nsam=10, high=high, iso=false, norm=norm)
Data_Utils.invert_covmat(s2w) #1.41E18, #1.26E-20
Data_Utils.invert_covmat(s2w, 1e-10) #39k, #1.26E-20
Data_Utils.invert_covmat(s2w, 1e-5) #4.92, #1.26E-20 Try this!

Data_Utils.invert_covmat(s2cov) #2e33, #1e6 CAN'T DO THIS!
Data_Utils.invert_covmat(s2cov, 1e-10) #800k
Data_Utils.invert_covmat(s2cov, 1e-5) #9.24, 1e-19 Try this!

#Incl all S1, S2
mask = trues(2+Nf+Nf^2)
mask[1:2] .= false
std_sim=std(img)
s2w, s2cov = Data_Utils.S2_whitenoiseweights(img, fhash, Nsam=10, loc=0.0, sig=std_sim, smooth=false, smoothval =0.8, coeff_mask=mask) #S2_uniweights(img, fhash, Nsam=10, high=high, iso=false, norm=norm)
Data_Utils.invert_covmat(s2w) #1.41E28, #1.26E-20
Data_Utils.invert_covmat(s2w, 1e-5) #6.2, 1-20
Data_Utils.invert_covmat(s2cov) #2e43, 1e11 CAN'T DO THIS!
Data_Utils.invert_covmat(s2cov, 1e-5) #12, 1e-19 Try this!




#S20
Nx=64
img = readdust(Nx)
fhash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1, Omega=true)
(Nf,) = size(fhash["filt_index"])
s2targ = DHC_compute(img, fhash, doS2=false, doS20=true, doS12=false, norm=false, iso=false)
#Including all S20
mask=trues(2+Nf+Nf^2)
mask[1:Nf+2] .= false
std_sim=std(img)
s2w, s2cov = Data_Utils.S20_whitenoiseweights(img, fhash, Nsam=10, loc=0.0, sig=std_sim, smooth=false, smoothval =0.8, coeff_choice="S20", coeff_mask=mask) #S2_uniweights(img, fhash, Nsam=10, high=high, iso=false, norm=norm)
Data_Utils.invert_covmat(s2w) #513, #1.26E-20
Data_Utils.invert_covmat(s2w, 1e-5) #6.7, #1.26E-20 Try this!

Data_Utils.invert_covmat(s2cov) #Breaks
Data_Utils.invert_covmat(s2cov, 1e-10) #5e6, 1e-10
Data_Utils.invert_covmat(s2cov, 1e-5) #55, 1e-17 Choosing this!

#Incl only S20 that are larger than 1e-5
mask = s2targ .>= 1e-5
mask[1:Nf+2] .= false
std_sim=std(img)
s2w, s2cov = Data_Utils.S20_whitenoiseweights(img, fhash, Nsam=10, loc=0.0, sig=std_sim, smooth=false, smoothval =0.8, coeff_choice="S20", coeff_mask=mask) #S2_uniweights(img, fhash, Nsam=10, high=high, iso=false, norm=norm)
Data_Utils.invert_covmat(s2w) #361
Data_Utils.invert_covmat(s2w, 1e-5) #6
Data_Utils.invert_covmat(s2cov) #e7, Breaks
Data_Utils.invert_covmat(s2cov, 1e-5) #51


#S12
Nx=64
img = readdust(Nx)
fhash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1, Omega=true)
(Nf,) = size(fhash["filt_index"])
s2targ = DHC_compute(img, fhash, doS2=false, doS20=false, doS12=true, norm=false, iso=false)
#Including all S12:
mask=trues(2+Nf+Nf^2)
mask[1:Nf+2] .= false
std_sim=std(img)
s2w, s2cov = Data_Utils.S20_whitenoiseweights(img, fhash, Nsam=10, loc=0.0, sig=std_sim, smooth=false, smoothval =0.8, coeff_choice="S12", coeff_mask=mask) #S2_uniweights(img, fhash, Nsam=10, high=high, iso=false, norm=norm)
Data_Utils.invert_covmat(s2w, 1e-5) #e7
Data_Utils.invert_covmat(s2cov, 1e-5) #e8

#Including only S12 larger than some eps
mask = s2targ .>=1e-5
mask[1:Nf+2] .= false
std_sim=std(img)
s2w, s2cov = Data_Utils.S20_whitenoiseweights(img, fhash, Nsam=10, loc=0.0, sig=std_sim, smooth=false, smoothval =0.8, coeff_choice="S12", coeff_mask=mask) #S2_uniweights(img, fhash, Nsam=10, high=high, iso=false, norm=norm)
Data_Utils.invert_covmat(s2w, 1e-5) #e7
Data_Utils.invert_covmat(s2cov, 1e-5) #e8
#appear to be some coefficients which aren't too small O(0.6) but still have 0 std
findall(s2w .<1e-5)
s2targ[mask][findall(s2w .<1e-5)]
#Exclude all below 1e0
mask = s2targ .>=1 #This does not exclude any of the diagonal elements (S1)
mask[1:Nf+2] .= false
std_sim=std(img)
s2w, s2cov = Data_Utils.S20_whitenoiseweights(img, fhash, Nsam=10, loc=0.0, sig=std_sim, smooth=false, smoothval =0.8, coeff_choice="S12", coeff_mask=mask) #S2_uniweights(img, fhash, Nsam=10, high=high, iso=false, norm=norm)
Data_Utils.invert_covmat(s2w, 1e-5) #100k Try this!
Data_Utils.invert_covmat(s2cov, 1e-5) #e8, Num Err=1e-8


#NOTE: Trying all the variants with reasonable condition numbers
#S2: Diagonal, All S1, S2 coeffs greater than eps=1e-5: 22->7 Works well
img, init, recon = imgreconS2test(64, falses((64, 64)), norm=false, sim_smoothed=false, input_smoothed=false, coeff_choice="S2", invcov="Diagonal+Eps")
Visualization.plot_synth_QA(img, init, recon, fname="scratch_NM/TestPlots/CoeffCombinations_WhiteNoise_Nosmooth/S2_Diag_allgreaterthaneps.png")
mean(abs.(init - img)), mean(abs.(recon - img))

#S2: FullCov, All S1, S2 coeffs greater than eps=1e-5: Breaks
img, init, recon = imgreconS2test(64, falses((64, 64)), norm=false, sim_smoothed=false, input_smoothed=false, coeff_choice="S2", invcov="Full+Eps")
Visualization.plot_synth_QA(img, init, recon, fname="scratch_NM/TestPlots/CoeffCombinations_WhiteNoise_Nosmooth/S2_Cov_allgreaterthaneps.png")
mean(abs.(init - img)), mean(abs.(recon - img))


#S2: Incl All S1+S2, adding an eps to the super small coeffs
#Diagonal mat,  22->8 Works well
Nx=64
img = readdust(Nx)
fhash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1, Omega=true)
(Nf,) = size(fhash["filt_index"])
s2targ = DHC_compute(img, fhash, doS2=true, doS20=false, doS12=false, norm=false, iso=false)
mask = trues(2+Nf+Nf^2)
mask[1:2] .= false
img, init, recon = imgreconS2test(64, falses((64, 64)), norm=false, sim_smoothed=false, input_smoothed=false, coeff_choice="S2", invcov="Diagonal+Eps",mask=mask)
Visualization.plot_synth_QA(img, init, recon, fname="scratch_NM/TestPlots/CoeffCombinations_WhiteNoise_Nosmooth/S2_DiagEps_allincl.png")
mean(abs.(init - img)), mean(abs.(recon - img))

#Cov, 22->8 Works well
img, init, recon = imgreconS2test(64, falses((64, 64)), norm=false, sim_smoothed=false, input_smoothed=false, coeff_choice="S2", invcov="Full+Eps",mask=mask)
Visualization.plot_synth_QA(img, init, recon, fname="scratch_NM/TestPlots/CoeffCombinations_WhiteNoise_Nosmooth/S2_CovEps_allincl.png")
mean(abs.(init - img)), mean(abs.(recon - img))


#S20: All S20 only incl: Best so far.
img, init, recon = imgreconS2test(64, falses((64, 64)), norm=false, sim_smoothed=false, input_smoothed=false, coeff_choice="S20", invcov="Diagonal+Eps")
Visualization.plot_synth_QA(img, init, recon, fname="scratch_NM/TestPlots/CoeffCombinations_WhiteNoise_Nosmooth/S20_DiagEps_allincl.png")
mean(abs.(init - img)), mean(abs.(recon - img))

#S20: All S20 only incl: Breaks
img, init, recon = imgreconS2test(64, falses((64, 64)), norm=false, sim_smoothed=false, input_smoothed=false, coeff_choice="S20", invcov="Full+Eps")
Visualization.plot_synth_QA(img, init, recon, fname="scratch_NM/TestPlots/CoeffCombinations_WhiteNoise_Nosmooth/S20_FullEps_allincl.png")
mean(abs.(init - img)), mean(abs.(recon - img))

#S12:

#NOTE: With Smoothing
#S2
#Cov, 22->8 Works well
mask = trues(2+Nf+Nf^2)
mask[1:2] .= false
img, init, recon = imgreconS2test(64, falses((64, 64)), norm=false, sim_smoothed=true, input_smoothed=true, coeff_choice="S2", invcov="Full+Eps",mask=mask)
Visualization.plot_synth_QA(img, init, recon, fname="scratch_NM/TestPlots/CoeffCombinations_WhiteNoise_Smoothed/S2_CovEps_allincl.png")
mean(abs.(init - img)), mean(abs.(recon - img))

#S20: All S20 only incl: Best so far. All S20 incl.
img, init, recon = imgreconS2test(64, falses((64, 64)), norm=false, sim_smoothed=true, input_smoothed=true, coeff_choice="S20", invcov="Diagonal+Eps")
Visualization.plot_synth_QA(img, init, recon, fname="scratch_NM/TestPlots/CoeffCombinations_WhiteNoise_Smoothed/S20_DiagEps_allincl.png")
mean(abs.(init - img)), mean(abs.(recon - img))

#S12: All S12 only incl
Nx=64
img = readdust(Nx)
mean(img)
fhash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1, Omega=true)
(Nf,) = size(fhash["filt_index"])
s12targ = DHC_compute(img, fhash, doS2=false, doS20=false, doS12=true, norm=false, iso=false)
mask = s2targ .>=1 #This does not exclude any of the diagonal elements (S1)
mask[1:Nf+2] .= false
img, init, recon = imgreconS2test(64, falses((64, 64)), norm=false, sim_smoothed=true, input_smoothed=true, coeff_choice="S12", invcov="Diagonal+Eps", mask=mask)
Visualization.plot_synth_QA(img, init, recon, fname="scratch_NM/TestPlots/CoeffCombinations_WhiteNoise_Smoothed/S12_DiagEps_allincl.png")
mean(abs.(init - img)), mean(abs.(recon - img))

#NOTE: With the log of the image's coefficients
img = readdust(64)
img, init, recon = logimgreconS2test(64, falses((64, 64)), norm=false, std_added=std(img)/5.0, std_sim=std(img)/5.0, sim_smoothed=false, input_smoothed=false, coeff_choice="S20", invcov="Diagonal+Eps")
Visualization.plot_synth_QA(img, init, recon, fname="scratch_NM/TestPlots/CoeffCombinations_WhiteNoise_Nosmooth/S20_LogCoeff_DiagEps_allincl.png")
mean(abs.(init - img)), mean(abs.(recon - img))

img, init, recon = imgreconS2test(64, falses((64, 64)), norm=false, sim_smoothed=false, std_added=std(img)/5.0, std_sim=std(img)/5.0, input_smoothed=false, coeff_choice="S20", invcov="Diagonal+Eps")
Visualization.plot_synth_QA(img, init, recon, fname="scratch_NM/TestPlots/CoeffCombinations_WhiteNoise_Nosmooth/S20_DiagEps_allincl_comparison-with-log.png")
mean(abs.(init - img)), mean(abs.(recon - img))

#NOTE: Can it correct a bias?
img = readdust(64)
img, init, recon = imgreconS2test(64, falses((64, 64)), norm=false, bias=mean(img)/10.0, sim_smoothed=false, input_smoothed=false, coeff_choice="S20", invcov="Diagonal+Eps")
mean(abs.(init - img)), mean(abs.(recon - img))
Visualization.plot_diffscales([img, init, recon, recon - img], ["Ground Truth", "Init (N(0.1<I_gt>, std(I_gt))", "Reconstruction", "Reconstruction - Ground Truth"], fname="scratch_NM/TestPlots/CoeffCombinations_WhiteNoise_Nosmooth/S20_DiagEps_allincl_BiasedWhiteNoise")

#NOTE: What effect does adding a regularizing term have?
img, init, recon = imgreconS2test(64, falses((64, 64)), norm=false, sim_smoothed=false, input_smoothed=false, coeff_choice="S20", invcov="Diagonal+Eps")
Visualization.plot_synth_QA(img, init, recon, fname="scratch_NM/3-10_Plots/NoReg_1ksamps.png")
mean(abs.(init - img)), mean(abs.(recon - img))

img, init, recon = imgreconS2test(64, falses((64, 64)), norm=false, sim_smoothed=false, input_smoothed=false, coeff_choice="S20", invcov="Diagonal+Eps", lambda=0.1)
Visualization.plot_synth_QA(img, init, recon, fname="scratch_NM/3-10_Plots/Reg_1ksamps_0-1.png")
mean(abs.(init - img)), mean(abs.(recon - img))


#NOTE: Using the Gaussian covariance of fits
im, covar = mldust(64)
im, covar = mldust(64, logbool=False)
true_img = im[:, :, 1]
Nx=64
noise = rand(Normal(0.0, std(true_img)), Nx, Nx)
init = true_img + noise
filter_hash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1, Omega=true)
Ncovsamp = 10000
(Nf, ) = size(filter_hash["filt_index"])
s20_dbn = zeros(Float64, Ncovsamp, 2+Nf+Nf^2)
for idx=1:Ncovsamp
    s20_dbn[idx, :] = DHC_compute(im[:, :, idx], filter_hash, doS2=false, doS20=true, norm=false, iso=false)
end

s_targ_mean = mean(s20_dbn, dims=1)
scov  = (s20_dbn .- s_targ_mean)' * (s20_dbn .- s_targ_mean) ./(Ncovsamp-1)

sfddbn_experiment(im[:, :, 1], init, scov, falses(Nx, Nx), norm=false,) #complete

mean(im[:, :, 1])
#####################################################################################################
#Using NEW Reconstruction wrapper and SFD dbn

img, covar = mldust(64, logbool=false)


true_img = im[:, :, 1]
Nx=64
noise = rand(Normal(0.0, std(true_img)), Nx, Nx)
init = true_img + noise
filter_hash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1, Omega=true)
(Nf, ) = size(filter_hash["filt_index"])
coeff_mask = falses(2+Nf+Nf^2)
coeff_mask[Nf+3:end] .= true

dhc_args = Dict(:doS2=>false, :doS12=>false, :doS20=>true, :apodize=>false)
white_noise_args = Dict(:loc=>0.0, :sig=>1.0, :Nsam=>1000, :iso=>false, :norm=>false, :smooth=>false, :smoothval=>0.8) #only if you're using noise based covar
optim_settings = Dict([("iterations", 1000), ("norm", false)])
recon_settings = Dict([("target_type", "ground truth"),("covar_type", "white_noise"), ("log", false), ("GaussianLoss", true), ("Invcov_matrix", "Diagonal+Eps"),
    ("Covar_type", "sfd_dbn"), ("apd", "non_apd"), ("optim_settings", optim_settings), ("lambda", (std(true_img).^(-2))./100), ("white_noise_args", white_nois)]) #Add constraints
#"white_noise_args": args for white_noise sims if GaussianLoss and Covar_type='white_noise'. :loc...

s2mean, s2icov = meancov_generator(true_img, filter_hash, dhc_args, coeff_mask, recon_settings)
recon_settings["s_targ_mean"] = s2mean
recon_settings["s_invcov"] = s2icov

recon_img = reconstruction_wrapper(true_img, init, filter_hash, dhc_args, coeff_mask, recon_settings)
#Visualization.plot_synth_QA(true_img, init, recon_img, fname="scratch_NM/NewWrapper/sfd_dbntest.png")
Visualization.plot_diffscales([true_img, init, recon_img, recon_img - true_img], ["GT", "Init", "Reconstruction", "Residual"], fname="scratch_NM/NewWrapper/sfd_dbntest_std-by100.png")


@benchmark dhc = DHC_compute_apd(true_img, filter_hash; Dict(:doS2=>false, :doS12=>false, :doS20=>true, :apodize=>false)...)
@benchmark dhc_apd = DHC_compute_apd(true_img, filter_hash; Dict(:doS2=>false, :doS12=>false, :doS20=>true, :apodize=>true)...)



fname = "scratch_NM/data/dust10000.fits"
f = FITS(fname, "r")
big = read(f[1])
big64 = imresize(big, 64, 64, 10000)
img_old, covar = mldust(64, logbool=false)
mean(big64[:, :, 1])
bigf64 = imresize(Float64.(big), 64, 64, 10000)



(_,__,Nslice) = size(big)
println(Nslice, " slices")
size(big)
minimum(big64)
minimum(im)
heatmap(big[:, :, 1])
heatmap(im[:, :, 1])
##############################################################################################################################

#Using NEW Reconstruction wrapper and SFD dbn

im, covar = mldust(, logbool=false)
true_img = im[:, :, 1]
Nx=64
noise = rand(Normal(0.0, std(true_img)), Nx, Nx)
init = imfilter(true_img + noise, Kernel.gaussian(0.8))
filter_hash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1, Omega=true)
(Nf, ) = size(filter_hash["filt_index"])
coeff_mask = falses(2+Nf+Nf^2)
coeff_mask[Nf+3:end] .= true

dhc_args = Dict(:doS2=>false, :doS12=>false, :doS20=>true, :apodize=>true)
white_noise_args = Dict(:loc=>0.0, :sig=>1.0, :Nsam=>1000, :iso=>false, :norm=>false, :smooth=>false, :smoothval=>0.8) #only if you're using noise based covar
optim_settings = Dict([("iterations", 100), ("norm", false)])
recon_settings = Dict([("target_type", "ground truth"),("covar_type", "white_noise"), ("log", false), ("GaussianLoss", true), ("Invcov_matrix", "Diagonal+Eps"),
 ("apd", "non_apd"), ("optim_settings", optim_settings), ("lambda", 0.0), ("white_noise_args", white_noise_args)]) #Add constraints
#"white_noise_args": args for white_noise sims if GaussianLoss and Covar_type='white_noise'. :loc...

s2mean, s2icov = meancov_generator(true_img, filter_hash, dhc_args, coeff_mask, recon_settings)
recon_settings["s_targ_mean"] = s2mean
recon_settings["s_invcov"] = s2icov

recon_img = reconstruction_wrapper(true_img, init, filter_hash, dhc_args, coeff_mask, recon_settings)
Visualization.plot_diffscales([true_img, init, recon_img, recon_img - true_img], ["GT", "Init", "Reconstruction", "Residual"], fname="scratch_NM/NewWrapper/sfd_gtwhitenoisecov_lam0_smoothinit.png")

#Old image##############################################################################
Nx=64
im = readdust(Nx)
true_img = im

noise = rand(Normal(0.0, std(true_img)), Nx, Nx)
init = imfilter(true_img + noise, Kernel.gaussian(0.8))
filter_hash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1, Omega=true)
(Nf, ) = size(filter_hash["filt_index"])
coeff_mask = falses(2+Nf+Nf^2)
coeff_mask[Nf+3:end] .= true

dhc_args = Dict(:doS2=>false, :doS12=>false, :doS20=>true, :apodize=>false)
white_noise_args = Dict(:loc=>0.0, :sig=>1.0, :Nsam=>1000, :iso=>false, :norm=>false, :smooth=>false, :smoothval=>0.8) #only if you're using noise based covar
optim_settings = Dict([("iterations", 1000), ("norm", false)])
recon_settings = Dict([("target_type", "ground truth"),("covar_type", "white_noise"), ("log", false), ("GaussianLoss", true), ("Invcov_matrix", "Diagonal+Eps"),
 ("apd", "non_apd"), ("optim_settings", optim_settings), ("lambda", 0.0), ("white_noise_args", white_noise_args)]) #Add constraints
#"white_noise_args": args for white_noise sims if GaussianLoss and Covar_type='white_noise'. :loc...

s2mean, s2icov = meancov_generator(true_img, filter_hash, dhc_args, coeff_mask, recon_settings)
recon_settings["s_targ_mean"] = s2mean
recon_settings["s_invcov"] = s2icov

recon_img = reconstruction_wrapper(true_img, init, filter_hash, dhc_args, coeff_mask, recon_settings)
Visualization.plot_diffscales([true_img, init, recon_img, recon_img - true_img], ["GT", "Init", "Reconstruction", "Residual"], fname="scratch_NM/NewWrapper/sfd_gtwhitenoisecov_lam0_smoothinit_apd.png")



##############################################################################################################################
using StatsBase

image = readdust(256)
apdimg_fixed = apodizer(image)
apdimg = apodizer_ori(image, 1, 1, 255)
mean(image)
mean(apdimg_fixed)

heatmap(image)
heatmap(apdimg)

image = readdust(64)
apdimg = apodizer(image, 1, 1, 63)

heatmap(image)
heatmap(apdimg)

big64

apdimg_new = apodizer_new(image)
heatmap(apdimg_new)
mean(image)

mean(apdimg_new)
mean(apdimg)

#Apodization func check
image = readdust(64)
Nx=64
Amat = wind_2d(Nx)
datad_w = fweights(Amat);#why semi-col?#BUG: Replace 256 with im_size? #<A>
meanVal = mean(image,datad_w) #<AF>
temp2d_a = (image.-meanVal).*wind_2d(Nx).+meanVal #<AF>
mean(Amat.* (image .- meanVal)) #E-14


##After modifying dS20 derivsum

filter_hash = fink_filter_hash(1, 8, nx=64, Omega=true)

dS20sum_test(filter_hash)
