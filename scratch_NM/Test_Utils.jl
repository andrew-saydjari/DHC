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

function imgreconS2test(Nx, pixmask; norm=norm, sim_smoothed=false, sim_smoothedval=0.8, input_smoothed=false, coeff_choice="S2")
    if coeff_choice=="S2"
        epsilon=1e-5
        img = readdust(Nx)
        fhash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1, Omega=true)
        #img = (img .- mean(img))./std(img) #normalized
        high = quantile(img[:], 0.75) - quantile(img[:], 0.5)
        println("Added noise scale ", high)
        #Std Normal
        std_added = std(img)
        noise = reshape(rand(Normal(0.0, std_added), Nx^2), (Nx, Nx))
        init = img+noise
        #Uniform Added Noise
        #noise = rand(Nx, Nx).*(2*high) .- high
        #init = copy(img)+noise
        #init[1, 2] += 10.0
        #init[23, 5] -= 25.0
        if input_smoothed init = imfilter(init, Kernel.gaussian(0.8)) end
        std_sim = std(img)
        s2w, s2icov = Data_Utils.S2_whitenoiseweights(img, fhash, Nsam=10, loc=0.0, sig=std_sim, smooth=sim_smoothed, smoothval = sim_smoothedval) #S2_uniweights(img, fhash, Nsam=10, high=high, iso=false, norm=norm)
        s2targ = DHC_compute(img, fhash, doS2=true, norm=norm, iso=false)
        mask = s2targ .>= epsilon

        mask[1]=false
        mask[2]=false
        println("NF=", size(fhash["filt_index"]), "Sel Coeffs", count((i->(i==true)), mask), size(mask), " Size s2targ, type ", typeof(s2targ[mask]), " Size s2w ", typeof(s2w[mask]))
        recon_img = Deriv_Utils_New.image_recon_derivsum(init, fhash, Float64.(s2targ[mask]), s2icov[mask, mask], pixmask, "S2", coeff_mask=mask, optim_settings=Dict([("iterations", 1000), ("norm", norm)]))
        #recon_img = Deriv_Utils_New.img_reconfunc_coeffmask(init, fhash, s2targ[mask], s2icov[mask, mask], pixmask, Dict([("iterations", 500)]), coeff_mask=mask)
    else
        if coeff_choice!="S20" error("Not Implemented") end
        img = readdust(Nx)
        fhash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1, Omega=true)
        (Nf,) = size(fhash["filt_index"])
        #img = (img .- mean(img))./std(img) #normalized
        high = quantile(img[:], 0.75) - quantile(img[:], 0.5)
        println("Added noise scale ", high)
        #Std Normal
        std_added = std(img)
        noise = reshape(rand(Normal(0.0, std_added), Nx^2), (Nx, Nx))
        init = img+noise
        #Uniform Added Noise
        #noise = rand(Nx, Nx).*(2*high) .- high
        #init = copy(img)+noise
        #init[1, 2] += 10.0
        #init[23, 5] -= 25.0
        if input_smoothed init = imfilter(init, Kernel.gaussian(0.8)) end
        std_sim = std(img)
        s2w, s2icov = Data_Utils.S20_whitenoiseweights(img, fhash, Nsam=10, loc=0.0, sig=std_sim, smooth=sim_smoothed, smoothval = sim_smoothedval) #S2_uniweights(img, fhash, Nsam=10, high=high, iso=false, norm=norm)
        mask = trues(2+Nf+Nf^2)
        mask[1:2+Nf] .= false
        s2targ = DHC_compute(img, fhash, doS2=false, doS20=true, norm=norm, iso=false)
        println("NF=", size(fhash["filt_index"]))
        recon_img = Deriv_Utils_New.image_recon_derivsum(init, fhash, Float64.(s2targ[mask]), s2icov[mask, mask], pixmask, "S20", coeff_mask=nothing, optim_settings=Dict([("iterations", 1000), ("norm", norm)]))

    end
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
img, init, recon = imgreconS2test(64, falses((64, 64)), norm=false, sim_smoothed=false, input_smoothed=false, coeff_choice="S2")
Visualization.plot_synth_QA(img, init, recon, fink_filter_hash(1, 8, nx=32, pc=1, wd=1, Omega=true), fname="scratch_NM/TestPlots/WhiteNoiseS2tests/S20_didnt_pass_chisq.png")
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
