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
using Random
using Distributions
using FITSIO
using SparseArrays


push!(LOAD_PATH, pwd()*"/main")
using DHC_2DUtils
#include("Deriv_Utils_New.jl")
#import Deriv_Utils_New
push!(LOAD_PATH, pwd()*"/scratch_NM")
using Deriv_Utils_New
using Data_Utils
using Visualization
using ReconFuncs

Random.seed!(3)

Nx=64
im = readdust(Nx)
true_img = im[:, :, 1]

noise = rand(Normal(0.0, std(true_img)), Nx, Nx)
init = imfilter(true_img + noise, Kernel.gaussian(0.8))
filter_hash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1, Omega=true)
(Nf, ) = size(filter_hash["filt_index"])
coeff_mask = falses(2+Nf+Nf^2)
coeff_mask[Nf+3:end] .= true

dhc_args = Dict(:doS2=>false, :doS12=>false, :doS20=>true, :apodize=>true) #CHANGE: Change sig for the sfd data since the noise model is super high and the tiny values make sense
white_noise_args = Dict(:loc=>0.0, :sig=>1.0, :Nsam=>1000, :iso=>false, :norm=>false, :smooth=>true, :smoothval=>0.8) #only if you're using noise based covar
optim_settings = Dict([("iterations", 20), ("norm", false)])
recon_settings = Dict([("target_type", "ground truth"), ("covar_type", "white_noise"), ("log", false), ("GaussianLoss", true), ("Invcov_matrix", "Diagonal+Eps"),
  ("optim_settings", optim_settings), ("lambda", 0.0), ("white_noise_args", white_noise_args)]) #Add constraints
#"white_noise_args": args for white_noise sims if GaussianLoss and Covar_type='white_noise'. :loc...

s2mean, s2icov = meancov_generator(true_img, filter_hash, dhc_args, coeff_mask, recon_settings)
recon_settings["s_targ_mean"] = s2mean
recon_settings["s_invcov"] = s2icov

recon_img = reconstruction_wrapper(true_img, init, filter_hash, dhc_args, coeff_mask, recon_settings)


Visualization.plot_diffscales([true_img, init, recon_img, recon_img - true_img], ["GT", "Init", "Reconstruction", "Residual"], fname="scratch_NM/NewWrapper/sfd_gtwhitenoisecov_lam0_smoothinit_apd.png")

dhc_args_nonapd = Dict(:doS2=>false, :doS12=>false, :doS20=>true, :apodize=>false)
dhc_args_apd = dhc_args
@benchmark s_apd = DHC_compute_apd(true_img, filter_hash; dhc_args_apd...)
@benchmark s_nonapd = DHC_compute_apd(true_img, filter_hash; dhc_args_nonapd...)



heatmap(true_img)
heatmap(init)

minimum(true_img)


function adaptive_apodizer(input_image::Array{Float64, 2}, dhc_args)#SPEED
        if dhc_args[:apodize]
            apd_image = apodizer(input_image)
        else
            apd_image = input_image
        end
        return apd_image
end

function adaptive_apodizer_fast(input_image::Array{Float64, 2}, dhc_args)#SPEED
        if dhc_args[:apodize]
            input_image .= apodizer(input_image)
        else

        end
        return input_image
end


@benchmark apdval = adaptive_apodizer(true_img, dhc_args_nonapd)
@benchmark apdvalfast = adaptive_apodizer_fast(true_img, dhc_args_nonapd)
#No difference

@benchmark apdimg = apodizer(true_img) #39 μs
@benchmark img = DHC_compute_apd(true_img, filter_hash; dhc_args_nonapd...) #3.475 ms
@benchmark img = DHC_compute_apd(true_img, filter_hash; dhc_args_apd...) #3.595 ms

Profile.clear()
@profile img = DHC_compute_apd(true_img, filter_hash; dhc_args_apd...)
Juno.profiler()

using StatsBase
arr1 = rand(16, 16)
wts = fweights(arr1)

#3/21###########################################################################

function dS20sum_test(im, fhash) #Adapted from sandbox.jl
    (Nf, )    = size(fhash["filt_index"])
    Nx        = fhash["npix"]
    #im = rand(Nx,Nx)
    mywts = rand(Nf*Nf)

    # this is symmetrical, but not because S20 is symmetrical!
    wtgrid = reshape(mywts, Nf, Nf) + reshape(mywts, Nf, Nf)'
    wtvec  = reshape(wtgrid, Nf*Nf)

    # Use new faster code
    sum1 = Deriv_Utils_New.wst_S20_deriv_sum(im, fhash, wtgrid, FFTthreads=1)

    # Compare to established code
    dS20 = reshape(Deriv_Utils_New.wst_S20_deriv(im, fhash, FFTthreads=1),Nx,Nx,Nf*Nf)
    sum2 = zeros(Float64, Nx, Nx)
    for i=1:Nf*Nf sum2 += (dS20[:,:,i].*wtvec[i]) end #THIS NEEDS TO BE WTGRID
    sum2 = reshape(sum2, (Nx^2, 1))
    println("Abs mean", mean(abs.(sum2 - sum1)))
    println("Abs mean", std(sum2 - sum1))
    println("Ratio: ", mean(sum2./sum1))
    return
end

function dapodizer_test(img)
    eps = 1e-4
    im1 = copy(img)
    im0 = copy(img)
    im1[6, 30] += (eps/2)
    im0[6, 30] -= (eps/2)
    brute = ((apodizer(im1) - apodizer(im0))./ eps)[6, 30]
    Apmat = wind_2d(size(img)[1])
    c = sum(Apmat)
    ct = 1.0 + (1.0/c)
    term2 = (Apmat./c)
    Apmat .*= (ones(size(Apmat)).*ct .- term2)
    clever = Apmat[6, 30]
    println("Brute ", brute)
    println("Clever ", clever)
    println("diff= ", clever - brute)
end

function dwtapdtest(img, filter_hash)
    (Nf, ) = size(filter_hash["filt_index"])
    eps = 1e-4
    im1 = copy(img)
    im0 = copy(img)
    im1[6, 30] += (eps/2)
    im0[6, 30] -= (eps/2)
    brute = ((apodizer(im1) - apodizer(im0))./ eps)[6, 30]
    Apmat = wind_2d(size(img)[1])
    c = sum(Apmat)
    ct = 1.0 + (1.0/c)
    term2 = (Apmat./c)
    Apmat .*= (ones(size(Apmat)).*ct .- term2)
    clever = Apmat[6, 30]
    println("Brute ", brute)
    println("Clever ", clever)
    println("diff= ", clever - brute)

    mywts = rand(Nf*Nf)
    # this is symmetrical, but not because S20 is symmetrical!
    wtgrid = reshape(mywts, Nf, Nf) + reshape(mywts, Nf, Nf)'
    wtvec  = reshape(wtgrid, Nf*Nf)

    sum1  = Deriv_Utils_New.wst_S20_deriv_sum(apodizer(img), filter_hash, wtgrid, FFTthreads=1)
    Sjac = reshape(Deriv_Utils_New.wst_S20_deriv(apodizer(img), filter_hash, FFTthreads=1), Nx,Nx,Nf*Nf)
    sum2  = zeros(Float64, Nx, Nx)
    for i=1:Nf*Nf sum2 += (Sjac[:,:,i].*wtvec[i]) end #THIS NEEDS TO BE WTGRID
    sum2 = reshape(sum2, (Nx^2, 1))
    println("Abs mean", mean(abs.(sum2 - sum1)))
    println("Abs mean", std(sum2 - sum1))
    println("Ratio: ", mean(sum2./sum1))
end



dapodizer_test(true_img)
dS20sum_test(adaptive_apodizer(true_img), filter_hash)
dwtapdtest(true_img, filter_hash)

#######DEBUG ZONE##############
tonorm = false
s_targ_mean, s_targ_invcov = s2mean, s2icov
function adaptive_apodizer(input_image::Array{Float64, 2})
    if dhc_args[:apodize]
        apd_image = apodizer(input_image)
    else
        apd_image = input_image
    end
    return apd_image
end
#For the derivative term dA*P/dP
if dhc_args[:apodize]
    #=
    Apmat = wind_2d(Nx)
    c = sum(Apmat)
    ct = 1.0 + (1.0/c)
    term2 = (Apmat./c)
    Apmat = Apmat .* (ct .- term2)
    =#
    Ap = wind_2d(Nx)
    cA = sum(Ap)
    Apflat = reshape(Ap, Nx^2)
    od_nz_idx = findall((x->((x!=0) & (x!=1))), Apflat)
    #tol = 1e-3
    #od_nz_idx2 = findall((x->((abs(x)>tol) & (abs(x-1)>tol))), Apflat)
    avals = Apflat[od_nz_idx]
    dA = zeros(Nx^2, Nx^2)
    dA[od_nz_idx, od_nz_idx] .= ((1.0 .- avals) * avals')./cA
    dA[diagind(dA)] += Apflat

else
    dA = I
end

lambda=0.0
FFTthreads=1
pixmask = falses(size(init))

s_targ_invcov = I
#DEB: input->init
function loss_func20(img_curr::Array{Float64, 2})
    s_curr = DHC_compute_apd(img_curr,  filter_hash, iso=false, norm=tonorm; dhc_args...)[coeff_mask]
    #regterm =  0.5*lambda*sum((adaptive_apodizer(img_curr) - adaptive_apodizer(init)).^2)
    lnlik = ( 0.5 .* (s_curr - s_targ_mean)' * s_targ_invcov * (s_curr - s_targ_mean))
    #println("Lnlik size | Reg Size", size(lnlik), size(regterm))
    neglogloss = lnlik[1] #+ regterm #BUG:??
    return s_curr - s_targ_mean, lnlik, neglogloss
end
#TODO: Need to replace the deriv_sums20 with a deriv_sum wrapper that handles all cases (S12, S20, S2)
function dloss20(storage_grad::Array{Float64, 2}, img_curr::Array{Float64, 2})
    s_curr = DHC_compute_apd(img_curr, filter_hash, norm=tonorm, iso=false; dhc_args...)[coeff_mask]
    diff = s_curr - s_targ_mean
    wt = reshape(convert(Array{Float64, 2}, transpose(diff) * s_targ_invcov), (Nf, Nf))
    apdimg_curr = adaptive_apodizer(img_curr)
    storage_grad .= reshape(Deriv_Utils_New.wst_S20_deriv_sum(apdimg_curr, filter_hash, wt, FFTthreads=FFTthreads)' * dA, (Nx, Nx)) #DEBUG
    #dnll_wrong = reshape(Deriv_Utils_New.wst_S20_deriv_sum(wind_2d(Nx).*img_curr, filter_hash, wt, FFTthreads=FFTthreads), (Nx, Nx)) #DEBUG
    #storage_grad .= Apmat .* (dnll + lambda.*(apdimg_curr - adaptive_apodizer(init)))
    storage_grad[pixmask] .= 0 # better way to do this by taking pixmask as an argument wst_s2_deriv_sum?
    return diff, wt, dA, dnll#, dnll_wrong
end

println("Diff check")
eps = zeros(size(init))
row, col = 24, 18 #convert(Int8, Nx/2), convert(Int8, Nx/2)+3
eps[row, col] = 1e-4
d1, ll1, chisq1 = loss_func20(init+eps./2) #DEB
d0, ll0, chisq0 = loss_func20(init-eps./2) #DEB
d, ll, chisq = loss_func20(init)
brute  = (chisq1-chisq0)/1e-4
#df_brute = DHC_compute(reshape(input, (Nx, Nx)), filter_hash, doS2=true, doS12=false, doS20=false, norm=false)[coeff_mask] - s_targ_mean
clever = zeros(size(init)) #DEB
meanval = mean(adaptive_apodizer(init)) ./ mean(wind_2d(Nx))
ddiff, dwt, dApmat, dnll = dloss20(clever, init) #DEB
println("Chisq Derve Check")
println("Brute:  ",brute)
println("Clever: ",clever[row, col], " Difference: ", brute - clever[row, col], " Mean ", meanval) #DEB

#diffs: same
brute_dnll = ((ll1 - ll0)/1e-4)[1]
dl_dnll = (Apmat .* dnll)[row, col]
brute_dnll - dl_dnll

dl_dnll_wrong = (Apmat .* dnll_wrong)[row, col]
brute_dnll - dl_dnll_wrong

#heatmap(Apmat)
heatmap(Apmat - wind_2d(Nx))


#3-22-23#############################
#Examining dA
Nx=64
Ap = wind_2d(Nx)
cA = sum(Ap)
true_img = readdust(Nx)
filter_hash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1, Omega=true)
Apflat = reshape(Ap, Nx^2)
od_nz_idx = findall((x->((x!=0) & (x!=1))), Apflat)
#tol = 1e-3
#od_nz_idx2 = findall((x->((abs(x)>tol) & (abs(x-1)>tol))), Apflat)
avals = Apflat[od_nz_idx]
dA = zeros(Nx^2, Nx^2)
dA[od_nz_idx, od_nz_idx] .= ((1.0 .- avals) * avals')./cA
dA[diagind(dA)] += Apflat
#dA[od_nz_idx, od_nz_idx][diagind(dA[od_nz_idx, od_nz_idx])] .+= avals
count((x->x!=0), dA) / length(dA)

spdA = sparse(dA)

mywts = reshape(rand(Nf*Nf), (Nf, Nf))
dls20 = Deriv_Utils_New.wst_S20_deriv_sum(true_img, filter_hash, mywts, FFTthreads=FFTthreads)'
@benchmark prod = dls20 * dA #21ms mem: 32KiB, allocs=2
@benchmark prod = dls20 * spdA #4ms mem / allocs same
#Conclusion: Sparse is definitely faster (factor of 5)
dA[od_nz_idx, od_nz_idx]
#check
xi = 12
xj = 500
dA[od_nz_idx, od_nz_idx][xj, xi]
dA[od_nz_idx, od_nz_idx][xi, xj]
dA[od_nz_idx, od_nz_idx][xj, xi]
axi = Apflat[od_nz_idx][xi]
axj = Apflat[od_nz_idx][xj]
dA[od_nz_idx, od_nz_idx][xi, xi] - axi*(1 + (1/cA) -(axi/cA))
dA[od_nz_idx, od_nz_idx][xj, xi] - axi*(1-axj)/cA
dA[od_nz_idx, od_nz_idx][xi, xj] - axj*(1-axi)/cA

#Difference STILL EXISTS
Nx=64
Ap = wind_2d(Nx)
cA = sum(Ap)
true_img = readdust(Nx)
filter_hash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1, Omega=true)
function dwtapdtest_single(img, filter_hash)
    (Nf, ) = size(filter_hash["filt_index"])
    eps = 1e-4
    im1 = copy(img)
    im0 = copy(img)
    row, col = 20, 39
    flat_ind = Nx*(col-1) + row
    im1[row, col] += (eps/2)
    im0[row, col] -= (eps/2)
    brute = ((apodizer(im1) - apodizer(im0))./ eps)#[row, col]
    #=Apmat = wind_2d(size(img)[1])
    c = sum(Apmat)
    ct = 1.0 + (1.0/c)
    term2 = (Apmat./c)
    Apmat .*= (ones(size(Apmat)).*ct .- term2)
    =#
    Ap = wind_2d(Nx)
    cA = sum(Ap)
    Apflat = reshape(Ap, Nx^2)
    od_nz_idx = findall(!iszero, Apflat) #findall((x->((x!=0) & (x!=1))), Apflat)
    #od_zidx = findall(iszero, Apflat)
    avals = Apflat[od_nz_idx]
    dA = zeros(Nx^2, Nx^2)
    dA[:, od_nz_idx] .= ((1.0 .- Apflat) * avals')./cA
    dA[diagind(dA)] += Apflat
    #tol = 1e-3
    #od_nz_idx2 = findall((x->((abs(x)>tol) & (abs(x-1)>tol))), Apflat)
    #avals = Apflat[od_nz_idx]
    #
    #dA[od_nz_idx, od_nz_idx] .= ((1.0 .- avals) * avals')./cA
    println("Check dA[:, xi]")
    clever = reshape(dA[:, flat_ind], (Nx, Nx))
    #println("Brute ", brute)
    #println("Clever ", clever)
    println("Ratio= ", mean(abs.(clever./brute)))
    println("Mean Abs diff= ", mean(abs.(clever - brute)))


    # this is symmetrical, but not because S20 is symmetrical!
    mywts = rand(Nf, Nf)
    wtgrid = reshape(mywts, Nf, Nf) + reshape(mywts, Nf, Nf)'
    #only one coeff
    #wtgrid = zeros(Nf, Nf)
    #wtgrid[1, 1] = 1.0
    wtvec  = reshape(wtgrid, Nf*Nf)

    sum1  = Deriv_Utils_New.wst_S20_deriv_sum(apodizer(img), filter_hash, wtgrid, FFTthreads=1) #BUG: Is apodizer image doing something weird??
    Sjac = reshape(Deriv_Utils_New.wst_S20_deriv(apodizer(img), filter_hash, FFTthreads=1), Nx,Nx,Nf*Nf)
    sum2  = zeros(Float64, Nx, Nx)
    for i=1:Nf*Nf sum2 += (Sjac[:,:,i].*wtvec[i]) end #THIS NEEDS TO BE WTGRID
    sum2 = reshape(sum2, (Nx^2, 1))
    println("Check wst_deriv_sum vs longer route")
    println("Abs mean", mean(abs.(sum2 - sum1)))
    println("Abs mean", std(sum2 - sum1))
    println("Ratio: ", mean(sum2./sum1))

    println("Check deriv of Weighted Sum{S(A*P)}")
    dll = reshape(sum1' * dA, (Nx, Nx))
    coeffmask = falses(2+Nf+Nf^2)
    coeffmask[Nf+3:end] .= true
    Sh = DHC_compute_apd(im1, filter_hash; doS2=false, doS20=true, doS12=false, apodize=true, norm=false, iso=false)[coeffmask]
    Su =  DHC_compute_apd(im0, filter_hash; doS2=false, doS20=true, doS12=false, apodize=true, norm=false, iso=false)[coeffmask]

    dll_brute = ((wtvec' * Sh) - (wtvec' * Su))./eps
    println("Diff", dll[row, col] - dll_brute)
    println("Ratio", dll[row, col]./dll_brute)
    #dll_brute = (Deriv_Utils_New.wst_S20_deriv_sum(apodizer(im1), filter_hash, wtgrid, FFTthreads=1) - Deriv_Utils_New.wst_S20_deriv_sum(apodizer(im0), filter_hash, wtgrid, FFTthreads=1))./eps
    return clever, brute, dll, dll_brute, dA
end

dAclev, dAbrute, dll_cl, dll_brute, dAjac = dwtapdtest_single(true_img, filter_hash)
heatmap(reshape(dll_cl, (Nx, Nx)))
heatmap(reshape(dll_brute, (Nx, Nx)))
heatmap(apodizer(true_img))

row, col=6, 30
flat_ind = Nx*(col-1) + row
println(dA[1, flat_ind])

maximum(dll_cl)
maximum(dll_brute)

#TODO:
#Replace with fixed mu and look at comparison
#Why is the brute force test and clever SO DIFFERENT?
#shifting>????
#not using coeffmask in one place but using it elsewhere??/
#Revisit window func


Ap = wind_2d(Nx)
cA = sum(Ap)
Apflat = reshape(Ap, Nx^2)
od_nz_idx = findall(!iszero, Apflat) #findall((x->((x!=0) & (x!=1))), Apflat)
#od_zidx = findall(iszero, Apflat)
avals = Apflat[od_nz_idx]
dA = zeros(Nx^2, Nx^2)
dA[:, od_nz_idx] .= ((1.0 .- Apflat) * avals')./cA
dA[diagind(dA)] += Apflat

count((x->x!=0), dA) / length(dA)

spdA = sparse(dA)

mywts = reshape(rand(Nf*Nf), (Nf, Nf))
dls20 = Deriv_Utils_New.wst_S20_deriv_sum(true_img, filter_hash, mywts, FFTthreads=1)'
@benchmark prod = dls20 * dA #22ms mem: 32KiB, allocs=2
@benchmark prod = dls20 * spdA #12ms mem / allocs same

#3/24###########################################################################
#test wrapper
Nx=16
fhash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1, Omega=true)
(Nf, ) = size(fhash["filt_index"])
coeff_mask = falses(2+Nf+Nf^2)
coeff_mask[Nf+3:end] .= true
function test_wrapper(Nx, filter_hash, coeff_mask)
    true_img = readdust(Nx)
    swrap = DHC_compute_wrapper(true_img, filter_hash, coeff_mask=coeff_mask)
    sapd = DHC_compute_apd(true_img, filter_hash)[coeff_mask]
    print("Mean Abs Diff ", mean(abs.(swrap - sapd)))
end


test_wrapper(16, fhash, coeff_mask)

#=
filter_hash = fink_filter_hash(1, 8, nx=64, pc=1, wd=1, Omega=true)
Nx=64
true_img = readdust(Nx)
filter_hash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1, Omega=true)
DHC_compute(true_img, filter_hash, filter_hash2=filter_hash)

function test(a, b, c=b; d=3)
    return (a/b) -c + d
end

test(3, 4, 2, d=2)
=#

##3-28: Adding Iso and propagating changes everywhere
Nx=16
im = readsfd(Nx)
true_img = im[:, :, 1]
fhash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1, Omega=true)
siso = DHC_compute(true_img, fhash, iso=true, norm=false, doS2=true)

#S20+NoIsoChecking that you didnt break old code
fname_save = "scratch_NM/NewWrapper/IsoTests/s20_noiso"
Nx=64
im = readsfd(Nx)
true_img = im[:, :, 1]

noise = rand(Normal(0.0, std(true_img)), Nx, Nx)
init = true_img + noise
#heatmap(init)
#heatmap(true_img)

filter_hash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1, Omega=true)
(Nf, ) = size(filter_hash["filt_index"])
coeff_mask = falses(2+Nf+Nf^2)
coeff_mask[Nf+3:end] .= true

dhc_args = Dict(:doS2=>false, :doS12=>false, :doS20=>true, :apodize=>true, :iso=>false) #Iso #CHANGE: Change sig for the sfd data since the noise model is super high and the tiny values make sense
white_noise_args = Dict(:loc=>0.0, :sig=>std(true_img), :Nsam=>1000, :norm=>false, :smooth=>false, :smoothval=>0.8) #Iso #only if you're using noise based covar
optim_settings = Dict([("iterations", 10), ("norm", false), ("minmethod", ConjugateGradient())])
recon_settings = Dict([("target_type", "ground_truth"), ("covar_type", "white_noise"), ("log", false), ("GaussianLoss", true), ("Invcov_matrix", "Diagonal+Eps"),
  ("optim_settings", optim_settings), ("lambda", std(apodizer(true_img)).^(-2)), ("white_noise_args", white_noise_args)]) #Add constraints
regs_true = DHC_compute_wrapper(true_img, filter_hash; norm=false, dhc_args...)[coeff_mask]
s2mean, s2icov = meancov_generator(true_img, filter_hash, dhc_args, coeff_mask, recon_settings, safety=regs_true)
#Check
#s2mean - s_true

recon_settings["s_targ_mean"] = s2mean
recon_settings["s_invcov"] = s2icov
recon_settings["safemode"] = true
recon_settings["fname_save"] = fname_save * ".jld2"
recon_settings["optim_settings"] = optim_settings
resobj, recon_img = reconstruction_wrapper(true_img, init, filter_hash, dhc_args, coeff_mask, recon_settings)

p= plot([t.value for t in Optim.trace(resobj)])
plot!(title="Loss: S20 | Targ = S(GT) | Cov = SFD Covariance", xlabel = "No. Iterations", ylabel = "Loss")
savefig(p, fname_save * "_trace.png")

Visualization.plot_diffscales([apodizer(true_img), apodizer(init), apodizer(recon_img), apodizer(recon_img) - apodizer(true_img)], ["GT", "Init", "Reconstruction", "Residual"], fname=fname_save*".png")
Visualization.plot_synth_QA(apodizer(true_img), apodizer(init), apodizer(recon_img), fname=fname_save*"_6p.png")

println("Mean Abs Res: Init-True = ", mean(abs.(init - true_img)), " Recon-True = ", mean(abs.(recon_img - true_img)))
println("Mean Abs Frac Res", mean(abs.((init - true_img)./true_img)), " Recon-True=", mean(abs.((recon_img - true_img)./true_img)))


#S20 + Iso check
#S20+NoIsoChecking that you didnt break old code
fname_save = "scratch_NM/NewWrapper/IsoTests/s20_iso"
Nx=64
im = readsfd(Nx)
true_img = im[:, :, 1]

noise = rand(Normal(0.0, std(true_img)), Nx, Nx)
init = true_img + noise
#heatmap(init)
#heatmap(true_img)

filter_hash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1, Omega=true)
(S1iso, Nf) = size(filter_hash["S1_iso_mat"])
(S2iso, Nf) = size(filter_hash["S2_iso_mat"])
coeff_mask = falses(2+S1iso+S2iso)
coeff_mask[S1iso+3:end] .= true

dhc_args = Dict(:doS2=>false, :doS12=>false, :doS20=>true, :apodize=>true, :iso=>true) #Iso #CHANGE: Change sig for the sfd data since the noise model is super high and the tiny values make sense
white_noise_args = Dict(:loc=>0.0, :sig=>std(true_img), :Nsam=>1000, :norm=>false, :smooth=>false, :smoothval=>0.8) #Iso #only if you're using noise based covar
optim_settings = Dict([("iterations", 10), ("norm", false), ("minmethod", ConjugateGradient())])
recon_settings = Dict([("target_type", "ground_truth"), ("covar_type", "white_noise"), ("log", false), ("GaussianLoss", true), ("Invcov_matrix", "Diagonal+Eps"),
  ("optim_settings", optim_settings), ("lambda", std(apodizer(true_img)).^(-2)), ("white_noise_args", white_noise_args)]) #Add constraints
regs_true = DHC_compute_wrapper(true_img, filter_hash; norm=false, dhc_args...)[coeff_mask]
s2mean, s2icov = meancov_generator(true_img, filter_hash, dhc_args, coeff_mask, recon_settings, safety=regs_true)
#Check
#s2mean - s_true

recon_settings["s_targ_mean"] = s2mean
recon_settings["s_invcov"] = s2icov
recon_settings["safemode"] = false
recon_settings["fname_save"] = fname_save * ".jld2"
recon_settings["optim_settings"] = optim_settings
resobj, recon_img = reconstruction_wrapper(true_img, init, filter_hash, dhc_args, coeff_mask, recon_settings)
