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
using StatsBase
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

##How does full cov in the log-im case make a difference? vs Diag (Getting the high density and the other artefacts)
fname = "scratch_NM/StandardizedExp/Nx64/noisy_stdtrue/SFDTargSFDCov/log_apd_noiso/1000_triu"
gttarget = load(fname*".jld2")
#Calc 1dps log
Nx = size(gttarget["true_img"])[1]
kbins = collect(1:Nx/2.0)

true_img = gttarget["true_img"]
init = gttarget["init"]
recon_img =gttarget["recon"]


println("Mean Abs Res: Init-True = ", mean(abs.(init - true_img)), " Recon-True = ", mean(abs.(recon_img - true_img)))
println("Mean Abs Frac Res", mean(abs.((init - true_img)./true_img)), " Recon-True=", mean(abs.((recon_img - true_img)./true_img)))
println("RMSE Init-True= ", (mean((init - true_img).^2)).^0.5, " Recon-True=", (mean((recon_img - true_img).^2)).^0.5)

true_lps = calc_1dps(apodizer(log.(gttarget["true_img"])), kbins)
recon_lps = calc_1dps(apodizer(log.(gttarget["recon"])), kbins)
init_lps = calc_1dps(apodizer(log.(gttarget["init"])), kbins)
p = plot(log.(kbins), log.(true_lps), label="True")
plot!(log.(kbins), log.(recon_lps), label="Recon")
plot!(log.(kbins), log.(init_lps), label="Init")
plot!(title="DiagCov Noiso: P(k) of Log Image")
xlabel!("lnk")
ylabel!("lnP(k)")

true_ps = calc_1dps(apodizer(gttarget["true_img"]), kbins)
recon_ps = calc_1dps(apodizer(gttarget["recon"]), kbins)
init_ps = calc_1dps(apodizer(gttarget["init"]), kbins)
p = plot(log.(kbins), log.(true_ps), label="True")
plot!(log.(kbins), log.(recon_ps), label="Recon")
plot!(log.(kbins), log.(init_ps), label="Init")
plot!(title="DiagCov Noiso: P(k) of Image")
xlabel!("lnk")
ylabel!("lnP(k)")


##With the Full Cov:
fname = "scratch_NM/StandardizedExp/Nx64/noisy_stdtrue/SFDTargSFDCov/log_apd_noiso/1000_fullcov_triu"
gttarget = load(fname*".jld2")
#Calc 1dps log
Nx = size(gttarget["true_img"])[1]
kbins = collect(1:Nx/2.0)

true_img = gttarget["true_img"]
init = gttarget["init"]
recon_img =gttarget["recon"]


println("Mean Abs Res: Init-True = ", mean(abs.(init - true_img)), " Recon-True = ", mean(abs.(recon_img - true_img)))
println("Mean Abs Frac Res", mean(abs.((init - true_img)./true_img)), " Recon-True=", mean(abs.((recon_img - true_img)./true_img)))
println("RMSE Init-True= ", (mean((init - true_img).^2)).^0.5, " Recon-True=", (mean((recon_img - true_img).^2)).^0.5)

true_lps = calc_1dps(apodizer(log.(gttarget["true_img"])), kbins)
recon_lps = calc_1dps(apodizer(log.(gttarget["recon"])), kbins)
init_lps = calc_1dps(apodizer(log.(gttarget["init"])), kbins)
p = plot(log.(kbins), log.(true_lps), label="True")
plot!(log.(kbins), log.(recon_lps), label="Recon")
plot!(log.(kbins), log.(init_lps), label="Init")
plot!(title="FullCov Noiso: P(k) of Log Image")
xlabel!("lnk")
ylabel!("lnP(k)")

true_ps = calc_1dps(apodizer(gttarget["true_img"]), kbins)
recon_ps = calc_1dps(apodizer(gttarget["recon"]), kbins)
init_ps = calc_1dps(apodizer(gttarget["init"]), kbins)
p = plot(log.(kbins), log.(true_ps), label="True")
plot!(log.(kbins), log.(recon_ps), label="Recon")
plot!(log.(kbins), log.(init_ps), label="Init")
plot!(title="FullCov Noiso: P(k) of Image")
xlabel!("lnk")
ylabel!("lnP(k)")

##Why Don't Reg plots move away?? Actually it ends up closer to the target than the true image.
#Maybe loss in reg doesn't differentiate noisy from not noisy?
fname = "scratch_NM/StandardizedExp/Nx64/noisy_stdtrue/SFDTargSFDCov/reg_apd_noiso/1000_fullcov_triu"
s_targ_mean, s_targ_invcov = gttarget["dict"]["s_targ_mean"], gttarget["dict"]["s_invcov"]
dhc_args = Dict(:doS2=>false, :doS12=>false, :doS20=>true, :apodize=>true, :iso=>false)

altrue = apodizer(gttarget["true_img"])
alinit = apodizer(gttarget["init"])
alrecon = apodizer(gttarget["recon"])

coeff_mask = gttarget["coeff_mask"]
s_true = DHC_compute_wrapper(gttarget["true_img"], filter_hash, norm=false; dhc_args...)[coeff_mask]
s_init = DHC_compute_wrapper(gttarget["init"], filter_hash, norm=false; dhc_args...)[coeff_mask]
s_recon = DHC_compute_wrapper(gttarget["recon"], filter_hash, norm=false; dhc_args...)[coeff_mask]
ltruecheck = ( 0.5 .* (s_true - s_targ_mean)' * s_targ_invcov * (s_true - s_targ_mean))[1]
l2true = 0.5*gttarget["dict"]["lambda"]*sum((alinit - altrue).^2)
#7.82, 2.32

l1init = ( 0.5 .* (s_init - s_targ_mean)' * s_targ_invcov * (s_init - s_targ_mean))[1]
l2init = 0.5*gttarget["dict"]["lambda"]*sum((alinit - alinit).^2)
#11.2, 0

l1recon = ( 0.5 .* (s_recon - s_targ_mean)' * s_targ_invcov * (s_recon - s_targ_mean))[1]
l2recon = 0.5*gttarget["dict"]["lambda"]*sum((alrecon - alinit).^2)
#7.61, 3.57
ltruecheck + l2true
l1init + l2init
l1recon + l2recon

##Log
fname = "scratch_NM/StandardizedExp/Nx64/noisy_stdtrue/SFDTargSFDCov/log_apd_noiso/1000_fullcov_triu"
s_targ_mean, s_targ_invcov = gttarget["dict"]["s_targ_mean"], gttarget["dict"]["s_invcov"]
dhc_args = Dict(:doS2=>false, :doS12=>false, :doS20=>true, :apodize=>true, :iso=>false)

altrue = apodizer(log.(gttarget["true_img"]))
alinit = apodizer(log.(gttarget["init"]))
alrecon = apodizer(log.(gttarget["recon"]))

coeff_mask = gttarget["coeff_mask"]
s_true = DHC_compute_wrapper(log.(gttarget["true_img"]), filter_hash, norm=false; dhc_args...)[coeff_mask]
s_init = DHC_compute_wrapper(log.(gttarget["init"]), filter_hash, norm=false; dhc_args...)[coeff_mask]
s_recon = DHC_compute_wrapper(log.(gttarget["recon"]), filter_hash, norm=false; dhc_args...)[coeff_mask]
ltruecheck = ( 0.5 .* (s_true - s_targ_mean)' * s_targ_invcov * (s_true - s_targ_mean))[1]
l2true = 0.5*gttarget["dict"]["lambda"]*sum((alinit - altrue).^2)
#446, 84503

l1init = ( 0.5 .* (s_init - s_targ_mean)' * s_targ_invcov * (s_init - s_targ_mean))[1]
l2init = 0.5*gttarget["dict"]["lambda"]*sum((alinit - alinit).^2)
#6e9, 0

l1recon = ( 0.5 .* (s_recon - s_targ_mean)' * s_targ_invcov * (s_recon - s_targ_mean))[1]
l2recon = 0.5*gttarget["dict"]["lambda"]*sum((alrecon - alinit).^2)
#2.5e3, 6e4

ztrue_recon = (s_recon - s_true) .* diag(s_targ_invcov)
ztrue_target = (s_true - s_targ_mean) .* diag(s_targ_invcov)
zinit_target = (s_init - s_targ_mean) .* diag(s_targ_invcov)

scatter(log.(abs.(zinit_target)), log.(abs.(ztrue_recon)))
xlabel!("ln Abs Z score: Coeffs Target-Init")
ylabel!("ln Abs Z score: Coeffs Recon-True")

zscoremat = zeros(34, 34)
zscoremat[reshape(gttarget["coeff_mask"][3+34:end], (34, 34))] .= log.(abs.(ztrue_target))
heatmap(zscoremat, title="Im 1000: Ln Abs Z-Score b/w True & Target")

gttarget["fhash"]


zscoremat
#basically the J=3 scales that are problematic

filename = normpath("scratch_NM/NewWrapper/Weave", "4-7.jmd")
weave(filename, out_path = :pwd)
using Pkg
Pkg.add("IJulia")

using IJulia
IJulia.notebook(dir="/home/nayantara/Desktop/GalacticDustProject/WST/DHC/scratch_NM/NewWrapper/4-7/")


##ANALYZING Z SCORES AND COVARIANCES
#Log coeff, log im
Nx=64
dbnimg = readsfd(Nx, logbool=false)
fname = "scratch_NM/StandardizedExp/Nx64/noisy_stdtrue/SFDTargSFDCov/reg_apd_noiso/LogCoeff/1000_0-01_full_triu"
gttarget = load(fname*".jld2")
dhc_args = gttarget["dhc_args"]
coeff_mask = gttarget["coeff_mask"]
fhash = gttarget["fhash"]
dbn1 = log.(get_dbn_coeffs(dbnimg[:, :, 1:5000], fhash, dhc_args, coeff_mask = coeff_mask))
dbn2 = log.(get_dbn_coeffs(dbnimg[:, :, 5001:end], fhash, dhc_args, coeff_mask= coeff_mask))


#Mean dia, cov
Ncovsamp = 5000
s_targ_mean = mean(dbn1, dims=1)
scov  = (dbn1 .- s_targ_mean)' * (dbn1 .- s_targ_mean) ./(Ncovsamp-1)
smean1, sdiag1, scov1 = s_targ_mean, diag(scov), scov
s_targ_mean = mean(dbn2, dims=1)
scov  = (dbn2 .- s_targ_mean)' * (dbn2 .- s_targ_mean) ./(Ncovsamp-1)
smean2, sdiag2, scov2 = s_targ_mean, diag(scov), scov


scov1inv = invert_covmat(scov1, 1e-8)
scov2inv = invert_covmat(scov2, 1e-8)

diff = dbn1 .- reshape(smean1, (1, 595))
chisq1 = sum((diff * scov1inv) .* diff, dims=2)


diff = dbn2 .- reshape(smean2, (1, 595))
chisq2 = sum((diff * scov2inv) .* diff, dims=2)
zscores = (dbn1 .- s_targ_mean).* reshape(diag(scov1inv).^0.5, (1, 595))
p = histogram(chisq1[:], label="Dbn1")
histogram!(sum(zscores, dims=2)./595, label="Z_Dbn1")
histogram!(chisq2[:], label="Dbn2")
chisqsamp = rand(Distributions.Chisq(595), 5000)
histogram!(chisqsamp, label="ChiSq")




fname = "scratch_NM/StandardizedExp/Nx64/noisy_stdtrue/SFDTargSFDCov/reg_apd_noiso/LogCoeff/1000_0-01_full_triu"
gttarget = load(fname*".jld2")
filter_hash = gttarget["fhash"]
s_targ_mean, s_targ_invcov = gttarget["dict"]["fs_targ_mean"], gttarget["dict"]["fs_invcov"]
dhc_args = Dict(:doS2=>false, :doS12=>false, :doS20=>true, :apodize=>true, :iso=>false)

altrue = apodizer(gttarget["true_img"])
alinit = apodizer(gttarget["init"])
alrecon = apodizer(gttarget["recon"])

coeff_mask = gttarget["coeff_mask"]
s_true = DHC_compute_wrapper(gttarget["true_img"], filter_hash, norm=false; dhc_args...)[coeff_mask])
s_init = DHC_compute_wrapper(gttarget["init"], filter_hash, norm=false; dhc_args...)[coeff_mask]
s_recon = DHC_compute_wrapper(gttarget["recon"], filter_hash, norm=false; dhc_args...)[coeff_mask]

struesq = zeros(34, 34)
struesq[reshape(coeff_mask[37:end], (34, 34))] .= s_true
sinitsq = zeros(34, 34)
sinitsq[reshape(coeff_mask[37:end], (34, 34))] .= s_init
sreconsq = zeros(34, 34)
sreconsq[reshape(coeff_mask[37:end], (34, 34))] .= s_recon
stargsq = zeros(34, 34)
stargsq[reshape(coeff_mask[37:end], (34, 34))] .= exp.(gttarget["dict"]["fs_targ_mean"])


s_true = log.(s_true)
s_init  = log.(s_init)
s_recon = log.(s_recon)
ltruecheck = ( 0.5 .* (s_true - s_targ_mean)' * s_targ_invcov * (s_true - s_targ_mean))[1]
l2true = 0.5*gttarget["dict"]["lambda"]*sum((alinit - altrue).^2)
println("True", ltruecheck)
println(l2true)

l1init = ( 0.5 .* (s_init - s_targ_mean)' * s_targ_invcov * (s_init - s_targ_mean))[1]
l2init = 0.5*gttarget["dict"]["lambda"]*sum((alinit - alinit).^2)
println("Init", l1init)
println(l2init)
println("From true", ( 0.5 .* (s_init - s_true)' * s_targ_invcov * (s_init - s_true))[1])

l1recon = ( 0.5 .* (s_recon - s_targ_mean)' * s_targ_invcov * (s_recon - s_targ_mean))[1]
l2recon = 0.5*gttarget["dict"]["lambda"]*sum((alrecon - alinit).^2)
println("Recon", l1recon)
println(l2recon)
println("From true", ( 0.5 .* (s_recon - s_true)' * s_targ_invcov * (s_recon - s_true))[1])



##Z-Score for mv gaussians test case
covmat = reshape([30, -15, -1, -15, 21, 3, -1, 3, 2], (3, 3))
mvg = Distributions.MultivariateNormal([-100, 3, 0.05], covmat)
samps = rand(mvg, 100)
samean = mean(samps, dims=2)
sacov = (samps .- samean) * (samps .- samean)'
sacov = sacov ./(99)
zscores = (samps .- samean) ./ (diag(sacov).^0.5)
#these do look like z scores


##Are the larger J value coefficients close to the true??
Nx=64
dbnimg = readsfd(Nx, logbool=false)
fname = "scratch_NM/StandardizedExp/Nx64/noisy_stdtrue/SFDTargSFDCov/reg_apd_noiso/LogCoeff/1000_0-01_full_triu"
gttarget = load(fname*".jld2")
dhc_args = gttarget["dhc_args"]
coeff_mask = gttarget["coeff_mask"]
fhash = gttarget["fhash"]
dbn = get_dbn_coeffs(dbnimg, fhash, dhc_args, coeff_mask = nothing)

s_true = DHC_compute_wrapper(gttarget["true_img"], fhash, norm=false; dhc_args...)
s_init = DHC_compute_wrapper(gttarget["init"], fhash, norm=false; dhc_args...)
s_recon = DHC_compute_wrapper(gttarget["recon"], fhash, norm=false; dhc_args...)
ffb, ffinfo = fink_filter_bank(1, 8, nx=64, Omega=true)



function J_hashindices(J_values)
    jindlist = []
    for jval in J_values
        push!(jindlist, findall(fhash["J_L"][:, 1].==jval))
    end
    return vcat(jindlist'...)
end

function J_S1indices(J_values)
    #Assumes this is applied to an object of length 2+Nf+Nf^2 or 2+Nf
    return J_hashindices(J_values) .+ 2
end

function distribution_percentiles(dbn, idx, perc)
    idxperc = zeros(size(idx))
    idxperc .= (x-> percentile(dbn[:, x], perc)).(idx)
    return idxperc
end

Jfiltidx = J_hashindices([0, 1, 2, 3])
JS1ind = J_S1indices([0, 1, 2, 3])
p25 = distribution_percentiles(dbn, JS1ind, 25)
p75 = distribution_percentiles(dbn, JS1ind, 75)
strueselect = (x->s_true[x]).(JS1ind)
sinitselect = (x->s_init[x]).(JS1ind)
sreconselect = (x->s_recon[x]).(JS1ind)
truebool = (strueselect .> p25) .& (strueselect .<p75)
initbool = (sinitselect .> p25) .& (sinitselect .<p75)
reconbool = (sreconselect .> p25) .& (sreconselect .<p75)


frac_devn_init_true = (sinitselect .- strueselect)./strueselect
frac_devn_recon_true = (sreconselect .- strueselect)./strueselect
dbn_mean = mean(dbn, dims=1)[:]
meanselect = (x->dbn_mean[x]).(JS1ind)
frac_devn_sfd_true = (meanselect .- strueselect)./strueselect
frac_devn_75_25 = (p75 .- p25)./p25

dbn_median = distribution_percentiles(dbn, JS1ind, 50)
frac_devn_medsfd_true = (dbn_median .- strueselect)./strueselect
clim = (minimum(frac_devn_medsfd_true), maximum(frac_devn_medsfd_true))
heatmap(frac_devn_init_true, title="Frac Devn Init-True", xlabel="L", ylabel="J", xticks=collect(1:8))
heatmap(frac_devn_sfd_true, title="Frac Devn SFDMean-True", xlabel="L", ylabel="J", xticks=collect(1:8))
heatmap(frac_devn_75_25, title="Frac Devn 75-25", xlabel="L", ylabel="J", xticks=collect(1:8))
heatmap(frac_devn_medsfd_true, title="Frac Devn SFDMedian-True", xlabel="L", ylabel="J", xticks=collect(1:8))
frac_devn_mean_med = (meanselect .- dbn_median)./dbn_median
heatmap(frac_devn_mean_med, title="Frac Devn Mean-Median", xlabel="L", ylabel="J", xticks=collect(1:8))

heatmap(frac_devn_medsfd_true, title="Frac Devn SFDMedian-True", xlabel="L", ylabel="J", xticks=collect(1:8), clim=clim)
heatmap(frac_devn_init_true, title="Frac Devn Init-True", xlabel="L", ylabel="J", xticks=collect(1:8), clim=clim)

##Does mean-subtraction make a difference?
meanim = mean(dbnimg, dims=[1, 2])
dbnimg_ms = dbnimg .- meanim
fname = "scratch_NM/StandardizedExp/Nx64/noisy_stdtrue/SFDTargSFDCov/reg_apd_noiso/LogCoeff/1000_0-01_full_triu"
gttarget = load(fname*".jld2")
dhc_args = gttarget["dhc_args"]
coeff_mask = gttarget["coeff_mask"]
fhash = gttarget["fhash"]
dbn = get_dbn_coeffs(dbnimg_ms, fhash, dhc_args, coeff_mask = nothing)

s_true = DHC_compute_wrapper(gttarget["true_img"] .- mean(gttarget["true_img"]), fhash, norm=false; dhc_args...)
s_init = DHC_compute_wrapper(gttarget["init"] .- mean(gttarget["init"]), fhash, norm=false; dhc_args...)
s_recon = DHC_compute_wrapper(gttarget["recon"] .- mean(gttarget["recon"]), fhash, norm=false; dhc_args...)
ffb, ffinfo = fink_filter_bank(1, 8, nx=64, Omega=true)


Jfiltidx = J_hashindices([0, 1, 2, 3])
JS1ind = J_S1indices([0, 1, 2, 3])
p25 = distribution_percentiles(dbn, JS1ind, 25)
p75 = distribution_percentiles(dbn, JS1ind, 75)
strueselect = (x->s_true[x]).(JS1ind)
sinitselect = (x->s_init[x]).(JS1ind)
sreconselect = (x->s_recon[x]).(JS1ind)
truebool = (strueselect .> p25) .& (strueselect .<p75)
initbool = (sinitselect .> p25) .& (sinitselect .<p75)
reconbool = (sreconselect .> p25) .& (sreconselect .<p75)


frac_devn_init_true = (sinitselect .- strueselect)./strueselect
frac_devn_recon_true = (sreconselect .- strueselect)./strueselect
dbn_mean = mean(dbn, dims=1)[:]
meanselect = (x->dbn_mean[x]).(JS1ind)
frac_devn_sfd_true = (meanselect .- strueselect)./strueselect
frac_devn_75_25 = (p75 .- p25)./p25

dbn_median = distribution_percentiles(dbn, JS1ind, 50)
frac_devn_medsfd_true = (dbn_median .- strueselect)./strueselect
heatmap(frac_devn_init_true, title="Frac Devn Init-True MeanSub", xlabel="L", ylabel="J", xticks=collect(1:8))
heatmap(frac_devn_sfd_true, title="Frac Devn SFDMean-True MeanSub", xlabel="L", ylabel="J", xticks=collect(1:8))
heatmap(frac_devn_75_25, title="Frac Devn 75-25 MeanSub", xlabel="L", ylabel="J", xticks=collect(1:8))
heatmap(frac_devn_medsfd_true, title="Frac Devn SFDMedian-True MeanSub", xlabel="L", ylabel="J", xticks=collect(1:8))
frac_devn_mean_med = (meanselect .- dbn_median)./dbn_median
heatmap(frac_devn_mean_med, title="Frac Devn Mean-Median MeanSub", xlabel="L", ylabel="J", xticks=collect(1:8))


##Making sure the log of coefficients didn't have to do with the results
Nx=64
dbnimg = readsfd(Nx, logbool=false)
fname = "scratch_NM/StandardizedExp/Nx64/noisy_stdtrue/SFDTargSFDCov/reg_apd_noiso/LambdaVary/1000_tunedlam_0-01_fullcov_triu"
gttarget = load(fname*".jld2")
dhc_args = gttarget["dhc_args"]
coeff_mask = gttarget["coeff_mask"]
fhash = gttarget["fhash"]
dbn = get_dbn_coeffs(dbnimg, fhash, dhc_args, coeff_mask = nothing)

s_true = DHC_compute_wrapper(gttarget["true_img"], fhash, norm=false; dhc_args...)
s_init = DHC_compute_wrapper(gttarget["init"], fhash, norm=false; dhc_args...)
s_recon = DHC_compute_wrapper(gttarget["recon"], fhash, norm=false; dhc_args...)
ffb, ffinfo = fink_filter_bank(1, 8, nx=64, Omega=true)



Jfiltidx = J_hashindices([0, 1, 2, 3])
JS1ind = J_S1indices([0, 1, 2, 3])
p25 = distribution_percentiles(dbn, JS1ind, 25)
p75 = distribution_percentiles(dbn, JS1ind, 75)
strueselect = (x->s_true[x]).(JS1ind)
sinitselect = (x->s_init[x]).(JS1ind)
sreconselect = (x->s_recon[x]).(JS1ind)
truebool = (strueselect .> p25) .& (strueselect .<p75)
initbool = (sinitselect .> p25) .& (sinitselect .<p75)
reconbool = (sreconselect .> p25) .& (sreconselect .<p75)


frac_devn_init_true = (sinitselect .- strueselect)./strueselect
frac_devn_recon_true = (sreconselect .- strueselect)./strueselect
dbn_mean = mean(dbn, dims=1)[:]
meanselect = (x->dbn_mean[x]).(JS1ind)
frac_devn_sfd_true = (meanselect .- strueselect)./strueselect
frac_devn_75_25 = (p75 .- p25)./p25

dbn_median = distribution_percentiles(dbn, JS1ind, 50)
frac_devn_medsfd_true = (dbn_median .- strueselect)./strueselect
p= heatmap(frac_devn_init_true, title="Frac Devn Init-True", xlabel="L", ylabel="J", xticks=collect(1:8))
savefig(p, "scratch_NM/NewWrapper/4-10/fdevn-init-true.png")
p=heatmap(frac_devn_sfd_true, title="Frac Devn SFDMean-True", xlabel="L", ylabel="J", xticks=collect(1:8))
savefig(p, "scratch_NM/NewWrapper/4-10/fdevn-sfdmean-true.png")
p= heatmap(frac_devn_75_25, title="Frac Devn 75-25", xlabel="L", ylabel="J", xticks=collect(1:8))
p= heatmap(frac_devn_medsfd_true, title="Frac Devn SFDMedian-True", xlabel="L", ylabel="J", xticks=collect(1:8))
savefig(p, "scratch_NM/NewWrapper/4-10/fdevn-sfdmedian-true.png")
frac_devn_mean_med = (meanselect .- dbn_median)./dbn_median
heatmap(frac_devn_mean_med, title="Frac Devn Mean-Median", xlabel="L", ylabel="J", xticks=collect(1:8))

p99 = distribution_percentiles(dbn, JS1ind, 90)
meanselect .> p99

##WHEN you used the log
fname = "scratch_NM/StandardizedExp/Nx64/noisy_stdtrue/SFDTargSFDCov/reg_apd_noiso/LambdaVary/1000_tunedlam_0-01_fullcov_triu"
gttarget = load(fname*".jld2")
dhc_args = gttarget["dhc_args"]
coeff_mask = gttarget["coeff_mask"]
fhash = gttarget["fhash"]
dbn = log.(get_dbn_coeffs(dbnimg, fhash, dhc_args, coeff_mask = nothing))

s_true = log.(DHC_compute_wrapper(gttarget["true_img"], fhash, norm=false; dhc_args...))
s_init = log.(DHC_compute_wrapper(gttarget["init"], fhash, norm=false; dhc_args...))
s_recon = log.(DHC_compute_wrapper(gttarget["recon"], fhash, norm=false; dhc_args...))
ffb, ffinfo = fink_filter_bank(1, 8, nx=64, Omega=true)



Jfiltidx = J_hashindices([0, 1, 2, 3])
JS1ind = J_S1indices([0, 1, 2, 3])
p25 = distribution_percentiles(dbn, JS1ind, 25)
p75 = distribution_percentiles(dbn, JS1ind, 75)
strueselect = (x->s_true[x]).(JS1ind)
sinitselect = (x->s_init[x]).(JS1ind)
sreconselect = (x->s_recon[x]).(JS1ind)
truebool = (strueselect .> p25) .& (strueselect .<p75)
initbool = (sinitselect .> p25) .& (sinitselect .<p75)
reconbool = (sreconselect .> p25) .& (sreconselect .<p75)


frac_devn_init_true = (sinitselect .- strueselect)./abs.(strueselect)
frac_devn_recon_true = (sreconselect .- strueselect)./abs.(strueselect)
dbn_mean = mean(dbn, dims=1)[:]
meanselect = (x->dbn_mean[x]).(JS1ind)
frac_devn_sfd_true = (meanselect .- strueselect)./abs.(strueselect)
frac_devn_75_25 = (p75 .- p25)./abs.(p25)

dbn_median = distribution_percentiles(dbn, JS1ind, 50)
frac_devn_medsfd_true = (dbn_median .- strueselect)./abs.(strueselect)
p= heatmap(frac_devn_init_true, title="Frac Devn Init-True", xlabel="L", ylabel="J", xticks=collect(1:8))
savefig(p, "scratch_NM/NewWrapper/4-10/LogCoeff/fdevn-init-true.png")
p=heatmap(frac_devn_sfd_true, title="Frac Devn SFDMean-True", xlabel="L", ylabel="J", xticks=collect(1:8))
savefig(p, "scratch_NM/NewWrapper/4-10/LogCoeff/fdevn-sfdmean-true.png")
p= heatmap(frac_devn_75_25, title="Frac Devn 75-25", xlabel="L", ylabel="J", xticks=collect(1:8))
p= heatmap(frac_devn_medsfd_true, title="Frac Devn SFDMedian-True", xlabel="L", ylabel="J", xticks=collect(1:8))
savefig(p, "scratch_NM/NewWrapper/4-10/LogCoeff/fdevn-sfdmedian-true.png")
frac_devn_mean_med = (meanselect .- dbn_median)./abs.(dbn_median)
heatmap(frac_devn_mean_med, title="Frac Devn Mean-Median", xlabel="L", ylabel="J", xticks=collect(1:8))

p99 = distribution_percentiles(dbn, JS1ind, 90)
meanselect .> p99

##Z-scores
Nx=64
dbnimg = readsfd(Nx, logbool=false)
fname = "scratch_NM/StandardizedExp/Nx64/noisy_stdtrue/SFDTargSFDCov/reg_apd_noiso/LambdaVary/1000_tunedlam_0-01_fullcov_triu"
gttarget = load(fname*".jld2")
dhc_args = gttarget["dhc_args"]
coeff_mask = gttarget["coeff_mask"]
fhash = gttarget["fhash"]
dbn = get_dbn_coeffs(dbnimg, fhash, dhc_args, coeff_mask = nothing)

sigdbn = std(dbn, dims=1)

s_true = DHC_compute_wrapper(gttarget["true_img"], fhash, norm=false; dhc_args...)
s_init = DHC_compute_wrapper(gttarget["init"], fhash, norm=false; dhc_args...)
s_recon = DHC_compute_wrapper(gttarget["recon"], fhash, norm=false; dhc_args...)
ffb, ffinfo = fink_filter_bank(1, 8, nx=64, Omega=true)


function J_hashindices(J_values)
    jindlist = []
    for jval in J_values
        push!(jindlist, findall(fhash["J_L"][:, 1].==jval))
    end
    return vcat(jindlist'...)
end

function J_S1indices(J_values)
    #Assumes this is applied to an object of length 2+Nf+Nf^2 or 2+Nf
    return J_hashindices(J_values) .+ 2
end

function zscores(a, b, std, idx)
    return (a[idx] .- b[idx])./std[idx]
end

Jfiltidx = J_hashindices([0, 1, 2, 3])
JS1ind = J_S1indices([0, 1, 2, 3])
strueselect = (x->s_true[x]).(JS1ind)
sinitselect = (x->s_init[x]).(JS1ind)
sreconselect = (x->s_recon[x]).(JS1ind)
stdselect = (x->sigdbn[x]).(JS1ind)


z_init_true = (sinitselect .- strueselect)./stdselect
z_recon_true = (sreconselect .- strueselect)./stdselect
dbn_mean = mean(dbn, dims=1)[:]
meanselect = (x->dbn_mean[x]).(JS1ind)
z_sfdmean_true = (meanselect .- strueselect)./stdselect


function distribution_percentiles(dbn, idx, perc)
    idxperc = zeros(size(idx))
    idxperc .= (x-> percentile(dbn[:, x], perc)).(idx)
    return idxperc
end

dbn_median = distribution_percentiles(dbn, JS1ind, 50)
z_sfdmed_true = (dbn_median .- strueselect)./stdselect

heatmap(z_init_true, title="Z Init-True", xlabel="L", ylabel="J", xticks=collect(1:8))
heatmap(z_sfdmean_true, title="Z SFDMean-True", xlabel="L", ylabel="J", xticks=collect(1:8))
heatmap(z_sfdmed_true, title="Z SFDMedian-True", xlabel="L", ylabel="J", xticks=collect(1:8))
z_mean_med = (meanselect .- dbn_median)./stdselect
heatmap(z_mean_med, title="Z Mean-Median", xlabel="L", ylabel="J", xticks=collect(1:8))

clim = (minimum(z_sfdmed_true), maximum(z_sfdmed_true))
heatmap(z_sfdmed_true, title="Z SFDMedian-True", xlabel="L", ylabel="J", xticks=collect(1:8), clim=clim)
heatmap(z_init_true, title="Z Init-True", xlabel="L", ylabel="J", xticks=collect(1:8), clim=clim)

badz_init_true = z_init_true./stdselect


##Log Coeff-Zscores

logdbn = log.(dbn)
sigdbn = std(logdbn, dims=1)

s_true = log.(DHC_compute_wrapper(gttarget["true_img"], fhash, norm=false; dhc_args...))
s_init = log.(DHC_compute_wrapper(gttarget["init"], fhash, norm=false; dhc_args...))
s_recon = log.(DHC_compute_wrapper(gttarget["recon"], fhash, norm=false; dhc_args...))
ffb, ffinfo = fink_filter_bank(1, 8, nx=64, Omega=true)


Jfiltidx = J_hashindices([0, 1, 2, 3])
JS1ind = J_S1indices([0, 1, 2, 3])
strueselect = (x->s_true[x]).(JS1ind)
sinitselect = (x->s_init[x]).(JS1ind)
sreconselect = (x->s_recon[x]).(JS1ind)
stdselect = (x->sigdbn[x]).(JS1ind)


z_init_true = (sinitselect .- strueselect)./stdselect
z_recon_true = (sreconselect .- strueselect)./stdselect
dbn_mean = mean(logdbn, dims=1)[:]
meanselect = (x->dbn_mean[x]).(JS1ind)
z_sfdmean_true = (meanselect .- strueselect)./stdselect


function distribution_percentiles(dbn, idx, perc)
    idxperc = zeros(size(idx))
    idxperc .= (x-> percentile(dbn[:, x], perc)).(idx)
    return idxperc
end

dbn_median = distribution_percentiles(logdbn, JS1ind, 50)
z_sfdmed_true = (dbn_median .- strueselect)./stdselect

heatmap(z_init_true, title="Z Init-True", xlabel="L", ylabel="J", xticks=collect(1:8))
heatmap(z_sfdmean_true, title="Z SFDMean-True", xlabel="L", ylabel="J", xticks=collect(1:8))
heatmap(z_sfdmed_true, title="Z SFDMedian-True", xlabel="L", ylabel="J", xticks=collect(1:8))
z_mean_med = (meanselect .- dbn_median)./stdselect
heatmap(z_mean_med, title="Z Mean-Median", xlabel="L", ylabel="J", xticks=collect(1:8))

clim = (minimum(z_sfdmed_true), maximum(z_sfdmed_true))
heatmap(z_sfdmed_true, title="Z SFDMedian-True", xlabel="L", ylabel="J", xticks=collect(1:8), clim=clim)
heatmap(z_init_true, title="Z Init-True", xlabel="L", ylabel="J", xticks=collect(1:8), clim=clim)

badz_init_true = z_init_true./stdselect


fnamelog = "scratch_NM/StandardizedExp/Nx64/noisy_stdtrue/SFDTargSFDCov/reg_apd_noiso/LogCoeff/1000_0-01_full_triu"
gtlog = load(fnamelog*".jld2")
s_invcov = gtlog["dict"]["fs_invcov"]
diag(s_invcov)
sigdbn[coeff_mask].^(-2)

##SOME DISCREPANCY BETWEEN THE FSTARG_INVCOV AND THE ONE YOU CALCULATE HERE FOR BOTH S_TARG_INVCOV AND LOGS_TARG_INVCOV:
#Resolved in check_invcov. Numerical instability issues.

Nx=64
dbnimg = readsfd(Nx, logbool=false)
fname = "scratch_NM/StandardizedExp/Nx64/noisy_stdtrue/SFDTargSFDCov/reg_apd_noiso/LambdaVary/1000_tunedlam_0-01_fullcov_triu"
gttarget = load(fname*".jld2")
dhc_args = gttarget["dhc_args"]
coeff_mask = gtlog["coeff_mask"]
fhash = gttarget["fhash"]
dbn = get_dbn_coeffs(dbnimg, fhash, dhc_args, coeff_mask = nothing)

sigdbn = std(dbn, dims=1)
saved = gttarget["dict"]["s_invcov"]
diag(saved)
sigdbn[coeff_mask].^(-2)
(dbn[:, coeff_mask]' * dbn[:, coeff_mask]) ./9999 #roughly consistent with sigdbn.^2

#INCONSISTENT WITH DIAG(SAVED)
datfile = "scratch_NM/StandardizedExp/Nx64/Data_10000.jld2" #Replace w SLURM array
loaddf = load(datfile)
true_img = loaddf["true_img"]
init = loaddf["init"]

filter_hash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1, Omega=true)
(S1iso, Nf) = size(filter_hash["S1_iso_mat"])
(S2iso, Nfsq) = size(filter_hash["S2_iso_mat"])

dhc_args = Dict(:doS2=>false, :doS12=>false, :doS20=>true, :apodize=>true, :iso=>false) #Iso #CHANGE: Change sig for the sfd data since the noise model is super high and the tiny values make sense
if dhc_args[:iso]
    coeff_mask = falses(2+S1iso+S2iso)
    coeff_mask[S1iso+3:end] .= true
else #Not iso
    coeff_mask = falses(2+Nf+Nf^2)
    coeff_mask[Nf+3:end] .= true
end

white_noise_args = Dict(:loc=>0.0, :sig=>std(true_img), :Nsam=>1000, :norm=>false, :smooth=>false, :smoothval=>0.8) #Iso #only if you're using noise based covar
optim_settings = Dict([("iterations", 1000), ("norm", false), ("minmethod", ConjugateGradient())])
recon_settings = Dict([("target_type", "sfd_dbn"), ("covar_type", "sfd_dbn"), ("log", false), ("GaussianLoss", true), ("Invcov_matrix", "Diagonal+Eps"),
  ("optim_settings", optim_settings), ("white_noise_args", white_noise_args)]) #Add constraints


if recon_settings["log"] & dhc_args[:apodize]
    lval = std(apodizer(log.(true_img))).^(-2)
elseif !recon_settings["log"] & dhc_args[:apodize]
    lval = std(apodizer(true_img)).^(-2)
elseif recon_settings["log"] & !dhc_args[:apodize]
    lval = std(log.(true_img)).^(-2)
else
    lval = std(true_img).^(-2)
end

recon_settings["lambda"] = lval
coeff_mask = gtlog["coeff_mask"]
regs_true = DHC_compute_wrapper(log.(true_img), filter_hash; norm=false, dhc_args...)[coeff_mask]
s2mean, s2icov = meancov_generator(true_img, filter_hash, dhc_args, coeff_mask, recon_settings, safety=nothing)
diag(s2icov)
##THIS Is consistent but the saved s_targ isn't???? HOW
