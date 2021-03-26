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
using StatsBase
using FITSIO
using SparseArrays
using MultivariateStats
using FileIO

push!(LOAD_PATH, pwd()*"/main")
using DHC_2DUtils
#include("Deriv_Utils_New.jl")
#import Deriv_Utils_New
push!(LOAD_PATH, pwd()*"/scratch_NM")
using Deriv_Utils_New
using Data_Utils
using Visualization
using ReconFuncs

##How Different are the dbns of SFD images, SFD+white_noise, SFD+white_noise+smoothing
Nx=128
filter_hash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1, Omega=true)
(Nf, ) = size(filter_hash["filt_index"])
coeff_mask = falses(2+Nf+Nf^2)
coeff_mask[Nf+3:end] .= true
dhc_args = Dict(:doS2=>false, :doS12=>false, :doS20=>true, :apodize=>true) #CHANGE: Change sig for the sfd data since the noise model is super high and the tiny values make sense

#In S20
sfdall = readsfd(Nx)
Num=2000
sfdsub = sfdall[:, :, 1:Num]
oridbn = get_dbn_coeffs(sfdsub, filter_hash, dhc_args, coeff_mask)
noisydbn = zeros(Num, size(oridbn)[2])
smoothdbn1 = zeros(Num, size(oridbn)[2])
noisysub = zeros(Nx, Nx, Num)
smoothsub = zeros(Nx, Nx, Num)
for s=1:Num
  tim = sfdsub[:,:, s]
  noise = reshape(rand(Normal(0.0, std(tim)/5),Nx^2), (Nx, Nx))
  init = tim+noise
  noisysub[:, :, s] .= init
  smoothsub[:, :, s] .= imfilter(init, Kernel.gaussian(1.0))
  noisydbn[s, :] .= DHC_compute_apd(init, filter_hash; dhc_args...)
  smoothdbn1[s, :] .= DHC_compute_apd(imfilter(init, Kernel.gaussian(1.0)), filter_hash; dhc_args...)
end

#sfdvecv = collect(eachslice(sfdsub, dims=3))
#apdsfd = mapslices(apodizer, sfdsub, dims=[1, 2])
#apddbn = get_dbn_coeffs(apdsfd, filter_hash, dhc_args, coeff_mask)
#Skip the below section

##3 dbns
#=
alldbns = vcat(oridbn, noisydbn, apddbn)
M = fit(PCA, alldbns'; maxoutdim=3)
s20_pca = transform(M, alldbns')

p = scatter(s20_pca[1,1:Num],s20_pca[2,1:Num], marker=:circle,linewidth=0,label="Ori")
scatter!(s20_pca[1, (2*Num)+1:end],s20_pca[2, (2*Num)+1:end], marker=:circle,linewidth=0,label="Apd")
#scatter!(s20_pca[1, Num+1:Num*2],s20_pca[2, Num+1:Num*2], marker=:circle,linewidth=0,label="Smoothed Noisy")
scatter!(s20_pca[1, Num+1:Num*2],s20_pca[2, Num+1:Num*2], marker=:circle,linewidth=0,label="Noisy")
#plot!(xlims=(-1e4, 1e3), ylims=(-100, 1e3))
plot!(p,xlabel="PC1",ylabel="PC2")
plot!(title="Reduction computed over all 3 distributions")

#4 dbns
alldbns = vcat(oridbn, apddbn, smoothdbn1, noisydbn)'
M = fit(PCA, alldbns; maxoutdim=3)
s20_pca = transform(M, alldbns)

p = scatter(s20_pca[1,1:Num],s20_pca[2,1:Num], marker=:circle,linewidth=0,label="Ori")
scatter!(s20_pca[1, Num+1:2*Num],s20_pca[2, Num+1:2*Num], marker=:circle,linewidth=0,label="Apd")
scatter!(s20_pca[1, (2*Num)+1:Num*3],s20_pca[2, (2*Num)+1:Num*3], marker=:circle,linewidth=0,label="Smoothed Noisy")
scatter!(s20_pca[1, (3*Num)+1:end],s20_pca[2, (3*Num)+1:end], marker=:circle,linewidth=0,label="Noisy")
#plot!(xlims=(-1e4, 1e3), ylims=(-100, 1e3))
plot!(p,xlabel="PC1",ylabel="PC2")
plot!(p, legend=:bottomleft)
plot!(title="Reduction computed over all 4 distributions")

p=scatter(s20_pca[1, (2*Num)+1:Num*3],s20_pca[2, (2*Num)+1:Num*3], marker=:circle,linewidth=0,label="Smoothed Noisy")
scatter!(s20_pca[1, (3*Num)+1:end],s20_pca[2, (3*Num)+1:end], marker=:circle,linewidth=0,label="Noisy")
#plot!(xlims=(-1e4, 1e3), ylims=(-100, 1e3))
plot!(p,xlabel="PC1",ylabel="PC2")
plot!(p, legend=:bottomleft)
plot!(title="Reduction computed over all 4 distributions: plotting only 2")
=#
##4 dbns
alldbns = vcat(oridbn, smoothdbn1, noisydbn)'

function pca_norm(inputdbn)
  #inputdbn: Rows are features, Cols are observations
  normed = (inputdbn .- mean(inputdbn, dims=2))./std(inputdbn, dims=2)
  M = fit(PCA, normed; maxoutdim=3)
  return mean(inputdbn, dims=2), std(inputdbn, dims=2), M, transform(M, normed)
end

dbn_mean, dbn_std, M, s20_pca = pca_norm(alldbns)

p = scatter(s20_pca[1,1:Num],s20_pca[2,1:Num], marker=:circle,linewidth=0,label="Ori")
scatter!(s20_pca[1, Num+1:2*Num],s20_pca[2, Num+1:2*Num], marker=:circle,linewidth=0,label="Smoothed Noisy")
scatter!(s20_pca[1, (2*Num)+1:Num*3],s20_pca[2, (2*Num)+1:Num*3], marker=:circle,linewidth=0,label="Noisy")
#scatter!(s20_pca[1, (3*Num)+1:end],s20_pca[2, (3*Num)+1:end], marker=:circle,linewidth=0,label="Noisy")
#plot!(xlims=(-1e4, 1e3), ylims=(-100, 1e3))
plot!(p,xlabel="PC1",ylabel="PC2")
plot!(p, legend=:bottomleft)
plot!(title="S20 PCA of apodized images")
plot!(xlims=(-50, 50), ylims=(-50, 50))
savefig(p, "scratch_NM/PCA/sfd_apd_nolog_std20per.png")
#very little variance along PC3
#=
p = scatter(s20_pca[1,1:Num],s20_pca[2,1:Num],s20_pca[3,1:Num], marker=:circle,linewidth=0,label="Ori")
scatter!(s20_pca[1, (2*Num)+1:end],s20_pca[2, (2*Num)+1:end],s20_pca[3, (2*Num)+1:end], marker=:circle,linewidth=0,label="Apd")
scatter!(s20_pca[1, Num+1:Num*2],s20_pca[2, Num+1:Num*2],s20_pca[3, Num+1:Num*2], marker=:circle,linewidth=0,label="Noisy")
plot!(xlims=(-1e4, 1e3), ylims=(-100, 1e3))
plot!(p,xlabel="PC1",ylabel="PC2", zlabel="PC3")
plot!(title="Reduction computed over all 3 distributions")
=#



##S20+Log

ltz = findall(noisysub.<0)
noisysub_bd = copy(noisysub)
noisysub_bd[ltz] .= 1e-6

ltz = findall(smoothsub.<0)
smoothsub_bd = copy(smoothsub)
smoothsub_bd[ltz] .= 1e-6

loridbn = get_dbn_coeffs(log.(sfdsub), filter_hash, dhc_args, coeff_mask)
lnoisydbn = get_dbn_coeffs(log.(noisysub_bd), filter_hash, dhc_args, coeff_mask)
lsmoothdbn1 = get_dbn_coeffs(log.(smoothsub_bd), filter_hash, dhc_args, coeff_mask)


alldbns_log = vcat(loridbn, lsmoothdbn1, lnoisydbn)'
ldbn_mean, ldbn_std, M, ls20_pca = pca_norm(alldbns_log)

p = scatter(ls20_pca[1,1:Num],ls20_pca[2,1:Num], marker=:circle,linewidth=0,label="Ori")
scatter!(ls20_pca[1, Num+1:2*Num],ls20_pca[2, Num+1:2*Num], marker=:circle,linewidth=0,label="Smoothed Noisy")
scatter!(ls20_pca[1, (2*Num)+1:Num*3],ls20_pca[2, (2*Num)+1:Num*3], marker=:circle,linewidth=0,label="Noisy")
#plot!(xlims=(-1e4, 1e3), ylims=(-100, 1e3))
plot!(p,xlabel="PC1",ylabel="PC2")
plot!(p, legend=:bottomleft)
plot!(title="S20 PCA of Apodized Log Images")
plot!(xlims=(-50, 50), ylims=(-50, 50))
savefig(p, "scratch_NM/PCA/sfd_apd_log.png")

##Considering ONLY S20
coeff_mask = falses(2+Nf+Nf^2)
coeff_mask[Nf+3:end] .= true
alldbns20 = alldbns[coeff_mask, :]

dbn_mean, dbn_std, M, s20_pca = pca_norm(alldbns20)
p = scatter(s20_pca[1,1:Num],s20_pca[2,1:Num], marker=:circle,linewidth=0,label="Ori")
scatter!(s20_pca[1, Num+1:2*Num],s20_pca[2, Num+1:2*Num], marker=:circle,linewidth=0,label="Smoothed Noisy")
scatter!(s20_pca[1, (2*Num)+1:Num*3],s20_pca[2, (2*Num)+1:Num*3], marker=:circle,linewidth=0,label="Noisy")
#scatter!(s20_pca[1, (3*Num)+1:end],s20_pca[2, (3*Num)+1:end], marker=:circle,linewidth=0,label="Noisy")
#plot!(xlims=(-1e4, 1e3), ylims=(-100, 1e3))
plot!(p,xlabel="PC1",ylabel="PC2")
plot!(p, legend=:bottomleft)
plot!(title="S20 PCA of apodized images")
plot!(xlims=(-50, 50), ylims=(-50, 50))
savefig(p, "scratch_NM/PCA/sfd_apd_nolog_std20per.png")

#Log
alldbns_logs20 = alldbns_log[coeff_mask, :]
ldbn_mean, ldbn_std, M, ls20_pca = pca_norm(alldbns_logs20)
p = scatter(ls20_pca[1,1:Num],ls20_pca[2,1:Num], marker=:circle,linewidth=0,label="Ori")
scatter!(ls20_pca[1, Num+1:2*Num],ls20_pca[2, Num+1:2*Num], marker=:circle,linewidth=0,label="Smoothed Noisy")
scatter!(ls20_pca[1, (2*Num)+1:Num*3],ls20_pca[2, (2*Num)+1:Num*3], marker=:circle,linewidth=0,label="Noisy")
#plot!(xlims=(-1e4, 1e3), ylims=(-100, 1e3))
plot!(p,xlabel="PC1",ylabel="PC2")
plot!(p, legend=:bottomleft)
plot!(title="S20 PCA of Apodized Log Images")
plot!(xlims=(-50, 50), ylims=(-50, 50))
savefig(p, "scratch_NM/PCA/sfd_apd_log.png")


##Reconstruction
Nx=64
im = readsfd(Nx)
true_img = im[:, :, 1]

noise = rand(Normal(0.0, std(true_img)/5.0), Nx, Nx)
init = imfilter(true_img + noise, Kernel.gaussian(1.0))
#heatmap(init)
#heatmap(true_img)
#PCA
#ltz = findall(init .< 0)
#init_bd = copy(init)
#init_bd[ltz] .= 1e-6

#trues = DHC_compute_apd(log.(true_img), filter_hash; dhc_args...)[coeff_mask]
#inits = DHC_compute_apd(log.(init_bd), filter_hash; dhc_args...)[coeff_mask]

#trues_norm = (trues .- ldbn_mean)./ldbn_std
#inits_norm = (inits .- ldbn_mean)./ldbn_std
#trues_pca = transform(M, trues_norm)
#inits_pca = transform(M, inits_norm)


filter_hash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1, Omega=true)
(Nf, ) = size(filter_hash["filt_index"])
coeff_mask = falses(2+Nf+Nf^2)
coeff_mask[Nf+3:end] .= true

dhc_args = Dict(:doS2=>false, :doS12=>false, :doS20=>true, :apodize=>true) #CHANGE: Change sig for the sfd data since the noise model is super high and the tiny values make sense
white_noise_args = Dict(:loc=>0.0, :sig=>std(true_img)/5.0, :Nsam=>1000, :iso=>false, :norm=>false, :smooth=>true, :smoothval=>1.0) #only if you're using noise based covar
optim_settings = Dict([("iterations", 1000), ("norm", false)])
recon_settings = Dict([("target_type", "ground_truth"), ("covar_type", "white_noise"), ("log", true), ("GaussianLoss", true), ("Invcov_matrix", "Diagonal+Eps"),
  ("optim_settings", optim_settings), ("lambda", std(true_img).^(-2)), ("white_noise_args", white_noise_args)]) #Add constraints
#"white_noise_args": args for white_noise sims if GaussianLoss and Covar_type='white_noise'. :loc...

s2mean, s2icov = meancov_generator(true_img, filter_hash, dhc_args, coeff_mask, recon_settings)
recon_settings["s_targ_mean"] = s2mean
recon_settings["s_invcov"] = s2icov
fname_save = "scratch_NM/PCA/log_apd_gt.jld2"
recon_settings["safemode"] = true
recon_settings["fname_save"] = fname_save
recon_img = reconstruction_wrapper(true_img, init, filter_hash, dhc_args, coeff_mask, recon_settings)


Visualization.plot_diffscales([true_img, init, recon_img, recon_img - true_img], ["GT", "Init", "Reconstruction", "Residual"], fname="scratch_NM/NewWrapper/3-24/kernel1smoothing/sfd_dbntargwhitenoisecov_lam0-1_smoothinit_apd_logspace.png")

println("Mean Abs Res", mean(abs.(init - true_img)), " ", mean(abs.(recon_img - true_img)))

aptrue = apodizer(true_img)
apinit = apodizer(init)
aprecon = apodizer(recon_img)
Visualization.plot_diffscales([apodizer(true_img), apodizer(init), apodizer(recon_img), apodizer(recon_img) - apodizer(true_img)], ["GT", "Init", "Reconstruction", "Residual"], fname="scratch_NM/NewWrapper/3-24/kernel1smoothing/APDsfd_dbntargwhitenoisecov_lam0-1_smoothinit_apd_logspacekern1.png")
Visualization.plot_synth_QA(apodizer(true_img), apodizer(init), apodizer(recon_img), fname="scratch_NM/NewWrapper/3-24/kernel1smoothing/APDsfd_dbntargwhitenoisecov_lam0-1_smoothinit_apd_logspace_kern1_6p.png")
println("Mean Abs Res: Init = ", mean(abs.(apinit - aptrue)), " Recon = ", mean(abs.(aprecon - aptrue)))



##########################################
Nx=64
im = readsfd(Nx)
true_img = im[:, :, 1]

noise = rand(Normal(0.0, std(true_img)/5.0), Nx, Nx)
init = imfilter(true_img + noise, Kernel.gaussian(1.0))
#heatmap(init)
#heatmap(true_img)
#PCA
#ltz = findall(init .< 0)
#init_bd = copy(init)
#init_bd[ltz] .= 1e-6

#trues = DHC_compute_apd(log.(true_img), filter_hash; dhc_args...)[coeff_mask]
#inits = DHC_compute_apd(log.(init_bd), filter_hash; dhc_args...)[coeff_mask]

#trues_norm = (trues .- ldbn_mean)./ldbn_std
#inits_norm = (inits .- ldbn_mean)./ldbn_std
#trues_pca = transform(M, trues_norm)
#inits_pca = transform(M, inits_norm)


filter_hash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1, Omega=true)
(Nf, ) = size(filter_hash["filt_index"])
coeff_mask = falses(2+Nf+Nf^2)
coeff_mask[Nf+3:end] .= true

dhc_args = Dict(:doS2=>false, :doS12=>false, :doS20=>true, :apodize=>true) #CHANGE: Change sig for the sfd data since the noise model is super high and the tiny values make sense
white_noise_args = Dict(:loc=>0.0, :sig=>std(true_img)/5.0, :Nsam=>1000, :iso=>false, :norm=>false, :smooth=>true, :smoothval=>1.0) #only if you're using noise based covar
optim_settings = Dict([("iterations", 1000), ("norm", false)])
recon_settings = Dict([("target_type", "ground_truth"), ("covar_type", "white_noise"), ("log", true), ("GaussianLoss", true), ("Invcov_matrix", "Diagonal+Eps"),
  ("optim_settings", optim_settings), ("lambda", std(true_img).^(-2)), ("white_noise_args", white_noise_args)]) #Add constraints
#"white_noise_args": args for white_noise sims if GaussianLoss and Covar_type='white_noise'. :loc...

s2mean, s2icov = meancov_generator(true_img, filter_hash, dhc_args, coeff_mask, recon_settings)
recon_settings["s_targ_mean"] = s2mean
recon_settings["s_invcov"] = s2icov
fname_save = "scratch_NM/PCA/log_apd_gt.jld2"
recon_settings["safemode"] = true
recon_settings["fname_save"] = fname_save
recon_img = reconstruction_wrapper(true_img, init, filter_hash, dhc_args, coeff_mask, recon_settings)


Visualization.plot_diffscales([true_img, init, recon_img, recon_img - true_img], ["GT", "Init", "Reconstruction", "Residual"], fname="scratch_NM/NewWrapper/3-24/kernel1smoothing/sfd_dbntargwhitenoisecov_lam0-1_smoothinit_apd_logspace.png")

println("Mean Abs Res", mean(abs.(init - true_img)), " ", mean(abs.(recon_img - true_img)))

aptrue = apodizer(true_img)
apinit = apodizer(init)
aprecon = apodizer(recon_img)
Visualization.plot_diffscales([apodizer(true_img), apodizer(init), apodizer(recon_img), apodizer(recon_img) - apodizer(true_img)], ["GT", "Init", "Reconstruction", "Residual"], fname="scratch_NM/NewWrapper/3-24/kernel1smoothing/APDsfd_dbntargwhitenoisecov_lam0-1_smoothinit_apd_logspacekern1.png")
Visualization.plot_synth_QA(apodizer(true_img), apodizer(init), apodizer(recon_img), fname="scratch_NM/NewWrapper/3-24/kernel1smoothing/APDsfd_dbntargwhitenoisecov_lam0-1_smoothinit_apd_logspace_kern1_6p.png")
println("Mean Abs Res: Init = ", mean(abs.(apinit - aptrue)), " Recon = ", mean(abs.(aprecon - aptrue)))
