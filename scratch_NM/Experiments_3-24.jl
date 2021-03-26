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
white_noise_args = Dict(:loc=>0.0, :sig=>std(true_img), :Nsam=>1000, :iso=>false, :norm=>false, :smooth=>true, :smoothval=>0.8) #only if you're using noise based covar
optim_settings = Dict([("iterations", 20), ("norm", false)])
recon_settings = Dict([("target_type", "ground truth"), ("covar_type", "white_noise"), ("log", false), ("GaussianLoss", true), ("Invcov_matrix", "Diagonal+Eps"),
  ("optim_settings", optim_settings), ("lambda", 0.0), ("white_noise_args", white_noise_args)]) #Add constraints
#"white_noise_args": args for white_noise sims if GaussianLoss and Covar_type='white_noise'. :loc...

s2mean, s2icov = meancov_generator(true_img, filter_hash, dhc_args, coeff_mask, recon_settings)
recon_settings["s_targ_mean"] = s2mean
recon_settings["s_invcov"] = s2icov

recon_img = reconstruction_wrapper(true_img, init, filter_hash, dhc_args, coeff_mask, recon_settings)
#################################

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
white_noise_args = Dict(:loc=>0.0, :sig=>std(true_img), :Nsam=>1000, :iso=>false, :norm=>false, :smooth=>true, :smoothval=>0.8) #only if you're using noise based covar
optim_settings = Dict([("iterations", 20), ("norm", false)])
recon_settings = Dict([("target_type", "ground truth"), ("covar_type", "white_noise"), ("log", false), ("GaussianLoss", true), ("Invcov_matrix", "Diagonal+Eps"),
  ("optim_settings", optim_settings), ("lambda", 0.1), ("white_noise_args", white_noise_args)]) #Add constraints
#"white_noise_args": args for white_noise sims if GaussianLoss and Covar_type='white_noise'. :loc...

s2mean, s2icov = meancov_generator(true_img, filter_hash, dhc_args, coeff_mask, recon_settings)
recon_settings["s_targ_mean"] = s2mean
recon_settings["s_invcov"] = s2icov

recon_img = reconstruction_wrapper(true_img, init, filter_hash, dhc_args, coeff_mask, recon_settings)
#################################

Nx=64
im = readsfd(Nx)
true_img = im[:, :, 1]

noise = rand(Normal(0.0, std(true_img)), Nx, Nx)
init = imfilter(true_img + noise, Kernel.gaussian(0.8))
filter_hash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1, Omega=true)
(Nf, ) = size(filter_hash["filt_index"])
coeff_mask = falses(2+Nf+Nf^2)
coeff_mask[Nf+3:end] .= true

dhc_args = Dict(:doS2=>false, :doS12=>false, :doS20=>true, :apodize=>true) #CHANGE: Change sig for the sfd data since the noise model is super high and the tiny values make sense
white_noise_args = Dict(:loc=>0.0, :sig=>std(true_img), :Nsam=>1000, :iso=>false, :norm=>false, :smooth=>true, :smoothval=>0.8) #only if you're using noise based covar
optim_settings = Dict([("iterations", 20), ("norm", false)])
recon_settings = Dict([("target_type", "ground truth"), ("covar_type", "white_noise"), ("log", false), ("GaussianLoss", true), ("Invcov_matrix", "Diagonal+Eps"),
  ("optim_settings", optim_settings), ("lambda", 0.1), ("white_noise_args", white_noise_args)]) #Add constraints
#"white_noise_args": args for white_noise sims if GaussianLoss and Covar_type='white_noise'. :loc...

s2mean, s2icov = meancov_generator(true_img, filter_hash, dhc_args, coeff_mask, recon_settings)
recon_settings["s_targ_mean"] = s2mean
recon_settings["s_invcov"] = s2icov

recon_img = reconstruction_wrapper(true_img, init, filter_hash, dhc_args, coeff_mask, recon_settings)

########################
#Weird behavior here: just memory issues -- works afetr closing window and reopening

Nx=64
im = readsfd(Nx)
true_img = im[:, :, 1]

noise = rand(Normal(0.0, std(true_img)), Nx, Nx)
init = imfilter(true_img + noise, Kernel.gaussian(1.0))
heatmap(init)
#heatmap(true_img)

filter_hash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1, Omega=true)
(Nf, ) = size(filter_hash["filt_index"])
coeff_mask = falses(2+Nf+Nf^2)
coeff_mask[Nf+3:end] .= true

dhc_args = Dict(:doS2=>false, :doS12=>false, :doS20=>true, :apodize=>true) #CHANGE: Change sig for the sfd data since the noise model is super high and the tiny values make sense
white_noise_args = Dict(:loc=>0.0, :sig=>std(true_img), :Nsam=>1000, :iso=>false, :norm=>false, :smooth=>true, :smoothval=>0.8) #only if you're using noise based covar
optim_settings = Dict([("iterations", 1000), ("norm", false)])
recon_settings = Dict([("target_type", "sfd_dbn"), ("covar_type", "sfd_dbn"), ("log", true), ("GaussianLoss", true), ("Invcov_matrix", "Diagonal+Eps"),
  ("optim_settings", optim_settings), ("lambda", std(true_img).^(-2)), ("white_noise_args", white_noise_args)]) #Add constraints
#"white_noise_args": args for white_noise sims if GaussianLoss and Covar_type='white_noise'. :loc...

s2mean, s2icov = meancov_generator(true_img, filter_hash, dhc_args, coeff_mask, recon_settings)
recon_settings["s_targ_mean"] = s2mean
recon_settings["s_invcov"] = s2icov
fname_save = "scratch_NM/NewWrapper/3-24/Exp1.jld2"
recon_settings["safemode"] = true
recon_settings["fname_save"] = fname_save
recon_img = reconstruction_wrapper(true_img, init, filter_hash, dhc_args, coeff_mask, recon_settings)
Visualization.plot_diffscales([true_img, init, recon_img, recon_img - true_img], ["GT", "Init", "Reconstruction", "Residual"], fname="scratch_NM/PCA/sfd_dbntargwhitenoisecov_lam_imglevel_smoothinit_apd_logspace.png")

println("Mean Abs Res", mean(abs.(init - true_img)), " ", mean(abs.(recon_img - true_img)))

aptrue = apodizer(true_img)
apinit = apodizer(init)
aprecon = apodizer(recon_img)
Visualization.plot_diffscales([apodizer(true_img), apodizer(init), apodizer(recon_img), apodizer(recon_img) - apodizer(true_img)], ["GT", "Init", "Reconstruction", "Residual"], fname="scratch_NM/PCA/sfd_dbntargwhitenoisecov_lam_imglevel_smoothinit_apd_logspacekern1.png")
Visualization.plot_synth_QA(apodizer(true_img), apodizer(init), apodizer(recon_img), fname="scratch_NM/PCA/sfd_dbntargwhitenoisecov_lam_imglevel_smoothinit_apd_logspace_6p.png")
println("Mean Abs Res: Init = ", mean(abs.(apinit - aptrue)), " Recon = ", mean(abs.(aprecon - aptrue)))


##3-35###################################
#Testing JLD
using JLD2
@save "scratch_NM/NewWrapper/3-24/test.jld2" true_img init recon_img recon_settings


@load "scratch_NM/NewWrapper/3-24/test.jld2" load_gt=true_img load_init=init load_recon=recon_img load_dict=recon_settings

using FileIO

save("scratch_NM/NewWrapper/3-24/test_fiolarge.jld2", Dict("true_img"=>true_img, "init"=>init, "recon"=>recon_img, "dict"=>recon_settings))

load("scratch_NM/")



##3-26###############################################################################################################
Nx=64
im = readsfd(Nx)
true_img = im[:, :, 1]

noise = rand(Normal(0.0, std(true_img)), Nx, Nx)
init = imfilter(true_img + noise, Kernel.gaussian(1.0))
heatmap(init)
#heatmap(true_img)

filter_hash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1, Omega=true)
(Nf, ) = size(filter_hash["filt_index"])
coeff_mask = falses(2+Nf+Nf^2)
coeff_mask[Nf+3:end] .= true

dhc_args = Dict(:doS2=>false, :doS12=>false, :doS20=>true, :apodize=>false) #CHANGE: Change sig for the sfd data since the noise model is super high and the tiny values make sense
white_noise_args = Dict(:loc=>0.0, :sig=>std(true_img), :Nsam=>1000, :iso=>false, :norm=>false, :smooth=>true, :smoothval=>0.8) #only if you're using noise based covar
optim_settings = Dict([("iterations", 1000), ("norm", false)])
recon_settings = Dict([("target_type", "sfd_dbn"), ("covar_type", "sfd_dbn"), ("log", true), ("GaussianLoss", true), ("Invcov_matrix", "Diagonal+Eps"),
  ("optim_settings", optim_settings), ("lambda", std(true_img).^(-2)), ("white_noise_args", white_noise_args)]) #Add constraints
#"white_noise_args": args for white_noise sims if GaussianLoss and Covar_type='white_noise'. :loc...

s2mean, s2icov = meancov_generator(true_img, filter_hash, dhc_args, coeff_mask, recon_settings)
recon_settings["s_targ_mean"] = s2mean
recon_settings["s_invcov"] = s2icov
fname_save = "scratch_NM/NewWrapper/3-24/Exp1.jld2"
recon_settings["safemode"] = true
recon_settings["fname_save"] = fname_save
recon_img = reconstruction_wrapper(true_img, init, filter_hash, dhc_args, coeff_mask, recon_settings)
#Stops after 5 iterations

#Without apodization | SFD | SFDCovar
Nx=64
im = readsfd(Nx)
true_img = im[:, :, 1]

noise = rand(Normal(0.0, std(true_img)), Nx, Nx)
init = imfilter(true_img + noise, Kernel.gaussian(1.0))
heatmap(init)
#heatmap(true_img)

filter_hash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1, Omega=true)
(Nf, ) = size(filter_hash["filt_index"])
coeff_mask = falses(2+Nf+Nf^2)
coeff_mask[Nf+3:end] .= true

dhc_args = Dict(:doS2=>false, :doS12=>false, :doS20=>true, :apodize=>false) #CHANGE: Change sig for the sfd data since the noise model is super high and the tiny values make sense
white_noise_args = Dict(:loc=>0.0, :sig=>std(true_img), :Nsam=>1000, :iso=>false, :norm=>false, :smooth=>true, :smoothval=>0.8) #only if you're using noise based covar
optim_settings = Dict([("iterations", 1000), ("norm", false)])
recon_settings = Dict([("target_type", "sfd_dbn"), ("covar_type", "sfd_dbn"), ("log", true), ("GaussianLoss", true), ("Invcov_matrix", "Diagonal+Eps"),
  ("optim_settings", optim_settings), ("lambda", std(true_img).^(-2)), ("white_noise_args", white_noise_args)]) #Add constraints
#"white_noise_args": args for white_noise sims if GaussianLoss and Covar_type='white_noise'. :loc...

s2mean, s2icov = meancov_generator(true_img, filter_hash, dhc_args, coeff_mask, recon_settings)
recon_settings["s_targ_mean"] = s2mean
recon_settings["s_invcov"] = s2icov
fname_save = "scratch_NM/NewWrapper/3-24/Exp1.jld2"
recon_settings["safemode"] = true
recon_settings["fname_save"] = fname_save
recon_img = reconstruction_wrapper(true_img, init, filter_hash, dhc_args, coeff_mask, recon_settings)


#With apd | SFD | SFDCovar
Nx=64
im = readsfd(Nx)
true_img = im[:, :, 1]

noise = rand(Normal(0.0, std(true_img)), Nx, Nx)
init = imfilter(true_img + noise, Kernel.gaussian(1.0))
heatmap(init)
#heatmap(true_img)

filter_hash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1, Omega=true)
(Nf, ) = size(filter_hash["filt_index"])
coeff_mask = falses(2+Nf+Nf^2)
coeff_mask[Nf+3:end] .= true

dhc_args = Dict(:doS2=>false, :doS12=>false, :doS20=>true, :apodize=>true) #CHANGE: Change sig for the sfd data since the noise model is super high and the tiny values make sense
white_noise_args = Dict(:loc=>0.0, :sig=>std(true_img), :Nsam=>1000, :iso=>false, :norm=>false, :smooth=>true, :smoothval=>0.8) #only if you're using noise based covar
optim_settings = Dict([("iterations", 1000), ("norm", false)])
recon_settings = Dict([("target_type", "sfd_dbn"), ("covar_type", "sfd_dbn"), ("log", true), ("GaussianLoss", true), ("Invcov_matrix", "Diagonal+Eps"),
  ("optim_settings", optim_settings), ("lambda", std(true_img).^(-2)), ("white_noise_args", white_noise_args)]) #Add constraints
#"white_noise_args": args for white_noise sims if GaussianLoss and Covar_type='white_noise'. :loc...

s2mean, s2icov = meancov_generator(true_img, filter_hash, dhc_args, coeff_mask, recon_settings)
recon_settings["s_targ_mean"] = s2mean
recon_settings["s_invcov"] = s2icov
fname_save = "scratch_NM/NewWrapper/3-24/SfdExp1.jld2"
recon_settings["safemode"] = true
recon_settings["fname_save"] = fname_save
recon_img = reconstruction_wrapper(true_img, init, filter_hash, dhc_args, coeff_mask, recon_settings)
Visualization.plot_diffscales([apodizer(true_img), apodizer(init), apodizer(recon_img), apodizer(recon_img) - apodizer(true_img)], ["GT", "Init", "Reconstruction", "Residual"], fname="scratch_NM/NewWrapper/3-24/Exp1.png")
Visualization.plot_synth_QA(apodizer(true_img), apodizer(init), apodizer(recon_img), fname="scratch_NM/NewWrapper/3-24/Exp1_6p.png")
println("Mean abs Recon-True", round(mean(abs.(recon_img - true_img)), digits=6))
println("Mean abs Recon-Init", round(mean(abs.(recon_img - init)), digits=6))
println("Mean abs Recon-True", round(mean(abs.(init- true_img)), digits=6))


#With apd | GT Target | SFDCovar
Nx=64
im = readsfd(Nx)
true_img = im[:, :, 1]

noise = rand(Normal(0.0, std(true_img)), Nx, Nx)
init = imfilter(true_img + noise, Kernel.gaussian(1.0))
heatmap(init)
#heatmap(true_img)

filter_hash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1, Omega=true)
(Nf, ) = size(filter_hash["filt_index"])
coeff_mask = falses(2+Nf+Nf^2)
coeff_mask[Nf+3:end] .= true

dhc_args = Dict(:doS2=>false, :doS12=>false, :doS20=>true, :apodize=>true) #CHANGE: Change sig for the sfd data since the noise model is super high and the tiny values make sense
white_noise_args = Dict(:loc=>0.0, :sig=>std(true_img), :Nsam=>1000, :iso=>false, :norm=>false, :smooth=>true, :smoothval=>0.8) #only if you're using noise based covar
optim_settings = Dict([("iterations", 1000), ("norm", false)])
recon_settings = Dict([("target_type", "ground_truth"), ("covar_type", "sfd_dbn"), ("log", true), ("GaussianLoss", true), ("Invcov_matrix", "Diagonal+Eps"),
  ("optim_settings", optim_settings), ("lambda", std(true_img).^(-2)), ("white_noise_args", white_noise_args)]) #Add constraints
#"white_noise_args": args for white_noise sims if GaussianLoss and Covar_type='white_noise'. :loc...

s2mean, s2icov = meancov_generator(true_img, filter_hash, dhc_args, coeff_mask, recon_settings)
recon_settings["s_targ_mean"] = s2mean
recon_settings["s_invcov"] = s2icov
fname_save = "scratch_NM/NewWrapper/3-24/Sfd_gttargExp2.jld2"
recon_settings["safemode"] = false
recon_settings["fname_save"] = fname_save
recon_settings["optim_settings"] = optim_settings
resobj, recon_img = reconstruction_wrapper(true_img, init, filter_hash, dhc_args, coeff_mask, recon_settings)

p= plot([t.value for t in Optim.trace(resobj)])
plot!(title="Loss: S20 | Targ = S(GT) | Cov = SFD Dbn Covariance", xlabel = "No. Iterations", ylabel = "Loss")
savefig(p, "scratch_NM/NewWrapper/3-24/Trace_Sfd_gttargExp2.png")

Visualization.plot_diffscales([apodizer(true_img), apodizer(init), apodizer(recon_img), apodizer(recon_img) - apodizer(true_img)], ["GT", "Init", "Reconstruction", "Residual"], fname="scratch_NM/NewWrapper/3-24/Sfd_gttargExp2.png")
Visualization.plot_synth_QA(apodizer(true_img), apodizer(init), apodizer(recon_img), fname="scratch_NM/NewWrapper/3-24/Sfd_gttargExp2_6p.png")
convert(Array, Optim.trace(resobj))


#With apd | GT Target | GTCovar
fname_save = "scratch_NM/NewWrapper/3-24/GTtargWNcov"
Nx=64
im = readsfd(Nx)
true_img = im[:, :, 1]

noise = rand(Normal(0.0, std(true_img)), Nx, Nx)
init = imfilter(true_img + noise, Kernel.gaussian(1.0))
heatmap(init)
#heatmap(true_img)

filter_hash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1, Omega=true)
(Nf, ) = size(filter_hash["filt_index"])
coeff_mask = falses(2+Nf+Nf^2)
coeff_mask[Nf+3:end] .= true

dhc_args = Dict(:doS2=>false, :doS12=>false, :doS20=>true, :apodize=>true) #CHANGE: Change sig for the sfd data since the noise model is super high and the tiny values make sense
white_noise_args = Dict(:loc=>0.0, :sig=>std(true_img), :Nsam=>1000, :iso=>false, :norm=>false, :smooth=>true, :smoothval=>0.8) #only if you're using noise based covar
optim_settings = Dict([("iterations", 1000), ("norm", false)])
recon_settings = Dict([("target_type", "ground_truth"), ("covar_type", "white_noise"), ("log", true), ("GaussianLoss", true), ("Invcov_matrix", "Diagonal+Eps"),
  ("optim_settings", optim_settings), ("lambda", std(true_img).^(-2)), ("white_noise_args", white_noise_args)]) #Add constraints
#"white_noise_args": args for white_noise sims if GaussianLoss and Covar_type='white_noise'. :loc...

s2mean, s2icov = meancov_generator(true_img, filter_hash, dhc_args, coeff_mask, recon_settings)
recon_settings["s_targ_mean"] = s2mean
recon_settings["s_invcov"] = s2icov
recon_settings["safemode"] = true
recon_settings["fname_save"] = fname_save * ".jld2"
recon_settings["optim_settings"] = optim_settings
resobj, recon_img = reconstruction_wrapper(true_img, init, filter_hash, dhc_args, coeff_mask, recon_settings)

p= plot([t.value for t in Optim.trace(resobj)])
plot!(title="Loss: S20 | Targ = S(GT) | Cov = White Noise Covariance", xlabel = "No. Iterations", ylabel = "Loss")
savefig(p, fname_save * "_trace.png")

Visualization.plot_diffscales([apodizer(true_img), apodizer(init), apodizer(recon_img), apodizer(recon_img) - apodizer(true_img)], ["GT", "Init", "Reconstruction", "Residual"], fname="scratch_NM/NewWrapper/3-24/Sfd_gttargExp2.png")
Visualization.plot_synth_QA(apodizer(true_img), apodizer(init), apodizer(recon_img), fname="scratch_NM/NewWrapper/3-24/Sfd_gttargExp2_6p.png")
convert(Array, Optim.trace(resobj))
