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

codefile = "scratch_NM/Experiments_3-24.jl"
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
fname_save = "scratch_NM/NewWrapper/3-24/WNtargWNcov"
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
optim_settings = Dict([("iterations", 1000), ("norm", false), ("minmethod", ConjugateGradient())])
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


#Same as above with LBFGS(): Does the descent not stall?
#With apd | GT Target | WNCovar
fname_save = "scratch_NM/NewWrapper/3-24/GTtargWNcov_LBFGS"
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
optim_settings = Dict([("iterations", 1000), ("norm", false), ("minmethod", LBFGS())])
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


#GT Targ | SFD cov | Lambda=0: DECENT-ISH
fname_save = "scratch_NM/NewWrapper/3-24/GTtargSFDcov_noreg_CG"
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
optim_settings = Dict([("iterations", 1000), ("norm", false), ("minmethod", ConjugateGradient())])
recon_settings = Dict([("target_type", "ground_truth"), ("covar_type", "sfd_dbn"), ("log", true), ("GaussianLoss", true), ("Invcov_matrix", "Diagonal+Eps"),
  ("optim_settings", optim_settings), ("lambda", 0.0), ("white_noise_args", white_noise_args)]) #Add constraints
#"white_noise_args": args for white_noise sims if GaussianLoss and Covar_type='white_noise'. :loc...
s_true = DHC_compute_apd(log.(true_img), filter_hash;norm=false, iso=false, dhc_args...)[coeff_mask]
s2mean, s2icov = meancov_generator(true_img, filter_hash, dhc_args, coeff_mask, recon_settings, safety=s_true)
#Check
regs_true = DHC_compute_apd(true_img, filter_hash; dhc_args...)[coeff_mask]
s2mean - s_true

recon_settings["s_targ_mean"] = s2mean
recon_settings["s_invcov"] = s2icov
recon_settings["safemode"] = false
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

#GT Targ | SFD cov
fname_save = "scratch_NM/NewWrapper/3-24/GTtargSFDcov_lamcorr_LBFGS"
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
optim_settings = Dict([("iterations", 1000), ("norm", false), ("minmethod", LBFGS())])
recon_settings = Dict([("target_type", "ground_truth"), ("covar_type", "sfd_dbn"), ("log", true), ("GaussianLoss", true), ("Invcov_matrix", "Diagonal+Eps"),
  ("optim_settings", optim_settings), ("lambda", std(apodizer(log.(true_img))).^(-2)), ("white_noise_args", white_noise_args)]) #Add constraints
#"white_noise_args": args for white_noise sims if GaussianLoss and Covar_type='white_noise'. :loc...

s2mean, s2icov = meancov_generator(true_img, filter_hash, dhc_args, coeff_mask, recon_settings)
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

#GT Targ | WN Cov
#Same as above with LBFGS(): Does the descent not stall?
#With apd | GT Target | WNCovar
fname_save = "scratch_NM/NewWrapper/3-24/GTtargWNcov_CGExp3"
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
optim_settings = Dict([("iterations", 1000), ("norm", false), ("minmethod", ConjugateGradient())])
recon_settings = Dict([("target_type", "ground_truth"), ("covar_type", "white_noise"), ("log", true), ("GaussianLoss", true), ("Invcov_matrix", "Diagonal+Eps"),
  ("optim_settings", optim_settings), ("lambda", 1.0), ("white_noise_args", white_noise_args)]) #Add constraints
#"white_noise_args": args for white_noise sims if GaussianLoss and Covar_type='white_noise'. :loc...

s2mean, s2icov = meancov_generator(true_img, filter_hash, dhc_args, coeff_mask, recon_settings)
recon_settings["s_targ_mean"] = s2mean
recon_settings["s_invcov"] = s2icov
recon_settings["safemode"] = true
recon_settings["fname_save"] = fname_save * ".jld2"
recon_settings["optim_settings"] = optim_settings
resobj, recon_img = reconstruction_wrapper(true_img, init, filter_hash, dhc_args, coeff_mask, recon_settings)

p= plot([t.value for t in Optim.trace(resobj)])
plot!(title="Loss: S20 | Targ = S(GT) | Cov = SFD Cov Lam", xlabel = "No. Iterations", ylabel = "Loss")
savefig(p, fname_save * "_trace.png")


##3-26 What's wrong with the sfd dbn loss?
#Are you already closer than 1 std to the true image?
fname = "scratch_NM/NewWrapper/3-24/GTtargSFDcov_lamcorr_LBFGS"
gttarget = load(fname*".jld2")
#order: A*log*Im
ltrue = log.(gttarget["true_img"])
linit = log.(gttarget["init"])
lrecon =log.(gttarget["recon"])

altrue = apodizer(log.(gttarget["true_img"]))
alinit = apodizer(log.(gttarget["init"]))
alrecon = apodizer(log.(gttarget["recon"]))

lam_corr = std(altrue).^(-2)
lam_old = gttarget["dict"]["lambda"]

#Loss contrib
s_targ_mean, s_targ_invcov = gttarget["dict"]["s_targ_mean"], gttarget["dict"]["s_invcov"]
#init
l2init = 0.5*lam_old*sum((alinit - altrue).^2)
s_init = DHC_compute_apd(linit, filter_hash; dhc_args...)[coeff_mask]
s_recon = DHC_compute_apd(lrecon, filter_hash; dhc_args...)[coeff_mask]
l1init = ( 0.5 .* (s_init - s_targ_mean)' * s_targ_invcov * (s_init - s_targ_mean))[1]
#819, 322

#recon
l2init = 0.5*lam_old*sum((alrecon - altrue).^2)
l1init = ( 0.5 .* (s_recon - s_targ_mean)' * s_targ_invcov * (s_recon - s_targ_mean))[1]
#812, 439


#Are you already closer than 1 std to the true image?
fname = "scratch_NM/NewWrapper/3-24/GTtargSFDcov_noreg_LBFGS"
gttarget = load(fname*".jld2")
#order: A*log*Im
ltrue = log.(gttarget["true_img"])
linit = log.(gttarget["init"])
lrecon =log.(gttarget["recon"])

altrue = apodizer(log.(gttarget["true_img"]))
alinit = apodizer(log.(gttarget["init"]))
alrecon = apodizer(log.(gttarget["recon"]))

lam_corr = std(altrue).^(-2)
lam_old = gttarget["dict"]["lambda"]

#Loss contrib
s_targ_mean, s_targ_invcov = gttarget["dict"]["s_targ_mean"], gttarget["dict"]["s_invcov"]
#init
l2init = 0.5*lam_old*sum((alinit - altrue).^2)
s_init = DHC_compute_apd(linit, filter_hash; dhc_args...)[coeff_mask]
s_recon = DHC_compute_apd(lrecon, filter_hash; dhc_args...)[coeff_mask]
l1init = ( 0.5 .* (s_init - s_targ_mean)' * s_targ_invcov * (s_init - s_targ_mean))[1]
#819

#recon
l2init = 0.5*lam_old*sum((alrecon - altrue).^2)
l1init = ( 0.5 .* (s_recon - s_targ_mean)' * s_targ_invcov * (s_recon - s_targ_mean))[1]
#821


##
fname = "scratch_NM/NewWrapper/3-24/GTtargSFDcov_noreg_CG"
gttarget = load(fname*".jld2")
#order: A*log*Im
ltrue = log.(gttarget["true_img"])
linit = log.(gttarget["init"])
lrecon =log.(gttarget["recon"])

altrue = apodizer(log.(gttarget["true_img"]))
alinit = apodizer(log.(gttarget["init"]))
alrecon = apodizer(log.(gttarget["recon"]))

lam_corr = std(altrue).^(-2)
lam_old = gttarget["dict"]["lambda"]

#Loss contrib
s_targ_mean, s_targ_invcov = gttarget["dict"]["s_targ_mean"], gttarget["dict"]["s_invcov"]
#init
l2init = 0.5*lam_old*sum((alinit - altrue).^2)
s_init = DHC_compute_apd(linit, filter_hash; norm=false, iso=false, dhc_args...)[coeff_mask]
s_recon = DHC_compute_apd(lrecon, filter_hash; norm=false, iso=false, dhc_args...)[coeff_mask]
s_true = DHC_compute_apd(ltrue, filter_hash; norm=false, iso=false, dhc_args...)[coeff_mask]
l1init = ( 0.5 .* (s_init - s_targ_mean)' * s_targ_invcov * (s_init - s_targ_mean))[1]
ltruecheck = ( 0.5 .* (s_true - s_targ_mean)' * s_targ_invcov * (s_true - s_targ_mean))[1]
#16.1

#recon
l2init = 0.5*lam_old*sum((alrecon - altrue).^2)
l1recon = ( 0.5 .* (s_recon - s_targ_mean)' * s_targ_invcov * (s_recon - s_targ_mean))[1]
#1e-5

##


#SFD Targ | SFD cov | Lambda=Corr
fname_save = "scratch_NM/NewWrapper/3-24/SFDtargSFDcov_lamcorr_CG"
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
optim_settings = Dict([("iterations", 1000), ("norm", false), ("minmethod", ConjugateGradient())])
recon_settings = Dict([("target_type", "sfd_dbn"), ("covar_type", "sfd_dbn"), ("log", true), ("GaussianLoss", true), ("Invcov_matrix", "Diagonal+Eps"),
  ("optim_settings", optim_settings), ("lambda", std(apodizer(log.(true_img))).^(-2)), ("white_noise_args", white_noise_args)]) #Add constraints
#"white_noise_args": args for white_noise sims if GaussianLoss and Covar_type='white_noise'. :loc...
#s_true = DHC_compute_apd(log.(true_img), filter_hash;norm=false, iso=false, dhc_args...)[coeff_mask]
s2mean, s2icov = meancov_generator(true_img, filter_hash, dhc_args, coeff_mask, recon_settings, safety=nothing)
#Check
#regs_true = DHC_compute_apd(true_img, filter_hash; dhc_args...)[coeff_mask]
#s2mean - s_true

recon_settings["s_targ_mean"] = s2mean
recon_settings["s_invcov"] = s2icov
recon_settings["safemode"] = false
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

#Testing

fname = "scratch_NM/NewWrapper/3-24/SFDtargSFDcov_lamcorr_CG"
gttarget = load(fname*".jld2")
#order: A*log*Im
ltrue = log.(gttarget["true_img"])
linit = log.(gttarget["init"])
lrecon =log.(gttarget["recon"])

altrue = apodizer(log.(gttarget["true_img"]))
alinit = apodizer(log.(gttarget["init"]))
alrecon = apodizer(log.(gttarget["recon"]))

lam_corr = std(altrue).^(-2)
lam_old = gttarget["dict"]["lambda"]

#Loss contrib
s_targ_mean, s_targ_invcov = gttarget["dict"]["s_targ_mean"], gttarget["dict"]["s_invcov"]
#init
l2init = 0.5*lam_old*sum((alinit - alinit).^2)
s_init = DHC_compute_apd(linit, filter_hash; norm=false, iso=false, dhc_args...)[coeff_mask]
s_recon = DHC_compute_apd(lrecon, filter_hash; norm=false, iso=false, dhc_args...)[coeff_mask]
s_true = DHC_compute_apd(ltrue, filter_hash; norm=false, iso=false, dhc_args...)[coeff_mask]
l1init = ( 0.5 .* (s_init - s_targ_mean)' * s_targ_invcov * (s_init - s_targ_mean))[1]
ltruecheck = ( 0.5 .* (s_true - s_targ_mean)' * s_targ_invcov * (s_true - s_targ_mean))[1]
#True: 277,

#recon
l2init = 0.5*lam_old*sum((alrecon - altrue).^2)
l1recon = ( 0.5 .* (s_recon - s_targ_mean)' * s_targ_invcov * (s_recon - s_targ_mean))[1]
#1e-5

##No smooth
#GT Targ | SFD cov | Lambda=0: DECENT-ISH
fname_save = "scratch_NM/NewWrapper/3-24/GTtargSFDcov_nosmooth_CG"
Nx=64
im = readsfd(Nx)
true_img = im[:, :, 1]

noise = rand(Normal(0.0, std(true_img)), Nx, Nx)
init = true_img + noise
heatmap(init)
#heatmap(true_img)

filter_hash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1, Omega=true)
(Nf, ) = size(filter_hash["filt_index"])
coeff_mask = falses(2+Nf+Nf^2)
coeff_mask[Nf+3:end] .= true

dhc_args = Dict(:doS2=>false, :doS12=>false, :doS20=>true, :apodize=>true) #CHANGE: Change sig for the sfd data since the noise model is super high and the tiny values make sense
white_noise_args = Dict(:loc=>0.0, :sig=>std(true_img), :Nsam=>1000, :iso=>false, :norm=>false, :smooth=>true, :smoothval=>0.8) #only if you're using noise based covar
optim_settings = Dict([("iterations", 1000), ("norm", false), ("minmethod", ConjugateGradient())])
recon_settings = Dict([("target_type", "ground_truth"), ("covar_type", "sfd_dbn"), ("log", true), ("GaussianLoss", true), ("Invcov_matrix", "Diagonal+Eps"),
  ("optim_settings", optim_settings), ("lambda", std(apodizer(log.(true_img))).^(-2)), ("white_noise_args", white_noise_args)]) #Add constraints
#"white_noise_args": args for white_noise sims if GaussianLoss and Covar_type='white_noise'. :loc...
s_true = DHC_compute_apd(log.(true_img), filter_hash;norm=false, iso=false, dhc_args...)[coeff_mask]
s2mean, s2icov = meancov_generator(true_img, filter_hash, dhc_args, coeff_mask, recon_settings, safety=s_true)
#Check
regs_true = DHC_compute_apd(true_img, filter_hash; dhc_args...)[coeff_mask]
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

##SFD Targ | SFD Cov
fname_save = "scratch_NM/NewWrapper/3-24/SFDtargSFDcov_lamcorr_nosmooth_CG_recheck"
Nx=64
im = readsfd(Nx)
true_img = im[:, :, 1]

noise = rand(Normal(0.0, std(true_img)), Nx, Nx)
init = true_img + noise
heatmap(init)
#heatmap(true_img)

filter_hash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1, Omega=true)
(Nf, ) = size(filter_hash["filt_index"])
coeff_mask = falses(2+Nf+Nf^2)
coeff_mask[Nf+3:end] .= true

dhc_args = Dict(:doS2=>false, :doS12=>false, :doS20=>true, :apodize=>true) #CHANGE: Change sig for the sfd data since the noise model is super high and the tiny values make sense
white_noise_args = Dict(:loc=>0.0, :sig=>std(true_img), :Nsam=>1000, :iso=>false, :norm=>false, :smooth=>true, :smoothval=>0.8) #only if you're using noise based covar
optim_settings = Dict([("iterations", 1000), ("norm", false), ("minmethod", ConjugateGradient())])
recon_settings = Dict([("target_type", "sfd_dbn"), ("covar_type", "sfd_dbn"), ("log", true), ("GaussianLoss", true), ("Invcov_matrix", "Diagonal+Eps"),
  ("optim_settings", optim_settings), ("lambda", std(apodizer(log.(true_img))).^(-2)), ("white_noise_args", white_noise_args)]) #Add constraints
#"white_noise_args": args for white_noise sims if GaussianLoss and Covar_type='white_noise'. :loc...
#s_true = DHC_compute_apd(log.(true_img), filter_hash;norm=false, iso=false, dhc_args...)[coeff_mask]
s2mean, s2icov = meancov_generator(true_img, filter_hash, dhc_args, coeff_mask, recon_settings, safety=nothing)
#Check
regs_true = DHC_compute_apd(true_img, filter_hash; dhc_args...)[coeff_mask]
#s2mean - s_true

recon_settings["s_targ_mean"] = s2mean
recon_settings["s_invcov"] = s2icov
recon_settings["safemode"] = true
recon_settings["fname_save"] = fname_save * ".jld2"
recon_settings["optim_settings"] = optim_settings
resobj, recon_img = reconstruction_wrapper(true_img, init, filter_hash, dhc_args, coeff_mask, recon_settings)

p= plot([t.value for t in Optim.trace(resobj)])
plot!(title="Loss: S20 | Targ = S(SFD) | Cov = SFD Covariance", xlabel = "No. Iterations", ylabel = "Loss")
savefig(p, fname_save * "_trace.png")

Visualization.plot_diffscales([apodizer(true_img), apodizer(init), apodizer(recon_img), apodizer(recon_img) - apodizer(true_img)], ["GT", "Init", "Reconstruction", "Residual"], fname=fname_save*".png")
Visualization.plot_synth_QA(apodizer(true_img), apodizer(init), apodizer(recon_img), fname=fname_save*"_6p.png")

println("Mean Abs Res: Init-True = ", mean(abs.(init - true_img)), " Recon-True = ", mean(abs.(recon_img - true_img)))
println("Mean Abs Frac Res", mean(abs.((init - true_img)./true_img)), " Recon-True=", mean(abs.((recon_img - true_img)./true_img)))
maximum(im)
mean(im)
minimum(im)



##SFD Targ | SFD Cov | No Log
fname_save = "scratch_NM/NewWrapper/3-24/SFDtargSFDcov_lamcorr_nosmooth_reg_CG"
Nx=64
im = readsfd(Nx)
true_img = im[:, :, 1]

noise = rand(Normal(0.0, std(true_img)), Nx, Nx)
init = true_img + noise
heatmap(init)
#heatmap(true_img)

filter_hash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1, Omega=true)
(Nf, ) = size(filter_hash["filt_index"])
coeff_mask = falses(2+Nf+Nf^2)
coeff_mask[Nf+3:end] .= true

dhc_args = Dict(:doS2=>false, :doS12=>false, :doS20=>true, :apodize=>true) #CHANGE: Change sig for the sfd data since the noise model is super high and the tiny values make sense
white_noise_args = Dict(:loc=>0.0, :sig=>std(true_img), :Nsam=>1000, :iso=>false, :norm=>false, :smooth=>true, :smoothval=>0.8) #only if you're using noise based covar
optim_settings = Dict([("iterations", 1000), ("norm", false), ("minmethod", ConjugateGradient())])
recon_settings = Dict([("target_type", "sfd_dbn"), ("covar_type", "sfd_dbn"), ("log", false), ("GaussianLoss", true), ("Invcov_matrix", "Diagonal+Eps"),
  ("optim_settings", optim_settings), ("lambda", std(apodizer(true_img)).^(-2)), ("white_noise_args", white_noise_args)]) #Add constraints

s2mean, s2icov = meancov_generator(true_img, filter_hash, dhc_args, coeff_mask, recon_settings, safety=nothing)
#Check
regs_true = DHC_compute_apd(true_img, filter_hash; dhc_args...)[coeff_mask]
#s2mean - s_true

recon_settings["s_targ_mean"] = s2mean
recon_settings["s_invcov"] = s2icov
recon_settings["safemode"] = true
recon_settings["fname_save"] = fname_save * ".jld2"
recon_settings["optim_settings"] = optim_settings
resobj, recon_img = reconstruction_wrapper(true_img, init, filter_hash, dhc_args, coeff_mask, recon_settings)

p= plot([t.value for t in Optim.trace(resobj)])
plot!(title="Loss: S20 | Targ = S(SFD) | Cov = SFD Covariance", xlabel = "No. Iterations", ylabel = "Loss")
savefig(p, fname_save * "_trace.png")

Visualization.plot_diffscales([apodizer(true_img), apodizer(init), apodizer(recon_img), apodizer(recon_img) - apodizer(true_img)], ["GT", "Init", "Reconstruction", "Residual"], fname=fname_save*".png")
Visualization.plot_synth_QA(apodizer(true_img), apodizer(init), apodizer(recon_img), fname=fname_save*"_6p.png")

println("Mean Abs Res: Init-True = ", mean(abs.(init - true_img)), " Recon-True = ", mean(abs.(recon_img - true_img)))
println("Mean Abs Frac Res", mean(abs.((init - true_img)./true_img)), " Recon-True=", mean(abs.((recon_img - true_img)./true_img)))
maximum(im)
mean(im)
minimum(im)


##GT Targ | SFD Cov | Reg
fname_save = "scratch_NM/NewWrapper/3-24/GTtargSFDcov_lam0_nosmooth_reg_CG"
Nx=64
im = readsfd(Nx)
true_img = im[:, :, 1]

noise = rand(Normal(0.0, std(true_img)), Nx, Nx)
init = true_img + noise
heatmap(init)
#heatmap(true_img)

filter_hash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1, Omega=true)
(Nf, ) = size(filter_hash["filt_index"])
coeff_mask = falses(2+Nf+Nf^2)
coeff_mask[Nf+3:end] .= true

dhc_args = Dict(:doS2=>false, :doS12=>false, :doS20=>true, :apodize=>true) #CHANGE: Change sig for the sfd data since the noise model is super high and the tiny values make sense
white_noise_args = Dict(:loc=>0.0, :sig=>std(true_img), :Nsam=>1000, :iso=>false, :norm=>false, :smooth=>true, :smoothval=>0.8) #only if you're using noise based covar
optim_settings = Dict([("iterations", 1000), ("norm", false), ("minmethod", ConjugateGradient())])
recon_settings = Dict([("target_type", "ground_truth"), ("covar_type", "sfd_dbn"), ("log", false), ("GaussianLoss", true), ("Invcov_matrix", "Diagonal+Eps"),
  ("optim_settings", optim_settings), ("lambda", 0.0), ("white_noise_args", white_noise_args)]) #Add constraints

s2mean, s2icov = meancov_generator(true_img, filter_hash, dhc_args, coeff_mask, recon_settings, safety=nothing)
#Check
regs_true = DHC_compute_apd(true_img, filter_hash; dhc_args...)[coeff_mask]
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

##GT Targ | GT Cov | Reg
fname_save = "scratch_NM/NewWrapper/3-24/GTtargWNcov_lamcorr_nosmooth_reg_CG"
Nx=64
im = readsfd(Nx)
true_img = im[:, :, 1]

noise = rand(Normal(0.0, std(true_img)), Nx, Nx)
init = true_img + noise
heatmap(init)
#heatmap(true_img)

filter_hash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1, Omega=true)
(Nf, ) = size(filter_hash["filt_index"])
coeff_mask = falses(2+Nf+Nf^2)
coeff_mask[Nf+3:end] .= true

dhc_args = Dict(:doS2=>false, :doS12=>false, :doS20=>true, :apodize=>true, :coeff_mask=>coeff_mask) #CHANGE: Change sig for the sfd data since the noise model is super high and the tiny values make sense
white_noise_args = Dict(:loc=>0.0, :sig=>std(true_img), :Nsam=>1000, :iso=>false, :norm=>false, :smooth=>false, :smoothval=>0.8) #only if you're using noise based covar
optim_settings = Dict([("iterations", 1000), ("norm", false), ("minmethod", ConjugateGradient())])
recon_settings = Dict([("target_type", "ground_truth"), ("covar_type", "white_noise"), ("log", false), ("GaussianLoss", true), ("Invcov_matrix", "Diagonal+Eps"),
  ("optim_settings", optim_settings), ("lambda", std(apodizer(true_img)).^(-2)), ("white_noise_args", white_noise_args)]) #Add constraints

s2mean, s2icov = meancov_generator(true_img, filter_hash, dhc_args, coeff_mask, recon_settings, safety=nothing)
#Check
regs_true = DHC_compute_apd(true_img, filter_hash; dhc_args...)[coeff_mask]
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

#Testing: Where it DID work
fname = "scratch_NM/NewWrapper/3-24/GTtargWNcov_lamcorr_nosmooth_reg_CG"
gttarget = load(fname * ".jld2")
#order: A*log*Im
true_img = gttarget["true_img"]
init = gttarget["init"]
recon =gttarget["recon"]

altrue = apodizer(gttarget["true_img"])
alinit = apodizer(gttarget["init"])
alrecon = apodizer(gttarget["recon"])

println("Mean Abs Res: Init-True = ", mean(abs.(alinit - altrue)), " Recon-True = ", mean(abs.(alrecon - altrue)))
println("Mean Abs Frac Res", mean(abs.((alinit - altrue)./altrue)), " Recon-True=", mean(abs.((alrecon - altrue)./altrue)))

lam_corr = std(altrue).^(-2)
lam_old = gttarget["dict"]["lambda"]

#Loss contrib
s_targ_mean, s_targ_invcov = gttarget["dict"]["s_targ_mean"], gttarget["dict"]["s_invcov"]
#init
l2init = 0.5*lam_old*sum((alinit - alinit).^2)
s_init = DHC_compute_apd(init, filter_hash; norm=false, iso=false, dhc_args...)[coeff_mask]
s_recon = DHC_compute_apd(recon, filter_hash; norm=false, iso=false, dhc_args...)[coeff_mask]
s_true = DHC_compute_apd(true_img, filter_hash; norm=false, iso=false, dhc_args...)[coeff_mask]
l1init = ( 0.5 .* (s_init - s_targ_mean)' * s_targ_invcov * (s_init - s_targ_mean))[1]
ltruecheck = ( 0.5 .* (s_true - s_targ_mean)' * s_targ_invcov * (s_true - s_targ_mean))[1]
l2true = 0.5*lam_old*sum((alinit - altrue).^2)
#True: 0, 2280
#Init: 21321, 0

#recon
l2init = 0.5*lam_old*sum((alrecon - alinit).^2)
l1recon = ( 0.5 .* (s_recon - s_targ_mean)' * s_targ_invcov * (s_recon - s_targ_mean))[1]
#Recon: 173, 959

#Testing: Where it DID NOT work
fname = "scratch_NM/NewWrapper/3-24/GTtargSFDcov_lamcorr_nosmooth_reg_CG"
gttarget = load(fname * ".jld2")
#order: A*log*Im
true_img = gttarget["true_img"]
init = gttarget["init"]
recon =gttarget["recon"]

altrue = apodizer(gttarget["true_img"])
alinit = apodizer(gttarget["init"])
alrecon = apodizer(gttarget["recon"])

lam_corr = std(altrue).^(-2)
lam_old = gttarget["dict"]["lambda"]

#Loss contrib
s_targ_mean, s_targ_invcov = gttarget["dict"]["s_targ_mean"], gttarget["dict"]["s_invcov"]
#init
s_init = DHC_compute_apd(init, filter_hash; norm=false, iso=false, dhc_args...)[coeff_mask]
s_recon = DHC_compute_apd(recon, filter_hash; norm=false, iso=false, dhc_args...)[coeff_mask]
s_true = DHC_compute_apd(true_img, filter_hash; norm=false, iso=false, dhc_args...)[coeff_mask]
l1init = ( 0.5 .* (s_init - s_targ_mean)' * s_targ_invcov * (s_init - s_targ_mean))[1]
l2init = 0.5*lam_old*sum((alinit - alinit).^2)
l1truecheck = ( 0.5 .* (s_true - s_targ_mean)' * s_targ_invcov * (s_true - s_targ_mean))[1]
l2true = 0.5*lam_old*sum((alinit - altrue).^2)
#True: 0, 2327
#Init: 0.0009, 0

#recon
l1recon = ( 0.5 .* (s_recon - s_targ_mean)' * s_targ_invcov * (s_recon - s_targ_mean))[1]
l2recon = 0.5*lam_old*sum((alrecon - alinit).^2)
#Recon: 0.0009, 1e-9


#Testing where it did work but in log space.
fname = "scratch_NM/NewWrapper/3-24/GTtargSFDcov_nosmooth_CG"
gttarget = load(fname*".jld2")
#order: A*log*Im
ltrue = log.(gttarget["true_img"])
linit = log.(gttarget["init"])
lrecon =log.(gttarget["recon"])

altrue = apodizer(log.(gttarget["true_img"]))
alinit = apodizer(log.(gttarget["init"]))
alrecon = apodizer(log.(gttarget["recon"]))

lam_corr = std(altrue).^(-2)
lam_old = gttarget["dict"]["lambda"]

#Loss contrib
s_targ_mean, s_targ_invcov = gttarget["dict"]["s_targ_mean"], gttarget["dict"]["s_invcov"]
#init
s_init = DHC_compute_apd(linit, filter_hash; norm=false, iso=false, dhc_args...)[coeff_mask]
s_recon = DHC_compute_apd(lrecon, filter_hash; norm=false, iso=false, dhc_args...)[coeff_mask]
s_true = DHC_compute_apd(ltrue, filter_hash; norm=false, iso=false, dhc_args...)[coeff_mask]
ltruecheck = ( 0.5 .* (s_true - s_targ_mean)' * s_targ_invcov * (s_true - s_targ_mean))[1]
l2true = 0.5*lam_old*sum((alinit - altrue).^2)
l1init = ( 0.5 .* (s_init - s_targ_mean)' * s_targ_invcov * (s_init - s_targ_mean))[1]
l2init = 0.5*lam_old*sum((alinit - alinit).^2)
#True: 0, 2716
#Init: 3717, 0

#recon
l1recon = ( 0.5 .* (s_recon - s_targ_mean)' * s_targ_invcov * (s_recon - s_targ_mean))[1]
l2recon = 0.5*lam_old*sum((alrecon - alinit).^2)
#306, 549

true_img = gttarget["true_img"]
init = gttarget["init"]
recon_img =gttarget["recon"]


fname = "scratch_NM/NewWrapper/3-24/GTtargSFDcov_noreg_nosmooth_CG"
gttarget = load(fname*".jld2")
println("Mean Abs Res: Init-True = ", mean(abs.(init - true_img)), " Recon-True = ", mean(abs.(recon_img - true_img)))
println("Mean Abs Frac Res", mean(abs.((init - true_img)./true_img)), " Recon-True=", mean(abs.((recon_img - true_img)./true_img)))


##4-1: Generating standardized init images and experiments
Nx=64
imall = readsfd(Nx)
Random.seed!(41)

imlist = [1, 5, 10, 50, 100, 500, 1000, 5000, 10000]
for imid in imlist
  fname_save = "scratch_NM/StandardizedExp/Nx64/" * "Data_" * string(imid) * ".jld2"
  true_img = imall[:, :, imid]
  stdval = std(true_img)
  noise = rand(Normal(0.0, stdval), Nx, Nx)
  init = true_img + noise
  zind = findall(init.<0)
  println("Percentage set to 1e-6", 100*length(zind)/length(init))
  init[zind] .= 1e-6
  save(fname_save, Dict("true_img"=>true_img, "seed"=>41, "init"=>init, "noise model"=>"White noise, No smoothing. sigma=std(true_img)", "std"=>stdval))
end

savexp = load("scratch_NM/StandardizedExp/Nx64/Data_5000.jld2")
heatmap(savexp["true_img"])
heatmap(savexp["init"])


##4-2 Comparing SFD-SFD runs with and without apd
#Log-Apd-Iso
fname = "scratch_NM/NewWrapper/3-29/SFDTargSFDCov/log_apd_iso"
gttarget = load(fname*".jld2")
#order: A*log*Im
ltrue = log.(gttarget["true_img"])
linit = log.(gttarget["init"])
lrecon =log.(gttarget["recon"])

altrue = apodizer(log.(gttarget["true_img"]))
alinit = apodizer(log.(gttarget["init"]))
alrecon = apodizer(log.(gttarget["recon"]))

lam_corr = std(altrue).^(-2)
lam_old = gttarget["dict"]["lambda"]

#Loss contrib
s_targ_mean, s_targ_invcov = gttarget["dict"]["s_targ_mean"], gttarget["dict"]["s_invcov"]
filter_hash = fink_filter_hash(1, 8, nx=64, pc=1, wd=1, Omega=true)
(S1iso, Nf) = size(filter_hash["S1_iso_mat"])
(S2iso, Nfsq) = size(filter_hash["S2_iso_mat"])
coeff_mask = falses(2+S1iso+S2iso)
coeff_mask[S1iso+3:end] .= true
dhc_args = Dict(:doS2=>false, :doS12=>false, :doS20=>true, :apodize=>true, :iso=>true)
length(coeff_mask)
#init
s_init = DHC_compute_wrapper(linit, filter_hash; norm=false,  dhc_args...)[coeff_mask]
s_recon = DHC_compute_wrapper(lrecon, filter_hash; norm=false, dhc_args...)[coeff_mask]
s_true = DHC_compute_wrapper(ltrue, filter_hash; norm=false, dhc_args...)[coeff_mask]
ltruecheck = ( 0.5 .* (s_true - s_targ_mean)' * s_targ_invcov * (s_true - s_targ_mean))[1]
l2true = 0.5*lam_old*sum((alinit - altrue).^2)
l1init = ( 0.5 .* (s_init - s_targ_mean)' * s_targ_invcov * (s_init - s_targ_mean))[1]
l2init = 0.5*lam_old*sum((alinit - alinit).^2)
#True:43.1, 2505
#Init: 903, 0

#recon
l1recon = ( 0.5 .* (s_recon - s_targ_mean)' * s_targ_invcov * (s_recon - s_targ_mean))[1]
l2recon = 0.5*lam_old*sum((alrecon - alinit).^2)
#228, 184

true_img = gttarget["true_img"]
init = gttarget["init"]
recon_img =gttarget["recon"]


println("Mean Abs Res: Init-True = ", mean(abs.(init - true_img)), " Recon-True = ", mean(abs.(recon_img - true_img)))
println("Mean Abs Frac Res", mean(abs.((init - true_img)./true_img)), " Recon-True=", mean(abs.((recon_img - true_img)./true_img)))
println("RMSE Init-True= ", (mean((init - true_img).^2)).^0.5, " Recon-True=", (mean((recon_img - true_img).^2)).^0.5)


##Log-Apd-NoIso
fname = "scratch_NM/NewWrapper/3-29/SFDTargSFDCov/log_apd_iso"
gttarget = load(fname*".jld2")
#order: A*log*Im
ltrue = log.(gttarget["true_img"])
linit = log.(gttarget["init"])
lrecon =log.(gttarget["recon"])

altrue = apodizer(log.(gttarget["true_img"]))
alinit = apodizer(log.(gttarget["init"]))
alrecon = apodizer(log.(gttarget["recon"]))

lam_corr = std(altrue).^(-2)
lam_old = gttarget["dict"]["lambda"]

#Loss contrib
s_targ_mean, s_targ_invcov = gttarget["dict"]["s_targ_mean"], gttarget["dict"]["s_invcov"]
filter_hash = fink_filter_hash(1, 8, nx=64, pc=1, wd=1, Omega=true)
(S1iso, Nf) = size(filter_hash["S1_iso_mat"])
(S2iso, Nfsq) = size(filter_hash["S2_iso_mat"])
coeff_mask = falses(2+S1iso+S2iso)
coeff_mask[S1iso+3:end] .= true
dhc_args = Dict(:doS2=>false, :doS12=>false, :doS20=>true, :apodize=>true, :iso=>true)
length(coeff_mask)
#init
s_init = DHC_compute_wrapper(linit, filter_hash; norm=false,  dhc_args...)[coeff_mask]
s_recon = DHC_compute_wrapper(lrecon, filter_hash; norm=false, dhc_args...)[coeff_mask]
s_true = DHC_compute_wrapper(ltrue, filter_hash; norm=false, dhc_args...)[coeff_mask]
ltruecheck = ( 0.5 .* (s_true - s_targ_mean)' * s_targ_invcov * (s_true - s_targ_mean))[1]
l2true = 0.5*lam_old*sum((alinit - altrue).^2)
l1init = ( 0.5 .* (s_init - s_targ_mean)' * s_targ_invcov * (s_init - s_targ_mean))[1]
l2init = 0.5*lam_old*sum((alinit - alinit).^2)
#True:43.1, 2505
#Init: 903, 0

#recon
l1recon = ( 0.5 .* (s_recon - s_targ_mean)' * s_targ_invcov * (s_recon - s_targ_mean))[1]
l2recon = 0.5*lam_old*sum((alrecon - alinit).^2)
#228, 184

true_img = gttarget["true_img"]
init = gttarget["init"]
recon_img =gttarget["recon"]


println("Mean Abs Res: Init-True = ", mean(abs.(init - true_img)), " Recon-True = ", mean(abs.(recon_img - true_img)))
println("Mean Abs Frac Res", mean(abs.((init - true_img)./true_img)), " Recon-True=", mean(abs.((recon_img - true_img)./true_img)))
println("RMSE Init-True= ", (mean((init - true_img).^2)).^0.5, " Recon-True=", (mean((recon_img - true_img).^2)).^0.5)


##Log-Apd-Noiso
fname = "scratch_NM/StandardizedExp/Nx64/noisy_stdtrue/SFDTargSFDCov/log_apd_noiso/nophi_lamcorr_recon_1000"
gttarget = load(fname*".jld2")
(Nf,) = size(filter_hash["filt_index"])
reshape(gttarget["coeff_mask"][37:end], (Nf, Nf))
#order: A*log*Im
ltrue = log.(gttarget["true_img"])
linit = log.(gttarget["init"])
lrecon =log.(gttarget["recon"])

altrue = apodizer(log.(gttarget["true_img"]))
alinit = apodizer(log.(gttarget["init"]))
alrecon = apodizer(log.(gttarget["recon"]))

lam_corr = std(altrue).^(-2)
lam_old = gttarget["dict"]["lambda"]

#Loss contrib
s_targ_mean, s_targ_invcov = gttarget["dict"]["s_targ_mean"], gttarget["dict"]["s_invcov"]
filter_hash = fink_filter_hash(1, 8, nx=64, pc=1, wd=1, Omega=true)
(S1iso, Nf) = size(filter_hash["S1_iso_mat"])
(S2iso, Nfsq) = size(filter_hash["S2_iso_mat"])
coeff_mask = falses(2+S1iso+S2iso)
coeff_mask[S1iso+3:end] .= true
dhc_args = Dict(:doS2=>false, :doS12=>false, :doS20=>true, :apodize=>true, :iso=>true)
length(coeff_mask)
#init
s_init = DHC_compute_wrapper(linit, filter_hash; norm=false,  dhc_args...)[coeff_mask]
s_recon = DHC_compute_wrapper(lrecon, filter_hash; norm=false, dhc_args...)[coeff_mask]
s_true = DHC_compute_wrapper(ltrue, filter_hash; norm=false, dhc_args...)[coeff_mask]
ltruecheck = ( 0.5 .* (s_true - s_targ_mean)' * s_targ_invcov * (s_true - s_targ_mean))[1]
l2true = 0.5*lam_old*sum((alinit - altrue).^2)
l1init = ( 0.5 .* (s_init - s_targ_mean)' * s_targ_invcov * (s_init - s_targ_mean))[1]
l2init = 0.5*lam_old*sum((alinit - alinit).^2)
#True:43.1, 2505
#Init: 903, 0

#recon
l1recon = ( 0.5 .* (s_recon - s_targ_mean)' * s_targ_invcov * (s_recon - s_targ_mean))[1]
l2recon = 0.5*lam_old*sum((alrecon - alinit).^2)
#228, 184

true_img = gttarget["true_img"]
init = gttarget["init"]
recon_img =gttarget["recon"]


println("Mean Abs Res: Init-True = ", mean(abs.(init - true_img)), " Recon-True = ", mean(abs.(recon_img - true_img)))
println("Mean Abs Frac Res", mean(abs.((init - true_img)./true_img)), " Recon-True=", mean(abs.((recon_img - true_img)./true_img)))
println("RMSE Init-True= ", (mean((init - true_img).^2)).^0.5, " Recon-True=", (mean((recon_img - true_img).^2)).^0.5)


##4-2
#Log Apd Iso
fname = "scratch_NM/StandardizedExp/Nx64/noisy_stdtrue/SFDTargSFDCov/log_apd_iso/lamcorr_recon_1000"
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
plot!(title="P(k) of Log Image")


true_ps = calc_1dps(apodizer(gttarget["true_img"]), kbins)
recon_ps = calc_1dps(apodizer(gttarget["recon"]), kbins)
init_ps = calc_1dps(apodizer(gttarget["init"]), kbins)
p = plot(log.(kbins), log.(true_ps), label="True")
plot!(log.(kbins), log.(recon_ps), label="Recon")
plot!(log.(kbins), log.(init_ps), label="Init")
plot!(title="P(k) of Image")

s_targ_mean, s_targ_invcov = gttarget["dict"]["s_targ_mean"], gttarget["dict"]["s_invcov"]
dhc_args = Dict(:doS2=>false, :doS12=>false, :doS20=>true, :apodize=>true, :iso=>true)

coeff_mask = gttarget["coeff_mask"]
s_true = DHC_compute_wrapper(log.(gttarget["true_img"]), filter_hash, norm=false; dhc_args...)[coeff_mask]
s_init = DHC_compute_wrapper(log.(gttarget["init"]), filter_hash, norm=false; dhc_args...)[coeff_mask]
s_recon = DHC_compute_wrapper(log.(gttarget["recon"]), filter_hash, norm=false; dhc_args...)[coeff_mask]
ltruecheck = ( 0.5 .* (s_true - s_targ_mean)' * s_targ_invcov * (s_true - s_targ_mean))[1]
l2true = 0.5*lam_old*sum((alinit - altrue).^2)
#78.2, 2505

l1init = ( 0.5 .* (s_init - s_targ_mean)' * s_targ_invcov * (s_init - s_targ_mean))[1]
l2init = 0.5*lam_old*sum((alinit - alinit).^2)
#2.68e7, 0

l1recon = ( 0.5 .* (s_recon - s_targ_mean)' * s_targ_invcov * (s_recon - s_targ_mean))[1]
l2recon = 0.5*lam_old*sum((alrecon - alinit).^2)
#4168, 183.61




#Log, Apd, Noiso

fname = "scratch_NM/StandardizedExp/Nx64/noisy_stdtrue/SFDTargSFDCov/log_apd_noiso/lamcorr_recon_1000"
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
plot!(title="P(k) of Log Image: Noiso")


true_ps = calc_1dps(apodizer(gttarget["true_img"]), kbins)
recon_ps = calc_1dps(apodizer(gttarget["recon"]), kbins)
init_ps = calc_1dps(apodizer(gttarget["init"]), kbins)
p = plot(log.(kbins), log.(true_ps), label="True")
plot!(log.(kbins), log.(recon_ps), label="Recon")
plot!(log.(kbins), log.(init_ps), label="Init")
plot!(title="P(k) of Image: NoIso")

s_targ_mean, s_targ_invcov = gttarget["dict"]["s_targ_mean"], gttarget["dict"]["s_invcov"]
dhc_args = Dict(:doS2=>false, :doS12=>false, :doS20=>true, :apodize=>true, :iso=>false)
gttarget["coeff_mask"]

s_true = DHC_compute_wrapper(log.(gttarget["true_img"]), filter_hash, norm=false; dhc_args...)[gttarget["coeff_mask"]]
s_init = DHC_compute_wrapper(log.(gttarget["init"]), filter_hash, norm=false; dhc_args...)[gttarget["coeff_mask"]]
s_recon = DHC_compute_wrapper(log.(gttarget["recon"]), filter_hash, norm=false; dhc_args...)[gttarget["coeff_mask"]]

ltruecheck = ( 0.5 .* (s_true - s_targ_mean)' * s_targ_invcov * (s_true - s_targ_mean))[1]
l2true = 0.5*lam_old*sum((alinit - altrue).^2)
#648, 2505

l1init = ( 0.5 .* (s_init - s_targ_mean)' * s_targ_invcov * (s_init - s_targ_mean))[1]
l2init = 0.5*lam_old*sum((alinit - alinit).^2)
#7.25e7, 0

l1recon = ( 0.5 .* (s_recon - s_targ_mean)' * s_targ_invcov * (s_recon - s_targ_mean))[1]
l2recon = 0.5*lam_old*sum((alrecon - alinit).^2)
#2948, 184

sig = (diag(s_targ_invcov)).^(-0.5)
z_true_targ = (s_true - s_targ_mean)./sig
z_recon_targ = (s_recon - s_targ_mean)./sig
z_recon_true = (s_recon - s_true)./sig
z_init_targ = (s_init - s_targ_mean)./sig

zscores = hcat(z_true_targ, z_recon_targ, z_recon_true)
Nf = size(filter_hash["filt_index"])[1]
images = [reshape(z_true_targ, (Nf, Nf)), reshape(z_recon_targ, (Nf, Nf)), reshape(z_recon_true, (Nf, Nf)), reshape(z_init_targ, (Nf, Nf))]
titles= ["True-Target", "Recon-Target", "Recon-True", "Init-Targ"]
pl12 = plot(
        heatmap(images[1], title=titles[1]),
        heatmap(images[2], title=titles[2]),
        heatmap(images[3],title= titles[3]),
        heatmap(images[4],title= titles[4]))
plot!(suptitle="S20 Z Scores")


clim = (minimum(z_true_targ), maximum(z_true_targ))
pl12 = plot(
        heatmap(images[1], title=titles[1], clim=clim),
        heatmap(images[2], title=titles[2], clim=clim),
        heatmap(images[3],title= titles[3], clim=clim),
        heatmap(images[4],title= titles[4], clim=clim))
plot!(suptitle="S20 Z Scores")
z_recon_targ = reshape(z_recon_targ, (Nf, Nf))

##Covariance
Nx=64
dbnimg = readsfd(Nx, logbool=true)
fname = "scratch_NM/StandardizedExp/Nx64/noisy_stdtrue/SFDTargSFDCov/log_apd_noiso/lamcorr_recon_1000"
gttarget = load(fname*".jld2")
fhash = gttarget["filter_hash"]
dhc_args = Dict(:doS2=>false, :doS12=>false, :doS20=>true, :apodize=>true, :iso=>false)
coeff_mask = gttarget["coeff_mask"]
smean1, sdiag1, scov1 = dbn_coeffs_calc(dbnimg[:, :, 1:5000], fhash, dhc_args, coeff_mask)
smean2, sdiag2, scov2 = dbn_coeffs_calc(dbnimg[:, :, 5000:end], fhash, dhc_args, coeff_mask)

dbn1 = get_dbn_coeffs(dbnimg[:, :, 1:5000], fhash, dhc_args, coeff_mask = coeff_mask)
dbn2 = get_dbn_coeffs(dbnimg[:, :, 5001:end], fhash, dhc_args, coeff_mask= coeff_mask)
scov1inv = invert_covmat(scov1, 1e-6)
scov2inv = invert_covmat(scov2, 1e-6)


diff = dbn1 .- reshape(smean1, (1, 1156))
chisq1 = sum((diff * scov1inv) .* diff, dims=2)


diff = dbn2 .- reshape(smean2, (1, 1156))
chisq2 = sum((diff * scov2inv) .* diff, dims=2)

p = histogram(chisq1[:])
p = histogram(chisq2[:])
chisqsamp = rand(Distributions.Chisq(1156), 5000)
histogram!(chisqsamp)
########################

sd1inv = invert_covmat(sdiag1, 1e-10)
sd2inv = invert_covmat(sdiag2, 1e-10)


diff = dbn1 .- reshape(smean1, (1, 1156))
chisq1 = sum((diff * sd1inv) .* diff, dims=2)


diff = dbn2 .- reshape(smean2, (1, 1156))
chisq2 = sum((diff * sd2inv) .* diff, dims=2)

p = histogram(chisq1[:], label="Subset1")
p= histogram(chisq2[:], label="Subset2")
chisqsamp = rand(Distributions.Chisq(1156), 5000)
p = histogram(chisqsamp, label="True Chisq")


####################################
Nx=64
dbnimg = readsfd(Nx, logbool=true)
meanimg = mean(dbnimg, dims=[1, 2])
dbnimgms = dbnimg .- meanimg
dbn1 = get_dbn_coeffs(dbnimgms[:, :, 1:5000], fhash, dhc_args, coeff_mask = coeff_mask)
dbn2 = get_dbn_coeffs(dbnimgms[:, :, 5001:end], fhash, dhc_args, coeff_mask= coeff_mask)
smean1, sdiag1, scov1 = dbn_coeffs_calc(dbnimgms[:, :, 1:5000], fhash, dhc_args, coeff_mask)
smean2, sdiag2, scov2 = dbn_coeffs_calc(dbnimgms[:, :, 5001:end], fhash, dhc_args, coeff_mask)

scov1inv = invert_covmat(scov1, 1e-6)
scov2inv = invert_covmat(scov2, 1e-6)

diff = dbn1 .- reshape(smean1, (1, 1156))
chisq1 = sum((diff * scov1inv) .* diff, dims=2)


diff = dbn2 .- reshape(smean2, (1, 1156))
chisq2 = sum((diff * scov2inv) .* diff, dims=2)

p = histogram(chisq1[:])
p = histogram(chisq2[:])
chisqsamp = rand(Distributions.Chisq(1156), 5000)
histogram!(chisqsamp)

#########################################
diff = dbn1 .- reshape(smean1, (1, 1156))
chisq12 = sum((diff * scov2inv) .* diff, dims=2)


diff = dbn2 .- reshape(smean2, (1, 1156))
chisq21 = sum((diff * scov1inv) .* diff, dims=2)

p = histogram(chisq1[:], xlims=(0, 5000), label="ChiSq1-1")
histogram!(chisq12[:], xlims=(0, 5000), label="ChiSq1-2")

p = histogram(chisq2[:], xlims=(0, 5000), label="ChiSq2")
histogram!(chisq21[:], xlims=(0, 5000), label="ChiSq2-1")

chisqsamp = rand(Distributions.Chisq(1156), 5000)
histogram!(chisqsamp)
#########################################
#Log coeff, log im
Nx=64
dbnimg = readsfd(Nx, logbool=true)
#dbnimgms = dbnimg .- meanimg
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

diff = dbn1 .- reshape(smean1, (1, 1156))
chisq1 = sum((diff * scov1inv) .* diff, dims=2)


diff = dbn2 .- reshape(smean2, (1, 1156))
chisq2 = sum((diff * scov2inv) .* diff, dims=2)

p = histogram(chisq1[:])
p = histogram(chisq2[:])
chisqsamp = rand(Distributions.Chisq(1156/2), 5000)
histogram!(chisqsamp)


diff = dbn1 .- reshape(smean1, (1, 1156))
chisq12 = sum((diff * scov2inv) .* diff, dims=2)


diff = dbn2 .- reshape(smean2, (1, 1156))
chisq21 = sum((diff * scov1inv) .* diff, dims=2)

p = histogram(chisq1[:], label="ChiSq1-1")
histogram!(chisq12[:], label="ChiSq1-2")
histogram!(chisqsamp, label="ChiSqExpected")


p = histogram(chisq2[:],label="ChiSq2")
histogram!(chisq21[:],  label="ChiSq2-1")
histogram!(chisqsamp, label="ChiSqExpected")


##Save for MAF
Nx=64
dbnimg = readsfd(Nx, logbool=true)
fname = "scratch_NM/StandardizedExp/Nx64/noisy_stdtrue/SFDTargSFDCov/log_apd_noiso/lamcorr_recon_1000"
gttarget = load(fname*".jld2")
fhash = gttarget["filter_hash"]
Nf = size(fhash["filt_index"])[1]
coeff_mask = falses(2+Nf+Nf^2)
coeff_maskS20 = trues((Nf, Nf))
coeff_mask[3+Nf:end] .= triu(coeff_maskS20)[:]
dhc_args = Dict(:doS2=>false, :doS12=>false, :doS20=>true, :apodize=>true, :iso=>false)
dbn = get_dbn_coeffs(dbnimg, fhash, dhc_args, coeff_mask= coeff_mask)


f = FITS("scratch_NM/dbncoeffs/S20sym_noiso_log.fits", "w")
write(f, dbn)
close(f)


##Implementing log S20 derivatives
Nx=64
im = readsfd(Nx)
true_img = im[:, :, 1]

noise = rand(Normal(0.0, std(true_img)), Nx, Nx)
init = true_img + noise
#heatmap(init)
#heatmap(true_img)

filter_hash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1, Omega=true)
(S1iso, Nf) = size(filter_hash["S1_iso_mat"])
(S2iso, Nfsq) = size(filter_hash["S2_iso_mat"])
dhc_args = Dict(:doS2=>false, :doS12=>false, :doS20=>true, :apodize=>true, :iso=>false) #Iso #CHANGE: Change sig for the sfd data since the noise model is super high and the tiny values make sense
if dhc_args[:iso]
    coeff_mask = falses(2+S1iso+S2iso)
    coeff_mask[S1iso+3:end] .= true
else #Not iso
    coeff_mask = falses(2+Nf+Nf^2)
    coeff_maskS20 = trues((Nf, Nf))
    coeff_mask[3+Nf:end] .= true
end

white_noise_args = Dict(:loc=>0.0, :sig=>std(true_img), :Nsam=>1000, :norm=>false, :smooth=>false, :smoothval=>0.8) #Iso #only if you're using noise based covar
optim_settings = Dict([("iterations", 1000), ("norm", false), ("minmethod", ConjugateGradient())])
recon_settings = Dict([("target_type", "sfd_dbn"), ("covar_type", "sfd_dbn"), ("log", true), ("GaussianLoss", false),("TransformedGaussianLoss", true), ("Invcov_matrix", "Diagonal+Eps"),
  ("optim_settings", optim_settings), ("lambda", 0.00456)]) #Add constraints

dbnocffs = log.(get_dbn_coeffs(log.(im), filter_hash, dhc_args, coeff_mask = coeff_mask))
fmean = mean(dbnocffs, dims=1)
fcov = (dbnocffs .- fmean)' * (dbnocffs .- fmean) ./(size(dbnocffs)[1] -1)
fcovinv = invert_covmat(fcov, 1e-8)

recon_settings["fs_targ_mean"] = fmean[:]
recon_settings["fs_invcov"] = fcovinv
recon_settings["safemode"] = true
fname_save = "scratch_NM/NewWrapper/TransfGaussian/test"
recon_settings["fname_save"] = fname_save * ".jld2"
recon_settings["optim_settings"] = optim_settings
recon_settings["transform_func"] = Data_Utils.fnlog
recon_settings["transform_dfunc"] = Data_Utils.fndlog
resobj, recon_img = reconstruction_wrapper(true_img, init, filter_hash, dhc_args, coeff_mask, recon_settings)
ar1 = convert(Array{Any, 1}, exp.(rand(2)))
Data_Utils.fnlog(ar1)
p= plot([t.value for t in Optim.trace(resobj)])
plot!(title="Loss: S20 | Targ = S(SFD) | Cov = SFD Covariance", xlabel = "No. Iterations", ylabel = "Loss")
savefig(p, fname_save * "_trace.png")

Visualization.plot_diffscales([apodizer(true_img), apodizer(init), apodizer(recon_img), apodizer(recon_img) - apodizer(true_img)], ["GT", "Init", "Reconstruction", "Residual"], fname=fname_save*".png")
Visualization.plot_synth_QA(apodizer(true_img), apodizer(init), apodizer(recon_img), fname=fname_save*"_6p.png")

println("Mean Abs Res: Init-True = ", mean(abs.(init - true_img)), " Recon-True = ", mean(abs.(recon_img - true_img)))
println("Mean Abs Frac Res", mean(abs.((init - true_img)./true_img)), " Recon-True=", mean(abs.((recon_img - true_img)./true_img)))


##########################################################################
##DEBUG
tonorm = false
input = init
fs_targ_mean = fmean[:]
fs_targ_invcov = fcovinv
lambda = 0.004
Nx=64
coeff_mask = falses(2+Nf+Nf^2)
coeff_mask[3+Nf:end] .= true
if dhc_args[:apodize]
    Ap = wind_2d(Nx)
    cA = sum(Ap)
    Apflat = reshape(Ap, Nx^2)
    od_nz_idx = findall(!iszero, Apflat) #findall((x->((x!=0) & (x!=1))), Apflat)
    #od_zidx = findall(iszero, Apflat)
    avals = Apflat[od_nz_idx]
    dA = zeros(Nx^2, Nx^2)
    dA[:, od_nz_idx] .= ((1.0 .- Apflat) * avals')./cA
    dA[diagind(dA)] += Apflat
else
    dA = I
end

function adaptive_apodizer(input_image::Array{Float64, 2})
    if dhc_args[:apodize]
        apd_image = apodizer(input_image)
    else
        apd_image = input_image
    end
    return apd_image
end

if (dhc_args[:doS20]) & (dhc_args[:iso])
    iso2nf2mask = zeros(Int64, Nf^2)
    M20 = filter_hash["S2_iso_mat"]
    for id=1:Nf^2 iso2nf2mask[M20.colptr[id]] = M20.rowval[id] end
    coeff_masks20 = coeff_mask[3+N1iso:end]
else# (dhc_args[:doS20]) & (!dhc_args[:iso])
    coeff_masks20 = coeff_mask[3+Nf:end]
end

function augment_weights(inpwt::Array{Float64})
    if (dhc_args[:doS20]) & (dhc_args[:iso]) #wt (Nc) -> |S2_iso| -> |Nf^2|
        w_s2iso = zeros(size(filter_hash["S2_iso_mat"])[1])
        w_s2iso[coeff_masks20] .= inpwt
        w_nf2 = zeros(Nf^2)
        w_nf2 .= w_s2iso[iso2nf2mask]
        return reshape(w_nf2, (Nf, Nf))
    else#if (dhc_args[:doS20]) & (!dhc_args[:iso])
        w_nf2 = zeros(Nf^2)
        w_nf2[coeff_masks20] .= inpwt
        return reshape(w_nf2, (Nf, Nf))
    end
end

function loss_func20(img_curr::Array{Float64, 2})
    #println(typeof(DHC_compute_wrapper(img_curr, filter_hash, norm=tonorm; dhc_args...)[coeff_mask]))
    s_curr = DHC_compute_wrapper(img_curr, filter_hash, norm=tonorm; dhc_args...)[coeff_mask] #Edit
    regterm =  0.5*lambda*sum((adaptive_apodizer(img_curr) - adaptive_apodizer(input)).^2)
    println(size(s_curr), size(fs_targ_mean), size(fs_targ_invcov))
    lnlik = ( 0.5 .* (s_curr - fs_targ_mean)' * fs_targ_invcov * (s_curr - fs_targ_mean))
    neglogloss = lnlik[1] + regterm
    return neglogloss
end
#TODO: Need to replace the deriv_sums20 with a deriv_sum wrapper that handles all cases (S12, S20, S2)

pixmask=falses(Nx, Nx)
function dloss20(storage_grad::Array{Float64, 2}, img_curr::Array{Float64, 2})
    s_curr = DHC_compute_wrapper(img_curr, filter_hash, norm=tonorm; dhc_args...)[coeff_mask]
    diff = s_curr - fs_targ_mean
    #Add branches here
    wt1 = convert(Array{Float64,1}, reshape(transpose(diff) * fs_targ_invcov, (length(diff),)))
    println(size(wt1), typeof(wt1))
    wt = wt1# .* fill(1.0, length(diff))#(1.0./s_curr)
    wtaug = augment_weights(wt)
    #println(wt - wt1)
    apdimg_curr = adaptive_apodizer(img_curr)
    storage_grad .= reshape((Deriv_Utils_New.wst_S20_deriv_sum(apdimg_curr, filter_hash, wtaug, FFTthreads=1)' + reshape(lambda.*(apdimg_curr - adaptive_apodizer(input)), (1, Nx^2))) * dA, (Nx, Nx))
    storage_grad[pixmask] .= 0 # better way to do this by taking pixmask as an argument wst_s2_deriv_sum?
    return
end


println("Diff check")
eps = zeros(size(input))
row, col = 24, 18 #convert(Int8, Nx/2), convert(Int8, Nx/2)+3
eps[row, col] = 1e-6
chisq1 = loss_func20(input+eps./2) #DEB
chisq0 = loss_func20(input-eps./2) #DEB
brute  = (chisq1-chisq0)/1e-6
#df_brute = DHC_compute(reshape(input, (Nx, Nx)), filter_hash, doS2=true, doS12=false, doS20=false, norm=false)[coeff_mask] - s_targ_mean
clever = zeros(size(input)) #DEB
meanval = mean(adaptive_apodizer(input)) ./ mean(wind_2d(Nx))
_bar = dloss20(clever, input) #DEB
println("Chisq Derve Check")
println("Brute:  ",brute)
println("Clever: ",clever[row, col], " Difference: ", brute - clever[row, col]) #DEB

#####################################
##Debug: Check old code: Works with triangular mask
Nx=64
im = readsfd(Nx)
true_img = im[:, :, 1]

noise = rand(Normal(0.0, std(true_img)), Nx, Nx)
init = true_img + noise
#heatmap(init)
#heatmap(true_img)

filter_hash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1, Omega=true)
(S1iso, Nf) = size(filter_hash["S1_iso_mat"])
(S2iso, Nfsq) = size(filter_hash["S2_iso_mat"])
dhc_args = Dict(:doS2=>false, :doS12=>false, :doS20=>true, :apodize=>true, :iso=>false) #Iso #CHANGE: Change sig for the sfd data since the noise model is super high and the tiny values make sense
if dhc_args[:iso]
    coeff_mask = falses(2+S1iso+S2iso)
    coeff_mask[S1iso+3:end] .= true
else #Not iso
    coeff_mask = falses(2+Nf+Nf^2)
    coeff_maskS20 = trues((Nf, Nf))
    coeff_mask[3+Nf:end] .= triu(coeff_maskS20)[:]
end


white_noise_args = Dict(:loc=>0.0, :sig=>std(true_img), :Nsam=>1000, :norm=>false, :smooth=>false, :smoothval=>0.8) #Iso #only if you're using noise based covar
optim_settings = Dict([("iterations", 10), ("norm", false), ("minmethod", ConjugateGradient())])
recon_settings = Dict([("target_type", "sfd_dbn"), ("covar_type", "sfd_dbn"), ("log", true), ("GaussianLoss", true),("TransformedGaussianLoss", false), ("Invcov_matrix", "Diagonal+Eps"),
  ("optim_settings", optim_settings), ("lambda", 0.00456)]) #Add constraints

dbnocffs = get_dbn_coeffs(log.(im), filter_hash, dhc_args, coeff_mask = coeff_mask) #removed log from here
fmean = mean(dbnocffs, dims=1)
fcov = (dbnocffs .- fmean)' * (dbnocffs .- fmean) ./(size(dbnocffs)[1] -1)
fcovinv = invert_covmat(fcov, 1e-5)

recon_settings["s_targ_mean"] = fmean[:]
recon_settings["s_invcov"] = fcovinv
recon_settings["safemode"] = true
fname_save = "scratch_NM/NewWrapper/TransfGaussian/nof_test"
recon_settings["fname_save"] = fname_save * ".jld2"
recon_settings["optim_settings"] = optim_settings

resobj, recon_img = reconstruction_wrapper(true_img, init, filter_hash, dhc_args, coeff_mask, recon_settings)


p= plot([t.value for t in Optim.trace(resobj)])
plot!(title="Loss: S20 | Targ = S(SFD) | Cov = SFD Covariance", xlabel = "No. Iterations", ylabel = "Loss")
savefig(p, fname_save * "_trace.png")

Visualization.plot_diffscales([apodizer(true_img), apodizer(init), apodizer(recon_img), apodizer(recon_img) - apodizer(true_img)], ["GT", "Init", "Reconstruction", "Residual"], fname=fname_save*".png")
Visualization.plot_synth_QA(apodizer(true_img), apodizer(init), apodizer(recon_img), fname=fname_save*"_6p.png")

println("Mean Abs Res: Init-True = ", mean(abs.(init - true_img)), " Recon-True = ", mean(abs.(recon_img - true_img)))
println("Mean Abs Frac Res", mean(abs.((init - true_img)./true_img)), " Recon-True=", mean(abs.((recon_img - true_img)./true_img)))


##Log-Log check: Works

Nx=64
im = readsfd(Nx)
true_img = im[:, :, 1]

noise = rand(Normal(0.0, std(true_img)), Nx, Nx)
init = true_img + noise
#heatmap(init)
#heatmap(true_img)

filter_hash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1, Omega=true)
(S1iso, Nf) = size(filter_hash["S1_iso_mat"])
(S2iso, Nfsq) = size(filter_hash["S2_iso_mat"])
dhc_args = Dict(:doS2=>false, :doS12=>false, :doS20=>true, :apodize=>true, :iso=>false) #Iso #CHANGE: Change sig for the sfd data since the noise model is super high and the tiny values make sense
if dhc_args[:iso]
    coeff_mask = falses(2+S1iso+S2iso)
    coeff_mask[S1iso+3:end] .= true
else #Not iso
    coeff_mask = falses(2+Nf+Nf^2)
    coeff_maskS20 = trues((Nf, Nf))
    coeff_mask[3+Nf:end] .= triu(coeff_maskS20)[:]
end


white_noise_args = Dict(:loc=>0.0, :sig=>std(true_img), :Nsam=>1000, :norm=>false, :smooth=>false, :smoothval=>0.8) #Iso #only if you're using noise based covar
optim_settings = Dict([("iterations", 1000), ("norm", false), ("minmethod", ConjugateGradient())])
recon_settings = Dict([("target_type", "sfd_dbn"), ("covar_type", "sfd_dbn"), ("log", true), ("GaussianLoss", false),("TransformedGaussianLoss", true), ("Invcov_matrix", "Diagonal+Eps"),
  ("optim_settings", optim_settings), ("lambda", std(apodizer(log.(init)))^(-2) )]) #Add constraints

dbnocffs = get_dbn_coeffs(log.(im), filter_hash, dhc_args, coeff_mask = coeff_mask) #removed log from here
fmean = mean(dbnocffs, dims=1)
fcov = (dbnocffs .- fmean)' * (dbnocffs .- fmean) ./(size(dbnocffs)[1] -1)
fcovinv = invert_covmat(fcov, 1e-6)

recon_settings["transform_func"] = Data_Utils.fnlog
recon_settings["transform_dfunc"] = Data_Utils.fndlog
recon_settings["fs_targ_mean"] = fmean[:]
recon_settings["fs_invcov"] = fcovinv
recon_settings["safemode"] = false
fname_save = "scratch_NM/NewWrapper/TransfGaussian/nof_test"
recon_settings["fname_save"] = fname_save * ".jld2"
recon_settings["optim_settings"] = optim_settings

resobj, recon_img = reconstruction_wrapper(true_img, init, filter_hash, dhc_args, coeff_mask, recon_settings)


p= plot([t.value for t in Optim.trace(resobj)])
plot!(title="Loss: S20 | Targ = S(SFD) | Cov = SFD Covariance", xlabel = "No. Iterations", ylabel = "Loss")
savefig(p, fname_save * "_trace.png")

Visualization.plot_diffscales([apodizer(true_img), apodizer(init), apodizer(recon_img), apodizer(recon_img) - apodizer(true_img)], ["GT", "Init", "Reconstruction", "Residual"], fname=fname_save*".png")
Visualization.plot_synth_QA(apodizer(true_img), apodizer(init), apodizer(recon_img), fname=fname_save*"_6p.png")

println("Mean Abs Res: Init-True = ", mean(abs.(init - true_img)), " Recon-True = ", mean(abs.(recon_img - true_img)))
println("Mean Abs Frac Res", mean(abs.((init - true_img)./true_img)), " Recon-True=", mean(abs.((recon_img - true_img)./true_img)))
