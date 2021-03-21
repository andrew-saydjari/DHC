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
using FITSIO


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

dhc_args = Dict(:doS2=>false, :doS12=>false, :doS20=>true, :apodize=>false)
white_noise_args = Dict(:loc=>0.0, :sig=>1.0, :Nsam=>1000, :iso=>false, :norm=>false, :smooth=>false, :smoothval=>0.8) #only if you're using noise based covar
optim_settings = Dict([("iterations", 20), ("norm", false)])
recon_settings = Dict([("target_type", "ground truth"),("covar_type", "white_noise"), ("log", false), ("GaussianLoss", true), ("Invcov_matrix", "Diagonal+Eps"),
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
