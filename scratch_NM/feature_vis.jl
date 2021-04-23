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

#Rewritten
logbool= false                                                #Calculate coefficients of the log of the image?
apdbool = true                                                #Apodize?
isobool = false                                               #Iso?

#Image you want to start with
Nx=64
sfdall = readsfd_fromsrc("scratch_NM/data/dust10000.fits", Nx, logbool=false)
input = sfdall[:, :, 5]                     #Less of a memory crunch if you just read in the one image you want to start with

filter_hash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1, Omega=true)
dhc_args = Dict(:doS2=>false, :doS20=>true, :apodize=>apdbool, :iso=>isobool)
lambda = 0.0
optim_settings = Dict([("iterations", 1000), ("norm", false), ("minmethod", ConjugateGradient())])

#Coefficient to optimize wrt
Jvalue = 1
Lvalue = 0
#Code currently requires you to access S1 through the S2 matrix diagonal
mask_nf = (filter_hash["J_L"][:, 1] .== Jvalue) .& (filter_hash["J_L"][:, 2] .== Lvalue)
mask_nf2 = Diagonal(mask_nf)
(Nf,) = size(filter_hash["filt_index"])
coeff_mask = falses(2+Nf+Nf^2)
coeff_mask[3+Nf:end] .= mask_nf2[:]

#Setting a target coefficient: currently I'm taking the 10*true target coefficient
#=

if logbool
    procdbn = log.(sfdall)
else
    procdbn = sfdall
end

dbncoeffs = get_dbn_coeffs(procdbn, filter_hash, dhc_args, coeff_mask = coeff_mask)
sigval = std(dbncoeffs, dims=1)
=#

if logbool
    s_targ_mean = 0.1*DHC_compute_wrapper(log.(input), filter_hash, norm=false; dhc_args...)[coeff_mask]             #target coefficient vector = twice the current wst value
    s_targ_invcov = Diagonal([0.1.^(-2)])
    res, recon_img = Deriv_Utils_New.image_recon_derivsum_regularized(log.(input), filter_hash, s_targ_mean, s_targ_invcov, falses(Nx, Nx), dhc_args, coeff_mask=coeff_mask, optim_settings=optim_settings, lambda=lambda)
    recon_img = exp.(recon_img)
else
    s_targ_mean = 0.1*DHC_compute_wrapper(input, filter_hash, norm=false; dhc_args...)[coeff_mask]             #target coefficient vector = twice the current wst value
    s_targ_invcov = Diagonal([0.0015.^(-2)])          #Might need to tune. inverse covariance matrix with sigma= true_coeff/10 for J=1 L=2
    res, recon_img = Deriv_Utils_New.image_recon_derivsum_regularized(input, filter_hash, s_targ_mean, s_targ_invcov, falses(Nx, Nx), dhc_args, optim_settings=optim_settings, coeff_mask=coeff_mask, lambda=lambda)
end

heatmap(input)
heatmap(recon_img)


#Iso
#Rewritten
logbool= false                                               #Calculate coefficients of the log of the image?
apdbool = true                                               #Apodize?
isobool = true                                               #Iso?

#Image you want to start with
Nx=64
sfdall = readsfd_fromsrc("scratch_NM/data/dust10000.fits", Nx, logbool=false)
input = sfdall[:, :, 5]                     #Less of a memory crunch if you just read in the one image you want to start with

filter_hash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1, Omega=true)
dhc_args = Dict(:doS2=>false, :doS20=>true, :apodize=>apdbool, :iso=>isobool)
lambda = 0.0
optim_settings = Dict([("iterations", 1000), ("norm", false), ("minmethod", ConjugateGradient())])

#Coefficient to optimize wrt
Jvalue = 1
Lvalue = 0
filter_hash["S2_iso_mat"]
#Code currently requires you to access S1 through the S2 matrix diagonal
mask_nf = (filter_hash["J_L"][:, 1] .== Jvalue) .& (filter_hash["J_L"][:, 2] .== Lvalue)
mask_nf2 = Diagonal(mask_nf)
(Nf,) = size(filter_hash["filt_index"])
coeff_mask = falses(2+Nf+Nf^2)
coeff_mask[3+Nf:end] .= mask_nf2[:]


if logbool
    s_targ_mean = 0.1*DHC_compute_wrapper(log.(input), filter_hash, norm=false; dhc_args...)[coeff_mask]             #target coefficient vector = twice the current wst value
    s_targ_invcov = Diagonal([0.1.^(-2)])
    res, recon_img = Deriv_Utils_New.image_recon_derivsum_regularized(log.(input), filter_hash, s_targ_mean, s_targ_invcov, falses(Nx, Nx), dhc_args, coeff_mask=coeff_mask, optim_settings=optim_settings, lambda=lambda)
    recon_img = exp.(recon_img)
else
    s_targ_mean = 0.1*DHC_compute_wrapper(input, filter_hash, norm=false; dhc_args...)[coeff_mask]             #target coefficient vector = twice the current wst value
    s_targ_invcov = Diagonal([0.0015.^(-2)])          #Might need to tune. inverse covariance matrix with sigma= true_coeff/10 for J=1 L=2
    res, recon_img = Deriv_Utils_New.image_recon_derivsum_regularized(input, filter_hash, s_targ_mean, s_targ_invcov, falses(Nx, Nx), dhc_args, optim_settings=optim_settings, coeff_mask=coeff_mask, lambda=lambda)
end

heatmap(input)
heatmap(recon_img)
