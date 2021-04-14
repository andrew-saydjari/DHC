using Statistics
using Plots
using FFTW
using Statistics
using Optim
using Images, FileIO, ImageIO
using Printf
using Revise
using Profile
using LinearAlgebra
using JLD2
using Random
using Distributions
using FITSIO
using SparseArrays

push!(LOAD_PATH, pwd()*"/../main")
using DHC_2DUtils
push!(LOAD_PATH, pwd()*"/../scratch_NM")
using Deriv_Utils_New
using Data_Utils
using Visualization
using ReconFuncs

function kldiv_from_samples(samps, qGaussian)
    #size(samps) = (Nsamp, Ndim)
    #qGaussian is a function which acts on each row of samps
    #MC estimate of KLP||Q: cant do this naively
    Nsamps = size(samps)[1]
    logqeval = zeros(Nsamps)
    for s=1:Nsamps
        qgeval = qGaussian(samps[s, 1:end])
        println(size(qgeval), " Val", qgeval)
        logqeval[s] = qgeval
    end
    println(logqeval)
    logpeval = fill(-log.(Nsamps), Nsamps)
    kld = sum(logpeval .- logqeval)./Nsamps
    return kld
end

function Gaussian_expr(meantarg, invcov)
    function Gwrapper(x)
        return -0.5.* (x- meantarg)' * invcov * (x- meantarg)
    end
    return Gwrapper
end

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
    coeff_mask[Nf+3:end] .= true
end

white_noise_args = Dict(:loc=>0.0, :sig=>std(true_img), :Nsam=>1000, :norm=>false, :smooth=>false, :smoothval=>0.8) #Iso #only if you're using noise based covar
optim_settings = Dict([("iterations", 1000), ("norm", false), ("minmethod", ConjugateGradient())])
recon_settings = Dict([("target_type", "sfd_dbn"), ("covar_type", "sfd_dbn"), ("log", true), ("GaussianLoss", true), ("Invcov_matrix", "Diagonal+Eps"),
  ("optim_settings", optim_settings), ("lambda", 0.00456), ("white_noise_args", white_noise_args)]) #Add constraints
regs_true = DHC_compute_wrapper(log.(true_img), filter_hash; norm=false, dhc_args...)[coeff_mask]
s2mean, s2icov = meancov_generator(true_img, filter_hash, dhc_args, coeff_mask, recon_settings, safety=nothing)
qg = Gaussian_expr(s2mean, s2icov)
psamps = get_dbn_coeffs(log.(im), filter_hash, dhc_args, coeff_mask=coeff_mask)
#println("Check ", qg())
println("KLDiv = ", kldiv_from_samples(psamps, qg))


