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
using LineSearches
using SparseArrays

push!(LOAD_PATH, pwd()*"/../../../main")
using DHC_2DUtils
push!(LOAD_PATH, pwd()*"/../../../scratch_NM")
using Deriv_Utils_New
using Data_Utils
using Visualization
using ReconFuncs


numfile = 1000

logbool = false
apdbool = true
isobool = false
invcovtype = "Full+Eps"

direc = "scratch_NM/StandardizedExp/Nx64/" #Change
datfile = direc * "Data_" * string(numfile) * ".jld2" #Replace w SLURM array
loaddf = load(datfile)
true_img = loaddf["true_img"]
init = loaddf["init"]

fname_save = direc * "SFDTargSFDCov/" * string(logbool) * "_" * string(apdbool) * "_" * string(isobool) * "/LambdaVary/" * string(numfile) * "rand_init"  #Change
Nx=size(true_img)[1]
filter_hash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1, Omega=true)
(S1iso, Nf) = size(filter_hash["S1_iso_mat"])
(S2iso, Nfsq) = size(filter_hash["S2_iso_mat"])
dhc_args = Dict(:doS2=>false, :doS20=>true, :apodize=>apdbool, :iso=>isobool) #Iso #CHANGE: Change sig for the sfd data since the noise model is super high and the tiny values make sense
if dhc_args[:iso]
    coeff_mask = falses(2+S1iso+S2iso)
    coeff_mask[S1iso+3:end] .= true
else #Not iso
    coeff_mask = falses(2+Nf+Nf^2)
    coeff_mask[Nf+3:end] .= triu(trues(Nf, Nf))[:]
end

white_noise_args = Dict(:loc=>0.0, :sig=>std(true_img), :Nsam=>1000, :norm=>false, :smooth=>false, :smoothval=>0.8) #Iso #only if you're using noise based covar
optim_settings = Dict([("iterations", 1000), ("norm", false), ("minmethod", ConjugateGradient(linesearch=LineSearches.BackTracking()))])
recon_settings = Dict([("target_type", "ground_truth"), ("covar_type", "sfd_dbn"), ("log", logbool), ("GaussianLoss", false), ("iteration_init", randn(Nx, Nx)), ("TransformedGaussianLoss", false), ("RandomInit", true), ("Invcov_matrix", invcovtype), ("epsvalue", 1e-8),
  ("optim_settings", optim_settings), ("white_noise_args", white_noise_args)]) #Add constraints

recon_settings["datafile"] = datfile

regs_true = DHC_compute_wrapper(log.(true_img), filter_hash; norm=false, dhc_args...)[coeff_mask]
s2mean, s2icov = meancov_generator(true_img, filter_hash, dhc_args, coeff_mask, recon_settings, safety=nothing)
#Check
#s2mean - s_true

if recon_settings["log"] & dhc_args[:apodize]
    lval = std(apodizer(log.(true_img))).^(-2)
elseif !recon_settings["log"] & dhc_args[:apodize]
    s_init = DHC_compute_wrapper(apodizer(init), filter_hash, norm=false; dhc_args...)[coeff_mask]
    l1init = ( 0.5 .* (s_init - s2mean)' * s2icov * (s_init - s2mean))[1]
    println("L1init", l1init)
    est1 = std(apodizer(true_img)).^(-2)
    wn_exp = mean(wind_2d(Nx).^2)*Nx*Nx*(loaddf["std"])^2
    lval = minimum([est1, 0.01*(l1init - 1.0)/(wn_exp)])
elseif recon_settings["log"] & !dhc_args[:apodize]
    lval = std(log.(true_img)).^(-2)
else
    s_init = DHC_compute_wrapper(init, filter_hash, norm=false; dhc_args...)[coeff_mask]
    l1init = ( 0.5 .* (s_init - s2mean)' * s2icov * (s_init - s2mean))[1]
    println("L1init", l1init)
    est1 = std(true_img).^(-2)
    wn_exp = Nx*Nx*(loaddf["std"])^2
    lval = minimum([est1, 0.01*(l1init - 1.0)/(wn_exp)])
end

recon_settings["lambda"] = lval
println("Regularizer Lambda=", round(lval, sigdigits=3))

recon_settings["s_targ_mean"] = s2mean
recon_settings["s_invcov"] = s2icov
recon_settings["safemode"] = false
recon_settings["fname_save"] = fname_save * ".jld2"
recon_settings["optim_settings"] = optim_settings
resobj, recon_img = reconstruction_wrapper(true_img, imfilter(init, Kernel.gaussian(0.8)), filter_hash, dhc_args, coeff_mask, recon_settings)

plot(heatmap(true_img), heatmap(recon_settings["iteration_init"]), heatmap(recon_img))



DHC_compute_wrapper(recon_img, filter_hash, norm=false; dhc_args...) .- DHC_compute_wrapper(true_img, filter_hash, norm=false; dhc_args...)



p= plot([t.value for t in Optim.trace(resobj)])
plot!(title="Loss: S20 | Targ = S(SFD) | Cov = SFD Covariance", xlabel = "No. Iterations", ylabel = "Loss")
savefig(p, fname_save * "_trace.png")

Visualization.plot_diffscales([apodizer(true_img), apodizer(init), apodizer(recon_img), apodizer(recon_img) - apodizer(true_img)], ["GT", "Init", "Reconstruction", "Residual"], fname=fname_save*".png")
Visualization.plot_synth_QA(apodizer(true_img), apodizer(init), apodizer(recon_img), fname=fname_save*"_6p.png")

println("Input Data", datfile)
println("Output File", fname_save)
println("Mean Abs Res: Init-True = ", mean(abs.(init - true_img)), " Recon-True = ", mean(abs.(recon_img - true_img)))
println("Mean Abs Frac Res", mean(abs.((init - true_img)./true_img)), " Recon-True=", mean(abs.((recon_img - true_img)./true_img)))
println("Mean Abs Res: Init-True = ", mean((init - true_img).^2).^0.5, " Recon-True = ", mean((recon_img - true_img).^2).^0.5)
