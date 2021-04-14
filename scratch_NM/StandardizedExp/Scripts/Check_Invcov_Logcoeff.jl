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

push!(LOAD_PATH, pwd()*"/../../../main")
using DHC_2DUtils
push!(LOAD_PATH, pwd()*"/../../../scratch_NM")
using Deriv_Utils_New
using Data_Utils
using Visualization
using ReconFuncs

println("Log of Coefficients Check: Coeff_mask added in get_dbn line in invvar2")
numfile = Base.parse(Int, ENV["SLURM_ARRAY_TASK_ID"]) 
println(numfile, ARGS[1], ARGS[2])

if ARGS[1]=="log"
    logbool=true
else
    if ARGS[1]!="reg" error("Invalid log arg") end
    logbool=false
end

if ARGS[2]=="apd"
    apdbool = true
else
    if ARGS[2]!="nonapd" error("Invalid apd arg") end
    apdbool=false
end

if ARGS[3]=="iso"
    isobool = true
else
    if ARGS[3]!="noiso" error("Invalid iso arg") end
    isobool=false
end

#read in data
direc = ARGS[4]
datfile = direc * "Data_" * string(numfile) * ".jld2" #Replace w SLURM array
loaddf = load(datfile)
true_img = loaddf["true_img"]
init = loaddf["init"]

fname_save = direc * "SFDTargSFDCov/" * ARGS[1] * "_" * ARGS[2] * "_" * ARGS[3] * "/LogCoeff/" * string(numfile) * ARGS[5] #Change
Nx=size(true_img)[1]
filter_hash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1, Omega=true)
(S1iso, Nf) = size(filter_hash["S1_iso_mat"])
(S2iso, Nfsq) = size(filter_hash["S2_iso_mat"])
dhc_args = Dict(:doS2=>false, :doS12=>false, :doS20=>true, :apodize=>apdbool, :iso=>isobool) #Iso #CHANGE: Change sig for the sfd data since the noise model is super high and the tiny values make sense
if dhc_args[:iso]
    coeff_mask = falses(2+S1iso+S2iso)
    coeff_mask[S1iso+3:end] .= true
else #Not iso
    coeff_mask = falses(2+Nf+Nf^2)
    coeff_mask[Nf+3:end] .= triu(trues(Nf, Nf))[:]
end

recon_settings = Dict([("log", logbool), ("covar_type", "sfd_dbn"), ("target_type", "sfd_dbn"), ("Invcov_matrix", ARGS[6]), ("epsvalue", 1e-6)])
println("Apdbool", apdbool)
println("Isobool", isobool)
println(recon_settings)


sfdall = readsfd(Nx, logbool=logbool)
dbnocffs = log.(get_dbn_coeffs(sfdall, filter_hash, dhc_args, coeff_mask = coeff_mask))
fmean = mean(dbnocffs, dims=1)
fcov = (dbnocffs .- fmean)' * (dbnocffs .- fmean) ./(size(dbnocffs)[1] -1)
sig2 = diag(fcov)
if recon_settings["Invcov_matrix"]=="Diagonal"
    fcovinv = invert_covmat(sig2)
elseif recon_settings["Invcov_matrix"]=="Diagonal+Eps"
    fcovinv = invert_covmat(sig2, recon_settings["epsvalue"]) #DEBUG
elseif recon_settings["Invcov_matrix"]=="Full+Eps"
    fcovinv = invert_covmat(fcov, recon_settings["epsvalue"])
else#s2icov
    fcovinv = invert_covmat(fcov)
end

invvar1 = diag(fcovinv)

#Check wrt calculation here
dbn = log.(get_dbn_coeffs(sfdall, filter_hash, dhc_args, coeff_mask = coeff_mask))
invvar2 = std(dbn, dims=1)[:]
println(size(invvar2))
invvar2 = (invvar2.^(-2))
println("InvVar1 ", invvar1[1:10])
println("InvVar2 ", invvar2[1:10])
println("Mean Abs Error = ", mean(abs.(invvar1 .- invvar2)))

var1 = diag(fcov)
var2 = std(dbn, dims=1)[:]
var2 = var2.^2
println("Var1 ", var1[1:10])
println("Var2 ", var2[1:10])
println("Mean Abs Error = ", mean(abs.(var1 .- var2)))
println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
