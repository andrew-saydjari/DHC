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

direc = ARGS[4]
datfile = direc * "Data_" * string(numfile) * ".jld2" #Replace w SLURM array
loaddf = load(datfile)
true_img = loaddf["true_img"]
init = loaddf["init"]
Nx=size(true_img)[1]
filter_hash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1, Omega=true)
Nf = size(filter_hash["filt_index"])[1]
dhc_args = Dict(:doS2=>false, :doS12=>false, :doS20=>true, :apodize=>apdbool, :iso=>isobool)
if dhc_args[:iso]
    coeff_mask = falses(2+S1iso+S2iso)
    coeff_mask[S1iso+3:end] .= true
else #Not iso
    coeff_mask = falses(2+Nf+Nf^2)
    coeff_mask[Nf+3:end] .= triu(trues(Nf, Nf))[:]
end
recon_settings = Dict([("log", logbool), ("covar_type", "sfd_dbn"), ("target_type", "sfd_dbn"), ("Invcov_matrix", ARGS[6]), ("epsvalue", 1e-10)])
println("Apdbool", apdbool)
println("Isobool", isobool)
println(recon_settings)

s2mean, s2icov = meancov_generator(true_img, filter_hash, dhc_args, coeff_mask, recon_settings, safety=nothing)
invvar1 = diag(s2icov)

#Check wrt calculation here
dbnimg = readsfd(Nx, logbool=logbool)
dbn = get_dbn_coeffs(dbnimg, filter_hash, dhc_args, coeff_mask = nothing)
invvar2 = std(dbn, dims=1)[:]
println(size(invvar2))
invvar2 = (invvar2.^(-2))[coeff_mask]
println("InvVar1", invvar1[:10])
println("InvVar2", invvar2[:10])
println("Mean Abs Error = ", mean(abs.(invvar1 .- invvar2)))
println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")