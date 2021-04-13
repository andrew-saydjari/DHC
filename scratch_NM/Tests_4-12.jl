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
using StatsBase
using LineSearches
using SparseArrays

push!(LOAD_PATH, pwd()*"/main")
using DHC_2DUtils
push!(LOAD_PATH, pwd()*"/scratch_NM")
using Deriv_Utils_New
using Data_Utils
using Visualization
using ReconFuncs



logbool=false
apdbool = true
isobool=false
invcovhandling = "Full"

datfile = "scratch_NM/StandardizedExp/Nx64/Data_1000.jld2" #Replace w SLURM array
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
recon_settings = Dict([("log", logbool), ("covar_type", "sfd_dbn"), ("target_type", "sfd_dbn"), ("Invcov_matrix", invcovhandling), ("epsvalue", 1e-10)])
println("Apdbool", apdbool)
println("Isobool", isobool)
println(recon_settings)

s2mean, s2icov = meancov_generator(true_img, filter_hash, dhc_args, coeff_mask, recon_settings, safety=nothing)
invvar1 = diag(s2icov)

#Check wrt calculation here
dbnimg = readsfd(Nx, logbool=logbool)
dbn = get_dbn_coeffs(dbnimg, filter_hash, dhc_args, coeff_mask = nothing)
dbntriu = dbn[:, coeff_mask]
fmean = mean(dbntriu, dims=1)
fcov = (dbntriu .- fmean)' * (dbntriu .- fmean) ./ 9999
println("CovMat Dynamic Range: ")
println("Max: ", maximum(fcov))
println("Min: ", minimum(fcov))
emax = eigmax(fcov)
emin = eigmin(fcov)
evals = eigvals(fcov)
findall(evals.>1e-6)
svd_out = svd(fcov)
lowrank_approx = svd_out.U[1:end, 1:497] * Diagonal(svd_out.S[1:497]) * svd_out.V'[1:497, 1:end]
lowrank_approx
res = sum((lowrank_approx .- fcov).^2)
sum(svd_out.S[498:end].^2)
cond(fcov)

fullrank_approx = svd_out.U * Diagonal(svd_out.S) * svd_out.V
res = sum((fullrank_approx .- fcov).^2)

fcov -> add eps to the diag -> invcov -> diag(invcov) ?= std(dbncoeffs).^(-2)


end

randmat = randn(1000, 595)
prod = randmat' * randmat ./999
svdrand = svd(prod)
frrand = svdrand.U * Diagonal(svdrand.S) * svdrand.V'
sum((frrand .- prod).^2)

dbntriu_scaled = standardize(UnitRangeTransform, dbntriu, dims=1)
fmean_sc = mean(dbntriu_scaled, dims=1)
fcovsc = (dbntriu_scaled .- fmean_sc)' * (dbntriu_scaled .- fmean_sc) ./ 9999
findmax(dbntriu_scaled[1:end, 1])
findmax(dbntriu_scaled[1:end, 2])
maxcoeff_imgs = (x->findmax(dbntriu_scaled[1:end, x])).(collect(1:595))
badimg  =hcat((x->[x[1], x[2]]).(maxcoeff_imgs)...)[2, 1:end]
badimg = convert(Array{Int16}, badimg)
for img in unique(badimg)
    p= heatmap(dbnimg[:, :, img])
    savefig(p, "scratch_NM/NewWrapper/4-12/Img_"*string(img))
end

#Remove outlier images
imgmask = trues(10000)
imgmask[unique(badimg)] .= false
dbn = get_dbn_coeffs(dbnimg[:, :, imgmask], filter_hash, dhc_args, coeff_mask = nothing)
dbntriu = dbn[:, coeff_mask]
fmean = mean(dbntriu, dims=1)
fcov = (dbntriu .- fmean)' * (dbntriu .- fmean) ./ 9984
svd_out = svd(fcov)
fullrank_approx = svd_out.U * Diagonal(svd_out.S) * svd_out.V
res = sum((fullrank_approx .- fcov).^2)


invvar2 = std(dbn, dims=1)[:]
println(size(invvar2))
invvar2 = (invvar2.^(-2))[coeff_mask]
println("InvVar1", invvar1[:10])
println("InvVar2", invvar2[:10])
println("Mean Abs Error = ", mean(abs.(invvar1 .- invvar2)))
println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

dhc_argsnorm = Dict(:doS2=>false, :doS12=>false, :doS20=>true, :apodize=>apdbool, :iso=>isobool, :norm=>true)
(N1iso,Nf) = size(filter_hash["S1_iso_mat"])
(N2iso,Nfsq) = size(filter_hash["S2_iso_mat"])
Ncovsamp = size(dbnimg)[3]
s20_dbn = zeros(Float64, Ncovsamp, 2+ (dhc_args[:iso] ? N1iso+N2iso : Nf+Nf^2))
for idx=1:Ncovsamp
    s20_dbn[idx, :] = DHC_compute_wrapper(dbnimg[:, :, idx], filter_hash; dhc_argsnorm...)
end
if coeff_mask!=nothing
    return s20_dbn[:, coeff_mask]
else
    return s20_dbn
end
