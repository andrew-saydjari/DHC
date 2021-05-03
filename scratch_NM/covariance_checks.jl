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
using Flux
using StatsBase
using SparseArrays

push!(LOAD_PATH, pwd()*"/main")
using DHC_2DUtils
push!(LOAD_PATH, pwd()*"/scratch_NM")
using Deriv_Utils_New
using Data_Utils
using Visualization
using ReconFuncs

using IJulia
IJulia.notebook(dir="/home/nayantara/Desktop/GalacticDustProject/WST/DHC/scratch_NM/NewWrapper/4-28/")
#=
#Reg, RegCoeff
logbool = false
apdbool=true
isobool = false
Nx=64
filter_hash = fink_filter_hash(1, 8, nx=Nx, t=1, wd=1, Omega=true)
sfdall = readsfd_fromsrc("data/dust10000.fits", Nx, logbool=logbool)
dhc_args = Dict(:doS2=>false, :doS20=>true, :apodize=>apdbool, :iso=>isobool)
dbnocffs = get_dbn_coeffs(sfdall, filter_hash, dhc_args)
fmean = mean(dbnocffs, dims=1)
scov= (dbnocffs .- fmean)' * (dbnocffs .- fmean) ./(size(dbnocffs)[1] -1)
cond(scov)
scovinv = invert_covmat(scov, 1e-5)
cholupp = cholesky(scovinv).U
dbnmean =
whitened_ms = (cholupp * (dbnocffs .-  mean(dbnocffs, dims=1))')'
selidx = [3, 4]
p = scatter(whitened[:, selidx[1]], whitened[:, selidx[2]], title="Real im, Not log coeffs")
xlabel!("J=1, L=0")
ylabel!("J=2, L=0")
filter_hash["J_L"]
selidx = [3, 7]
p = scatter(whitened[:, selidx[1]], whitened[:, selidx[2]], title="Real im, Not log coeffs")
xlabel!("J=1, L=0")
ylabel!("J=1, L=1")

#Reg, LogCoeff
logdbn = log.(dbnocffs)
flmean = mean(logdbn, dims=1)
scov= (logdbn .- flmean)' * (logdbn .- flmean) ./(size(logdbn)[1] -1)
cond(scov)
scovinv = invert_covmat(scov, 1e-10)
cholupp = cholesky(scovinv).U
whitened = (cholupp * logdbn')'
selidx = [3, 4]
p = scatter(whitened[:, selidx[1]], whitened[:, selidx[2]], title="Real im, Log coeffs")
xlabel!("J=1, L=0")
ylabel!("J=2, L=0")
filter_hash["J_L"]
selidx = [3, 7]
p = scatter(whitened[:, selidx[1]], whitened[:, selidx[2]], title="Real im, Log coeffs")
xlabel!("J=1, L=0")
ylabel!("J=1, L=1")

#Log
logbool = true
apdbool=true
isobool = false
Nx=64
filter_hash = fink_filter_hash(1, 8, nx=Nx, t=1, wd=1, Omega=true)
sfdall = readsfd_fromsrc("data/dust10000.fits", Nx, logbool=logbool)
dhc_args = Dict(:doS2=>false, :doS20=>true, :apodize=>apdbool, :iso=>isobool)
dbnocffs = get_dbn_coeffs(sfdall, filter_hash, dhc_args)
fmean = mean(dbnocffs, dims=1)
scov= (dbnocffs .- fmean)' * (dbnocffs .- fmean) ./(size(dbnocffs)[1] -1)
cond(scov)
scovinv = invert_covmat(scov, 1e-5)
=#

## Reg, RegCoeff
logbool = false
apdbool=true
isobool = true
Nx=64
filter_hash = fink_filter_hash(1, 8, nx=Nx, t=1, wd=1, Omega=true)
sfdall = readsfd_fromsrc("scratch_NM/data/dust10000.fits", Nx, logbool=logbool)
dhc_args = Dict(:doS2=>false, :doS20=>true, :apodize=>apdbool, :iso=>isobool)
dbnocffs = get_dbn_coeffs(sfdall, filter_hash, dhc_args)

#Subsection
fmean = mean(dbnocffs, dims=1)
scov= (dbnocffs .- fmean)' * (dbnocffs .- fmean) ./(size(dbnocffs)[1] -1)
cond(scov)
scovinv = invert_covmat(scov, 1e-5)
cholupp = cholesky(scovinv).U
whitened_ms = (cholupp * (dbnocffs .-  mean(dbnocffs, dims=1))')'
selidx = [3, 4]

p = plot(scatter(whitened_ms[:, selidx[1]], whitened_ms[:, selidx[2]], title="Real im, Not log coeffs", xlabel="J=1", ylabel = "J=2"), size=(600, 600))

eps = randn(156, 10000)
covsq = cholesky(scov+Diagonal(ones(156).*1e-5)).U
qsamps = fmean .+ (covsq*eps)'

p = scatter(qsamps[1:1000, selidx[1]], qsamps[1:1000, selidx[2]], label="Samples from Gaussian", color="red", xlabel="J=1", ylabel = "J=2", xlim=(-100, 100), ylim=(-100, 100))
scatter!(dbnocffs[1:1000, selidx[1]], dbnocffs[1:1000, selidx[2]], label="True Samples", color="black", xlabel="J=1", ylabel = "J=2", xlim=(-100, 100), ylim=(-100, 100))


##Using Phi cut to remove bad images
function distribution_percentiles(dbn, idx, perc)
    idxperc = zeros(size(idx))
    idxperc .= (x-> percentile(dbn[:, x], perc)).(idx)
    return idxperc
end

phipval = distribution_percentiles(dbnocffs, [7], 99)
cmask = (dbnocffs[:, 7] .> phipval)
badimg = findall(!iszero, cmask)
bidx = rand(1:length(badimg))
heatmap(sfdall[:, :, badimg[bidx]], title="Phi Value=" * string(dbnocffs[badimg[bidx], 7]))

##Removing bad images based on phi cut
goodimg = findall(iszero, cmask)
subdbn = dbnocffs[goodimg, :]
fmean = mean(subdbn, dims=1)
scov= (subdbn .- fmean)' * (subdbn .- fmean) ./(size(subdbn)[1] -1)
cond(scov)
scovinv = invert_covmat(scov, 1e-5)
cholupp = cholesky(scovinv).U
whitened_ms = (cholupp * (dbnocffs .-  mean(dbnocffs, dims=1))')'
selidx = [3, 4]

p = plot(scatter(whitened_ms[:, selidx[1]], whitened_ms[:, selidx[2]], title="Real im, Not log coeffs", xlabel="J=1", ylabel = "J=2"), size=(600, 600))

eps = randn(156, 10000)
covsq = cholesky(scov+Diagonal(ones(156).*1e-5)).U
qsamps = fmean .+ (covsq*eps)'

p = scatter(qsamps[1:1000, selidx[1]], qsamps[1:1000, selidx[2]], label="Samples from Gaussian", color="red", xlabel="J=1", ylabel = "J=2", xlim=(-100, 100), ylim=(-100, 100))
scatter!(subdbn[1:1000, selidx[1]], subdbn[1:1000, selidx[2]], label="True Samples", color="black", xlabel="J=1", ylabel = "J=2", xlim=(-100, 100), ylim=(-100, 100))

##wd=2
logbool = false
apdbool=true
isobool = false
Nx=64
filter_hash = fink_filter_hash(1, 8, nx=Nx, t=1, wd=2, Omega=true)
sfdall = readsfd_fromsrc("scratch_NM/data/dust10000.fits", Nx, logbool=logbool)
dhc_args = Dict(:doS2=>false, :doS20=>true, :apodize=>apdbool, :iso=>isobool)
dbnocffs = get_dbn_coeffs(sfdall, filter_hash, dhc_args)

#Subsection
fmean = mean(dbnocffs, dims=1)
scov= (dbnocffs .- fmean)' * (dbnocffs .- fmean) ./(size(dbnocffs)[1] -1)
cond(scov)
scovinv = invert_covmat(scov, 1e-5)
cholupp = cholesky(scovinv).U
whitened_ms = (cholupp * (dbnocffs .-  mean(dbnocffs, dims=1))')'
selidx = [3, 4]

p = plot(scatter(whitened_ms[:, selidx[1]], whitened_ms[:, selidx[2]], title="Real im, Not log coeffs", xlabel="J=1", ylabel = "J=2"), size=(600, 600))

eps = randn(1192, 10000)
covsq = cholesky(scov+Diagonal(ones(1192).*1e-5)).U
qsamps = fmean .+ (covsq*eps)'

p = scatter(qsamps[1:1000, selidx[1]], qsamps[1:1000, selidx[2]], label="Samples from Gaussian", color="red", xlabel="J=1", ylabel = "J=2", xlim=(-100, 100), ylim=(-100, 100))
scatter!(dbnocffs[1:1000, selidx[1]], dbnocffs[1:1000, selidx[2]], label="True Samples", color="black", xlabel="J=1", ylabel = "J=2", xlim=(-100, 100), ylim=(-100, 100))

dbnocffs= log.(dbnocffs)

fmean = mean(dbnocffs, dims=1)
scov= (dbnocffs .- fmean)' * (dbnocffs .- fmean) ./(size(dbnocffs)[1] -1)
cond(scov)
scovinv = invert_covmat(scov, 1e-5)
cholupp = cholesky(scovinv).U
whitened_ms = (cholupp * (dbnocffs .-  mean(dbnocffs, dims=1))')'
selidx = [3, 4]

p = plot(scatter(whitened_ms[:, selidx[1]], whitened_ms[:, selidx[2]], title="Real im, Log coeffs", xlabel="J=1", ylabel = "J=2"), size=(600, 600))

eps = randn(1192, 10000)
covsq = cholesky(scov+Diagonal(ones(1192).*1e-5)).U
qsamps = fmean .+ (covsq*eps)'

p = scatter(qsamps[1:1000, selidx[1]], qsamps[1:1000, selidx[2]], label="Samples from Gaussian", color="red", xlabel="J=1", ylabel = "J=2", xlim=(-100, 100), ylim=(-100, 100))
scatter!(dbnocffs[1:1000, selidx[1]], dbnocffs[1:1000, selidx[2]], label="True Samples", color="black", xlabel="J=1", ylabel = "J=2", xlim=(-100, 100), ylim=(-100, 100))
