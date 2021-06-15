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
using Distributed
push!(LOAD_PATH, pwd()*"/main")
using DHC_2DUtils
push!(LOAD_PATH, pwd()*"/scratch_NM")
using Deriv_Utils_New
using Data_Utils
using Visualization
using ReconFuncs
include("../../../main/compute.jl")


#Bruno-like denoising using wst

##5) Iterative: using empirical noisy dbn (for non-Omega) and true: with pixelwise regularization
ARGS_buffer = ["reg", "nonapd", "noiso", "scratch_NM/StandardizedExp/Nx64/", "full_3losstest", "Full+Eps"]
ENV_buffer= "1000"
numfile = Base.parse(Int, ENV_buffer)
println(numfile, ARGS_buffer[1], ARGS_buffer[2])

if ARGS_buffer[1]=="log"
    logbool=true
else
    if ARGS_buffer[1]!="reg" error("Invalid log arg") end
    logbool=false
end

if ARGS_buffer[2]=="apd"
    apdbool = true
else
    if ARGS_buffer[2]!="nonapd" error("Invalid apd arg") end
    apdbool=false
end

if ARGS_buffer[3]=="iso"
    isobool = true
else
    if ARGS_buffer[3]!="noiso" error("Invalid iso arg") end
    isobool=false
end


direc = ARGS_buffer[4] #"../StandardizedExp/Nx64/noisy_stdtrue/" #Change
datfile = direc * "Data_" * string(numfile) * ".jld2" #Replace w SLURM array
loaddf = load(datfile)
true_img = loaddf["true_img"]
init = loaddf["init"]

fname_save = direc * "scratch_NM/NewWrapper/5-30/" * string(numfile) * "_try1"  #Change
Nx=size(true_img)[1]
filter_hash = fink_filter_hash(1, 8, nx=Nx, t=1, wd=1, Omega=true)
(S1iso, Nf) = size(filter_hash["S1_iso_mat"])
(S2iso, Nfsq) = size(filter_hash["S2_iso_mat"])
dhc_args = Dict(:doS2=>false, :doS20=>true, :apodize=>apdbool, :iso=>isobool) #Iso #CHANGE: Change sig for the sfd data since the noise model is super high and the tiny values make sense

optim_settings = Dict([("iterations", 20), ("norm", false), ("minmethod", ConjugateGradient())])
recon_settings = Dict([("log", logbool), ("Invcov_matrix", ARGS_buffer[6]), ("optim_settings", optim_settings), ("eps_value_sfd", 1e-5), ("eps_value_init", 1e-10)]) #Add constraints

recon_settings["datafile"] = datfile

if dhc_args[:iso]
    error("Not constructed for iso")
else #Not iso
    coeffmask = falses(2+Nf+Nf^2)
    coeffmask[Nf+3:end] .= Diagonal(trues(Nf))[:] #Diagonal(triu(trues(Nf, Nf)))[:]
end


#Weight given to terms
lval2 = 0.0
lval3 = 0.0

println("Regularizer Lambda=", round(lval3, sigdigits=3))
#input::Array{Float64, 2}, filter_hash::Dict, s_targ_mean::Array{Float64, 1}, s_targ_invcov, dhc_args, LossFunc, dLossFunc;
#FFTthreads::Int=1, optim_settings=Dict([("iterations", 10)]), lambda=0.001, func_specific_params=nothing


#Generate fixed noise maps
Nr=10000
sigma = loaddf["std"]
noisemaps = []
for n=1:Nr
    noisyim = randn((Nx, Nx)).*sigma
    push!(noisemaps, noisyim)
end

s_noisy = DHC_compute_wrapper(loaddf["init"], filter_hash, norm=false; dhc_args...)[coeffmask]
starget = s_noisy #DHC_compute_wrapper(loaddf["true_img"], filter_hash, norm=false; dhc_args...)[coeffmask] #

@time pmap(image -> DHC_compute_wrapper(image, filter_hash; norm=false, dhc_args...)[coeffmask], noisemaps)

function ens_noise()
    empsn = []
    for n=1:Nr
        push!(empsn,  DHC_compute_wrapper(noisemaps[n], filter_hash, norm=false; dhc_args...)[coeffmask])
    end
    return empsn
end

@time ens_noise()

function Loss_noisyfast()
    empsn = []



function Loss_noisy(img_curr, filter_hash, dhc_args; noisemaps=nothing, coeffmask=nothing, starget=nothing)
    #func_specific_params should contain: coeff_mask2, target2, invcov2
    empsn = []
    M = 500
    totdiff = 0
    for n=1:M
        noisy = img_curr + noisemaps[n]
        diffsq = sum((DHC_compute_wrapper(noisy, filter_hash; norm=false, dhc_args...)[coeffmask] .- starget).^2)
        totdiff += diffsq
    end
    totdiff = totdiff/M
    return totdiff
end

function dLoss_noisy!(storage_grad, img_curr, filter_hash, dhc_args; noisemaps=nothing, coeffmask=nothing, starget=nothing, dA=nothing)
    #func_specific_params should contain: coeff_mask2, target2, invcov2
    empsn = []
    M = 500 #length(noisemaps)
    totdiff = zeros(Nx^2, 1)
    for n=1:M
        noisy = img_curr + noisemaps[n]
        diffsq = 2*(DHC_compute_wrapper(noisy, filter_hash; norm=false, dhc_args...)[coeffmask] .- starget)
        wts = zeros(2+Nf+Nf^2)
        wts[coeffmask] .= diffsq
        wts = reshape(wts[Nf+3:end], (Nf, Nf))
        #wts = ReconFuncs.augment_weights_S20(convert(Array{Float64,1}, diffsq), filter_hash, dhc_args, coeffmask)
        derve = Deriv_Utils_New.wst_S20_deriv_sum(noisy, filter_hash, wts)
        totdiff .+= derve
    end
    totdiff ./= M
    storage_grad .= reshape(totdiff, (Nx, Nx))
end

func_specific_params = Dict([(:coeffmask=> coeffmask), (:starget=>starget), (:noisemaps=> noisemaps)])

recon_settings["fname_save"] = fname_save * ".jld2"
recon_settings["optim_settings"] = optim_settings

if logbool
    error("Not implemented here")
end

res, recon_img = image_recon_derivsum_custom(init, filter_hash, dhc_args, Loss_noisy, dLoss_noisy!; optim_settings=optim_settings, func_specific_params)
heatmap(init)
heatmap(recon_img, title="Using true shift")
heatmap(true_img)

function calc_1dps_local(image, kbins::Array{Float64,1})
    #Assumes uniformly spaced kbins
    Nx = size(image)[1]
    fft_zerocenter = fftshift(fft(image))
    impf = abs2.(fft_zerocenter)
    x = (collect(1:Nx) * ones(Nx)') .- (Nx/2.0)
    y = (ones(Nx) * collect(1:Nx)') .- (Nx/2.0)
    krad  = (x.^2 + y.^2).^0.5
    meanpk = zeros(size(kbins))
    kdel = kbins[2] - kbins[1]
    #println(size(meanpk), " ", kdel)
    for k=1:size(meanpk)[1]
        filt = findall((krad .>= (kbins[k] - kdel./2.0)) .& (krad .<= (kbins[k] + kdel./2.0)))
        meanpk[k] = mean(impf[filt])
    end
    return meanpk
end

function J_hashindices(J_values, fhash)
    jindlist = []
    for jval in J_values
        push!(jindlist, findall(fhash["J_L"][:, 1].==jval))
    end
    return vcat(jindlist'...)
end

function J_S1indices(J_values, fhash)
    #Assumes this is applied to an object of length 2+Nf+Nf^2 or 2+Nf
    return J_hashindices(J_values, fhash) .+ 2
end

kbins=convert(Array{Float64, 1}, collect(1:32))
apdsmoothed = imfilter(init, Kernel.gaussian(1.0))
smoothps = calc_1dps_local(apdsmoothed, kbins)
true_ps = calc_1dps_local(true_img, kbins)
initps = calc_1dps_local(init, kbins)
recps = calc_1dps_local(recon_img, kbins)

JS1ind = J_S1indices([0, 1, 2, 3], filter_hash)
clim= (minimum(true_img), maximum(true_img))
p1 = heatmap(recon_img, title="Recon", clim=clim)
p3 = plot(log.(kbins), log.(true_ps), label="True")
plot!(log.(kbins), log.(recps), label="Recon")
plot!(log.(kbins), log.(initps), label="Init")
plot!(log.(kbins), log.(smoothps), label="Smoothed Init")
plot!(title="P(k): Denoising using Shift(init)")
xlabel!("lnk")
ylabel!("lnP(k)")
p2 = heatmap(true_img, title="True Img", clim=clim)
p4= heatmap(init, title="Init Img", clim=clim)
p5= heatmap(apdsmoothed, title="Smoothed init", clim=clim)
residual = recon_img- true_img
rlims = (minimum(residual), maximum(residual))
#symmax = maximum([abs(minimum(residual)), maximum(residual)])
#rg = cgrad(:bwr, [-symmax/2.0, symmax/2.0])
p6 = heatmap(residual, title="Residual: Recon - True", clims=rlims, c=:bwr)
p7 = heatmap(apdsmoothed- true_img, title="Residual: SmoothedInit - True", clims=rlims, c=:bwr)

struesel = Data_Utils.fnlog(DHC_compute_wrapper(true_img, filter_hash, norm=false; dhc_args...))
sinitsel = Data_Utils.fnlog(DHC_compute_wrapper(init, filter_hash, norm=false; dhc_args...))
ssmoothsel = Data_Utils.fnlog(DHC_compute_wrapper(apdsmoothed, filter_hash, norm=false; dhc_args...))
sreconsel = Data_Utils.fnlog(DHC_compute_wrapper(recon_img, filter_hash, norm=false; dhc_args...))
slims = (minimum(struesel[JS1ind]), maximum(struesel[JS1ind]))
cg = cgrad([:blue, :white, :red])
truephi, trueomg = round(struesel[2+filter_hash["phi_index"]], sigdigits=3), round(struesel[2+filter_hash["Omega_index"]], sigdigits=3)
reconphi, reconomg = round(sreconsel[2+filter_hash["phi_index"]], sigdigits=3), round(sreconsel[2+filter_hash["Omega_index"]], sigdigits=3)
smoothphi, smoothomg = round(ssmoothsel[2+filter_hash["phi_index"]], sigdigits=3), round(ssmoothsel[2+filter_hash["Omega_index"]], sigdigits=3)

p8 = heatmap(struesel[JS1ind], title="True Coeffs ϕ=" * string(truephi) * "Ω=" * string(trueomg) , clims=slims, c=cg)
p9 = heatmap(sreconsel[JS1ind], title="Recon Coeffs ϕ=" * string(reconphi) * "Ω=" * string(reconomg), clims=slims, c=cg)
p10 = heatmap(ssmoothsel[JS1ind], title="Smooth Init ϕ=" * string(smoothphi) * "Ω=" * string(smoothomg), clims=slims, c=cg)
p = plot(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, layout=(5, 2), size=(1800, 2400))
savefig(p, "scratch_NM/NewWrapper/5-30/wphlike_s2r_new.png")
fracres = (init .- true_img)./true_img
fps = (initps .- true_ps)./true_ps
println("Mean Abs Frac, Init = ", round(mean(abs.(fracres)), digits=3), "Smoothed = ", round(mean(abs.((apdsmoothed .- true_img)./true_img)), digits=3), "Recon = ", round(mean(abs.((recon_img .- true_img)./true_img)), digits=3))
println("MSE, Init = ", round(mean((init .- true_img).^2), digits=6), "Smoothed = ", round(mean((apdsmoothed .- true_img).^2), digits=6), "Recon = ", round(mean((recon_img .- true_img).^2), digits=6))
println("Power Spec Frac Res, Init = ", round(mean(abs.(fps)), digits=3), "Smoothed = ", round(mean(abs.((smoothps .- true_ps)./true_ps)), digits=3), "Recon = ", round(mean(abs.((recps .- true_ps)./true_ps)), digits=3))

##5) Iterative: using empirical noisy dbn (for non-Omega) and true: with pixelwise regularization
ARGS_buffer = ["reg", "nonapd", "noiso", "scratch_NM/StandardizedExp/Nx64/", "full_3losstest", "Full+Eps"]
ENV_buffer= "1000"
numfile = Base.parse(Int, ENV_buffer)
println(numfile, ARGS_buffer[1], ARGS_buffer[2])

if ARGS_buffer[1]=="log"
    logbool=true
else
    if ARGS_buffer[1]!="reg" error("Invalid log arg") end
    logbool=false
end

if ARGS_buffer[2]=="apd"
    apdbool = true
else
    if ARGS_buffer[2]!="nonapd" error("Invalid apd arg") end
    apdbool=false
end

if ARGS_buffer[3]=="iso"
    isobool = true
else
    if ARGS_buffer[3]!="noiso" error("Invalid iso arg") end
    isobool=false
end


direc = ARGS_buffer[4] #"../StandardizedExp/Nx64/noisy_stdtrue/" #Change
datfile = direc * "Data_" * string(numfile) * ".jld2" #Replace w SLURM array
loaddf = load(datfile)
true_img = loaddf["true_img"]
init = loaddf["init"]

fname_save = direc * "scratch_NM/NewWrapper/5-30/" * string(numfile) * "_try1"  #Change
Nx=size(true_img)[1]
filter_hash = fink_filter_hash(1, 8, nx=Nx, t=1, wd=1, Omega=true)
(S1iso, Nf) = size(filter_hash["S1_iso_mat"])
(S2iso, Nfsq) = size(filter_hash["S2_iso_mat"])
dhc_args = Dict(:doS2=>false, :doS20=>true, :apodize=>apdbool, :iso=>isobool) #Iso #CHANGE: Change sig for the sfd data since the noise model is super high and the tiny values make sense

optim_settings = Dict([("iterations", 20), ("norm", false), ("minmethod", ConjugateGradient())])
recon_settings = Dict([("log", logbool), ("Invcov_matrix", ARGS_buffer[6]), ("optim_settings", optim_settings), ("eps_value_sfd", 1e-5), ("eps_value_init", 1e-10)]) #Add constraints

recon_settings["datafile"] = datfile

if dhc_args[:iso]
    error("Not constructed for iso")
else #Not iso
    coeffmask = falses(2+Nf+Nf^2)
    coeffmask[Nf+3:end] .= Diagonal(trues(Nf))[:] #Diagonal(triu(trues(Nf, Nf)))[:]
end


#Weight given to terms
lval2 = 0.0
lval3 = 0.0

println("Regularizer Lambda=", round(lval3, sigdigits=3))
#input::Array{Float64, 2}, filter_hash::Dict, s_targ_mean::Array{Float64, 1}, s_targ_invcov, dhc_args, LossFunc, dLossFunc;
#FFTthreads::Int=1, optim_settings=Dict([("iterations", 10)]), lambda=0.001, func_specific_params=nothing


#Generate fixed noise maps
Nr=10000
sigma = loaddf["std"]
noisemaps = []
for n=1:Nr
    noisyim = randn((Nx, Nx)).*sigma
    push!(noisemaps, noisyim)
end

s_noisy = DHC_compute_wrapper(loaddf["init"], filter_hash, norm=false; dhc_args...)[coeffmask]
starget = s_noisy #DHC_compute_wrapper(loaddf["true_img"], filter_hash, norm=false; dhc_args...)[coeffmask] #

@time pmap(image -> DHC_compute_wrapper(image, filter_hash; norm=false, dhc_args...)[coeffmask], noisemaps)

function ens_noise()
    empsn = []
    for n=1:Nr
        push!(empsn,  DHC_compute_wrapper(noisemaps[n], filter_hash, norm=false; dhc_args...)[coeffmask])
    end
    return empsn
end

@time ens_noise()

function Loss_noisyfast()
    empsn = []



function Loss_noisy(img_curr, filter_hash, dhc_args; noisemaps=nothing, coeffmask=nothing, starget=nothing)
    #func_specific_params should contain: coeff_mask2, target2, invcov2
    empsn = []
    M = 500
    totdiff = 0
    for n=1:M
        noisy = img_curr + noisemaps[n]
        diffsq = sum((DHC_compute_wrapper(noisy, filter_hash; norm=false, dhc_args...)[coeffmask] .- starget).^2)
        totdiff += diffsq
    end
    totdiff = totdiff/M
    return totdiff
end

function dLoss_noisy!(storage_grad, img_curr, filter_hash, dhc_args; noisemaps=nothing, coeffmask=nothing, starget=nothing, dA=nothing)
    #func_specific_params should contain: coeff_mask2, target2, invcov2
    empsn = []
    M = 500 #length(noisemaps)
    totdiff = zeros(Nx^2, 1)
    for n=1:M
        noisy = img_curr + noisemaps[n]
        diffsq = 2*(DHC_compute_wrapper(noisy, filter_hash; norm=false, dhc_args...)[coeffmask] .- starget)
        wts = zeros(2+Nf+Nf^2)
        wts[coeffmask] .= diffsq
        wts = reshape(wts[Nf+3:end], (Nf, Nf))
        #wts = ReconFuncs.augment_weights_S20(convert(Array{Float64,1}, diffsq), filter_hash, dhc_args, coeffmask)
        derve = Deriv_Utils_New.wst_S20_deriv_sum(noisy, filter_hash, wts)
        totdiff .+= derve
    end
    totdiff ./= M
    storage_grad .= reshape(totdiff, (Nx, Nx))
end

func_specific_params = Dict([(:coeffmask=> coeffmask), (:starget=>starget), (:noisemaps=> noisemaps)])

recon_settings["fname_save"] = fname_save * ".jld2"
recon_settings["optim_settings"] = optim_settings

if logbool
    error("Not implemented here")
end

res, recon_img = image_recon_derivsum_custom(init, filter_hash, dhc_args, Loss_noisy, dLoss_noisy!; optim_settings=optim_settings, func_specific_params)
heatmap(init)
heatmap(recon_img, title="Using true shift")
heatmap(true_img)

function calc_1dps_local(image, kbins::Array{Float64,1})
    #Assumes uniformly spaced kbins
    Nx = size(image)[1]
    fft_zerocenter = fftshift(fft(image))
    impf = abs2.(fft_zerocenter)
    x = (collect(1:Nx) * ones(Nx)') .- (Nx/2.0)
    y = (ones(Nx) * collect(1:Nx)') .- (Nx/2.0)
    krad  = (x.^2 + y.^2).^0.5
    meanpk = zeros(size(kbins))
    kdel = kbins[2] - kbins[1]
    #println(size(meanpk), " ", kdel)
    for k=1:size(meanpk)[1]
        filt = findall((krad .>= (kbins[k] - kdel./2.0)) .& (krad .<= (kbins[k] + kdel./2.0)))
        meanpk[k] = mean(impf[filt])
    end
    return meanpk
end

function J_hashindices(J_values, fhash)
    jindlist = []
    for jval in J_values
        push!(jindlist, findall(fhash["J_L"][:, 1].==jval))
    end
    return vcat(jindlist'...)
end

function J_S1indices(J_values, fhash)
    #Assumes this is applied to an object of length 2+Nf+Nf^2 or 2+Nf
    return J_hashindices(J_values, fhash) .+ 2
end

kbins=convert(Array{Float64, 1}, collect(1:32))
apdsmoothed = imfilter(init, Kernel.gaussian(1.0))
smoothps = calc_1dps_local(apdsmoothed, kbins)
true_ps = calc_1dps_local(true_img, kbins)
initps = calc_1dps_local(init, kbins)
recps = calc_1dps_local(recon_img, kbins)

JS1ind = J_S1indices([0, 1, 2, 3], filter_hash)
clim= (minimum(true_img), maximum(true_img))
p1 = heatmap(recon_img, title="Recon", clim=clim)
p3 = plot(log.(kbins), log.(true_ps), label="True")
plot!(log.(kbins), log.(recps), label="Recon")
plot!(log.(kbins), log.(initps), label="Init")
plot!(log.(kbins), log.(smoothps), label="Smoothed Init")
plot!(title="P(k): Denoising using Shift(init)")
xlabel!("lnk")
ylabel!("lnP(k)")
p2 = heatmap(true_img, title="True Img", clim=clim)
p4= heatmap(init, title="Init Img", clim=clim)
p5= heatmap(apdsmoothed, title="Smoothed init", clim=clim)
residual = recon_img- true_img
rlims = (minimum(residual), maximum(residual))
#symmax = maximum([abs(minimum(residual)), maximum(residual)])
#rg = cgrad(:bwr, [-symmax/2.0, symmax/2.0])
p6 = heatmap(residual, title="Residual: Recon - True", clims=rlims, c=:bwr)
p7 = heatmap(apdsmoothed- true_img, title="Residual: SmoothedInit - True", clims=rlims, c=:bwr)

struesel = Data_Utils.fnlog(DHC_compute_wrapper(true_img, filter_hash, norm=false; dhc_args...))
sinitsel = Data_Utils.fnlog(DHC_compute_wrapper(init, filter_hash, norm=false; dhc_args...))
ssmoothsel = Data_Utils.fnlog(DHC_compute_wrapper(apdsmoothed, filter_hash, norm=false; dhc_args...))
sreconsel = Data_Utils.fnlog(DHC_compute_wrapper(recon_img, filter_hash, norm=false; dhc_args...))
slims = (minimum(struesel[JS1ind]), maximum(struesel[JS1ind]))
cg = cgrad([:blue, :white, :red])
truephi, trueomg = round(struesel[2+filter_hash["phi_index"]], sigdigits=3), round(struesel[2+filter_hash["Omega_index"]], sigdigits=3)
reconphi, reconomg = round(sreconsel[2+filter_hash["phi_index"]], sigdigits=3), round(sreconsel[2+filter_hash["Omega_index"]], sigdigits=3)
smoothphi, smoothomg = round(ssmoothsel[2+filter_hash["phi_index"]], sigdigits=3), round(ssmoothsel[2+filter_hash["Omega_index"]], sigdigits=3)

p8 = heatmap(struesel[JS1ind], title="True Coeffs ϕ=" * string(truephi) * "Ω=" * string(trueomg) , clims=slims, c=cg)
p9 = heatmap(sreconsel[JS1ind], title="Recon Coeffs ϕ=" * string(reconphi) * "Ω=" * string(reconomg), clims=slims, c=cg)
p10 = heatmap(ssmoothsel[JS1ind], title="Smooth Init ϕ=" * string(smoothphi) * "Ω=" * string(smoothomg), clims=slims, c=cg)
p = plot(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, layout=(5, 2), size=(1800, 2400))
savefig(p, "scratch_NM/NewWrapper/5-30/wphlike_s1only_new.png")
fracres = (init .- true_img)./true_img
fps = (initps .- true_ps)./true_ps
println("Mean Abs Frac, Init = ", round(mean(abs.(fracres)), digits=3), "Smoothed = ", round(mean(abs.((apdsmoothed .- true_img)./true_img)), digits=3), "Recon = ", round(mean(abs.((recon_img .- true_img)./true_img)), digits=3))
println("MSE, Init = ", round(mean((init .- true_img).^2), digits=6), "Smoothed = ", round(mean((apdsmoothed .- true_img).^2), digits=6), "Recon = ", round(mean((recon_img .- true_img).^2), digits=6))
println("Power Spec Frac Res, Init = ", round(mean(abs.(fps)), digits=3), "Smoothed = ", round(mean(abs.((smoothps .- true_ps)./true_ps)), digits=3), "Recon = ", round(mean(abs.((recps .- true_ps)./true_ps)), digits=3))
