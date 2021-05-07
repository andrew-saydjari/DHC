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

##Case 3a-f, h, hv-2, j
ARGS_buffer = ["reg", "apd", "noiso", "scratch_NM/StandardizedExp/Nx64/", "full_3losstest", "Full+Eps"]
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

fname_save = direc * "S1only/SFDTargSFDCov/" * ARGS_buffer[1] * "_" * ARGS_buffer[2] * "_" * ARGS_buffer[3] * "/LogCoeff/" * string(numfile) * ARGS_buffer[5]  #Change
Nx=size(true_img)[1]
filter_hash = fink_filter_hash(1, 8, nx=Nx, t=1, Omega=true)
(S1iso, Nf) = size(filter_hash["S1_iso_mat"])
(S2iso, Nfsq) = size(filter_hash["S2_iso_mat"])
dhc_args = Dict(:doS2=>false, :doS20=>true, :apodize=>apdbool, :iso=>isobool) #Iso #CHANGE: Change sig for the sfd data since the noise model is super high and the tiny values make sense

optim_settings = Dict([("iterations", 1000), ("norm", false), ("minmethod", ConjugateGradient())])
recon_settings = Dict([("log", logbool), ("Invcov_matrix", ARGS_buffer[6]), ("optim_settings", optim_settings), ("eps_value_sfd", 1e-5), ("eps_value_init", 1e-10)]) #Add constraints

recon_settings["datafile"] = datfile
#Read in SFD Dbn
sfdall = readsfd(Nx, logbool=logbool)
dbnocffs = log.(get_dbn_coeffs(sfdall, filter_hash, dhc_args))
fmean = mean(dbnocffs, dims=1)
size(dbnocffs)
#fmed = distribution_percentiles(dbnocffs, JS1ind, 50)


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

JS1ind = J_S1indices([0, 1, 2, 3], filter_hash)
function distribution_percentiles(dbn, idx, perc)
    idxperc = zeros(size(idx))
    idxperc .= (x-> percentile(dbn[:, x], perc)).(idx)
    return idxperc
end

#Construct SFD CoeffMask
if dhc_args[:iso]
    error("Not constructed for iso")
else #Not iso
    coeff_masksfd = falses(2+Nf+Nf^2)
    #lowjidx = (filter_hash["J_L"][:, 1] .== 0) .| (filter_hash["J_L"][:, 1] .== 1)
    #coeff_maskS1 = falses(Nf)
    #coeff_maskS1[lowjidx] .= true
    #coeff_maskS1[33] .= true
    #coeff_maskS1[34] = true
    coeff_masksfd[Nf+3:end] .= Diagonal(trues(Nf))[:] #Diagonal(triu(trues(Nf, Nf)))[:]
end

#Construct Init CoeffMask
if dhc_args[:iso]
    error("Not constructed for iso")
else #Not iso
    coeff_maskinit = falses(2+Nf+Nf^2)
    highjidx = (filter_hash["J_L"][:, 1] .== 2) .| (filter_hash["J_L"][:, 1] .== 3) .| (filter_hash["J_L"][:, 1] .== 1)
    coeff_maskS1 = falses(Nf)
    coeff_maskS1[highjidx] .= true
    coeff_maskS1[33] = true
    coeff_maskinit[Nf+3:end] .= Diagonal(coeff_maskS1)[:]
end
recon_settings["coeff_mask_sfd"] = coeff_masksfd
recon_settings["coeff_mask_init"] = coeff_maskinit

fcov = (dbnocffs .- fmean)' * (dbnocffs .- fmean) ./(size(dbnocffs)[1] -1)
fcovinv1 = invert_covmat(fcov[coeff_masksfd, coeff_masksfd], recon_settings["eps_value_sfd"])
fcovinv2 = invert_covmat(fcov[coeff_maskinit, coeff_maskinit], recon_settings["eps_value_init"])
ftargsfd = fmean[coeff_masksfd]
if logbool
    error("Not impl for log im")
end
ftarginit = log.(DHC_compute_wrapper(imfilter(init, Kernel.gaussian(0.8)), filter_hash; dhc_args...))
ftarginit = ftarginit[coeff_maskinit]

#Weight given to terms
lval2 = 10.0
lval3=0.00

println("Regularizer Lambda=", round(lval3, sigdigits=3))
#input::Array{Float64, 2}, filter_hash::Dict, s_targ_mean::Array{Float64, 1}, s_targ_invcov, dhc_args, LossFunc, dLossFunc;
#FFTthreads::Int=1, optim_settings=Dict([("iterations", 10)]), lambda=0.001, func_specific_params=nothing
func_specific_params = Dict([(:reg_input=> init), (:coeff_mask1=> coeff_masksfd), (:target1=>ftargsfd), (:invcov1=>fcovinv1), (:coeff_mask2=> coeff_maskinit), (:target2=>ftarginit), (:invcov2=>fcovinv2),
    (:func=> Data_Utils.fnlog), (:dfunc=> Data_Utils.fndlog)])
func_specific_params[:lambda2] = lval2
func_specific_params[:lambda3] = lval3

recon_settings["fname_save"] = fname_save * ".jld2"
recon_settings["optim_settings"] = optim_settings

if logbool
    error("Not implemented here")
end

res, recon_img = image_recon_derivsum_custom(init, filter_hash, dhc_args, ReconFuncs.Loss3Gaussian_transformed,
    ReconFuncs.dLoss3Gaussian_transformed!; optim_settings=optim_settings, func_specific_params)
#save(settings["fname_save"], Dict("true_img"=>true_img, "init"=>noisy_init, "recon"=>recon_img, "dict"=>settings, "trace"=>Optim.trace(res), "coeff_mask"=>coeff_mask, "fhash"=>fhash, "dhc_args"=>dhc_args))


function loss1(inp_img)
    s_curr = DHC_compute_wrapper(inp_img, filter_hash, norm=false; dhc_args...)
    s_curr1 = Data_Utils.fnlog(s_curr[coeff_masksfd])
    diff1 = s_curr1 - ftargsfd
    return ( 0.5 .* (diff1)' * fcovinv1 * (diff1))
end

function loss2(inp_img)
    s_curr = DHC_compute_wrapper(inp_img, filter_hash, norm=false; dhc_args...)
    s_curr1 = Data_Utils.fnlog(s_curr[coeff_maskinit])
    diff1 = s_curr1 - ftarginit
    return ( 0.5 *lval2 .* (diff1)' * fcovinv2 * (diff1))
end

function loss3(inp_img)
    return 0.5*lval3*sum((ReconFuncs.adaptive_apodizer(inp_img, dhc_args) - ReconFuncs.adaptive_apodizer(init, dhc_args)).^2)
end


println("Loss Term 1: SFDTarg")
println("True", loss1(true_img))
println("Init", loss1(init))
println("Recon", loss1(recon_img))

println("Loss Term 2: SmoothedInitTarg")
println("True", loss2(true_img))
println("Init", loss2(init))
println("Recon", loss2(recon_img))

println("Loss Term 3: Reg")
println("True", loss3(true_img))
println("Init", loss3(init))
println("Recon", loss3(recon_img))

save("scratch_NM/NewWrapper/4-28/1000_C5_S2sfd_highjinit.jld2", Dict("true_img"=>true_img, "init"=>init, "recon"=>recon_img, "dict"=>recon_settings, "trace"=>Optim.trace(res), "coeff_mask"=>[coeff_masksfd, coeff_maskinit], "fhash"=>filter_hash, "dhc_args"=>dhc_args, "func_specific_params"=>func_specific_params))



apdsmoothed = apodizer(imfilter(init, Kernel.gaussian(0.8)))
kbins= convert(Array{Float64}, collect(1:32))
clim=(minimum(apodizer(true_img)), maximum(apodizer(true_img)))
true_ps = Data_Utils.calc_1dps(apodizer(true_img), kbins)
initps = Data_Utils.calc_1dps(apodizer(init), kbins)
recps = Data_Utils.calc_1dps(apodizer(recon_img), kbins)
smoothps = Data_Utils.calc_1dps(apdsmoothed, kbins)
p1 = heatmap(apodizer(recon_img), title="Recon", clim=clim)
p3 = plot(log.(kbins), log.(true_ps), label="True")
plot!(log.(kbins), log.(recps), label="Recon")
plot!(log.(kbins), log.(initps), label="Init")
plot!(log.(kbins), log.(smoothps), label="Smoothed Init")
plot!(title="P(k): LogCoeffs 3 Loss")
xlabel!("lnk")
ylabel!("lnP(k)")
p2 = heatmap(apodizer(true_img), title="True Img", clim=clim)
p4= heatmap(apodizer(init), title="Init Img", clim=clim)
p5= heatmap(apodizer(imfilter(init, Kernel.gaussian(0.8))), title="Smoothed init", clim=clim)
residual = apodizer(recon_img)- apodizer(true_img)
rlims = (minimum(residual), maximum(residual))
#symmax = maximum([abs(minimum(residual)), maximum(residual)])
#rg = cgrad(:bwr, [-symmax/2.0, symmax/2.0])
p6 = heatmap(residual, title="Residual: Recon - True", clims=rlims, c=:bwr)
p7 = heatmap(apdsmoothed- apodizer(true_img), title="Residual: SmoothedInit - True", clims=rlims, c=:bwr)

struesel = Data_Utils.fnlog(DHC_compute_wrapper(true_img, filter_hash, norm=false; dhc_args...))
sinitsel = Data_Utils.fnlog(DHC_compute_wrapper(init, filter_hash, norm=false; dhc_args...))
ssmoothsel = Data_Utils.fnlog(DHC_compute_wrapper(imfilter(init, Kernel.gaussian(0.8)), filter_hash, norm=false; dhc_args...))
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
savefig(p, "scratch_NM/NewWrapper/5-3/Case3j-v2.png")
Visualization.plot_synth_QA(apodizer(true_img), apodizer(init), apodizer(recon_img), fname="scratch_NM/NewWrapper/4-28/1000_C5_S2sfd_highjinit.png")



clims=(-10, -2)
cg = cgrad([:blue, :white, :red])
p = plot(heatmap(struesel, title="True", clims=clims, c=cg),
heatmap(sinitsel, title="Init", clims=clims, c=cg),
heatmap(ssmoothsel, title="SmoothInit", clims=clims, c=cg),
heatmap(sreconsel, title="Recon", clims=clims, c=cg),
heatmap(fmean[JS1ind], title="SFDTargMean", clims=clims, c=cg),
heatmap(fmed, title="SFD Median", clims=clims, c=cg), size=(900, 600))

aptrue_img = apodizer(true_img)
apinit = apodizer(init)
aprecon_img = apodizer(recon_img)
println("Mean Abs Res: Init-True = ", mean(abs.(apinit - aptrue_img)), " Recon-True = ", mean(abs.(aprecon_img - aptrue_img)))
println("Mean Abs Frac Res", mean(abs.((apinit - aptrue_img)./aptrue_img)), " Recon-True=", mean(abs.((aprecon_img - aptrue_img)./aptrue_img)))
println("Mean L2 Res: Init-True = ", mean((apinit - aptrue_img).^2).^0.5, " Recon-True = ", mean((aprecon_img - aptrue_img).^2).^0.5)

##Case 3f: Different Image
ARGS_buffer = ["reg", "apd", "noiso", "scratch_NM/StandardizedExp/Nx64/", "full_3losstest", "Full+Eps"]
ENV_buffer= "100"
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

fname_save = direc * "S1only/SFDTargSFDCov/" * ARGS_buffer[1] * "_" * ARGS_buffer[2] * "_" * ARGS_buffer[3] * "/LogCoeff/" * string(numfile) * ARGS_buffer[5]  #Change
Nx=size(true_img)[1]
filter_hash = fink_filter_hash(1, 8, nx=Nx, t=1, wd=1, Omega=true)
(S1iso, Nf) = size(filter_hash["S1_iso_mat"])
(S2iso, Nfsq) = size(filter_hash["S2_iso_mat"])
dhc_args = Dict(:doS2=>false, :doS20=>true, :apodize=>apdbool, :iso=>isobool) #Iso #CHANGE: Change sig for the sfd data since the noise model is super high and the tiny values make sense

optim_settings = Dict([("iterations", 1000), ("norm", false), ("minmethod", ConjugateGradient())])
recon_settings = Dict([("log", logbool), ("Invcov_matrix", ARGS_buffer[6]), ("optim_settings", optim_settings), ("eps_value_sfd", 1e-5), ("eps_value_init", 1e-10)]) #Add constraints

recon_settings["datafile"] = datfile
#Read in SFD Dbn
sfdall = readsfd(Nx, logbool=logbool)
dbnocffs = log.(get_dbn_coeffs(sfdall, filter_hash, dhc_args))
fmean = mean(dbnocffs, dims=1)
size(dbnocffs)
fmed = distribution_percentiles(dbnocffs, JS1ind, 50)


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

JS1ind = J_S1indices([0, 1, 2, 3], filter_hash)
function distribution_percentiles(dbn, idx, perc)
    idxperc = zeros(size(idx))
    idxperc .= (x-> percentile(dbn[:, x], perc)).(idx)
    return idxperc
end

#Construct SFD CoeffMask
if dhc_args[:iso]
    error("Not constructed for iso")
else #Not iso
    coeff_masksfd = falses(2+Nf+Nf^2)
    #lowjidx = (filter_hash["J_L"][:, 1] .== 0) .| (filter_hash["J_L"][:, 1] .== 1)
    #coeff_maskS1 = falses(Nf)
    #coeff_maskS1[lowjidx] .= true
    #coeff_maskS1[33] .= true
    #coeff_maskS1[34] = true
    coeff_masksfd[Nf+3:end] .= Diagonal(trues(Nf))[:] #triu(trues(Nf, Nf))[:]
end

#Construct Init CoeffMask
if dhc_args[:iso]
    error("Not constructed for iso")
else #Not iso
    coeff_maskinit = falses(2+Nf+Nf^2)
    highjidx = (filter_hash["J_L"][:, 1] .== 2) .| (filter_hash["J_L"][:, 1] .== 3) .| (filter_hash["J_L"][:, 1] .== 1)
    coeff_maskS1 = falses(Nf)
    coeff_maskS1[highjidx] .= true
    coeff_maskS1[33] = true
    coeff_maskinit[Nf+3:end] .= Diagonal(coeff_maskS1)[:]
end
recon_settings["coeff_mask_sfd"] = coeff_masksfd
recon_settings["coeff_mask_init"] = coeff_maskinit

fcov = (dbnocffs .- fmean)' * (dbnocffs .- fmean) ./(size(dbnocffs)[1] -1)
fcovinv1 = invert_covmat(fcov[coeff_masksfd, coeff_masksfd], recon_settings["eps_value_sfd"])
fcovinv2 = invert_covmat(fcov[coeff_maskinit, coeff_maskinit], recon_settings["eps_value_init"])
ftargsfd = fmean[coeff_masksfd]
if logbool
    error("Not impl for log im")
end
ftarginit = log.(DHC_compute_wrapper(imfilter(init, Kernel.gaussian(0.8)), filter_hash; dhc_args...))
ftarginit = ftarginit[coeff_maskinit]

#Weight given to terms
lval2 = 10.0
lval3=0.00

println("Regularizer Lambda=", round(lval3, sigdigits=3))
#input::Array{Float64, 2}, filter_hash::Dict, s_targ_mean::Array{Float64, 1}, s_targ_invcov, dhc_args, LossFunc, dLossFunc;
#FFTthreads::Int=1, optim_settings=Dict([("iterations", 10)]), lambda=0.001, func_specific_params=nothing
func_specific_params = Dict([(:reg_input=> init), (:coeff_mask1=> coeff_masksfd), (:target1=>ftargsfd), (:invcov1=>fcovinv1), (:coeff_mask2=> coeff_maskinit), (:target2=>ftarginit), (:invcov2=>fcovinv2),
    (:func=> Data_Utils.fnlog), (:dfunc=> Data_Utils.fndlog)])
func_specific_params[:lambda2] = lval2
func_specific_params[:lambda3] = lval3

recon_settings["fname_save"] = fname_save * ".jld2"
recon_settings["optim_settings"] = optim_settings

if logbool
    error("Not implemented here")
end

res, recon_img = image_recon_derivsum_custom(init, filter_hash, dhc_args, ReconFuncs.Loss3Gaussian_transformed,
    ReconFuncs.dLoss3Gaussian_transformed!; optim_settings=optim_settings, func_specific_params)
#save(settings["fname_save"], Dict("true_img"=>true_img, "init"=>noisy_init, "recon"=>recon_img, "dict"=>settings, "trace"=>Optim.trace(res), "coeff_mask"=>coeff_mask, "fhash"=>fhash, "dhc_args"=>dhc_args))


function loss1(inp_img)
    s_curr = DHC_compute_wrapper(inp_img, filter_hash, norm=false; dhc_args...)
    s_curr1 = Data_Utils.fnlog(s_curr[coeff_masksfd])
    diff1 = s_curr1 - ftargsfd
    return ( 0.5 .* (diff1)' * fcovinv1 * (diff1))
end

function loss2(inp_img)
    s_curr = DHC_compute_wrapper(inp_img, filter_hash, norm=false; dhc_args...)
    s_curr1 = Data_Utils.fnlog(s_curr[coeff_maskinit])
    diff1 = s_curr1 - ftarginit
    return ( 0.5 *lval2 .* (diff1)' * fcovinv2 * (diff1))
end

function loss3(inp_img)
    return 0.5*lval3*sum((ReconFuncs.adaptive_apodizer(inp_img, dhc_args) - ReconFuncs.adaptive_apodizer(init, dhc_args)).^2)
end


println("Loss Term 1: SFDTarg")
println("True", loss1(true_img))
println("Init", loss1(init))
println("Recon", loss1(recon_img))

println("Loss Term 2: SmoothedInitTarg")
println("True", loss2(true_img))
println("Init", loss2(init))
println("Recon", loss2(recon_img))

println("Loss Term 3: Reg")
println("True", loss3(true_img))
println("Init", loss3(init))
println("Recon", loss3(recon_img))

save("scratch_NM/NewWrapper/4-28/1000_C5_S2sfd_highjinit.jld2", Dict("true_img"=>true_img, "init"=>init, "recon"=>recon_img, "dict"=>recon_settings, "trace"=>Optim.trace(res), "coeff_mask"=>[coeff_masksfd, coeff_maskinit], "fhash"=>filter_hash, "dhc_args"=>dhc_args, "func_specific_params"=>func_specific_params))


heatmap(recon_img)
apdsmoothed = apodizer(imfilter(init, Kernel.gaussian(0.8)))
kbins= convert(Array{Float64}, collect(1:32))
clim=(minimum(apodizer(true_img)), maximum(apodizer(true_img)))
true_ps = Data_Utils.calc_1dps(apodizer(true_img), kbins)
initps = Data_Utils.calc_1dps(apodizer(init), kbins)
recps = Data_Utils.calc_1dps(apodizer(recon_img), kbins)
smoothps = Data_Utils.calc_1dps(apdsmoothed, kbins)
p1 = heatmap(apodizer(recon_img), title="Recon", clim=clim)
p3 = plot(log.(kbins), log.(true_ps), label="True")
plot!(log.(kbins), log.(recps), label="Recon")
plot!(log.(kbins), log.(initps), label="Init")
plot!(log.(kbins), log.(smoothps), label="Smoothed Init")
plot!(title="P(k): LogCoeffs 3 Loss")
xlabel!("lnk")
ylabel!("lnP(k)")
p2 = heatmap(apodizer(true_img), title="True Img", clim=clim)
p4= heatmap(apodizer(init), title="Init Img", clim=clim)
p5= heatmap(apodizer(imfilter(init, Kernel.gaussian(0.8))), title="Smoothed init", clim=clim)
p6 = heatmap(apodizer(recon_img)- apodizer(true_img), title="Residual: Recon - True")
p7 = heatmap(apodizer(recon_img)- apodizer(true_img), title="Residual: SmoothedInit - True")

struesel = Data_Utils.fnlog(DHC_compute_wrapper(true_img, filter_hash, norm=false; dhc_args...))
sinitsel = Data_Utils.fnlog(DHC_compute_wrapper(init, filter_hash, norm=false; dhc_args...))
ssmoothsel = Data_Utils.fnlog(DHC_compute_wrapper(imfilter(init, Kernel.gaussian(0.8)), filter_hash, norm=false; dhc_args...))
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
savefig(p, "scratch_NM/NewWrapper/5-3/Case3f.png")
Visualization.plot_synth_QA(apodizer(true_img), apodizer(init), apodizer(recon_img), fname="scratch_NM/NewWrapper/4-28/1000_C5_S2sfd_highjinit.png")



clims=(-10, -2)
cg = cgrad([:blue, :white, :red])
p = plot(heatmap(struesel, title="True", clims=clims, c=cg),
heatmap(sinitsel, title="Init", clims=clims, c=cg),
heatmap(ssmoothsel, title="SmoothInit", clims=clims, c=cg),
heatmap(sreconsel, title="Recon", clims=clims, c=cg),
heatmap(fmean[JS1ind], title="SFDTargMean", clims=clims, c=cg),
heatmap(fmed, title="SFD Median", clims=clims, c=cg), size=(900, 600))

aptrue_img = apodizer(true_img)
apinit = apodizer(init)
aprecon_img = apodizer(recon_img)
println("Mean Abs Res: Init-True = ", mean(abs.(apinit - aptrue_img)), " Recon-True = ", mean(abs.(aprecon_img - aptrue_img)))
println("Mean Abs Frac Res", mean(abs.((apinit - aptrue_img)./aptrue_img)), " Recon-True=", mean(abs.((aprecon_img - aptrue_img)./aptrue_img)))
println("Mean L2 Res: Init-True = ", mean((apinit - aptrue_img).^2).^0.5, " Recon-True = ", mean((aprecon_img - aptrue_img).^2).^0.5)

#=
p= plot([t.value for t in Optim.trace(resobj)])
plot!(title="Loss: S20 | Targ = S(SFD) | Cov = SFD Covariance", xlabel = "No. Iterations", ylabel = "Loss")
savefig(p, fname_save * "_trace.png")

Visualization.plot_diffscales([apodizer(true_img), apodizer(init), apodizer(recon_img), apodizer(recon_img) - apodizer(true_img)], ["GT", "Init", "Reconstruction", "Residual"], fname=fname_save*".png")
Visualization.plot_synth_QA(apodizer(true_img), apodizer(init), apodizer(recon_img), fname=nothing)

println("Input Data", datfile)
println("Output File", fname_save)
println("Mean Abs Res: Init-True = ", mean(abs.(init - true_img)), " Recon-True = ", mean(abs.(recon_img - true_img)))
println("Mean Abs Frac Res", mean(abs.((init - true_img)./true_img)), " Recon-True=", mean(abs.((recon_img - true_img)./true_img)))
println("Mean Abs Res: Init-True = ", mean((init - true_img).^2).^0.5, " Recon-True = ", mean((recon_img - true_img).^2).^0.5)
=#

##Case 3i
ARGS_buffer = ["reg", "apd", "noiso", "scratch_NM/StandardizedExp/Nx64/", "full_3losstest", "Full+Eps"]
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

fname_save = direc * "S1only/SFDTargSFDCov/" * ARGS_buffer[1] * "_" * ARGS_buffer[2] * "_" * ARGS_buffer[3] * "/LogCoeff/" * string(numfile) * ARGS_buffer[5]  #Change
Nx=size(true_img)[1]
filter_hash = fink_filter_hash(1, 8, nx=Nx, t=1, wd=1, Omega=true)
(S1iso, Nf) = size(filter_hash["S1_iso_mat"])
(S2iso, Nfsq) = size(filter_hash["S2_iso_mat"])
dhc_args = Dict(:doS2=>false, :doS20=>true, :apodize=>apdbool, :iso=>isobool) #Iso #CHANGE: Change sig for the sfd data since the noise model is super high and the tiny values make sense

optim_settings = Dict([("iterations", 1000), ("norm", false), ("minmethod", ConjugateGradient())])
recon_settings = Dict([("log", logbool), ("Invcov_matrix", ARGS_buffer[6]), ("optim_settings", optim_settings), ("eps_value_sfd", 1e-5), ("eps_value_init", 1e-10)]) #Add constraints

recon_settings["datafile"] = datfile
#Read in SFD Dbn
sfdall = readsfd(Nx, logbool=logbool)
dbnocffs = log.(get_dbn_coeffs(sfdall, filter_hash, dhc_args))
fmean = mean(dbnocffs, dims=1)
size(dbnocffs)
#fmed = distribution_percentiles(dbnocffs, JS1ind, 50)


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

JS1ind = J_S1indices([0, 1, 2, 3], filter_hash)
function distribution_percentiles(dbn, idx, perc)
    idxperc = zeros(size(idx))
    idxperc .= (x-> percentile(dbn[:, x], perc)).(idx)
    return idxperc
end

#Construct SFD CoeffMask
if dhc_args[:iso]
    error("Not constructed for iso")
else #Not iso
    coeff_masksfd = falses(2+Nf+Nf^2)
    #lowjidx = (filter_hash["J_L"][:, 1] .== 0) .| (filter_hash["J_L"][:, 1] .== 1)
    #coeff_maskS1 = falses(Nf)
    #coeff_maskS1[lowjidx] .= true
    #coeff_maskS1[33] .= true
    #coeff_maskS1[34] = true
    coeff_masksfd[Nf+3:end] .= Diagonal(trues(Nf))[:] #triu(trues(Nf, Nf))[:]
end

#Construct Init CoeffMask
if dhc_args[:iso]
    error("Not constructed for iso")
else #Not iso
    coeff_maskinit = falses(2+Nf+Nf^2)
    highjidx = (filter_hash["J_L"][:, 1] .== 2) .| (filter_hash["J_L"][:, 1] .== 3) .| (filter_hash["J_L"][:, 1] .== 1)
    coeff_maskS1 = falses(Nf)
    coeff_maskS1[highjidx] .= true
    coeff_maskS1[33] = true
    coeff_maskinit[Nf+3:end] .= Diagonal(coeff_maskS1)[:]
end
recon_settings["coeff_mask_sfd"] = coeff_masksfd
recon_settings["coeff_mask_init"] = coeff_maskinit

fcov = (dbnocffs .- fmean)' * (dbnocffs .- fmean) ./(size(dbnocffs)[1] -1)
fcovinv1 = invert_covmat(fcov[coeff_masksfd, coeff_masksfd], recon_settings["eps_value_sfd"])
fcovinv2 = invert_covmat(fcov[coeff_maskinit, coeff_maskinit], recon_settings["eps_value_init"])
ftargsfd = fmean[coeff_masksfd]
if logbool
    error("Not impl for log im")
end
ftarginit = log.(DHC_compute_wrapper(init, filter_hash; dhc_args...))
ftarginit = ftarginit[coeff_maskinit]

#Weight given to terms
lval2 = 10.0
lval3=0.00

println("Regularizer Lambda=", round(lval3, sigdigits=3))
#input::Array{Float64, 2}, filter_hash::Dict, s_targ_mean::Array{Float64, 1}, s_targ_invcov, dhc_args, LossFunc, dLossFunc;
#FFTthreads::Int=1, optim_settings=Dict([("iterations", 10)]), lambda=0.001, func_specific_params=nothing
func_specific_params = Dict([(:reg_input=> init), (:coeff_mask1=> coeff_masksfd), (:target1=>ftargsfd), (:invcov1=>fcovinv1), (:coeff_mask2=> coeff_maskinit), (:target2=>ftarginit), (:invcov2=>fcovinv2),
    (:func=> Data_Utils.fnlog), (:dfunc=> Data_Utils.fndlog)])
func_specific_params[:lambda2] = lval2
func_specific_params[:lambda3] = lval3

recon_settings["fname_save"] = fname_save * ".jld2"
recon_settings["optim_settings"] = optim_settings

if logbool
    error("Not implemented here")
end

white_noise = randn(Nx, Nx)
res, recon_img = image_recon_derivsum_custom(white_noise, filter_hash, dhc_args, ReconFuncs.Loss3Gaussian_transformed,
    ReconFuncs.dLoss3Gaussian_transformed!; optim_settings=optim_settings, func_specific_params)
#save(settings["fname_save"], Dict("true_img"=>true_img, "init"=>noisy_init, "recon"=>recon_img, "dict"=>settings, "trace"=>Optim.trace(res), "coeff_mask"=>coeff_mask, "fhash"=>fhash, "dhc_args"=>dhc_args))


function loss1(inp_img)
    s_curr = DHC_compute_wrapper(inp_img, filter_hash, norm=false; dhc_args...)
    s_curr1 = Data_Utils.fnlog(s_curr[coeff_masksfd])
    diff1 = s_curr1 - ftargsfd
    return ( 0.5 .* (diff1)' * fcovinv1 * (diff1))
end

function loss2(inp_img)
    s_curr = DHC_compute_wrapper(inp_img, filter_hash, norm=false; dhc_args...)
    s_curr1 = Data_Utils.fnlog(s_curr[coeff_maskinit])
    diff1 = s_curr1 - ftarginit
    return ( 0.5 *lval2 .* (diff1)' * fcovinv2 * (diff1))
end

function loss3(inp_img)
    return 0.5*lval3*sum((ReconFuncs.adaptive_apodizer(inp_img, dhc_args) - ReconFuncs.adaptive_apodizer(init, dhc_args)).^2)
end


println("Loss Term 1: SFDTarg")
println("True", loss1(true_img))
println("Init", loss1(init))
println("Recon", loss1(recon_img))

println("Loss Term 2: SmoothedInitTarg")
println("True", loss2(true_img))
println("Init", loss2(init))
println("Recon", loss2(recon_img))

println("Loss Term 3: Reg")
println("True", loss3(true_img))
println("Init", loss3(init))
println("Recon", loss3(recon_img))

save("scratch_NM/NewWrapper/4-28/1000_C5_S2sfd_highjinit.jld2", Dict("true_img"=>true_img, "init"=>init, "recon"=>recon_img, "dict"=>recon_settings, "trace"=>Optim.trace(res), "coeff_mask"=>[coeff_masksfd, coeff_maskinit], "fhash"=>filter_hash, "dhc_args"=>dhc_args, "func_specific_params"=>func_specific_params))


heatmap(recon_img)
apdsmoothed = apodizer(imfilter(init, Kernel.gaussian(0.8)))
kbins= convert(Array{Float64}, collect(1:32))
clim=(minimum(apodizer(true_img)), maximum(apodizer(true_img)))
true_ps = Data_Utils.calc_1dps(apodizer(true_img), kbins)
initps = Data_Utils.calc_1dps(apodizer(white_noise), kbins)
recps = Data_Utils.calc_1dps(apodizer(recon_img), kbins)
smoothps = Data_Utils.calc_1dps(apdsmoothed, kbins)
p1 = heatmap(apodizer(recon_img), title="Recon", clim=clim)
p3 = plot(log.(kbins), log.(true_ps), label="True")
plot!(log.(kbins), log.(recps), label="Recon")
plot!(log.(kbins), log.(initps), label="White Noise")
plot!(log.(kbins), log.(smoothps), label="Smoothed Init (CoeffLoss)")
plot!(title="P(k): LogCoeffs 3 Loss")
xlabel!("lnk")
ylabel!("lnP(k)")
p2 = heatmap(apodizer(true_img), title="True Img", clim=clim)
p4= heatmap(apodizer(white_noise), title="Init Img", clim=clim)
p5= heatmap(apodizer(imfilter(init, Kernel.gaussian(0.8))), title="Smoothed init", clim=clim)
p6 = heatmap(apodizer(recon_img)- apodizer(true_img), title="Residual: Recon - True")
p7 = heatmap(apodizer(apdsmoothed)- apodizer(true_img), title="Residual: SmoothedInit - True")

struesel = Data_Utils.fnlog(DHC_compute_wrapper(true_img, filter_hash, norm=false; dhc_args...))
sinitsel = Data_Utils.fnlog(DHC_compute_wrapper(white_noise, filter_hash, norm=false; dhc_args...))
ssmoothsel = Data_Utils.fnlog(DHC_compute_wrapper(imfilter(init, Kernel.gaussian(0.8)), filter_hash, norm=false; dhc_args...))
sreconsel = Data_Utils.fnlog(DHC_compute_wrapper(recon_img, filter_hash, norm=false; dhc_args...))
slims = (minimum(struesel[JS1ind]), maximum(struesel[JS1ind]))
cg = cgrad([:blue, :white, :red])
truephi, trueomg = round(struesel[2+filter_hash["phi_index"]], sigdigits=3), round(struesel[2+filter_hash["Omega_index"]], sigdigits=3)
reconphi, reconomg = round(sreconsel[2+filter_hash["phi_index"]], sigdigits=3), round(sreconsel[2+filter_hash["Omega_index"]], sigdigits=3)
smoothphi, smoothomg = round(sinitsel[2+filter_hash["phi_index"]], sigdigits=3), round(sinitsel[2+filter_hash["Omega_index"]], sigdigits=3)

p8 = heatmap(struesel[JS1ind], title="True Coeffs ϕ=" * string(truephi) * "Ω=" * string(trueomg) , clims=slims, c=cg)
p9 = heatmap(sreconsel[JS1ind], title="Recon Coeffs ϕ=" * string(reconphi) * "Ω=" * string(reconomg), clims=slims, c=cg)
p10 = heatmap(ssmoothsel[JS1ind], title="White Noise Init ϕ=" * string(smoothphi) * "Ω=" * string(smoothomg), clims=slims, c=cg)
p = plot(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, layout=(5, 2), size=(1800, 2400))
savefig(p, "scratch_NM/NewWrapper/5-3/Case3i.png")

Visualization.plot_synth_QA(apodizer(true_img), apodizer(init), apodizer(recon_img), fname="scratch_NM/NewWrapper/4-28/1000_C5_S2sfd_highjinit.png")



clims=(-10, -2)
cg = cgrad([:blue, :white, :red])
p = plot(heatmap(struesel, title="True", clims=clims, c=cg),
heatmap(sinitsel, title="Init", clims=clims, c=cg),
heatmap(ssmoothsel, title="SmoothInit", clims=clims, c=cg),
heatmap(sreconsel, title="Recon", clims=clims, c=cg),
heatmap(fmean[JS1ind], title="SFDTargMean", clims=clims, c=cg),
heatmap(fmed, title="SFD Median", clims=clims, c=cg), size=(900, 600))

aptrue_img = apodizer(true_img)
apinit = apodizer(init)
aprecon_img = apodizer(recon_img)
println("Mean Abs Res: Init-True = ", mean(abs.(apinit - aptrue_img)), " Recon-True = ", mean(abs.(aprecon_img - aptrue_img)))
println("Mean Abs Frac Res", mean(abs.((apinit - aptrue_img)./aptrue_img)), " Recon-True=", mean(abs.((aprecon_img - aptrue_img)./aptrue_img)))
println("Mean L2 Res: Init-True = ", mean((apinit - aptrue_img).^2).^0.5, " Recon-True = ", mean((aprecon_img - aptrue_img).^2).^0.5)

##Distributions
Random.seed!(33)
randmat = randn(3, 3)
covmat = randmat * randmat'
smean = reshape(randn(3), (1, 3))

eps_samp = randn(1000, 3)
rsamps = smean .+ (sqrt(covmat) * eps_samp')'
rsamps_chol = smean .+ (cholesky(covmat).U * eps_samp')'
truedbn = Distributions.MvNormal(smean[:], covmat)
truesamps = rand(truedbn, 1000)'
p = plot(scatter(rsamps[:, 1], rsamps[:, 2], label="Samples using Cholesky"),
scatter!(truesamps[:, 1],truesamps[:, 2], label="Samples from True MVGaussian"))
whitened_ms = (cholesky(Symmetric(inv(covmat))).U * (truesamps .- smean)')'
p = plot(scatter(whitened_ms[:, 1], whitened_ms[:, 2]), label="Whitened true samples")

##Lognormal Distribution

truedbn = Distributions.MvNormal(smean[:], covmat)
truesamps = rand(truedbn, 1000)'
truesamps = exp.(truesamps)

sampmean = mean(truesamps, dims=1)
sampcov = (truesamps .- sampmean)' * (truesamps .- sampmean) ./ (size(truesamps)[1] - 1)

eps_samp = randn(1000, 3)
rsamps = sampmean .+ (cholesky(sampcov).U * eps_samp')'

p = plot(scatter(rsamps[:, 1], rsamps[:, 2], label="Samples using Cholesky", xlabel="Dim1", ylabel="Dim2"),
scatter!(truesamps[:, 1],truesamps[:, 2], label="Samples from True Log-MVGaussian", xlabel="Dim1", ylabel="Dim2"))

mean(rsamps, dims=1)
mean(truesamps, dims=1)
#rsampcov = (rsamps .- mean(rsamps, dims=1))' * (rsamps .- mean(rsamps, dims=1)) ./ (size(rsamps)[1] -1)
p = plot(scatter(rsamps[:, 2], rsamps[:, 3], label="Samples using Cholesky", xlabel="Dim2", ylabel="Dim3"),
scatter!(truesamps[:, 2],truesamps[:, 3], label="Samples from True Log-MVGaussian", xlabel="Dim2", ylabel="Dim3"))

#Whitened approx samples
covsqinv = cholesky(Symmetric(inv(sampcov))).U
whitened_ms = (covsqinv * (truesamps .- sampmean)')'
p = scatter(whitened_ms[:, 1], whitened_ms[:, 2], label="Whitened Samples from the Lognormal Distribution")

##MVNormal Distribution
Random.seed!(33)

truedbn = Distributions.MvNormal(smean[:], covmat)
truesamps = rand(truedbn, 1000)'

sampmean = mean(truesamps, dims=1)
sampcov = (truesamps .- sampmean)' * (truesamps .- sampmean) ./ (size(truesamps)[1] - 1)

eps_samp = randn(1000, 3)
rsamps = sampmean .+ (cholesky(sampcov).U * eps_samp')'

p = scatter(rsamps[:, 1], rsamps[:, 2], label="Samples using Cholesky", xlabel="Dim1", ylabel="Dim2", xlim=(-5, 5), ylim=(-5, 5))
scatter!(truesamps[:, 1],truesamps[:, 2], label="Samples from True MVGaussian", xlabel="Dim1", ylabel="Dim2", xlim=(-5, 5), ylim=(-5, 5))

mean(rsamps, dims=1)
mean(truesamps, dims=1)
#rsampcov = (rsamps .- mean(rsamps, dims=1))' * (rsamps .- mean(rsamps, dims=1)) ./ (size(rsamps)[1] -1)
p = scatter(rsamps[:, 2], rsamps[:, 3], label="Samples using Cholesky", xlabel="Dim2", ylabel="Dim3")
scatter!(truesamps[:, 2],truesamps[:, 3], label="Samples from True Log-MVGaussian", xlabel="Dim2", ylabel="Dim3"))

#Whitened approx samples
covsqinv = cholesky(invert_covmat(sampcov)).U
whitened_ms = (covsqinv * (truesamps .- sampmean)')'
p = scatter(whitened_ms[:, 2], whitened_ms[:, 3], label="Whitened Samples from the MVNormal Distribution", xlabel="Dim2", ylabel="Dim3", xlims=(-5, 5), ylims=(-5, 5))

##MVNormal but with sqrt
truedbn = Distributions.MvNormal(smean[:], covmat)
truesamps = rand(truedbn, 1000)'

sampmean = mean(truesamps, dims=1)
sampcov = (truesamps .- sampmean)' * (truesamps .- sampmean) ./ (size(truesamps)[1] - 1)

eps_samp = randn(1000, 3)
rsamps = sampmean .+ (sqrt(sampcov) * eps_samp')'

p = plot(scatter(rsamps[:, 1], rsamps[:, 2], label="Samples using Sqrt", xlabel="Dim1", ylabel="Dim2"),
scatter!(truesamps[:, 1],truesamps[:, 2], label="Samples from True MVGaussian", xlabel="Dim1", ylabel="Dim2"))

mean(rsamps, dims=1)
mean(truesamps, dims=1)
#rsampcov = (rsamps .- mean(rsamps, dims=1))' * (rsamps .- mean(rsamps, dims=1)) ./ (size(rsamps)[1] -1)
p = plot(scatter(rsamps[:, 2], rsamps[:, 3], label="Samples using Sqrt", xlabel="Dim2", ylabel="Dim3"),
scatter!(truesamps[:, 2],truesamps[:, 3], label="Samples from True Log-MVGaussian", xlabel="Dim2", ylabel="Dim3"))

#Whitened approx samples
covsqinv = sqrt(Symmetric(inv(sampcov)))
whitened_ms = (covsqinv * (truesamps .- sampmean)')'
p = scatter(whitened_ms[:, 1], whitened_ms[:, 2], xlims=(-10,10), ylims=(-10, 10), label="Whitened Samples from the MVNormal Distribution")



randmat = randn(3, 3)
covmat = randmat * randmat'
smean = reshape(randn(3), (1, 3))

eps_samp = randn(1000, 3)
rsamps = smean .+ (cholesky(covmat).U * eps_samp')'

truedbn = Distributions.MvNormal(smean[:], covmat)
truesamps = rand(truedbn, 1000)'
p = plot(scatter(rsamps[:, 1], rsamps[:, 2], label="Samples using Cholesky"),
scatter!(truesamps[:, 1],truesamps[:, 2], label="Samples from True MVGaussian"))
whitened_ms = (cholesky(Symmetric(inv(covmat))).U * (truesamps .- smean)')'
p = plot(scatter(whitened_ms[:, 1], whitened_ms[:, 2]), label="Whitened true samples")

##Tanh-normal Distribution

truedbn = Distributions.MvNormal(smean[:], covmat)
truesamps = rand(truedbn, 1000)'
truesamps = tanh.(truesamps)

sampmean = mean(truesamps, dims=1)
sampcov = (truesamps .- sampmean)' * (truesamps .- sampmean) ./ (size(truesamps)[1] - 1)

eps_samp = randn(1000, 3)
rsamps = sampmean .+ (cholesky(sampcov).U * eps_samp')'

p = plot(scatter(rsamps[:, 1], rsamps[:, 2], label="Samples using Cholesky", xlabel="Dim1", ylabel="Dim2"),
scatter!(truesamps[:, 1],truesamps[:, 2], label="Samples from True Log-MVGaussian", xlabel="Dim1", ylabel="Dim2"))

mean(rsamps, dims=1)
mean(truesamps, dims=1)
#rsampcov = (rsamps .- mean(rsamps, dims=1))' * (rsamps .- mean(rsamps, dims=1)) ./ (size(rsamps)[1] -1)
p = plot(scatter(rsamps[:, 2], rsamps[:, 3], label="Samples using Cholesky", xlabel="Dim2", ylabel="Dim3"),
scatter!(truesamps[:, 2],truesamps[:, 3], label="Samples from True Log-MVGaussian", xlabel="Dim2", ylabel="Dim3"))

#Whitened approx samples
covsqinv = cholesky(invert_covmat(sampcov)).U
whitened_ms = (covsqinv * (truesamps .- sampmean)')'
p = scatter(whitened_ms[:, 1], whitened_ms[:, 2], label="Whitened Samples from the Lognormal Distribution")

##Precompute and save covmats for Posterity
#Case 1
logbool = false
apdbool=true
isobool = false
Nx=64
filter_hash = fink_filter_hash(1, 8, nx=Nx, t=1, wd=1, Omega=true)
sfdall = readsfd_fromsrc("scratch_NM/data/dust10000.fits", Nx, logbool=logbool)
dhc_args = Dict(:doS2=>false, :doS20=>true, :apodize=>apdbool, :iso=>isobool)
truesamps = get_dbn_coeffs(sfdall, filter_hash, dhc_args)
sampmean = mean(truesamps, dims=1)
sampcov = (truesamps .- sampmean)' * (truesamps .- sampmean) ./ (size(truesamps)[1] - 1)
codestring = "logbool = false\n"*
"apdbool=true\n"*
"isobool = false\n"*
"Nx=64\n"*
"filter_hash = fink_filter_hash(1, 8, nx=Nx, t=1, wd=1, Omega=true)\n"*
"sfdall = readsfd_fromsrc(\"scratch_NM/data/dust10000.fits\", Nx, logbool=logbool)\n"*
"dhc_args = Dict(:doS2=>false, :doS20=>true, :apodize=>apdbool, :iso=>isobool)\n"*
"truesamps = get_dbn_coeffs(sfdall, filter_hash, dhc_args)\n"*
"sampmean = mean(truesamps, dims=1)\n"*
"sampcov = (truesamps .- sampmean)' * (truesamps .- sampmean) ./ (size(truesamps)[1] - 1))\n"

savdict = Dict([("logbool", logbool), ("apdbool", apdbool), ("isobool", isobool), ("dbncoeffs", truesamps),
    ("filter_hash", filter_hash), ("dhc_args", dhc_args), ("sampmean", sampmean), ("sampcov", sampcov), ("codestring", codestring)])
save("scratch_NM/SavedCovMats/reg_apd_noiso_nologcoeff.jld2", savdict)

#Case 2
logbool = false
apdbool=true
isobool = false
Nx=64
filter_hash = fink_filter_hash(1, 8, nx=Nx, t=1, wd=1, Omega=true)
sfdall = readsfd_fromsrc("scratch_NM/data/dust10000.fits", Nx, logbool=logbool)
dhc_args = Dict(:doS2=>false, :doS20=>true, :apodize=>apdbool, :iso=>isobool)
truesamps = log.(get_dbn_coeffs(sfdall, filter_hash, dhc_args))
sampmean = mean(truesamps, dims=1)
sampcov = (truesamps .- sampmean)' * (truesamps .- sampmean) ./ (size(truesamps)[1] - 1)
codestring = "logbool = false\n"*
"apdbool=true\n"*
"isobool = false\n"*
"Nx=64\n"*
"filter_hash = fink_filter_hash(1, 8, nx=Nx, t=1, wd=1, Omega=true)\n"*
"sfdall = readsfd_fromsrc(\"scratch_NM/data/dust10000.fits\", Nx, logbool=logbool)\n"*
"dhc_args = Dict(:doS2=>false, :doS20=>true, :apodize=>apdbool, :iso=>isobool)\n"*
"truesamps = log.(get_dbn_coeffs(sfdall, filter_hash, dhc_args))\n"*
"sampmean = mean(truesamps, dims=1)\n"*
"sampcov = (truesamps .- sampmean)' * (truesamps .- sampmean) ./ (size(truesamps)[1] - 1))\n"

savdict = Dict([("logbool", logbool), ("apdbool", apdbool), ("isobool", isobool), ("dbncoeffs", truesamps),
    ("filter_hash", filter_hash), ("dhc_args", dhc_args), ("sampmean", sampmean), ("sampcov", sampcov), ("codestring", codestring)])
save("scratch_NM/SavedCovMats/reg_apd_noiso_logcoeff.jld2", savdict)

##Now with the SFD Distribution
# Reg, RegCoeff
saved_sfddbn = load("scratch_NM/SavedCovMats/reg_apd_noiso_nologcoeff.jld2")
truesamps = saved_sfddbn["dbncoeffs"]
sampmean = mean(truesamps, dims=1)
EPS=1e-5
sampcov = ((truesamps .- sampmean)' * (truesamps .- sampmean) ./ (size(truesamps)[1] - 1))#
eps_samp = randn(1000, 1192)
svdcov = svd(sampcov +Diagonal(ones(1192)*EPS))
svdcov.U * Diagonal(svdcov.S) * svdcov.Vt
covsq = svdcov.U*Diagonal(sqrt.(svdcov.S))*svdcov.Vt
rsamps = sampmean .+ (covsq * eps_samp')'
rsamps_svd = copy(rsamps)
p = scatter(rsamps[1:1000, 3], rsamps[1:1000, 4], label="Samples from Q_Gaussian", xlabel="Dim3", ylabel="Dim4", c="orange", xlims=(-100, 100), ylims=(-100, 100))
scatter!(truesamps[1:1000, 3],truesamps[1:1000, 4], label="Samples from True Coeff Distribution", xlabel="Dim3", ylabel="Dim4", c="blue", xlims=(-100, 100), ylims=(-100, 100))
title!("Using SVD")

#rsampcov = (rsamps .- mean(rsamps, dims=1))' * (rsamps .- mean(rsamps, dims=1)) ./ (size(rsamps)[1] -1)
funcinv = invert_covmat(((truesamps .- sampmean)' * (truesamps .- sampmean) ./ (size(truesamps)[1] - 1)), 1e-5)

svdinv = svd(invert_covmat(sampcov, EPS))
svdinv.U * Diagonal(svdinv.S) * svdinv.Vt
covsqinv = svdinv.U*Diagonal(sqrt.(svdinv.S))*svdinv.Vt
#covsqinv = #cholesky(invert_covmat(sampcov, 1e-5)).U
whitened_ms = (covsqinv * (truesamps .- sampmean)')'
p = scatter(eps_samp[1:1000, 3], eps_samp[1:1000, 4], label="Actual White Noise samples", xlims=(-5, 5), ylims=(-5, 5), c="orange")
scatter!(whitened_ms[1:1000, 3], whitened_ms[1:1000, 4], label="Whitened Samples from the True Coeff Distribution", xlims=(-5, 5), ylims=(-5, 5), c="blue")
title!("Using SVD")

p = scatter(whitened_ms[1:1000, 3], whitened_ms[1:1000, 4], label="Whitened Samples from the True Coeff Distribution", xlims=(-5, 5), ylims=(-5, 5))
p=scatter(eps_samp[1:1000, 3], eps_samp[1:1000, 4], label="Actual White Noise samples", xlims=(-5, 5), ylims=(-5, 5))
title!("Using SVD")
maximum(abs.((covsqinv * covsq) - I))

del_eps = (whitened_ms[1:1000, 1:end] - eps_samp[1:1000, 1:end])'
del_d = covsq * del_eps


##Now with the SFD Distribution
# Reg, RegCoeff
saved_sfddbn = load("scratch_NM/SavedCovMats/reg_apd_noiso_nologcoeff.jld2")
truesamps = saved_sfddbn["dbncoeffs"]
sampmean = mean(truesamps, dims=1)
EPS=1e-5
sampcov = ((truesamps .- sampmean)' * (truesamps .- sampmean) ./ (size(truesamps)[1] - 1))#
eps_samp = randn(1000, 1192)
covsq = cholesky(sampcov+Diagonal(ones(1192)*EPS)).U
rsamps = sampmean .+ (covsq * eps_samp')'
rsamps_cholesky = copy(rsamps)
p = scatter(rsamps[1:1000, 3], rsamps[1:1000, 4], label="Samples from Q_Gaussian", xlabel="Dim3", ylabel="Dim4", xlims=(-100, 100), ylims=(-100, 100), c="orange")
scatter!(truesamps[1:1000, 3],truesamps[1:1000, 4], label="Samples from True Coeff Distribution", xlabel="Dim3", ylabel="Dim4", xlims=(-100, 100), ylims=(-100, 100), c="blue")
title!("Using Cholesky")

p = scatter(rsamps[1:1000, 3], rsamps[1:1000, 4], label="Samples from Q_Gaussian", xlabel="Dim3", ylabel="Dim4", xlims=(-0.1, 0.1), ylims=(-0.1, 0.1), c="orange")
scatter!(truesamps[1:1000, 3],truesamps[1:1000, 4], label="Samples from True Coeff Distribution", xlabel="Dim3", ylabel="Dim4", xlims=(-0.1, 0.1), ylims=(-0.1, 0.1), c="blue")
title!("Using Cholesky")
p = scatter(rsamps[:, 3], rsamps[:, 4], label="Samples from Q_Gaussian", xlabel="Dim3", ylabel="Dim4", c="orange")
scatter!(truesamps[:, 3],truesamps[:, 4], label="Samples from True Coeff Distribution", xlabel="Dim3", ylabel="Dim4", c="blue")
title!("Using Cholesky")

#rsampcov = (rsamps .- mean(rsamps, dims=1))' * (rsamps .- mean(rsamps, dims=1)) ./ (size(rsamps)[1] -1)
funcinv = invert_covmat(((truesamps .- sampmean)' * (truesamps .- sampmean) ./ (size(truesamps)[1] - 1)), 1e-5)

covsqinv = cholesky(invert_covmat(sampcov, EPS)).U
#covsqinv = #cholesky(invert_covmat(sampcov, 1e-5)).U
whitened_ms = (covsqinv * (truesamps .- sampmean)')'
p = scatter(eps_samp[1:1000, 3], eps_samp[1:1000, 4], label="Actual White Noise samples", xlims=(-5, 5), ylims=(-5, 5), c="orange")
scatter!(whitened_ms[1:1000, 3], whitened_ms[1:1000, 4], label="Whitened Samples from the True Coeff Distribution", xlims=(-5, 5), ylims=(-5, 5), c="blue")
title!("Using Cholesky")

p = scatter(whitened_ms[1:1000, 3], whitened_ms[1:1000, 4], label="Whitened Samples from the True Coeff Distribution", xlims=(-5, 5), ylims=(-5, 5), c="blue")
p=scatter(eps_samp[1:1000, 3], eps_samp[1:1000, 4], label="Actual White Noise samples", xlims=(-5, 5), ylims=(-5, 5), c="orange")
title!("Using Cholesky")
maximum(abs.(inv(covsqinv) - covsq))

##Now with the SFD Distribution:10k
# Reg, RegCoeff
saved_sfddbn = load("scratch_NM/SavedCovMats/reg_apd_noiso_nologcoeff.jld2")
truesamps = saved_sfddbn["dbncoeffs"]
sampmean = mean(truesamps, dims=1)
EPS=1e-5
sampcov = ((truesamps .- sampmean)' * (truesamps .- sampmean) ./ (size(truesamps)[1] - 1))#
eps_samp = randn(100000, 1192)
svdcov = svd(sampcov +Diagonal(ones(1192)*EPS))
svdcov.U * Diagonal(svdcov.S) * svdcov.Vt
covsq = svdcov.U*Diagonal(sqrt.(svdcov.S))*svdcov.Vt
rsamps = sampmean .+ (covsq * eps_samp')'
rsamps_svd = copy(rsamps)
svdrcov = (rsamps_svd .- mean(rsamps_svd, dims=1))' * (rsamps_svd .- mean(rsamps_svd, dims=1)) ./(size(rsamps_svd)[1] -1)
svdrcov

p = scatter(rsamps[1:1000, 3], rsamps[1:1000, 4], label="Samples from Q_Gaussian", xlabel="Dim3", ylabel="Dim4", c="orange", xlims=(-100, 100), ylims=(-100, 100))
scatter!(truesamps[1:1000, 3],truesamps[1:1000, 4], label="Samples from True Coeff Distribution", xlabel="Dim3", ylabel="Dim4", c="blue", xlims=(-100, 100), ylims=(-100, 100))
title!("Using SVD")
svdrcov = (rsamps_svd .- mean(rsamps_svd, dims=1))' * (rsamps_svd .- mean(rsamps_svd, dims=1)) ./(size(rsamps_svd)[1] -1)
#rsampcov = (rsamps .- mean(rsamps, dims=1))' * (rsamps .- mean(rsamps, dims=1)) ./ (size(rsamps)[1] -1)
funcinv = invert_covmat(((truesamps .- sampmean)' * (truesamps .- sampmean) ./ (size(truesamps)[1] - 1)), 1e-5)

svdinv = svd(invert_covmat(sampcov, EPS))
svdinv.U * Diagonal(svdinv.S) * svdinv.Vt
covsqinv = svdinv.U*Diagonal(sqrt.(svdinv.S))*svdinv.Vt
#covsqinv = #cholesky(invert_covmat(sampcov, 1e-5)).U
whitened_ms = (covsqinv * (truesamps .- sampmean)')'
p = scatter(eps_samp[1:1000, 3], eps_samp[1:1000, 4], label="Actual White Noise samples", xlims=(-5, 5), ylims=(-5, 5), c="orange")
scatter!(whitened_ms[1:1000, 3], whitened_ms[1:1000, 4], label="Whitened Samples from the True Coeff Distribution", xlims=(-5, 5), ylims=(-5, 5), c="blue")
title!("Using SVD")

p = scatter(whitened_ms[1:1000, 3], whitened_ms[1:1000, 4], label="Whitened Samples from the True Coeff Distribution", xlims=(-5, 5), ylims=(-5, 5))
p=scatter(eps_samp[1:1000, 3], eps_samp[1:1000, 4], label="Actual White Noise samples", xlims=(-5, 5), ylims=(-5, 5))
title!("Using SVD")
maximum(abs.((covsqinv * covsq) - I))

del_eps = (whitened_ms[1:1000, 1:end] - eps_samp[1:1000, 1:end])'
del_d = covsq * del_eps

##CHolesky-10k
saved_sfddbn = load("scratch_NM/SavedCovMats/reg_apd_noiso_nologcoeff.jld2")
truesamps = saved_sfddbn["dbncoeffs"]
sampmean = mean(truesamps, dims=1)
EPS=1e-5
sampcov = ((truesamps .- sampmean)' * (truesamps .- sampmean) ./ (size(truesamps)[1] - 1))#
eps_samp = randn(100000, 1192)
covsq = cholesky(sampcov+Diagonal(ones(1192)*EPS)).U
rsamps = sampmean .+ (covsq * eps_samp')'
rsamps_cholesky = copy(rsamps)
choleskyrcov = (rsamps_cholesky .- mean(rsamps_cholesky, dims=1))' * (rsamps_cholesky .- mean(rsamps_cholesky, dims=1)) ./(size(rsamps_cholesky)[1] -1)

##CHolesky-10k
saved_sfddbn = load("scratch_NM/SavedCovMats/reg_apd_noiso_nologcoeff.jld2")
truesamps = saved_sfddbn["dbncoeffs"]
sampmean = mean(truesamps, dims=1)
EPS=1e-5
sampcov = ((truesamps .- sampmean)' * (truesamps .- sampmean) ./ (size(truesamps)[1] - 1))#
eps_samp = randn(100000, 1192)
svdobj = svd(sampcov+ Diagonal(ones(1192)*EPS)) #
covsq = svdobj.U *Diagonal(sqrt.(svdobj.S))*svdobj.Vt
rsamps = sampmean .+ (covsq * eps_samp')'
rsamps_svd = copy(rsamps)
svdrcov = (rsamps_svd .- mean(rsamps_svd, dims=1))' * (rsamps_svd .- mean(rsamps_svd, dims=1)) ./(size(rsamps_svd)[1] -1)


##Looking at an MVGaussian dbn with a decent covariance matrix: also INCONSISTENT
Random.seed!(33)
smean = randn(1192)
amat = randn(1192, 1192)
scov = amat * amat'
truedbn = MvNormal(smean, scov)
truesamps = rand(truedbn, 10000)'

sampmean = mean(truesamps, dims=1)
EPS=1e-5
sampcov = ((truesamps .- sampmean)' * (truesamps .- sampmean) ./ (size(truesamps)[1] - 1))#
eps_samp = randn(10000, 1192)
covsq = cholesky(sampcov+Diagonal(ones(1192)*EPS)).U
rsamps = sampmean .+ (covsq * eps_samp')'

p = scatter(rsamps[1:1000, 3], rsamps[1:1000, 4], label="Samples from Q_Gaussian", xlabel="Dim3", ylabel="Dim4", xlims=(-100, 100), ylims=(-100, 100), c="orange")
scatter!(truesamps[1:1000, 3],truesamps[1:1000, 4], label="Samples from True Coeff Distribution", xlabel="Dim3", ylabel="Dim4", xlims=(-100, 100), ylims=(-100, 100), c="blue")
title!("Using Cholesky")

covsqinv = cholesky(invert_covmat(sampcov, EPS)).U
#covsqinv = #cholesky(invert_covmat(sampcov, 1e-5)).U
whitened_ms = (covsqinv * (truesamps .- sampmean)')'
p =scatter(eps_samp[1:1000, 3], eps_samp[1:1000, 4], label="Actual White Noise samples", xlims=(-5, 5), ylims=(-5, 5), c="orange")
scatter!(whitened_ms[1:1000, 3], whitened_ms[1:1000, 4], label="Whitened Samples from the True Coeff Distribution", xlims=(-5, 5), ylims=(-5, 5), c="blue")
title!("Using Cholesky")




truesamps = tanh.(truesamps)

sampmean = mean(truesamps, dims=1)
EPS=1e-5
sampcov = ((truesamps .- sampmean)' * (truesamps .- sampmean) ./ (size(truesamps)[1] - 1))#
eps_samp = randn(1000, 1192)
covsq = cholesky(sampcov+Diagonal(ones(1192)*EPS)).U
rsamps = sampmean .+ (covsq * eps_samp')'

p = scatter(rsamps[:, 3], rsamps[:, 4], label="Samples from Q_Gaussian", xlabel="Dim3", ylabel="Dim4", xlims=(-1, 1), ylims=(-1, 1), c="orange")
scatter!(truesamps[:, 3],truesamps[:, 4], label="Samples from True Coeff Distribution", xlabel="Dim3", ylabel="Dim4", xlims=(-1, 1), ylims=(-1, 1), c="blue")
title!("Using Cholesky")
p = scatter(truesamps[1:1000, 3],truesamps[1:1000, 4], label="Samples from True Coeff Distribution", xlabel="Dim3", ylabel="Dim4", xlims=(-100, 100), ylims=(-100, 100), c="blue")
scatter!(rsamps[:, 3], rsamps[:, 4], label="Samples from Q_Gaussian", xlabel="Dim3", ylabel="Dim4", xlims=(-100, 100), ylims=(-100, 100), c="orange")
title!("Using Cholesky")

covsqinv = cholesky(invert_covmat(sampcov, EPS)).U
#covsqinv = #cholesky(invert_covmat(sampcov, 1e-5)).U
whitened_ms = (covsqinv * (truesamps .- sampmean)')'
p = scatter(whitened_ms[1:1000, 3], whitened_ms[1:1000, 4], label="Whitened Samples from the True Coeff Distribution", xlims=(-5, 5), ylims=(-5, 5))
scatter!(eps_samp[:, 3], eps_samp[:, 4], label="Actual White Noise samples", xlims=(-5, 5), ylims=(-5, 5))
title!("Using Cholesky")


whitened_ms[1, 3]
truesamps[1, 3]
##With a Gaussian distribution with the same covar
Random.seed!(33)
smean = randn(4)
amat = randn(4, 4)
scov = amat * amat'
truedbn = MvNormal(smean, scov)
truesamps = rand(truedbn, 1000)'
truesamps = tanh.(truesamps)

sampmean = mean(truesamps, dims=1)
EPS=1e-5
sampcov = ((truesamps .- sampmean)' * (truesamps .- sampmean) ./ (size(truesamps)[1] - 1))#
eps_samp = randn(1000, 4)
covsq = cholesky(sampcov+Diagonal(ones(4)*EPS)).U
rsamps = sampmean .+ (covsq * eps_samp')'

p = scatter(rsamps[:, 3], rsamps[:, 4], label="Samples from Q_Gaussian", xlabel="Dim3", ylabel="Dim4", xlims=(-1, 1), ylims=(-1, 1), c="orange")
scatter!(truesamps[:, 3],truesamps[:, 4], label="Samples from True Coeff Distribution", xlabel="Dim3", ylabel="Dim4", xlims=(-1, 1), ylims=(-1, 1), c="blue")
title!("Using Cholesky")
p = scatter(truesamps[1:1000, 3],truesamps[1:1000, 4], label="Samples from True Coeff Distribution", xlabel="Dim3", ylabel="Dim4", xlims=(-100, 100), ylims=(-100, 100), c="blue")
scatter!(rsamps[:, 3], rsamps[:, 4], label="Samples from Q_Gaussian", xlabel="Dim3", ylabel="Dim4", xlims=(-100, 100), ylims=(-100, 100), c="orange")
title!("Using Cholesky")

covsqinv = cholesky(invert_covmat(sampcov, EPS)).U
#covsqinv = #cholesky(invert_covmat(sampcov, 1e-5)).U
whitened_ms = (covsqinv * (truesamps .- sampmean)')'
p = scatter(whitened_ms[1:1000, 3], whitened_ms[1:1000, 4], label="Whitened Samples from the True Coeff Distribution", xlims=(-5, 5), ylims=(-5, 5))
scatter!(eps_samp[:, 3], eps_samp[:, 4], label="Actual White Noise samples", xlims=(-5, 5), ylims=(-5, 5))
title!("Using Cholesky")

##Identically written lognormal distribution
Random.seed!(33)
randmat = randn(3, 3)
covmat = randmat * randmat'
smean = reshape(randn(3), (1, 3))

truedbn = Distributions.MvNormal(smean[:], covmat)
truesamps = rand(truedbn, 1000)'
truesamps = tanh.(truesamps)

sampmean = mean(truesamps, dims=1)
sampcov = (truesamps .- sampmean)' * (truesamps .- sampmean) ./ (size(truesamps)[1] - 1)

eps_samp = randn(1000, 3)
rsamps = sampmean .+ (cholesky(sampcov + Diagonal(ones(3)*EPS)).U * eps_samp')'

p = plot(scatter(rsamps[:, 1], rsamps[:, 2], label="Samples using Cholesky", xlabel="Dim1", ylabel="Dim2"),
scatter!(truesamps[:, 1],truesamps[:, 2], label="Samples from True Tanh-MVGaussian", xlabel="Dim1", ylabel="Dim2"))

mean(rsamps, dims=1)
mean(truesamps, dims=1)
#rsampcov = (rsamps .- mean(rsamps, dims=1))' * (rsamps .- mean(rsamps, dims=1)) ./ (size(rsamps)[1] -1)
p = plot(scatter(rsamps[:, 2], rsamps[:, 3], label="Samples using Cholesky", xlabel="Dim2", ylabel="Dim3"),
scatter!(truesamps[:, 2],truesamps[:, 3], label="Samples from True Log-MVGaussian", xlabel="Dim2", ylabel="Dim3"))

#Whitened approx samples
covsqinv = cholesky(invert_covmat(sampcov, EPS)).U
whitened_ms = (covsqinv * (truesamps .- sampmean)')'
p = scatter(whitened_ms[:, 1], whitened_ms[:, 2], label="Whitened Samples from the Tanh Distribution")

covsqinv' * covsqinv #equals invert_covmat(sampcov)
covsqinv * cholesky(sampcov + Diagonal(ones(3)*EPS)).U

##Identically written lognormal distribution
Random.seed!(33)
randmat = randn(3, 3)
covmat = randmat * randmat'
smean = reshape(randn(3), (1, 3))

truedbn = Distributions.MvNormal(smean[:], covmat)
truesamps = rand(truedbn, 1000)'
truesamps = tanh.(truesamps)

sampmean = mean(truesamps, dims=1)
sampcov = (truesamps .- sampmean)' * (truesamps .- sampmean) ./ (size(truesamps)[1] - 1)

eps_samp = randn(1000, 3)
svdcov = svd(sampcov)
svdcov.U * Diagonal(svdcov.S) * svdcov.Vt
covsq = svdcov.U*Diagonal(sqrt.(svdcov.S))*svdcov.Vt
covsq*covsq
rsamps = sampmean .+ (covsq * eps_samp')'

p = plot(scatter(rsamps[:, 1], rsamps[:, 2], label="Samples using Cholesky", xlabel="Dim1", ylabel="Dim2"),
scatter!(truesamps[:, 1],truesamps[:, 2], label="Samples from True Tanh-MVGaussian", xlabel="Dim1", ylabel="Dim2"))

mean(rsamps, dims=1)
mean(truesamps, dims=1)
#rsampcov = (rsamps .- mean(rsamps, dims=1))' * (rsamps .- mean(rsamps, dims=1)) ./ (size(rsamps)[1] -1)
p = plot(scatter(rsamps[:, 2], rsamps[:, 3], label="Samples using Cholesky", xlabel="Dim2", ylabel="Dim3"),
scatter!(truesamps[:, 2],truesamps[:, 3], label="Samples from True Log-MVGaussian", xlabel="Dim2", ylabel="Dim3"))

#Whitened approx samples
svdinv = svd(invert_covmat(sampcov))
covsqinv = svdinv.U * Diagonal(sqrt.(svdinv.S)) * svdinv.Vt
#covsqinv =   #cholesky(invert_covmat(sampcov, EPS)).U
whitened_ms = (covsqinv * (truesamps .- sampmean)')'
p = scatter(whitened_ms[:, 1], whitened_ms[:, 2], label="Whitened Samples from the Tanh Distribution")

covsqinv' * covsqinv #equals invert_covmat(sampcov)
covsqinv * cholesky(sampcov + Diagonal(ones(3)*EPS)).U

##Now with the SFD Distribution AND Symmetric Sampcov
# Reg, RegCoeff
logbool = false
apdbool=true
isobool = false
Nx=64
filter_hash = fink_filter_hash(1, 8, nx=Nx, t=1, wd=1, Omega=true)
sfdall = readsfd_fromsrc("scratch_NM/data/dust10000.fits", Nx, logbool=logbool)
dhc_args = Dict(:doS2=>false, :doS20=>true, :apodize=>apdbool, :iso=>isobool)
truesamps = get_dbn_coeffs(sfdall, filter_hash, dhc_args)
sampmean = mean(truesamps, dims=1)
EPS=1e-5
sampcov = Symmetric(((truesamps .- sampmean)' * (truesamps .- sampmean) ./ (size(truesamps)[1] - 1)) + Diagonal(ones(1192)*EPS))

eps_samp = randn(1000, 1192)
rsamps = sampmean .+ (cholesky(sampcov).U * eps_samp')'

p = plot(scatter(rsamps[:, 3], rsamps[:, 4], label="Samples using Cholesky", xlabel="Dim3", ylabel="Dim4"),
scatter!(truesamps[:, 3],truesamps[:, 4], label="Samples from True Log-MVGaussian", xlabel="Dim3", ylabel="Dim4"))

#rsampcov = (rsamps .- mean(rsamps, dims=1))' * (rsamps .- mean(rsamps, dims=1)) ./ (size(rsamps)[1] -1)
funcinv = invert_covmat(((truesamps .- sampmean)' * (truesamps .- sampmean) ./ (size(truesamps)[1] - 1)), 1e-5)

covsqinv = cholesky(inv(sampcov)).U
whitened_ms = (covsqinv * (truesamps .- sampmean)')'
p = scatter(whitened_ms[:, 3], whitened_ms[:, 4], label="Whitened Samples from the Lognormal Distribution")

#hereinv = inv(sampcov)
#maximum(abs.(funcinv .- hereinv)) = 0.02

#Differences from ipynb code:
#Symmetric
#inv instead of invert_covmat

#Doug
Ns = 100
mockx = randn(Ns)
mocky = randn(Ns)+mockx.*2 .+1
mydata = hcat(mockx,mocky)
data_mean = mean(mydata,dims=1)
data_meansub = mydata .- data_mean
cov = (data_meansub' * data_meansub)./(Ns-1)

U = cholesky(cov).U
ran = randn(Ns,2)

draw = (ran * U) .+ data_mean
scatter(mydata[:,1],mydata[:,2])
scatter!(draw[:,1],draw[:,2])

# Reg, RegCoeff
saved_sfddbn = load("scratch_NM/SavedCovMats/reg_apd_noiso_nologcoeff.jld2")
truesamps = saved_sfddbn["dbncoeffs"]
sampmean = mean(truesamps, dims=1)
EPS=1e-5
sampcov = ((truesamps .- sampmean)' * (truesamps .- sampmean) ./ (size(truesamps)[1] - 1))#
eps_samp = randn(1000, 1192)
svdcov = svd(sampcov +Diagonal(ones(1192)*EPS))
svdcov.U * Diagonal(svdcov.S) * svdcov.Vt
covsq = svdcov.U*Diagonal(sqrt.(svdcov.S))*svdcov.Vt
rsamps = sampmean .+ (covsq * eps_samp')'

p = scatter(rsamps[1:1000, 3], rsamps[1:1000, 4], label="Samples from Q_Gaussian", xlabel="Dim3", ylabel="Dim4", c="orange", xlims=(-100, 100), ylims=(-100, 100))
scatter!(truesamps[:, 3],truesamps[:, 4], label="Samples from True Coeff Distribution", xlabel="Dim3", ylabel="Dim4", c="blue", xlims=(-100, 100), ylims=(-100, 100))
title!("Using SVD:Imbalanced")

#rsampcov = (rsamps .- mean(rsamps, dims=1))' * (rsamps .- mean(rsamps, dims=1)) ./ (size(rsamps)[1] -1)
funcinv = invert_covmat(((truesamps .- sampmean)' * (truesamps .- sampmean) ./ (size(truesamps)[1] - 1)), 1e-5)

svdinv = svd(invert_covmat(sampcov, EPS))
svdinv.U * Diagonal(svdinv.S) * svdinv.Vt
covsqinv = svdinv.U*Diagonal(sqrt.(svdinv.S))*svdinv.Vt
#covsqinv = #cholesky(invert_covmat(sampcov, 1e-5)).U
whitened_ms = (covsqinv * (truesamps .- sampmean)')'
p = scatter(eps_samp[1:1000, 3], eps_samp[1:1000, 4], label="Actual White Noise samples", xlims=(-5, 5), ylims=(-5, 5), c="orange")
scatter!(whitened_ms[:, 3], whitened_ms[:, 4], label="Whitened Samples from the True Coeff Distribution", xlims=(-5, 5), ylims=(-5, 5), c="blue")
title!("Using SVD")

p = scatter(whitened_ms[1:1000, 3], whitened_ms[1:1000, 4], label="Whitened Samples from the True Coeff Distribution", xlims=(-5, 5), ylims=(-5, 5))
p=scatter(eps_samp[1:1000, 3], eps_samp[1:1000, 4], label="Actual White Noise samples", xlims=(-5, 5), ylims=(-5, 5))
title!("Using SVD")
maximum(abs.((covsqinv * covsq) - I))

del_eps = (whitened_ms[1:1000, 1:end] - eps_samp[1:1000, 1:end])'
del_d = covsq * del_eps

##5 / 6


##Which Subset of coefficients has the most decent covariance?
saved_sfddbn = load("scratch_NM/SavedCovMats/reg_apd_noiso_nologcoeff.jld2")
truesamps = saved_sfddbn["dbncoeffs"]
scov = (truesamps .- mean(truesamps, dims=1))' * (truesamps .- mean(truesamps, dims=1))./(size(truesamps)[1] -1)
filter_hash = saved_sfddbn["filter_hash"]
Nf = size(filter_hash["filt_index"])[1]

#Only S1
onlyS1 = scov[3:Nf+2, 3:Nf+2]
cond(onlyS1)
eigvals(onlyS1)

#Uppertriangular S2R
mask = falses(2+Nf+Nf^2)
mask[Nf+3:end] .= triu(trues(Nf, Nf))[:]
triuS2 = scov[mask, mask]
cond(triuS2)
findall(eigvals(triuS2) .<1e-5)
595-456
228*2

plot(log10.(eigvals(triuS2)))


##Adding S2R
ARGS_buffer = ["reg", "apd", "noiso", "scratch_NM/StandardizedExp/Nx64/", "full_3losstest", "Full+Eps"]
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

fname_save = direc * "S1only/SFDTargSFDCov/" * ARGS_buffer[1] * "_" * ARGS_buffer[2] * "_" * ARGS_buffer[3] * "/LogCoeff/" * string(numfile) * ARGS_buffer[5]  #Change
Nx=size(true_img)[1]
filter_hash = fink_filter_hash(1, 8, nx=Nx, t=1, Omega=true)
(S1iso, Nf) = size(filter_hash["S1_iso_mat"])
(S2iso, Nfsq) = size(filter_hash["S2_iso_mat"])
dhc_args = Dict(:doS2=>false, :doS20=>true, :apodize=>apdbool, :iso=>isobool) #Iso #CHANGE: Change sig for the sfd data since the noise model is super high and the tiny values make sense

optim_settings = Dict([("iterations", 1000), ("norm", false), ("minmethod", ConjugateGradient())])
recon_settings = Dict([("log", logbool), ("Invcov_matrix", ARGS_buffer[6]), ("optim_settings", optim_settings), ("eps_value_sfd", 1e-5), ("eps_value_init", 1e-10)]) #Add constraints

recon_settings["datafile"] = datfile
#Read in SFD Dbn
sfdall = readsfd(Nx, logbool=logbool)
dbnocffs = log.(get_dbn_coeffs(sfdall, filter_hash, dhc_args))
fmean = mean(dbnocffs, dims=1)
size(dbnocffs)
#fmed = distribution_percentiles(dbnocffs, JS1ind, 50)


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

JS1ind = J_S1indices([0, 1, 2, 3], filter_hash)
function distribution_percentiles(dbn, idx, perc)
    idxperc = zeros(size(idx))
    idxperc .= (x-> percentile(dbn[:, x], perc)).(idx)
    return idxperc
end

#Construct SFD CoeffMask
if dhc_args[:iso]
    error("Not constructed for iso")
else #Not iso
    coeff_masksfd = falses(2+Nf+Nf^2)
    #lowjidx = (filter_hash["J_L"][:, 1] .== 0) .| (filter_hash["J_L"][:, 1] .== 1)
    #coeff_maskS1 = falses(Nf)
    #coeff_maskS1[lowjidx] .= true
    #coeff_maskS1[33] .= true
    #coeff_maskS1[34] = true
    coeff_masksfd[Nf+3:end] .= triu(trues(Nf, Nf))[:] #Diagonal(trues(Nf))[:] #
end

#Construct Init CoeffMask
if dhc_args[:iso]
    error("Not constructed for iso")
else #Not iso
    coeff_maskinit = falses(2+Nf+Nf^2)
    highjidx = (filter_hash["J_L"][:, 1] .== 2) .| (filter_hash["J_L"][:, 1] .== 3) .| (filter_hash["J_L"][:, 1] .== 1)
    coeff_maskS1 = falses(Nf)
    coeff_maskS1[highjidx] .= true
    coeff_maskS1[33] = true
    coeff_maskinit[Nf+3:end] .= Diagonal(coeff_maskS1)[:]
end
recon_settings["coeff_mask_sfd"] = coeff_masksfd
recon_settings["coeff_mask_init"] = coeff_maskinit

fcov = (dbnocffs .- fmean)' * (dbnocffs .- fmean) ./(size(dbnocffs)[1] -1)
fcovinv1 = pinv(fcov[coeff_masksfd, coeff_masksfd])
fcovinv2 = pinv(fcov[coeff_maskinit, coeff_maskinit])
ftargsfd = fmean[coeff_masksfd]
if logbool
    error("Not impl for log im")
end
ftarginit = log.(DHC_compute_wrapper(imfilter(init, Kernel.gaussian(0.8)), filter_hash; dhc_args...))
ftarginit = ftarginit[coeff_maskinit]

#Weight given to terms
lval2 = 1.0
lval3=1.0

println("Regularizer Lambda=", round(lval3, sigdigits=3))
#input::Array{Float64, 2}, filter_hash::Dict, s_targ_mean::Array{Float64, 1}, s_targ_invcov, dhc_args, LossFunc, dLossFunc;
#FFTthreads::Int=1, optim_settings=Dict([("iterations", 10)]), lambda=0.001, func_specific_params=nothing
func_specific_params = Dict([(:reg_input=> init), (:coeff_mask1=> coeff_masksfd), (:target1=>ftargsfd), (:invcov1=>fcovinv1), (:coeff_mask2=> coeff_maskinit), (:target2=>ftarginit), (:invcov2=>fcovinv2),
    (:func=> Data_Utils.fnlog), (:dfunc=> Data_Utils.fndlog)])
func_specific_params[:lambda2] = lval2
func_specific_params[:lambda3] = lval3

recon_settings["fname_save"] = fname_save * ".jld2"
recon_settings["optim_settings"] = optim_settings

if logbool
    error("Not implemented here")
end

res, recon_img = image_recon_derivsum_custom(init, filter_hash, dhc_args, ReconFuncs.Loss3Gaussian_transformed,
    ReconFuncs.dLoss3Gaussian_transformed!; optim_settings=optim_settings, func_specific_params)
#save(settings["fname_save"], Dict("true_img"=>true_img, "init"=>noisy_init, "recon"=>recon_img, "dict"=>settings, "trace"=>Optim.trace(res), "coeff_mask"=>coeff_mask, "fhash"=>fhash, "dhc_args"=>dhc_args))


function loss1(inp_img)
    s_curr = DHC_compute_wrapper(inp_img, filter_hash, norm=false; dhc_args...)
    s_curr1 = Data_Utils.fnlog(s_curr[coeff_masksfd])
    diff1 = s_curr1 - ftargsfd
    return ( 0.5 .* (diff1)' * fcovinv1 * (diff1))
end

function loss2(inp_img)
    s_curr = DHC_compute_wrapper(inp_img, filter_hash, norm=false; dhc_args...)
    s_curr1 = Data_Utils.fnlog(s_curr[coeff_maskinit])
    diff1 = s_curr1 - ftarginit
    return ( 0.5 *lval2 .* (diff1)' * fcovinv2 * (diff1))
end

function loss3(inp_img)
    return 0.5*lval3*sum((ReconFuncs.adaptive_apodizer(inp_img, dhc_args) - ReconFuncs.adaptive_apodizer(init, dhc_args)).^2)
end


println("Loss Term 1: SFDTarg")
println("True", loss1(true_img))
println("Init", loss1(init))
println("Recon", loss1(recon_img))

println("Loss Term 2: SmoothedInitTarg")
println("True", loss2(true_img))
println("Init", loss2(init))
println("Recon", loss2(recon_img))

println("Loss Term 3: Reg")
println("True", loss3(true_img))
println("Init", loss3(init))
println("Recon", loss3(recon_img))

apdsmoothed = apodizer(imfilter(init, Kernel.gaussian(0.8)))
kbins= convert(Array{Float64}, collect(1:32))
clim=(minimum(apodizer(true_img)), maximum(apodizer(true_img)))
true_ps = Data_Utils.calc_1dps(apodizer(true_img), kbins)
initps = Data_Utils.calc_1dps(apodizer(init), kbins)
recps = Data_Utils.calc_1dps(apodizer(recon_img), kbins)
smoothps = Data_Utils.calc_1dps(apdsmoothed, kbins)
p1 = heatmap(apodizer(recon_img), title="Recon", clim=clim)
p3 = plot(log.(kbins), log.(true_ps), label="True")
plot!(log.(kbins), log.(recps), label="Recon")
plot!(log.(kbins), log.(initps), label="Init")
plot!(log.(kbins), log.(smoothps), label="Smoothed Init")
plot!(title="P(k): LogCoeffs 3 Loss")
xlabel!("lnk")
ylabel!("lnP(k)")
p2 = heatmap(apodizer(true_img), title="True Img", clim=clim)
p4= heatmap(apodizer(init), title="Init Img", clim=clim)
p5= heatmap(apodizer(imfilter(init, Kernel.gaussian(0.8))), title="Smoothed init", clim=clim)
residual = apodizer(recon_img)- apodizer(true_img)
rlims = (minimum(residual), maximum(residual))
#symmax = maximum([abs(minimum(residual)), maximum(residual)])
#rg = cgrad(:bwr, [-symmax/2.0, symmax/2.0])
p6 = heatmap(residual, title="Residual: Recon - True", clims=rlims, c=:bwr)
p7 = heatmap(apdsmoothed- apodizer(true_img), title="Residual: SmoothedInit - True", clims=rlims, c=:bwr)

struesel = Data_Utils.fnlog(DHC_compute_wrapper(true_img, filter_hash, norm=false; dhc_args...))
sinitsel = Data_Utils.fnlog(DHC_compute_wrapper(init, filter_hash, norm=false; dhc_args...))
ssmoothsel = Data_Utils.fnlog(DHC_compute_wrapper(imfilter(init, Kernel.gaussian(0.8)), filter_hash, norm=false; dhc_args...))
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
savefig(p, "scratch_NM/NewWrapper/5-3/Case4j.png")

dbnwhite = Distributions.Normal(0, 0.1)
ximap = reshape(rand(dbnwhite, 64*64), (64, 64))
fximap = fft(ximap)


##Using Pinv
saved_sfddbn = load("scratch_NM/SavedCovMats/reg_apd_noiso_nologcoeff.jld2")
truesamps = saved_sfddbn["dbncoeffs"]
sampmean = mean(truesamps, dims=1)
EPS=1e-5
sampcov = ((truesamps .- sampmean)' * (truesamps .- sampmean) ./ (size(truesamps)[1] - 1))#
eps_samp = randn(1000, 1192)
svdcov = svd(sampcov +Diagonal(ones(1192)*EPS))
svdcov.U * Diagonal(svdcov.S) * svdcov.Vt
covsq = svdcov.U*Diagonal(sqrt.(svdcov.S))*svdcov.Vt
rsamps = sampmean .+ (covsq * eps_samp')'
rsamps_svd = copy(rsamps)
p = scatter(rsamps[1:1000, 3], rsamps[1:1000, 4], label="Samples from Q_Gaussian", xlabel="Dim3", ylabel="Dim4", c="orange", xlims=(-100, 100), ylims=(-100, 100))
scatter!(truesamps[1:1000, 3],truesamps[1:1000, 4], label="Samples from True Coeff Distribution", xlabel="Dim3", ylabel="Dim4", c="blue", xlims=(-100, 100), ylims=(-100, 100))
title!("Using SVD")

#rsampcov = (rsamps .- mean(rsamps, dims=1))' * (rsamps .- mean(rsamps, dims=1)) ./ (size(rsamps)[1] -1)
funcinv = invert_covmat(((truesamps .- sampmean)' * (truesamps .- sampmean) ./ (size(truesamps)[1] - 1)), 1e-5)

svdinv = svd(invert_covmat(sampcov, EPS))
svdinv.U * Diagonal(svdinv.S) * svdinv.Vt
covsqinv = svdinv.U*Diagonal(sqrt.(svdinv.S))*svdinv.Vt
#covsqinv = #cholesky(invert_covmat(sampcov, 1e-5)).U
whitened_ms = (covsqinv * (truesamps .- sampmean)')'
p = scatter(eps_samp[1:1000, 3], eps_samp[1:1000, 4], label="Actual White Noise samples", xlims=(-5, 5), ylims=(-5, 5), c="orange")
scatter!(whitened_ms[1:1000, 3], whitened_ms[1:1000, 4], label="Whitened Samples from the True Coeff Distribution", xlims=(-5, 5), ylims=(-5, 5), c="blue")
title!("Using SVD, old")

p = scatter(whitened_ms[1:1000, 3], whitened_ms[1:1000, 4], label="Whitened Samples from the True Coeff Distribution", xlims=(-5, 5), ylims=(-5, 5))
p=scatter(eps_samp[1:1000, 3], eps_samp[1:1000, 4], label="Actual White Noise samples", xlims=(-5, 5), ylims=(-5, 5))
title!("Using SVD")
maximum(abs.((covsqinv * covsq) - I))

del_eps = (whitened_ms[1:1000, 1:end] - eps_samp[1:1000, 1:end])'
del_d = covsq * del_eps
