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
lval3=0.01

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
kbins= convert(Array{Float64}, collect(1:32))
clim=(minimum(apodizer(true_img)), maximum(apodizer(true_img)))
true_ps = Data_Utils.calc_1dps(apodizer(true_img), kbins)
initps = Data_Utils.calc_1dps(apodizer(init), kbins)
recps = Data_Utils.calc_1dps(apodizer(recon_img), kbins)
p1 = heatmap(apodizer(recon_img), title="Recon", clim=clim)
p3 = plot(log.(kbins), log.(true_ps), label="True")
plot!(log.(kbins), log.(recps), label="Recon")
plot!(log.(kbins), log.(initps), label="Init")
plot!(title="P(k): LogCoeffs 3 Loss")
xlabel!("lnk")
ylabel!("lnP(k)")
p2 = heatmap(apodizer(true_img), title="True Img", clim=clim)
p4= heatmap(apodizer(init), title="Init Img", clim=clim)
p = plot(p1, p2, p3, p4, layout=4, size=(1200, 1200))
Visualization.plot_synth_QA(apodizer(true_img), apodizer(init), apodizer(recon_img), fname="scratch_NM/NewWrapper/4-28/1000_C5_S2sfd_highjinit.png")


struesel = Data_Utils.fnlog(DHC_compute_wrapper(true_img, filter_hash, norm=false; dhc_args...))[JS1ind]
sinitsel = Data_Utils.fnlog(DHC_compute_wrapper(init, filter_hash, norm=false; dhc_args...))[JS1ind]
ssmoothsel = Data_Utils.fnlog(DHC_compute_wrapper(imfilter(init, Kernel.gaussian(0.8)), filter_hash, norm=false; dhc_args...))[JS1ind]
sreconsel = Data_Utils.fnlog(DHC_compute_wrapper(recon_img, filter_hash, norm=false; dhc_args...))[JS1ind]

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
