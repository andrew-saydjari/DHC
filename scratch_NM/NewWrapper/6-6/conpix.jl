using Statistics
using Plots
using FFTW
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
using Weave
using SparseArrays
using Distances

push!(LOAD_PATH, pwd()*"/../../../main")
using DHC_2DUtils
push!(LOAD_PATH, pwd()*"/../../../scratch_NM")
using Deriv_Utils_New
using Data_Utils
using Visualization
using ReconFuncs
include("../../../main/compute.jl")





#Functions
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

function compareS1(true_img, init_img, recon_img, filter_hash, dhc_args, tlist)
    JS1ind = J_S1indices([0, 1, 2, 3], filter_hash)
    strue = DHC_compute_wrapper(true_img, filter_hash; dhc_args...)
    s1true = strue[JS1ind]
    clim = (minimum(s1true), maximum(s1true))
    s1gaussian = DHC_compute_wrapper(gprecon, filter_hash; dhc_args...)[JS1ind]
    s1recon = DHC_compute_wrapper(recon_img, filter_hash; dhc_args...)[JS1ind]
    p1 = heatmap(s1true, title=tlist[1], clim=clim)
    p2 = heatmap(s1gaussian, title=tlist[2], clim=clim)
    p3 = heatmap(s1recon, title=tlist[3], clim=clim)
    p = plot(p1, p2, p3)
end


function plot_panel(true_img, gpinit, recon_img)
    clim = (minimum(true_img), maximum(true_img))
    kbins= convert(Array{Float64}, collect(1:32))
    JS1ind = J_S1indices([0, 1, 2, 3], filter_hash)
    true_ps = Data_Utils.calc_1dps(apodizer(true_img), kbins)
    gpps = Data_Utils.calc_1dps(apodizer(gpinit), kbins)
    recps = Data_Utils.calc_1dps(apodizer(recon_img), kbins)
    p1 = heatmap(recon_img, title="Recon", clim=clim)
    p3 = plot(log.(kbins), log.(true_ps), label="True")
    plot!(log.(kbins), log.(recps), label="Recon")
    plot!(log.(kbins), log.(gpps), label="Init")
    plot!(title="P(k)")
    xlabel!("lnk")
    ylabel!("lnP(k)")
    p2 = heatmap(true_img, title="True Img", clim=clim)
    p4= heatmap(gpinit, title="GPInit Img", clim=clim)

    residual = recon_img- true_img
    rlims = (minimum(residual), maximum(residual))
    p5 = heatmap(residual, title="Residual: Recon - True", clims=rlims, c=:bwr)
    p6 = heatmap(gpinit- true_img, title="Residual: GPInit - True", clims=rlims, c=:bwr)

    struesel = Data_Utils.fnlog(DHC_compute_wrapper(true_img, filter_hash, norm=false; dhc_args...))
    ssmoothsel = Data_Utils.fnlog(DHC_compute_wrapper(gpinit, filter_hash, norm=false; dhc_args...))
    sreconsel = Data_Utils.fnlog(DHC_compute_wrapper(recon_img, filter_hash, norm=false; dhc_args...))
    slims = (minimum(struesel[JS1ind]), maximum(struesel[JS1ind]))
    cg = cgrad([:blue, :white, :red])
    truephi, trueomg = round(struesel[2+filter_hash["phi_index"]], sigdigits=3), round(struesel[2+filter_hash["Omega_index"]], sigdigits=3)
    reconphi, reconomg = round(sreconsel[2+filter_hash["phi_index"]], sigdigits=3), round(sreconsel[2+filter_hash["Omega_index"]], sigdigits=3)
    smoothphi, smoothomg = round(ssmoothsel[2+filter_hash["phi_index"]], sigdigits=3), round(ssmoothsel[2+filter_hash["Omega_index"]], sigdigits=3)

    p7 = heatmap(struesel[JS1ind], title="True Coeffs ϕ=" * string(truephi) * "Ω=" * string(trueomg) , clims=slims, c=cg)
    p8 = heatmap(sreconsel[JS1ind], title="Recon Coeffs ϕ=" * string(reconphi) * "Ω=" * string(reconomg), clims=slims, c=cg)
    p9 = heatmap(ssmoothsel[JS1ind], title="GP Init ϕ=" * string(smoothphi) * "Ω=" * string(smoothomg), clims=slims, c=cg)
    p10 = heatmap(zeros(Nx, Nx))
    p = plot(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, layout = (5, 2), size=(1800, 2400))
end

##Common
loaddf = load("../../../scratch_NM/StandardizedExp/Nx64/Data_1000.jld2")
trueimg = loaddf["true_img"]
Nx=64
pixall = sample(1:Nx^2, Integer(round(0.1*4096)), replace = false)
flatmask = falses(Nx*Nx)
flatmask[pixall] .= true
pixmask = reshape(flatmask, (Nx, Nx))

filter_hash = fink_filter_hash(1, 8, nx=Nx, t=1, wd=1, Omega=true)
dhc_args = Dict(:doS2=>false, :doS20=>true, :apodize=>false, :iso=>false)
strue = DHC_compute_wrapper(trueimg, filter_hash; dhc_args...)

#targets
Nf = length(filter_hash["filt_index"])
coeffmask = falses(2+Nf+Nf^2)
coeffmask[3+Nf:end] .= Diagonal(trues(Nf, Nf))[:]
starget = log.(strue[coeffmask])
scovinv = I
img_guess = fill(mean(trueimg), (Nx, Nx))
img_guess[pixmask] .= trueimg[pixmask]

#GP starting point
gr1d = collect(1:Nx)
xmat = reshape(repeat(gr1d, outer=[Nx]), (Nx, Nx))
ymat = xmat'
xfl = xmat[:]
yfl = ymat[:]
posmat = hcat(xfl, yfl)'
pwise_full = pairwise(Distances.Euclidean(), posmat, dims=2)
function SqExpBasisKernel(d; length=1.0, scale=1.0)
    return scale.^2 * exp(-d.^2/(2* (length.^2)))
end

covmat_full = map((x->SqExpBasisKernel(x; length=5.0, scale=0.5)), pwise_full) #L, scale, sort of optimized
cov_ss = covmat_full[flatmask, flatmask] + 0.01*I
testmask = (x->!x).(flatmask)
cov_sv = covmat_full[flatmask, testmask]
cov_vs = covmat_full[testmask, flatmask]

pred_gaussian = mean(trueimg) .+ cov_vs * inv(cov_ss) * (reshape(trueimg[flatmask], (length(trueimg[flatmask]), 1)) .- mean(trueimg))
gprecon = zeros(Nx^2)
gprecon[flatmask] .= trueimg[flatmask]
gprecon[testmask] .= pred_gaussian[:]
gprecon = reshape(gprecon, (Nx, Nx))
heatmap(gprecon)
using Feather
my_df = Dict([("field", trueimg), ("source_mask", pixmask), ("gprecon_julia", gprecon)])
using HDF5
h5open("fieldsrcs.h5", "w") do file
    write(file, "field", trueimg)
    #write(file, "source_mask", pixmask)
    write(file, "gprecon_julia", gprecon)
end
save(my_df, ".hdf5")

#
lval2 = 0.0
lval3 = 0.0
optim_settings = Dict([("iterations", 1000), ("norm", false), ("minmethod", ConjugateGradient())])
func_specific_params = Dict([(:reg_input=> img_guess), (:coeff_mask1=> coeffmask), (:target1=>starget), (:invcov1=>scovinv), (:coeff_mask2=> coeffmask), (:target2=>starget), (:invcov2=>scovinv), (:lambda2=>lval2), (:lambda3=>lval3), (:pixmask=>pixmask), (:func=>Data_Utils.fnlog), (:dfunc=>Data_Utils.fndlog)])


res, recon_img = image_recon_derivsum_custom(img_guess, filter_hash, dhc_args, ReconFuncs.Loss3Gaussian_transformed, ReconFuncs.dLoss3Gaussian_transformed_cp!; optim_settings=optim_settings, func_specific_params)
#coeff_mask1=nothing, target1=nothing, invcov1=nothing, reg_input=nothing, coeff_mask2=nothing, target2=nothing, invcov2=nothing, func=nothing, dfunc=nothing, lambda2=nothing, lambda3=nothing)
compareS1(loaddf["true_img"], img_guess, recon_img, filter_hash, dhc_args, ["True", "InitPointswMean", "Recon"])
heatmap(recon_img)



#Start with gprecon, no regularization
lval2 = 0.0
lval3 = 0.0
optim_settings = Dict([("iterations", 1000), ("norm", false), ("minmethod", ConjugateGradient())])
func_specific_params = Dict([(:reg_input=> img_guess), (:coeff_mask1=> coeffmask), (:target1=>starget), (:invcov1=>scovinv), (:coeff_mask2=> coeffmask), (:target2=>starget), (:invcov2=>scovinv), (:lambda2=>lval2), (:lambda3=>lval3), (:pixmask=>pixmask), (:func=>Data_Utils.fnlog), (:dfunc=>Data_Utils.fndlog)])

res, recon_img = image_recon_derivsum_custom(gprecon, filter_hash, dhc_args, ReconFuncs.Loss3Gaussian_transformed, ReconFuncs.dLoss3Gaussian_transformed_cp!; optim_settings=optim_settings, func_specific_params)
heatmap(recon_img)
compareS1(loaddf["true_img"], gprecon, recon_img, filter_hash, dhc_args, ["True", "GPRecon", "S2R-Recon"])
println("GP Recon Mean Abs Frac", mean(abs.((gprecon .- trueimg)./trueimg)))
println("S2r Recon Mean Abs Frac", mean(abs.((recon_img .- trueimg)./trueimg)))
println("GP Recon MSE", mean((gprecon .- trueimg).^2))
println("S2r Recon MSE", mean((recon_img .- trueimg).^2))
plot_panel(trueimg, gprecon, recon_img)
#decent-ish but not better than without

#Starting with the GP recon, regularizing with respect to the mean everywhere and the point values at those places: BAD
#Gets the coefficients right? Worse than without reg
lval2 = 0.0
lval3 = 1.0
optim_settings = Dict([("iterations", 1000), ("norm", false), ("minmethod", ConjugateGradient())])
func_specific_params = Dict([(:reg_input=> img_guess), (:coeff_mask1=> coeffmask), (:target1=>starget), (:invcov1=>scovinv), (:coeff_mask2=> coeffmask), (:target2=>starget), (:invcov2=>scovinv), (:lambda2=>lval2), (:lambda3=>lval3), (:pixmask=>pixmask), (:func=>Data_Utils.fnlog), (:dfunc=>Data_Utils.fndlog)])

res, recon_img = image_recon_derivsum_custom(gprecon, filter_hash, dhc_args, ReconFuncs.Loss3Gaussian_transformed, ReconFuncs.dLoss3Gaussian_transformed_cp!; optim_settings=optim_settings, func_specific_params)
heatmap(recon_img)
println("GP Recon Mean Abs Frac ", mean(abs.((gprecon .- trueimg)./trueimg)))
println("S2r Recon Mean Abs Frac ", mean(abs.((recon_img .- trueimg)./trueimg)))
println("GP Recon MSE ", mean((gprecon .- trueimg).^2))
println("S2r Recon MSE ", mean((recon_img .- trueimg).^2))

#Starting with the GP recon, regularizing with respect to GP recon at those places
lval2 = 0.0
lval3 = 1.0
optim_settings = Dict([("iterations", 1000), ("norm", false), ("minmethod", ConjugateGradient())])
func_specific_params = Dict([(:reg_input=> gprecon), (:coeff_mask1=> coeffmask), (:target1=>starget), (:invcov1=>scovinv), (:coeff_mask2=> coeffmask), (:target2=>starget), (:invcov2=>scovinv), (:lambda2=>lval2), (:lambda3=>lval3), (:pixmask=>pixmask), (:func=>Data_Utils.fnlog), (:dfunc=>Data_Utils.fndlog)])

res, recon_img = image_recon_derivsum_custom(gprecon, filter_hash, dhc_args, ReconFuncs.Loss3Gaussian_transformed, ReconFuncs.dLoss3Gaussian_transformed_cp!; optim_settings=optim_settings, func_specific_params)
println("GP Recon Mean Abs Frac ", mean(abs.((gprecon .- trueimg)./trueimg)))
println("S2r Recon Mean Abs Frac ", mean(abs.((recon_img .- trueimg)./trueimg)))
println("GP Recon MSE ", mean((gprecon .- trueimg).^2))
println("S2r Recon MSE ", mean((recon_img .- trueimg).^2))
plot_panel(trueimg, gprecon, recon_img)

#All S2R
#Starting with the GP recon, regularizing with respect to GP recon at those places
coeffmask = falses(2+Nf+Nf^2)
coeffmask[3+Nf:end] .= triu(trues(Nf, Nf))[:]
starget = log.(strue[coeffmask])
lval2 = 0.0
lval3 = 1.0
optim_settings = Dict([("iterations", 1000), ("norm", false), ("minmethod", ConjugateGradient())])
func_specific_params = Dict([(:reg_input=> gprecon), (:coeff_mask1=> coeffmask), (:target1=>starget), (:invcov1=>scovinv), (:coeff_mask2=> coeffmask), (:target2=>starget), (:invcov2=>scovinv), (:lambda2=>lval2), (:lambda3=>lval3), (:pixmask=>pixmask), (:func=>Data_Utils.fnlog), (:dfunc=>Data_Utils.fndlog)])

res, recon_img = image_recon_derivsum_custom(gprecon, filter_hash, dhc_args, ReconFuncs.Loss3Gaussian_transformed, ReconFuncs.dLoss3Gaussian_transformed_cp!; optim_settings=optim_settings, func_specific_params)
println("GP Recon Mean Abs Frac ", mean(abs.((gprecon .- trueimg)./trueimg)))
println("S2r Recon Mean Abs Frac ", mean(abs.((recon_img .- trueimg)./trueimg)))
println("GP Recon MSE ", mean((gprecon .- trueimg).^2))
println("S2r Recon MSE ", mean((recon_img .- trueimg).^2))
p = plot_panel(trueimg, gprecon, recon_img)
savefig(p, "../../../scratch_NM/Conpix/6-10/Case2_alls2r.png")

#Pick the wrong coefficient target: Can I take a random dust image's vector and make it look like that?
anotherdict = load("../../../scratch_NM/StandardizedExp/Nx64/Data_10.jld2")
heatmap(anotherdict["true_img"])
another_s = DHC_compute_wrapper(anotherdict["true_img"], filter_hash; dhc_args...)

coeffmask = falses(2+Nf+Nf^2)
coeffmask[3+Nf:end] .= triu(trues(Nf, Nf))[:]
starget = log.(another_s[coeffmask])
lval2 = 0.0
lval3 = 1.0
optim_settings = Dict([("iterations", 1000), ("norm", false), ("minmethod", ConjugateGradient())])
func_specific_params = Dict([(:reg_input=> gprecon), (:coeff_mask1=> coeffmask), (:target1=>starget), (:invcov1=>scovinv), (:coeff_mask2=> coeffmask), (:target2=>starget), (:invcov2=>scovinv), (:lambda2=>lval2), (:lambda3=>lval3), (:pixmask=>pixmask), (:func=>Data_Utils.fnlog), (:dfunc=>Data_Utils.fndlog)])

res, recon_img = image_recon_derivsum_custom(gprecon, filter_hash, dhc_args, ReconFuncs.Loss3Gaussian_transformed, ReconFuncs.dLoss3Gaussian_transformed_cp!; optim_settings=optim_settings, func_specific_params)
println("GP Recon Mean Abs Frac ", mean(abs.((gprecon .- trueimg)./trueimg)))
println("S2r Recon Mean Abs Frac ", mean(abs.((recon_img .- trueimg)./trueimg)))
println("GP Recon MSE ", mean((gprecon .- trueimg).^2))
println("S2r Recon MSE ", mean((recon_img .- trueimg).^2))
p = plot_panel(trueimg, gprecon, recon_img)


#All S2R, Norm
#Starting with the GP recon, regularizing with respect to GP recon at those places
dhc_args = Dict(:doS2=>false, :doS20=>true, :apodize=>false, :iso=>false, :norm=>true)

coeffmask = falses(2+Nf+Nf^2)
coeffmask[3+Nf:end] .= triu(trues(Nf, Nf))[:]
starget = log.(strue[coeffmask])
lval2 = 0.0
lval3 = 1.0
optim_settings = Dict([("iterations", 1000), ("norm", false), ("minmethod", ConjugateGradient())])

func_specific_params = Dict([(:reg_input=> gprecon), (:coeff_mask1=> coeffmask), (:target1=>starget), (:invcov1=>scovinv), (:coeff_mask2=> coeffmask), (:target2=>starget), (:invcov2=>scovinv), (:lambda2=>lval2), (:lambda3=>lval3),
    (:pixmask=>pixmask), (:func=>Data_Utils.fnlog), (:dfunc=>Data_Utils.fndlog)])

res, recon_img = image_recon_derivsum_custom(gprecon, filter_hash, dhc_args, ReconFuncs.Loss3Gaussian_transformed, ReconFuncs.dLoss3Gaussian_transformed_cp!; optim_settings=optim_settings, func_specific_params)
println("GP Recon Mean Abs Frac ", mean(abs.((gprecon .- trueimg)./trueimg)))
println("S2r Recon Mean Abs Frac ", mean(abs.((recon_img .- trueimg)./trueimg)))
println("GP Recon MSE ", mean((gprecon .- trueimg).^2))
println("S2r Recon MSE ", mean((recon_img .- trueimg).^2))
p = plot_panel(trueimg, gprecon, recon_img)
#savefig(p, "../../../scratch_NM/Conpix/6-10/Case2_alls2r.png")



#Resave coefficients
coeffdict = load("../../SavedCovMats/reg_apd_noiso_logcoeff.jld2")
using HDF5
sfdall = readsfd_fromsrc("../../data/dust10000.fits", 64, logbool=false)
h5open("../../SavedCovMats/reg_apd_noiso_logcoeffs_img.h5", "w") do file
    write(file, "coeffdbn", coeffdict["dbncoeffs"])
    write(file, "images", sfdall)
end
