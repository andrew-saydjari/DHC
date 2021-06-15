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
using Weave
using SparseArrays

push!(LOAD_PATH, pwd()*"/../../../main")
using DHC_2DUtils
push!(LOAD_PATH, pwd()*"/../../../scratch_NM")
using Deriv_Utils_New
using Data_Utils
using Visualization
using ReconFuncs
include("../../../main/compute.jl")

# FUNCTIONS
function realspace_filter(Nx, f_i, f_v)

    zarr = zeros(ComplexF64, Nx, Nx)
    for i = 1:length(f_i)
        zarr[f_i[i]] = f_v[i] # filter*image in Fourier domain
    end
    filt = ifft(zarr)  # real space, complex
    return filt
end

function DHC_compute_S20r_noisy_so(image::Array{Float64,2}, filter_hash::Dict, sigim::Array{Float64,2};
    doS2::Bool=true, doS20::Bool=false, apodize=false, norm=false, iso=false, FFTthreads=1, filter_hash2::Dict=filter_hash, coeff_mask=nothing)
    #Not using coeff_mask here after all. ASSUMES ANY ONE of doS12, doS2 and doS20 are true, when using coeff_mask
    #@assert !iso "Iso not implemented yet"
    Nf = size(filter_hash["filt_index"])[1]
    Nx = size(image)[1]

    if apodize
        ap_img = apodizer(image)
        dA = get_dApodizer(Nx, Dict([(:apodize => apodize)]))
    else
        ap_img = image
        dA = get_dApodizer(Nx, Dict([(:apodize => apodize)]))
    end

    function DHC_compute_biased(image::Array{Float64,2}, filter_hash::Dict, filter_hash2::Dict, sigim::Array{Float64,2};
        doS2::Bool=true, doS20::Bool=false, norm=true, iso=false, FFTthreads=2, normS1::Bool=false, normS1iso::Bool=false)
        # image        - input for WST
        # filter_hash  - filter hash from fink_filter_hash
        # filter_hash2 - filters for second order.  Default to same as first order.
        # doS2         - compute S2 coeffs
        # doS20        - compute S2 coeffs
        # norm         - scale to mean zero, unit variance
        # iso          - sum over angles to obtain isotropic coeffs

        # Use 2 threads for FFT
        FFTW.set_num_threads(FFTthreads)

        (Nx, Ny)  = size(image)
        if Nx != Ny error("Input image must be square") end
        (Nf, )    = size(filter_hash["filt_index"])
        if Nf == 0  error("filter hash corrupted") end
        @assert Nx==filter_hash["npix"] "Filter size should match npix"
        @assert Nx==filter_hash2["npix"] "Filter2 size should match npix"
        @assert (normS1 && normS1iso) != 1 "normS1 and normS1iso are mutually exclusive"

        # allocate coeff arrays
        out_coeff = []
        S0  = zeros(Float64, 2)
        S1  = zeros(Float64, Nf)
        if doS2  S2  = zeros(Float64, Nf, Nf) end  # traditional 2nd order
        if doS20 S20 = zeros(Float64, Nf, Nf) end  # real space correlation
        anyM2 = doS2 | doS20
        anyrd = doS2 | doS20             # compute real domain with iFFT

        # allocate image arrays for internal use
        if anyrd
            im_rd_0_1  = Array{Float64, 3}(undef, Nx, Ny, Nf)
            ψsqfac = Array{Float64, 4}(undef, Nf, Nx, Ny, 3)
            Iψfac = Array{ComplexF64, 4}(undef, Nf, Nx, Ny, 2)
            ψfaccross = Array{Float64, 2}(undef, Nf, Nf)
            sozoterms = Array{Float64, 4}(undef, Nf, Nx, Ny, 2)
            rsψmat = Array{ComplexF64, 3}(undef, Nf, Nx, Ny)
        end
        varim  = sigim.^2
        Pf = plan_fft(varim)
        fvar = Pf*(varim)

        ## 0th Order
        S0[1]   = mean(image)
        norm_im = image.-S0[1]
        S0[2]   = sum(norm_im .* norm_im)/(Nx*Ny)
        if norm
            norm_im ./= sqrt(Nx*Ny*S0[2])
        else
            norm_im = copy(image)
        end

        append!(out_coeff,S0[:])

        ## 1st Order
        im_fd_0 = Pf*(norm_im)  # total power=1.0

        # unpack filter_hash
        f_ind   = filter_hash["filt_index"]  # (J, L) array of filters represented as index value pairs
        f_val   = filter_hash["filt_value"]

        zarr = zeros(ComplexF64, Nx, Ny)  # temporary array to fill with zvals

        # make a FFTW "plan" for an array of the given size and type
        if anyrd
            P = plan_ifft(im_fd_0) end  # P is an operator, P*im is ifft(im)

        ## Main 1st Order and Precompute 2nd Order
        for f = 1:Nf
            S1tot = 0.0
            f_i = f_ind[f]  # CartesianIndex list for filter
            f_v = f_val[f]  # Values for f_i
            # for (ind, val) in zip(f_i, f_v)   # this is slower!
            if length(f_i) > 0
                for i = 1:length(f_i)
                    ind       = f_i[i]
                    zval      = f_v[i] * im_fd_0[ind]
                    S1tot    += abs2(zval)
                    zarr[ind] = zval        # filter*image in Fourier domain
                end
                S1[f] = S1tot/(Nx*Ny)  # image power
                if anyrd
                    #psi_pow = sum(f_v.^2)./(Nx*Ny)
                    im_rd_0_1[:,:,f] .= abs2.(P*zarr)
                    rsψmat[f, :, :] = realspace_filter(Nx, f_i, f_v)
                    frealψ = Pf*(real.(rsψmat[f, :, :]))
                    fimψ = Pf*(imag.(rsψmat[f, :, :]))
                    ψsqfac[f, :, :, 1] = real.(rsψmat[f, :, :]) #check that real
                    ψsqfac[f, :, :, 2] = imag.(rsψmat[f, :, :])
                    #ψsqfac[f, :, :, 3] = P*(fvar .* powrs)
                    #ψsqfac[f, :, :, 4] = P*(fvar .* Pf*(imag.(rsψ).*real.(rsψ)))

                    Iψfac[f, :, :, 1] = P*(im_fd_0 .* frealψ) #(I ✪ ψR)
                    Iψfac[f, :, :, 2] = P*(im_fd_0 .* fimψ)   #(I ✪ ψI)
                    #Iψfac[f, :, :, 3] = P*(fvar .* fsqrsψ)
                    #Iψfac[f, :, :, 4] = P*(fvar .* fsqimψ)
                end

                zarr[f_i] .= 0
            end
        end

        append!(out_coeff, iso ? filter_hash["S1_iso_mat"]*S1 : S1)

        if normS1iso
            S1iso = vec(reshape(filter_hash["S1_iso_mat"]*S1,(1,:))*filter_hash["S1_iso_mat"])
        end

        # we stored the abs()^2, so take sqrt (this is faster to do all at once)
        if anyrd
            im_rd_0_1 .= sqrt.(im_rd_0_1)
            for f1=1:Nf
                #Precompute SO_λ1.ZO_λ2 terms
                fsqrsψ = Pf*(real.(rsψmat[f1, :, :]).^2) #F(ψR^2)
                fsqimψ = Pf*(imag.(rsψmat[f1, :, :]).^2) #F(ψI^2)
                powrs = fsqrsψ + fsqimψ #F(|ψ|^2)
                sozoterms[f1, :, :, 1] = (P*(fvar .* powrs)) ./ im_rd_0_1[:, :, f1]                   #(σ2 ✪ |ψ|^2)/|I ✪ ψ|
                sozoterms[f1, :, :, 2] = real.((Iψfac[f1, :, :, 1].^2) .* (P*(fvar .* fsqrsψ)))     #(I ✪ ψR)^2 . (σ2 ✪ ψR^2)
                sozoterms[f1, :, :, 2] += real.((Iψfac[f1, :, :, 2].^2) .* (P*(fvar .* fsqimψ)))    #(I ✪ ψI)^2 . (σ2 ✪ ψI^2)
                sozoterms[f1, :, :, 2] += real.(2*Iψfac[f1, :, :, 1].*Iψfac[f1, :, :, 2].* (P*(fvar .* (Pf*(imag.(rsψmat[f1, :, :]).*real.(rsψmat[f1, :, :]))))))  #2(I ✪ ψR)(I ✪ ψI) . (σ2 ✪ ψRψI)
                sozoterms[f1, :, :, 2] = sozoterms[f1, :, :, 2]./(im_rd_0_1[:, :, f1].^3)
                sozoterms[f1, :, :, 1] -= sozoterms[f1, :, :, 2]
            end
            for f1=1:Nf
                for f2=1:Nf
                    #println("f2", f2)
                    val1 = Pf*(ψsqfac[f1, :, :, 1] .* ψsqfac[f2, :, :, 1]) #F{ψ_λ1R.ψ_λ2R}
                    #println(size(val1), size(fvar))
                    term1 = (P*(fvar .* val1) .* Iψfac[f1, :, :, 1]) .* Iψfac[f2, :, :, 1]
                    term2 = (P*(fvar .* (Pf*(ψsqfac[f1, :, :, 1] .* ψsqfac[f2, :, :, 2]))).* Iψfac[f1, :, :, 1]) .* Iψfac[f2, :, :, 2]
                    term3 = (P*(fvar .* (Pf*(ψsqfac[f1, :, :, 2] .* ψsqfac[f2, :, :, 1]))).* Iψfac[f1, :, :, 2]) .* Iψfac[f2, :, :, 1]
                    term4 = (P*(fvar .* (Pf*(ψsqfac[f1, :, :, 2] .* ψsqfac[f2, :, :, 2]))).* Iψfac[f1, :, :, 2]) .* Iψfac[f2, :, :, 2]
                    so1zo2 = 0.5 .* im_rd_0_1[:, :, f2] .* sozoterms[f1, :, :, 1] #T0_λ2 . T2_λ1
                    so2zo1 = 0.5 .* im_rd_0_1[:, :, f1] .* sozoterms[f2, :, :, 1] #T0_λ1 . T2_λ2
                    #so(λ1) * zo(λ2)
                    #so1_1 = ψsqfac[f1, :, :, 3] ./ im_rd_0_1[:, :, f1]
                    #so1_2 = (Iψfac[f1, :, :, 1].^2) .* Iψfac[f1, :, :, 3]
                    #so1_2 += (Iψfac[f1, :, :, 2].^2) .* Iψfac[f1, :, :, 4]
                    #so1_2 += 2*Iψfac[f1, :, :, 1].*Iψfac[f1, :, :, 2].*ψsqfac[f1, :, :, 4]
                    #so1_2 = so1_2./im_rd_0_1[:, :, f1].^3

                    combined = (term1 + term2 + term3 + term4)./(im_rd_0_1[:, :, f1] .* im_rd_0_1[:, :, f2]) #fo1fo2
                    combined += (so2zo1 + so1zo2)
                    comsum = sum(combined)
                    println(imag(comsum))
                    ψfaccross[f1, f2] = real(comsum)
                end
            end
        end

        Mat2 = filter_hash["S2_iso_mat"]
        if doS2
            f_ind2   = filter_hash2["filt_index"]  # (J, L) array of filters represented as index value pairs
            f_val2   = filter_hash2["filt_value"]

            ## Traditional second order
            for f1 = 1:Nf
                thisim = fft(im_rd_0_1[:,:,f1])  # Could try rfft here
                # Loop over f2 and do second-order convolution
                if normS1
                    normS1pwr = S1[f1]
                elseif normS1iso
                    normS1pwr = S1iso[f1]
                else
                    normS1pwr = 1
                end

                for f2 = 1:Nf
                    f_i = f_ind2[f2]  # CartesianIndex list for filter
                    f_v = f_val2[f2]  # Values for f_i
                    # sum im^2 = sum(|fft|^2/npix)
                    #intfterm = fft((real.(rsψ1) + imag.(rsψ1)).*(real.(rsψ2) + imag.(rsψ2)))
                    S2[f1,f2] = sum(abs2.(f_v .* thisim[f_i]))/(Nx*Ny)/normS1pwr
                end
            end
            append!(out_coeff, iso ? Mat2*S2[:] : S2[:])
        end

        # Real domain 2nd order
        if doS20
            Amat = reshape(im_rd_0_1, Nx*Ny, Nf)
            S20  = Amat' * Amat
            #println(size(S20))
            S20 .+= ψfaccross #assuming the nx*ny factor above was for the parseval correction
            append!(out_coeff, iso ? Mat2*S20[:] : S20[:])
        end


        return out_coeff
    end

    sall = DHC_compute_biased(ap_img, filter_hash, filter_hash2, sigim, doS2=doS2, doS20=doS20, norm=norm, iso=iso, FFTthreads=FFTthreads)
    dS20dp = wst_S20_deriv(ap_img, filter_hash)
    ap_dS20dp = dA * reshape(dS20dp, Nx*Nx, Nf*Nf)
    G =  ap_dS20dp .* reshape(sigim, Nx*Nx, 1)
    cov = G'*G

    if coeff_mask!=nothing
        @assert count(!iszero, coeff_mask[1:Nf+2])==0 "This function only handles S20r"
        @assert length(coeff_mask)==length(sall) "Mask must have the same length as the total number of coefficients"
        s20rmask = coeff_mask[Nf+3:end]
        return sall[coeff_mask], cov[s20rmask, s20rmask]
    else
        return sall, cov
    end

end

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


##0) What smoothing kernel gives the best MSE, Abs Frac Res and PowSpec?
ARGS_buffer = ["reg", "nonapd", "noiso", "../../../scratch_NM/StandardizedExp/Nx64/", "full_3losstest", "Full+Eps"]
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

kbins=convert(Array{Float64, 1}, collect(1:32))
for l in [0.8, 0.9, 1.0, 1.1, 1.2]
    apdsmoothed = imfilter(init, Kernel.gaussian(l))
    true_ps = calc_1dps(true_img, kbins)
    smoothps = calc_1dps(apdsmoothed, kbins)
    fracres = (apdsmoothed .- true_img)./true_img
    fps = (smoothps .- true_ps)./true_ps
    println("Smoothing scale = ", l)
    println("Mean Abs Frac, Smoothed = ", round(mean(abs.((apdsmoothed .- true_img)./true_img)), digits=3))
    println("MSE = ", round(mean((apdsmoothed .- true_img).^2), digits=5))
    println("Power Spec Frac Res, Smoothed = ", round(mean(abs.(smoothps .- true_ps)), digits=3))
end

##3) Can true+emp noisy do better than smoothing with pixwise reg?
##3a) Only S1
ARGS_buffer = ["reg", "nonapd", "noiso", "../../../scratch_NM/StandardizedExp/Nx64/", "full_3losstest", "Full+Eps"]
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

optim_settings = Dict([("iterations", 1000), ("norm", false), ("minmethod", ConjugateGradient())])
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
lval3 = 100.0

println("Regularizer Lambda=", round(lval3, sigdigits=3))

img_guess = init
recon_img = img_guess

img_list = []
push!(img_list, img_guess)
num_rounds=5

#if YOU HAD THE TRUE IMAGE, calculating the shift empirically
Nr=10000
sigma = loaddf["std"]
empsn = []
for n=1:Nr
    noisyim = randn((Nx, Nx)).*sigma .+ true_img
    push!(empsn, DHC_compute_wrapper(noisyim, filter_hash; dhc_args...))
end
coeffsim = hcat(empsn...)'
ncoeffmean = mean(coeffsim, dims=1)
scov = (coeffsim .- ncoeffmean)' * (coeffsim .- ncoeffmean)./(Nr-1)
scov = scov[coeffmask, coeffmask]
snmean = ncoeffmean[:][coeffmask]
strue = DHC_compute_wrapper(loaddf["true_img"], filter_hash, norm=false; dhc_args...)[coeffmask]
shift_true = snmean .- strue
s_noisy = DHC_compute_wrapper(loaddf["init"], filter_hash, norm=false; dhc_args...)[coeffmask]
starget = s_noisy - shift_true #DHC_compute_wrapper(loaddf["true_img"], filter_hash, norm=false; dhc_args...)[coeffmask] #
scovinv = invert_covmat(scov, 1e-10)

ratio = starget./strue
println("Shift_true omega = ", shift_true[34])
println("Ratio = ", ratio[34])
println("Noisy init = ", s_noisy[34], " mean = ", snmean[34], "True = ", strue[34])
println("Omega stddev raw", sqrt(scov[34, 34]), " Post reg Omega stddev = ", sqrt(inv(scovinv[34, 34])))
println("Starget = ", starget[34])
zsc = (s_noisy[34] - snmean[34])/sqrt(inv(scovinv[34, 34]))

func_specific_params = Dict([(:reg_input=> init), (:coeff_mask1=> coeffmask), (:target1=>starget), (:invcov1=>scovinv), (:coeff_mask2=> coeffmask), (:target2=>starget), (:invcov2=>scovinv), (:lambda2=>lval2), (:lambda3=>lval3)])

recon_settings["fname_save"] = fname_save * ".jld2"
recon_settings["optim_settings"] = optim_settings

if logbool
    error("Not implemented here")
end

res, recon_img = image_recon_derivsum_custom(img_guess, filter_hash, dhc_args, ReconFuncs.Loss3Gaussian, ReconFuncs.dLoss3Gaussian!; optim_settings=optim_settings, func_specific_params)
heatmap(init)
heatmap(recon_img, title="Using true shift")
heatmap(true_img)


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
#savefig(p, "../../../scratch_NM/NewWrapper/6-6/denoising_w_strueempirical_lam100.png")
fracres = (init .- true_img)./true_img
fps = (initps .- true_ps)./true_ps
println("Mean Abs Frac, Init = ", round(mean(abs.(fracres)), digits=3), "Smoothed = ", round(mean(abs.((apdsmoothed .- true_img)./true_img)), digits=3), "Recon = ", round(mean(abs.((recon_img .- true_img)./true_img)), digits=3))
println("MSE, Init = ", round(mean((init .- true_img).^2), digits=6), "Smoothed = ", round(mean((apdsmoothed .- true_img).^2), digits=6), "Recon = ", round(mean((recon_img .- true_img).^2), digits=6))
println("Power Spectrum Frac Res, Init = ", round(mean(abs.(fps)), digits=3), "Smoothed = ", round(mean(abs.((smoothps .- true_ps)./true_ps)), digits=3), "Recon = ", round(mean(abs.(recps .- true_ps)./true_ps), digits=3))


##3b) S2R
ARGS_buffer = ["reg", "nonapd", "noiso", "../../../scratch_NM/StandardizedExp/Nx64/", "full_3losstest", "Full+Eps"]
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

optim_settings = Dict([("iterations", 1000), ("norm", false), ("minmethod", ConjugateGradient())])
recon_settings = Dict([("log", logbool), ("Invcov_matrix", ARGS_buffer[6]), ("optim_settings", optim_settings), ("eps_value_sfd", 1e-5), ("eps_value_init", 1e-10)]) #Add constraints

recon_settings["datafile"] = datfile

if dhc_args[:iso]
    error("Not constructed for iso")
else #Not iso
    coeffmask = falses(2+Nf+Nf^2)
    coeffmask[Nf+3:end] .= triu(trues(Nf, Nf))[:]
end


#Weight given to terms
lval2 = 0.0
lval3 = 1.0

println("Regularizer Lambda=", round(lval3, sigdigits=3))

img_guess = init


img_list = []
push!(img_list, img_guess)
num_rounds=5

#if YOU HAD THE TRUE IMAGE, calculating the shift empirically
Nr=10000
sigma = loaddf["std"]
empsn = []
for n=1:Nr
    noisyim = randn((Nx, Nx)).*sigma .+ true_img
    push!(empsn, DHC_compute_wrapper(noisyim, filter_hash; dhc_args...))
end
coeffsim = hcat(empsn...)'
ncoeffmean = mean(coeffsim, dims=1)
scov = (coeffsim .- ncoeffmean)' * (coeffsim .- ncoeffmean)./(Nr-1)
scov = scov[coeffmask, coeffmask]
snmean = ncoeffmean[:][coeffmask]
strue = DHC_compute_wrapper(loaddf["true_img"], filter_hash, norm=false; dhc_args...)[coeffmask]
shift_true = snmean .- strue
s_noisy = DHC_compute_wrapper(loaddf["init"], filter_hash, norm=false; dhc_args...)[coeffmask]
starget = s_noisy - shift_true #DHC_compute_wrapper(loaddf["true_img"], filter_hash, norm=false; dhc_args...)[coeffmask] #
scovinv = invert_covmat(scov, 1e-10)

ratio = starget./strue
println("Shift_true omega = ", shift_true[34])
println("Ratio = ", ratio[34])
println("Noisy init = ", s_noisy[34], " mean = ", snmean[34], "True = ", strue[34])
println("Omega stddev raw", sqrt(scov[34, 34]), " Post reg Omega stddev = ", sqrt(inv(scovinv[34, 34])))
println("Starget = ", starget[34])
zsc = (s_noisy[34] - snmean[34])/sqrt(inv(scovinv[34, 34]))

func_specific_params = Dict([(:reg_input=> init), (:coeff_mask1=> coeffmask), (:target1=>starget), (:invcov1=>scovinv), (:coeff_mask2=> coeffmask), (:target2=>starget), (:invcov2=>scovinv), (:lambda2=>lval2), (:lambda3=>lval3)])

recon_settings["fname_save"] = fname_save * ".jld2"
recon_settings["optim_settings"] = optim_settings

if logbool
    error("Not implemented here")
end
reshape(coeffmask[3+Nf:end], (Nf, Nf))
res, recon_img = image_recon_derivsum_custom(img_guess, filter_hash, dhc_args, ReconFuncs.Loss3Gaussian, ReconFuncs.dLoss3Gaussian!; optim_settings=optim_settings, func_specific_params)
heatmap(init)
heatmap(recon_img, title="Using true shift")
heatmap(true_img)


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
plot!(title="P(k)")
xlabel!("lnk")
ylabel!("lnP(k)")
p2 = heatmap(true_img, title="True Img", clim=clim)
p4= heatmap(init, title="Init Img", clim=clim)
p5= heatmap(apdsmoothed, title="Smoothed init", clim=clim)
residual = recon_img- true_img
rlims = (minimum(residual), maximum(residual))

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
#savefig(p, "../../../scratch_NM/NewWrapper/6-6/3b_denoising_w_strueempirical_lam100.png")
fracres = (init .- true_img)./true_img
fps = (initps .- true_ps)./true_ps
println("Mean Abs Frac, Init = ", round(mean(abs.(fracres)), digits=3), "Smoothed = ", round(mean(abs.((apdsmoothed .- true_img)./true_img)), digits=3), "Recon = ", round(mean(abs.((recon_img .- true_img)./true_img)), digits=3))
println("MSE, Init = ", round(mean((init .- true_img).^2), digits=5), "Smoothed = ", round(mean((apdsmoothed .- true_img).^2), digits=5), "Recon = ", round(mean((recon_img .- true_img).^2), digits=5))
println("Power Spec Frac Res, Init = ", round(mean(abs.(fps)), digits=3), "Smoothed = ", round(mean(abs.((smoothps .- true_ps)./true_ps)), digits=3), "Recon = ", round(mean(abs.(recps .- true_ps)./true_ps), digits=3))

#3c) S1 sequentially with S2R using S1 as starting point.
ARGS_buffer = ["reg", "nonapd", "noiso", "../../../scratch_NM/StandardizedExp/Nx64/", "full_3losstest", "Full+Eps"]
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

optim_settings = Dict([("iterations", 1000), ("norm", false), ("minmethod", ConjugateGradient())])
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
lval3 = 100.0

println("Regularizer Lambda=", round(lval3, sigdigits=3))

img_guess = init
recon_img = img_guess

img_list = []
push!(img_list, img_guess)
num_rounds=5

#if YOU HAD THE TRUE IMAGE, calculating the shift empirically
Nr=10000
sigma = loaddf["std"]
empsn = []
for n=1:Nr
    noisyim = randn((Nx, Nx)).*sigma .+ true_img
    push!(empsn, DHC_compute_wrapper(noisyim, filter_hash; dhc_args...))
end
coeffsim = hcat(empsn...)'
ncoeffmean = mean(coeffsim, dims=1)
scov = (coeffsim .- ncoeffmean)' * (coeffsim .- ncoeffmean)./(Nr-1)
scov = scov[coeffmask, coeffmask]
snmean = ncoeffmean[:][coeffmask]
strue = DHC_compute_wrapper(loaddf["true_img"], filter_hash, norm=false; dhc_args...)[coeffmask]
shift_true = snmean .- strue
s_noisy = DHC_compute_wrapper(loaddf["init"], filter_hash, norm=false; dhc_args...)[coeffmask]
starget = s_noisy - shift_true #DHC_compute_wrapper(loaddf["true_img"], filter_hash, norm=false; dhc_args...)[coeffmask] #
scovinv = invert_covmat(scov, 1e-10)

ratio = starget./strue
println("Shift_true omega = ", shift_true[34])
println("Ratio = ", ratio[34])
println("Noisy init = ", s_noisy[34], " mean = ", snmean[34], "True = ", strue[34])
println("Omega stddev raw", sqrt(scov[34, 34]), " Post reg Omega stddev = ", sqrt(inv(scovinv[34, 34])))
println("Starget = ", starget[34])
zsc = (s_noisy[34] - snmean[34])/sqrt(inv(scovinv[34, 34]))

func_specific_params = Dict([(:reg_input=> init), (:coeff_mask1=> coeffmask), (:target1=>starget), (:invcov1=>scovinv), (:coeff_mask2=> coeffmask), (:target2=>starget), (:invcov2=>scovinv), (:lambda2=>lval2), (:lambda3=>lval3)])

recon_settings["fname_save"] = fname_save * ".jld2"
recon_settings["optim_settings"] = optim_settings

if logbool
    error("Not implemented here")
end

res, recon_img = image_recon_derivsum_custom(img_guess, filter_hash, dhc_args, ReconFuncs.Loss3Gaussian, ReconFuncs.dLoss3Gaussian!; optim_settings=optim_settings, func_specific_params)

fracres = (init .- true_img)./true_img
fps = (initps .- true_ps)./true_ps
println("Mean Abs Frac, Init = ", round(mean(abs.(fracres)), digits=3), "Smoothed = ", round(mean(abs.((apdsmoothed .- true_img)./true_img)), digits=3), "Recon = ", round(mean(abs.((recon_img .- true_img)./true_img)), digits=3))
println("MSE, Init = ", round(mean((init .- true_img).^2), digits=5), "Smoothed = ", round(mean((apdsmoothed .- true_img).^2), digits=5), "Recon = ", round(mean((apdsmoothed .- true_img).^2), digits=5))
println("Power Spectrum Frac Res, Init = ", round(mean(abs.(fps)), digits=3), "Smoothed = ", round(mean(abs.((smoothps .- true_ps)./true_ps)), digits=3), "Recon = ", round(mean(abs.(recps .- true_ps)./true_ps), digits=3))

#S2R sequentially
if dhc_args[:iso]
    error("Not constructed for iso")
else #Not iso
    coeffmask = falses(2+Nf+Nf^2)
    coeffmask[Nf+3:end] .= (triu(trues(Nf, Nf)) - I)[:]
end


#Weight given to terms
lval2 = 0.0
lval3 = 0.0

println("Regularizer Lambda=", round(lval3, sigdigits=3))

img_guess = init


img_list = []
push!(img_list, img_guess)
num_rounds=5

#if YOU HAD THE TRUE IMAGE, calculating the shift empirically
Nr=10000
sigma = loaddf["std"]
empsn = []
for n=1:Nr
    noisyim = randn((Nx, Nx)).*sigma .+ true_img
    push!(empsn, DHC_compute_wrapper(noisyim, filter_hash; dhc_args...))
end
coeffsim = hcat(empsn...)'
ncoeffmean = mean(coeffsim, dims=1)
scov = (coeffsim .- ncoeffmean)' * (coeffsim .- ncoeffmean)./(Nr-1)
scov = scov[coeffmask, coeffmask]
snmean = ncoeffmean[:][coeffmask]
strue = DHC_compute_wrapper(loaddf["true_img"], filter_hash, norm=false; dhc_args...)[coeffmask]
shift_true = snmean .- strue
s_noisy = DHC_compute_wrapper(loaddf["init"], filter_hash, norm=false; dhc_args...)[coeffmask]
starget = s_noisy - shift_true #DHC_compute_wrapper(loaddf["true_img"], filter_hash, norm=false; dhc_args...)[coeffmask] #
scovinv = invert_covmat(scov, 1e-10)

ratio = starget./strue
println("Shift_true omega = ", shift_true[34])
println("Ratio = ", ratio[34])
println("Noisy init = ", s_noisy[34], " mean = ", snmean[34], "True = ", strue[34])
println("Omega stddev raw", sqrt(scov[34, 34]), " Post reg Omega stddev = ", sqrt(inv(scovinv[34, 34])))
println("Starget = ", starget[34])
zsc = (s_noisy[34] - snmean[34])/sqrt(inv(scovinv[34, 34]))

func_specific_params = Dict([(:reg_input=> init), (:coeff_mask1=> coeffmask), (:target1=>starget), (:invcov1=>scovinv), (:coeff_mask2=> coeffmask), (:target2=>starget), (:invcov2=>scovinv), (:lambda2=>lval2), (:lambda3=>lval3)])

recon_settings["fname_save"] = fname_save * ".jld2"
recon_settings["optim_settings"] = optim_settings

if logbool
    error("Not implemented here")
end
reshape(coeffmask[3+Nf:end], (Nf, Nf))
res, recon_imgs2ronly = image_recon_derivsum_custom(recon_img, filter_hash, dhc_args, ReconFuncs.Loss3Gaussian, ReconFuncs.dLoss3Gaussian!; optim_settings=optim_settings, func_specific_params)

heatmap(init)
heatmap(recon_img, title="Using true shift")
heatmap(true_img)


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
plot!(title="P(k)")
xlabel!("lnk")
ylabel!("lnP(k)")
p2 = heatmap(true_img, title="True Img", clim=clim)
p4= heatmap(init, title="Init Img", clim=clim)
p5= heatmap(apdsmoothed, title="Smoothed init", clim=clim)
residual = recon_img- true_img
rlims = (minimum(residual), maximum(residual))

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
savefig(p, "../../../scratch_NM/NewWrapper/6-6/3c_sequential_denoising_w_strueempirical_lam0.png")
fracres = (init .- true_img)./true_img
fps = (initps .- true_ps)./true_ps
println("Mean Abs Frac, Init = ", round(mean(abs.(fracres)), digits=3), "Smoothed = ", round(mean(abs.((apdsmoothed .- true_img)./true_img)), digits=3), "Recon = ", round(mean(abs.((recon_img .- true_img)./true_img)), digits=3))
println("MSE, Init = ", round(mean((init .- true_img).^2), digits=5), "Smoothed = ", round(mean((apdsmoothed .- true_img).^2), digits=5), "Recon = ", round(mean((recon_img .- true_img).^2), digits=5))
println("Power Spec Frac Res, Init = ", round(mean(abs.(fps)), digits=3), "Smoothed = ", round(mean(abs.((smoothps .- true_ps)./true_ps)), digits=3), "Recon = ", round(mean(abs.(recps .- true_ps)./true_ps), digits=3))


#=
function Loss3GaussianS2(img_curr, filter_hash, dhc_args; coeff_mask1=nothing, target1=nothing, invcov1=nothing, reg_input=nothing, coeff_mask2=nothing, target2=nothing, invcov2=nothing, lambda2=nothing, lambda3=nothing)
    #func_specific_params should contain: coeff_mask2, target2, invcov2
    s_curr = DHC_compute_wrapper(img_curr, filter_hash, norm=false; dhc_args...)
    s_curr1 = s_curr[coeff_mask1]
    s_curr2 = s_curr[coeff_mask2]
    regterm =  0.5*lambda3*sum((adaptive_apodizer(img_curr, dhc_args) - adaptive_apodizer(reg_input, dhc_args)).^2)
    lnlik_sfd = ( 0.5 .* (s_curr1 - target1)' * invcov1 * (s_curr1 - target1))
    s_curr2 = s_curr[coeff_mask2]
    lnlik_init = ( 0.5*lambda2 .* (s_curr2 - target2)' * invcov2 * (s_curr2 - target2))
    neglogloss = lnlik_sfd[1] + lnlik_init[1] + regterm
end

function dLoss3GaussianS2!(storage_grad, img_curr, filter_hash, dhc_args; coeff_mask1=nothing, target1=nothing, invcov1=nothing, reg_input=nothing, coeff_mask2=nothing, target2=nothing, invcov2=nothing, dA=nothing, lambda2=nothing, lambda3=nothing)
    #func_specific_params should contain: coeff_mask2, target2, invcov2
    Nx = size(img_curr)[1]
    s_curr = DHC_compute_wrapper(img_curr, filter_hash, norm=false; dhc_args...)
    s_curr1 = s_curr[coeff_mask1]
    s_curr2 = s_curr[coeff_mask2]

    diff1 = s_curr1 - target1
    diff2 = s_curr2 - target2

    #Add branches here

    wt1 = augment_weights_S20(convert(Array{Float64,1}, reshape(transpose(diff1) * invcov1, (length(diff1),))), filter_hash, dhc_args, coeff_mask1)
    wt2 = augment_weights_S20(convert(Array{Float64,1}, reshape(transpose(diff2) * invcov2, (length(diff2),))), filter_hash, dhc_args, coeff_mask2)
    apdimg_curr = adaptive_apodizer(img_curr, dhc_args)

    term1 = Deriv_Utils_New.wst_S20_deriv_sum(apdimg_curr, filter_hash, wt1)'
    term2 = lambda2 .* Deriv_Utils_New.wst_S20_deriv_sum(apdimg_curr, filter_hash, wt2)'
    term3= reshape(lambda3.*(apdimg_curr - adaptive_apodizer(reg_input, dhc_args)),(1, Nx^2))
    dsumterms = term1 + term2 + term3
    storage_grad .= reshape(dsumterms * dA, (Nx, Nx))
end
=#

##3d) Only S1 but using Init as a proxy
ARGS_buffer = ["reg", "nonapd", "noiso", "../../../scratch_NM/StandardizedExp/Nx64/", "full_3losstest", "Full+Eps"]
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

optim_settings = Dict([("iterations", 1000), ("norm", false), ("minmethod", ConjugateGradient())])
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
lval3 = 1000.0

println("Regularizer Lambda=", round(lval3, sigdigits=3))

img_guess = init
recon_img = img_guess

img_list = []
push!(img_list, img_guess)
num_rounds=5

#if YOU dont HAve THE TRUE IMAGE, calculating the shift empirically
Nr=10000
sigma = loaddf["std"]
empsn = []
for n=1:Nr
    noisyim = randn((Nx, Nx)).*sigma .+ init
    push!(empsn, DHC_compute_wrapper(noisyim, filter_hash; dhc_args...))
end
coeffsim = hcat(empsn...)'
ncoeffmean = mean(coeffsim, dims=1)
scov = (coeffsim .- ncoeffmean)' * (coeffsim .- ncoeffmean)./(Nr-1)
scov = scov[coeffmask, coeffmask]
snmean = ncoeffmean[:][coeffmask]
strue = DHC_compute_wrapper(loaddf["true_img"], filter_hash, norm=false; dhc_args...)[coeffmask]
shift_true = snmean .- strue
s_noisy = DHC_compute_wrapper(loaddf["init"], filter_hash, norm=false; dhc_args...)[coeffmask]
starget = s_noisy - shift_true #DHC_compute_wrapper(loaddf["true_img"], filter_hash, norm=false; dhc_args...)[coeffmask] #
scovinv = invert_covmat(scov, 1e-10)

ratio = starget./strue
println("Shift_true omega = ", shift_true[34])
println("Ratio = ", ratio[34])
println("Noisy init = ", s_noisy[34], " mean = ", snmean[34], "True = ", strue[34])
println("Omega stddev raw", sqrt(scov[34, 34]), " Post reg Omega stddev = ", sqrt(inv(scovinv[34, 34])))
println("Starget = ", starget[34])
zsc = (s_noisy[34] - snmean[34])/sqrt(inv(scovinv[34, 34]))

func_specific_params = Dict([(:reg_input=> init), (:coeff_mask1=> coeffmask), (:target1=>starget), (:invcov1=>scovinv), (:coeff_mask2=> coeffmask), (:target2=>starget), (:invcov2=>scovinv), (:lambda2=>lval2), (:lambda3=>lval3)])

recon_settings["fname_save"] = fname_save * ".jld2"
recon_settings["optim_settings"] = optim_settings

if logbool
    error("Not implemented here")
end

res, recon_img = image_recon_derivsum_custom(img_guess, filter_hash, dhc_args, ReconFuncs.Loss3Gaussian, ReconFuncs.dLoss3Gaussian!; optim_settings=optim_settings, func_specific_params)
heatmap(init)
heatmap(recon_img, title="Using true shift")
heatmap(true_img)


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
savefig(p, "../../../scratch_NM/NewWrapper/6-6/3d_denoising_w_sinitempirical_lam1000.png")
fracres = (init .- true_img)./true_img
fps = (initps .- true_ps)./true_ps
println("Mean Abs Frac, Init = ", round(mean(abs.(fracres)), digits=3), "Smoothed = ", round(mean(abs.((apdsmoothed .- true_img)./true_img)), digits=3), "Recon = ", round(mean(abs.((recon_img .- true_img)./true_img)), digits=3))
println("MSE, Init = ", round(mean((init .- true_img).^2), digits=6), "Smoothed = ", round(mean((apdsmoothed .- true_img).^2), digits=6), "Recon = ", round(mean((recon_img .- true_img).^2), digits=6))
println("Power Spectrum Frac Res, Init = ", round(mean(abs.(fps)), digits=3), "Smoothed = ", round(mean(abs.((smoothps .- true_ps)./true_ps)), digits=3), "Recon = ", round(mean(abs.(recps .- true_ps)./true_ps), digits=3))
mean(abs.(smoothps .- true_ps))
mean(abs.(recps .- true_ps))
abs.(recps .- true_ps)

##2 Comparing WST with the most perfect case (starget = strue) with smoothing

##2a) with only S1
ARGS_buffer = ["reg", "nonapd", "noiso", "../../../scratch_NM/StandardizedExp/Nx64/", "full_3losstest", "Full+Eps"]
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

optim_settings = Dict([("iterations", 1000), ("norm", false), ("minmethod", ConjugateGradient())])
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
lval3 = 1.0

println("Regularizer Lambda=", round(lval3, sigdigits=3))

img_guess = init
recon_img = img_guess

img_list = []
push!(img_list, img_guess)
num_rounds=5

#if YOU dont HAve THE TRUE IMAGE, calculating the shift empirically
Nr=10000
sigma = loaddf["std"]
empsn = []
for n=1:Nr
    noisyim = randn((Nx, Nx)).*sigma .+ init
    push!(empsn, DHC_compute_wrapper(noisyim, filter_hash; dhc_args...))
end
coeffsim = hcat(empsn...)'
ncoeffmean = mean(coeffsim, dims=1)
scov = (coeffsim .- ncoeffmean)' * (coeffsim .- ncoeffmean)./(Nr-1)
scov = scov[coeffmask, coeffmask]
strue = DHC_compute_wrapper(loaddf["true_img"], filter_hash, norm=false; dhc_args...)[coeffmask]
starget = strue #DHC_compute_wrapper(loaddf["true_img"], filter_hash, norm=false; dhc_args...)[coeffmask] #
scovinv = invert_covmat(scov, 1e-10)


func_specific_params = Dict([(:reg_input=> init), (:coeff_mask1=> coeffmask), (:target1=>starget), (:invcov1=>scovinv), (:coeff_mask2=> coeffmask), (:target2=>starget), (:invcov2=>scovinv), (:lambda2=>lval2), (:lambda3=>lval3)])

recon_settings["fname_save"] = fname_save * ".jld2"
recon_settings["optim_settings"] = optim_settings

if logbool
    error("Not implemented here")
end

res, recon_img = image_recon_derivsum_custom(img_guess, filter_hash, dhc_args, ReconFuncs.Loss3Gaussian, ReconFuncs.dLoss3Gaussian!; optim_settings=optim_settings, func_specific_params)
heatmap(init)
heatmap(recon_img, title="Using true shift")
heatmap(true_img)


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
savefig(p, "../../../scratch_NM/NewWrapper/6-6/2a_reg_lam1.png")
fracres = (init .- true_img)./true_img
fps = (initps .- true_ps)./true_ps
println("Mean Abs Frac, Init = ", round(mean(abs.(fracres)), digits=3), "Smoothed = ", round(mean(abs.((apdsmoothed .- true_img)./true_img)), digits=3), "Recon = ", round(mean(abs.((recon_img .- true_img)./true_img)), digits=3))
println("MSE, Init = ", round(mean((init .- true_img).^2), digits=5), "Smoothed = ", round(mean((apdsmoothed .- true_img).^2), digits=5), "Recon = ", round(mean((apdsmoothed .- true_img).^2), digits=5))
println("Power Spectrum Frac Res, Init = ", round(mean(abs.(fps)), digits=3), "Smoothed = ", round(mean(abs.((smoothps .- true_ps)./true_ps)), digits=3), "Recon = ", round(mean(abs.(recps .- true_ps)./true_ps), digits=3))


##2a) with only S1, no pixwise reg
ARGS_buffer = ["reg", "nonapd", "noiso", "../../../scratch_NM/StandardizedExp/Nx64/", "full_3losstest", "Full+Eps"]
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

optim_settings = Dict([("iterations", 1000), ("norm", false), ("minmethod", ConjugateGradient())])
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

img_guess = init
recon_img = img_guess

img_list = []
push!(img_list, img_guess)
num_rounds=5

#if YOU dont HAve THE TRUE IMAGE, calculating the shift empirically
Nr=10000
sigma = loaddf["std"]
empsn = []
for n=1:Nr
    noisyim = randn((Nx, Nx)).*sigma .+ init
    push!(empsn, DHC_compute_wrapper(noisyim, filter_hash; dhc_args...))
end
coeffsim = hcat(empsn...)'
ncoeffmean = mean(coeffsim, dims=1)
scov = (coeffsim .- ncoeffmean)' * (coeffsim .- ncoeffmean)./(Nr-1)
scov = scov[coeffmask, coeffmask]
strue = DHC_compute_wrapper(loaddf["true_img"], filter_hash, norm=false; dhc_args...)[coeffmask]
starget = strue #DHC_compute_wrapper(loaddf["true_img"], filter_hash, norm=false; dhc_args...)[coeffmask] #
scovinv = invert_covmat(scov, 1e-10)


func_specific_params = Dict([(:reg_input=> init), (:coeff_mask1=> coeffmask), (:target1=>starget), (:invcov1=>scovinv), (:coeff_mask2=> coeffmask), (:target2=>starget), (:invcov2=>scovinv), (:lambda2=>lval2), (:lambda3=>lval3)])

recon_settings["fname_save"] = fname_save * ".jld2"
recon_settings["optim_settings"] = optim_settings

if logbool
    error("Not implemented here")
end

res, recon_img = image_recon_derivsum_custom(img_guess, filter_hash, dhc_args, ReconFuncs.Loss3Gaussian, ReconFuncs.dLoss3Gaussian!; optim_settings=optim_settings, func_specific_params)
heatmap(init)
heatmap(recon_img, title="Using true shift")
heatmap(true_img)


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
savefig(p, "../../../scratch_NM/NewWrapper/6-6/2a.png")
fracres = (init .- true_img)./true_img
fps = (initps .- true_ps)./true_ps
println("Mean Abs Frac, Init = ", round(mean(abs.(fracres)), digits=3), "Smoothed = ", round(mean(abs.((apdsmoothed .- true_img)./true_img)), digits=3), "Recon = ", round(mean(abs.((recon_img .- true_img)./true_img)), digits=3))
println("MSE, Init = ", round(mean((init .- true_img).^2), digits=5), "Smoothed = ", round(mean((apdsmoothed .- true_img).^2), digits=5), "Recon = ", round(mean((apdsmoothed .- true_img).^2), digits=5))
println("Power Spectrum Frac Res, Init = ", round(mean(abs.(fps)), digits=3), "Smoothed = ", round(mean(abs.((smoothps .- true_ps)./true_ps)), digits=3), "Recon = ", round(mean(abs.(recps .- true_ps)./true_ps), digits=3))
mean(abs.((smoothps .- true_ps)))
mean(abs.(recps .- true_ps))


##2b) with S2R full, no pixwise reg
ARGS_buffer = ["reg", "nonapd", "noiso", "../../../scratch_NM/StandardizedExp/Nx64/", "full_3losstest", "Full+Eps"]
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

optim_settings = Dict([("iterations", 1000), ("norm", false), ("minmethod", ConjugateGradient())])
recon_settings = Dict([("log", logbool), ("Invcov_matrix", ARGS_buffer[6]), ("optim_settings", optim_settings), ("eps_value_sfd", 1e-5), ("eps_value_init", 1e-10)]) #Add constraints

recon_settings["datafile"] = datfile

if dhc_args[:iso]
    error("Not constructed for iso")
else #Not iso
    coeffmask = falses(2+Nf+Nf^2)
    coeffmask[Nf+3:end] .= triu(trues(Nf, Nf))[:]
end


#Weight given to terms
lval2 = 0.0
lval3 = 0.0

println("Regularizer Lambda=", round(lval3, sigdigits=3))

img_guess = init
recon_img = img_guess

img_list = []
push!(img_list, img_guess)
num_rounds=5

#if YOU dont HAve THE TRUE IMAGE, calculating the shift empirically
Nr=10000
sigma = loaddf["std"]
empsn = []
for n=1:Nr
    noisyim = randn((Nx, Nx)).*sigma .+ init
    push!(empsn, DHC_compute_wrapper(noisyim, filter_hash; dhc_args...))
end
coeffsim = hcat(empsn...)'
ncoeffmean = mean(coeffsim, dims=1)
scov = (coeffsim .- ncoeffmean)' * (coeffsim .- ncoeffmean)./(Nr-1)
scov = scov[coeffmask, coeffmask]
strue = DHC_compute_wrapper(loaddf["true_img"], filter_hash, norm=false; dhc_args...)[coeffmask]
starget = strue #DHC_compute_wrapper(loaddf["true_img"], filter_hash, norm=false; dhc_args...)[coeffmask] #
scovinv = invert_covmat(scov, 1e-10)


func_specific_params = Dict([(:reg_input=> init), (:coeff_mask1=> coeffmask), (:target1=>starget), (:invcov1=>scovinv), (:coeff_mask2=> coeffmask), (:target2=>starget), (:invcov2=>scovinv), (:lambda2=>lval2), (:lambda3=>lval3)])

recon_settings["fname_save"] = fname_save * ".jld2"
recon_settings["optim_settings"] = optim_settings

if logbool
    error("Not implemented here")
end

res, recon_img = image_recon_derivsum_custom(img_guess, filter_hash, dhc_args, ReconFuncs.Loss3Gaussian, ReconFuncs.dLoss3Gaussian!; optim_settings=optim_settings, func_specific_params)
heatmap(init)
heatmap(recon_img, title="Using true shift")
heatmap(true_img)


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
savefig(p, "../../../scratch_NM/NewWrapper/6-6/2b.png")
fracres = (init .- true_img)./true_img
fps = (initps .- true_ps)./true_ps
println("Mean Abs Frac, Init = ", round(mean(abs.(fracres)), digits=3), "Smoothed = ", round(mean(abs.((apdsmoothed .- true_img)./true_img)), digits=3), "Recon = ", round(mean(abs.((recon_img .- true_img)./true_img)), digits=3))
println("MSE, Init = ", round(mean((init .- true_img).^2), digits=8), "Smoothed = ", round(mean((apdsmoothed .- true_img).^2), digits=8), "Recon = ", round(mean((recon_img .- true_img).^2), digits=8))
println("Power Spectrum Frac Res, Init = ", round(mean(abs.(fps)), digits=3), "Smoothed = ", round(mean(abs.((smoothps .- true_ps)./true_ps)), digits=3), "Recon = ", round(mean(abs.(recps .- true_ps)./true_ps), digits=3))
mean(abs.((smoothps .- true_ps)))
mean(abs.(recps .- true_ps))
mean((apdsmoothed .- true_img).^2)

#Try on 128: is there a difference?
Random.seed!(41)
sfdall = readsfd_fromsrc("../../../scratch_NM/data/dust10000.fits", 128, logbool=false)
true128 = sfdall[:, :, 1000]
heatmap(true128)
sigma = 0.397*mean(true128)
init128 = true128 + rand(Normal(0, sigma), (128, 128))
savdict = Dict([("true_img"=>true128), ("init"=>init128), ("noise model", "White noise, No smoothing. sigma=0.397*mean(true_img)"), ("std"=>sigma), ("seed"=>41)])
save("../../../scratch_NM/StandardizedExp/Data1000_128.jld2", savdict)


##0) What smoothing kernel gives the best MSE, Abs Frac Res and PowSpec?
ARGS_buffer = ["reg", "nonapd", "noiso", "../../../scratch_NM/StandardizedExp/", "full_3losstest", "Full+Eps"]
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
datfile = direc * "Data" * string(numfile) * "_128.jld2" #Replace w SLURM array
loaddf = load(datfile)
true_img = loaddf["true_img"]
init = loaddf["init"]
Nx=size(true_img)[1]
println(Nx)
kbins=convert(Array{Float64, 1}, collect(1:Nx/2))
for l in [0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 1.8, 2.0]
    apdsmoothed = imfilter(init, Kernel.gaussian(l))
    true_ps = calc_1dps(true_img, kbins)
    smoothps = calc_1dps(apdsmoothed, kbins)
    fracres = (apdsmoothed .- true_img)./true_img
    fps = (smoothps .- true_ps)./true_ps
    println("Smoothing scale = ", l)
    println("Mean Abs Frac, Smoothed = ", round(mean(abs.((apdsmoothed .- true_img)./true_img)), digits=3))
    println("MSE = ", round(mean((apdsmoothed .- true_img).^2), digits=5))
    println("Power Spec Frac Res, Smoothed = ", round(mean(abs.(smoothps .- true_ps)), digits=3))
end


##2a)
##2a) with only S1
ARGS_buffer = ["reg", "nonapd", "noiso", "../../../scratch_NM/StandardizedExp/", "full_3losstest", "Full+Eps"]
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
datfile = direc * "Data" * string(numfile) * "_128.jld2" #Replace w SLURM array
loaddf = load(datfile)
true_img = loaddf["true_img"]
init = loaddf["init"]

fname_save = direc * "scratch_NM/NewWrapper/5-30/" * string(numfile) * "_try1"  #Change
Nx=size(true_img)[1]
filter_hash = fink_filter_hash(1, 8, nx=Nx, t=1, wd=1, Omega=true)
(S1iso, Nf) = size(filter_hash["S1_iso_mat"])
(S2iso, Nfsq) = size(filter_hash["S2_iso_mat"])
dhc_args = Dict(:doS2=>false, :doS20=>true, :apodize=>apdbool, :iso=>isobool) #Iso #CHANGE: Change sig for the sfd data since the noise model is super high and the tiny values make sense

optim_settings = Dict([("iterations", 1000), ("norm", false), ("minmethod", ConjugateGradient())])
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

img_guess = init
recon_img = img_guess

img_list = []
push!(img_list, img_guess)
num_rounds=5

#if YOU dont HAve THE TRUE IMAGE, calculating the shift empirically
Nr=10000
sigma = loaddf["std"]
empsn = []
for n=1:Nr
    noisyim = randn((Nx, Nx)).*sigma .+ init
    push!(empsn, DHC_compute_wrapper(noisyim, filter_hash; dhc_args...))
end
coeffsim = hcat(empsn...)'
ncoeffmean = mean(coeffsim, dims=1)
scov = (coeffsim .- ncoeffmean)' * (coeffsim .- ncoeffmean)./(Nr-1)
scov = scov[coeffmask, coeffmask]
strue = DHC_compute_wrapper(loaddf["true_img"], filter_hash, norm=false; dhc_args...)[coeffmask]
starget = strue #DHC_compute_wrapper(loaddf["true_img"], filter_hash, norm=false; dhc_args...)[coeffmask] #
scovinv = invert_covmat(scov, 1e-10)


func_specific_params = Dict([(:reg_input=> init), (:coeff_mask1=> coeffmask), (:target1=>starget), (:invcov1=>scovinv), (:coeff_mask2=> coeffmask), (:target2=>starget), (:invcov2=>scovinv), (:lambda2=>lval2), (:lambda3=>lval3)])

recon_settings["fname_save"] = fname_save * ".jld2"
recon_settings["optim_settings"] = optim_settings

if logbool
    error("Not implemented here")
end

res, recon_img = image_recon_derivsum_custom(img_guess, filter_hash, dhc_args, ReconFuncs.Loss3Gaussian, ReconFuncs.dLoss3Gaussian!; optim_settings=optim_settings, func_specific_params)
heatmap(init)
heatmap(recon_img, title="Using true shift")
heatmap(true_img)


kbins=convert(Array{Float64, 1}, collect(1:64))
apdsmoothed = imfilter(init, Kernel.gaussian(1.2))
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
savefig(p, "../../../scratch_NM/NewWrapper/6-6/2c_reg_lam0.png")
fracres = (init .- true_img)./true_img
fps = (initps .- true_ps)./true_ps
println("Mean Abs Frac, Init = ", round(mean(abs.(fracres)), digits=3), "Smoothed = ", round(mean(abs.((apdsmoothed .- true_img)./true_img)), digits=3), "Recon = ", round(mean(abs.((recon_img .- true_img)./true_img)), digits=3))
println("Power Spectrum Frac Res, Init = ", round(mean(abs.(fps)), digits=3), "Smoothed = ", round(mean(abs.((smoothps .- true_ps)./true_ps)), digits=3), "Recon = ", round(mean(abs.(recps .- true_ps)./true_ps), digits=3))
round(mean(abs.(fps)), digits=3)


##2c) with only S1
ARGS_buffer = ["reg", "nonapd", "noiso", "../../../scratch_NM/StandardizedExp/", "full_3losstest", "Full+Eps"]
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
datfile = direc * "Data" * string(numfile) * "_128.jld2" #Replace w SLURM array
loaddf = load(datfile)
true_img = loaddf["true_img"]
init = loaddf["init"]

fname_save = direc * "scratch_NM/NewWrapper/5-30/" * string(numfile) * "_try1"  #Change
Nx=size(true_img)[1]
filter_hash = fink_filter_hash(1, 8, nx=Nx, t=1, wd=1, Omega=true)
(S1iso, Nf) = size(filter_hash["S1_iso_mat"])
(S2iso, Nfsq) = size(filter_hash["S2_iso_mat"])
dhc_args = Dict(:doS2=>false, :doS20=>true, :apodize=>apdbool, :iso=>isobool) #Iso #CHANGE: Change sig for the sfd data since the noise model is super high and the tiny values make sense

optim_settings = Dict([("iterations", 1000), ("norm", false), ("minmethod", ConjugateGradient())])
recon_settings = Dict([("log", logbool), ("Invcov_matrix", ARGS_buffer[6]), ("optim_settings", optim_settings), ("eps_value_sfd", 1e-5), ("eps_value_init", 1e-10)]) #Add constraints

recon_settings["datafile"] = datfile

if dhc_args[:iso]
    error("Not constructed for iso")
else #Not iso
    coeffmask = falses(2+Nf+Nf^2)
    coeffmask[Nf+3:end] .= triu(trues(Nf, Nf))[:]
end


#Weight given to terms
lval2 = 0.0
lval3 = 0.0

println("Regularizer Lambda=", round(lval3, sigdigits=3))

img_guess = init
recon_img = img_guess

img_list = []
push!(img_list, img_guess)
num_rounds=5

#if YOU dont HAve THE TRUE IMAGE, calculating the shift empirically
Nr=10000
sigma = loaddf["std"]
empsn = []
for n=1:Nr
    noisyim = randn((Nx, Nx)).*sigma .+ init
    push!(empsn, DHC_compute_wrapper(noisyim, filter_hash; dhc_args...))
end
coeffsim = hcat(empsn...)'
ncoeffmean = mean(coeffsim, dims=1)
scov = (coeffsim .- ncoeffmean)' * (coeffsim .- ncoeffmean)./(Nr-1)
scov = scov[coeffmask, coeffmask]
strue = DHC_compute_wrapper(loaddf["true_img"], filter_hash, norm=false; dhc_args...)[coeffmask]
starget = strue #DHC_compute_wrapper(loaddf["true_img"], filter_hash, norm=false; dhc_args...)[coeffmask] #
scovinv = invert_covmat(scov, 1e-10)


func_specific_params = Dict([(:reg_input=> init), (:coeff_mask1=> coeffmask), (:target1=>starget), (:invcov1=>scovinv), (:coeff_mask2=> coeffmask), (:target2=>starget), (:invcov2=>scovinv), (:lambda2=>lval2), (:lambda3=>lval3)])

recon_settings["fname_save"] = fname_save * ".jld2"
recon_settings["optim_settings"] = optim_settings

if logbool
    error("Not implemented here")
end
coeffmask
res, recon_img = image_recon_derivsum_custom(img_guess, filter_hash, dhc_args, ReconFuncs.Loss3Gaussian, ReconFuncs.dLoss3Gaussian!; optim_settings=optim_settings, func_specific_params)
heatmap(init)
heatmap(recon_img, title="Using true shift")
heatmap(true_img)


kbins=convert(Array{Float64, 1}, collect(1:64))
apdsmoothed = imfilter(init, Kernel.gaussian(1.2))
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
savefig(p, "../../../scratch_NM/NewWrapper/6-6/2d_reg_lam0.png")
fracres = (init .- true_img)./true_img
fps = (initps .- true_ps)./true_ps
println("Mean Abs Frac, Init = ", round(mean(abs.(fracres)), digits=3), "Smoothed = ", round(mean(abs.((apdsmoothed .- true_img)./true_img)), digits=3), "Recon = ", round(mean(abs.((recon_img .- true_img)./true_img)), digits=3))
println("MSE, Init = ", round(mean((init .- true_img).^2), digits=6), "Smoothed = ", round(mean((apdsmoothed .- true_img).^2), digits=6), "Recon = ", round(mean((recon_img .- true_img).^2), digits=6))
println("Power Spectrum Frac Res, Init = ", round(mean(abs.(fps)), digits=3), "Smoothed = ", round(mean(abs.((smoothps .- true_ps)./true_ps)), digits=3), "Recon = ", round(mean(abs.(recps .- true_ps)./true_ps), digits=3))
round(mean(abs.(fps)), digits=3)
