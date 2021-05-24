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
include("../../../main/compute.jl")


##Doug functions

function my_DHC_compute(image::Array{Float64,2}, filter_hash::Dict, filter_hash2::Dict=filter_hash, sigim=nothing;
    doS2::Bool=true, doS12::Bool=false, doS20::Bool=false, norm=false, iso=false, FFTthreads=2)
    # image        - input for WST
    # filter_hash  - filter hash from fink_filter_hash
    # filter_hash2 - filters for second order.  Default to same as first order.
    # doS2         - compute S2 coeffs
    # doS12        - compute S2 coeffs
    # doS20        - compute S2 coeffs
    # norm         - scale to mean zero, unit variance
    # iso          - sum over angles to obtain isotropic coeffs

    # Use 2 threads for FFT
    FFTW.set_num_threads(FFTthreads)
    if norm println("Norm") end

    # array sizes
    (Nx, Ny)  = size(image)
    if Nx != Ny error("Input image must be square") end
    (Nf, )    = size(filter_hash["filt_index"])
    if Nf == 0  error("filter hash corrupted") end
    @assert Nx==filter_hash["npix"] "Filter size should match npix"
    @assert Nx==filter_hash2["npix"] "Filter2 size should match npix"

    # allocate coeff arrays
    out_coeff = []
    S0  = zeros(Float64, 2)
    S1  = zeros(Float64, Nf)
    if doS2  S2  = zeros(Float64, Nf, Nf) end  # traditional 2nd order
    if doS12 S12 = zeros(Float64, Nf, Nf) end  # Fourier correlation
    if doS20 S20 = zeros(Float64, Nf, Nf) end  # real space correlation
    anyM2 = doS2 | doS12 | doS20
    anyrd = doS2 | doS20             # compute real domain with iFFT

    # allocate image arrays for internal use
    if doS12 im_fdf_0_1 = zeros(Float64,           Nx, Ny, Nf) end   # this must be zeroed!
    if anyrd im_rd_0_1  = Array{Float64, 3}(undef, Nx, Ny, Nf) end

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
    im_fd_0 = fft(norm_im)  # total power=1.0

    # unpack filter_hash
    f_ind   = filter_hash["filt_index"]  # (J, L) array of filters represented as index value pairs
    f_val   = filter_hash["filt_value"]
    fvar = fft(sigim.^2)
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
                if doS12 im_fdf_0_1[ind,f] = abs(zval) end
            end
            S1[f] = S1tot/(Nx*Ny)  # image power
            if anyrd
                fpsi = fft(abs2.(realspace_filter(Nx, f_ind[f], f_val[f])))
                extrapower = ifft(fvar.*fpsi)
                im_rd_0_1[:,:,f] .= abs2.(P*zarr) .+ abs.(extrapower)
                println(maximum(abs.(extrapower)))
            #    im_rd_0_1[:,:,f] .= abs2.(P*zarr) .+ sigim.^2 .*(sum(f_v.^2)/(Nx*Nx))
            end
            zarr[f_i] .= 0
        end
    end

    append!(out_coeff, iso ? filter_hash["S1_iso_mat"]*S1 : S1)


    # we stored the abs()^2, so take sqrt (this is faster to do all at once)
    if anyrd im_rd_0_1 .= sqrt.(im_rd_0_1) end

    Mat2 = filter_hash["S2_iso_mat"]
    if doS2
        f_ind2   = filter_hash2["filt_index"]  # (J, L) array of filters represented as index value pairs
        f_val2   = filter_hash2["filt_value"]

        ## Traditional second order
        for f1 = 1:Nf
            thisim = fft(im_rd_0_1[:,:,f1])  # Could try rfft here
            # println("  f1",f1,"  sum(fft):",sum(abs2.(thisim))/Nx^2, "  sum(im): ",sum(abs2.(im_rd_0_1[:,:,f1])))
            # Loop over f2 and do second-order convolution
            for f2 = 1:Nf
                f_i = f_ind2[f2]  # CartesianIndex list for filter
                f_v = f_val2[f2]  # Values for f_i
                # sum im^2 = sum(|fft|^2/npix)
                S2[f1,f2] = sum(abs2.(f_v .* thisim[f_i]))/(Nx*Ny)
            end
        end
        append!(out_coeff, iso ? Mat2*S2[:] : S2[:])
    end

    # Fourier domain 2nd order
    if doS12
        Amat = reshape(im_fdf_0_1, Nx*Ny, Nf)
        S12  = Amat' * Amat
        append!(out_coeff, iso ? Mat2*S12[:] : S12[:])
    end

    # Real domain 2nd order
    if doS20
        Amat = reshape(im_rd_0_1, Nx*Ny, Nf)
        S20  = Amat' * Amat
        append!(out_coeff, iso ? Mat2*S20[:] : S20[:])
    end

    return out_coeff
end

function realspace_filter(Nx, f_i, f_v)

    zarr = zeros(ComplexF64, Nx, Nx)
    for i = 1:length(f_i)
        zarr[f_i[i]] = f_v[i] # filter*image in Fourier domain
    end
    filt = ifft(zarr)  # real space, complex
    return filt
end

function S20_noisecovar(im, fhash, σmap, Nsam=10; iso=iso, doS2=false, doS20=false)

    (Nx, Ny)  = size(im)
    if Nx != Ny error("Input image must be square") end
    (N1iso, Nf) = size(fhash["S1_iso_mat"])
    (N2iso, _)  = size(fhash["S2_iso_mat"])
    # fhash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1)
    if iso
        Sarr = zeros(Float64, N2iso, Nsam)
        i0 = 3+N1iso
        Smask = nothing # not implemented yet
    else
        # triangle mask
        tri_mask = reshape(tril(trues(Nf,Nf)),Nf*Nf)
        #tri_mask[1,1] = false
        Smask = vcat(falses(2+Nf), tri_mask)
        # sum(S1[jmax,l=odd]) = sum(S1[jmax,l=even]) so we must remove one more to avoid redundancy
        #Smask[19] = false
        Sarr  = zeros(Float64, sum(Smask), Nsam)
    end

    for j=1:Nsam
        noise = randn(Nx,Nx) .* σmap
        Scoeff = eqws_compute(im+noise, fhash, doS2=doS2, doS20=doS20, norm=false, iso=iso)
        Sarr[:,j] = (Scoeff[Smask])
    end
    Smean = mean(Sarr,dims=2)
    Smean = reshape(Smean, length(Smean))
    ΔS    = Sarr .- Smean
    Scov  = ΔS*ΔS' ./ (Nsam-1)
    return Smean, Scov, Smask
end



##Examining biased coeff distributions
#16x16
Random.seed!(33)
Nx=16
logbool=false
apdbool=true
isobool=false
filter_hash = fink_filter_hash(1, 8, nx=Nx, t=1, Omega=true)

sfdall = readsfd(Nx, logbool=logbool)

Nr=10000
true_img = sfdall[:, :, 1000]
sigma = mean(true_img)
noisemod = Normal(0.0, sigma)
noise = rand(noisemod, (Nr, Nx, Nx))
noisyims = noise .+ reshape(true_img, (1, Nx, Nx))
dhc_args = Dict(:doS2=>false, :doS20=>true, :iso=>isobool, :norm=> false) #Iso #CHANGE: Change sig for the sfd data since the noise model is super high and the tiny values make sense

z_noisy =[]
for n=1:Nr
    zlams = eqws_compute_convmap(true_img, filter_hash, filter_hash; dhc_args...)[3:end]
    push!(z_noisy, zlams)
end

#Compare actual and analytical mean of the ZLambdas
P = plan_ifft(z_noisy[1][1])
#Sim
empsn = []
for n=1:Nr
    push!(empsn, DHC_compute_wrapper(noisyims[n, :, :], filter_hash; dhc_args...))
end
coeffsim = hcat(empsn...)'
ncoeffmean = mean(coeffsim, dims=1)


expcoeffmean = my_DHC_compute(true_img, filter_hash, filter_hash, fill(sigma, (Nx, Nx)))

p = plot(collect(1:length(ncoeffmean)), ncoeffmean[:], label="Empirical Mean")
plot!(collect(1:length(ncoeffmean)), expcoeffmean[:], label="Analytical Mean")

p = plot(collect(1:length(ncoeffmean)), ncoeffmean[:], label="Empirical Mean")
plot!(collect(1:length(ncoeffmean)), 0.66*expcoeffmean[:], label="Analytical Mean Corrected by 0.66")

#Ans: Not equal

#Why were they equal in doug's case?
#He did 128x128
Nx=64
logbool=false
apdbool=true
isobool=false
filter_hash = fink_filter_hash(1, 8, nx=Nx, t=1, Omega=true)

Nr=40000
truefile = load("scratch_NM/StandardizedExp/Nx64/Data_1000.jld2")
true_img = truefile["true_img"]

sigma = mean(true_img)
noisemod = Normal(0.0, sigma)

dhc_args = Dict(:doS2=>false, :doS20=>true, :iso=>isobool, :norm=> false) #Iso #CHANGE: Change sig for the sfd data since the noise model is super high and the tiny values make sense
#=
z_noisy =[]
for n=1:Nr
    zlams = eqws_compute_convmap(true_img, filter_hash, filter_hash; dhc_args...)[3:end]
    push!(z_noisy, zlams)
end
=#
#Compare actual and analytical mean of the ZLambdas
#P = plan_ifft(z_noisy[1][1])
#Sim

empsn = []
for n=1:Nr
    noisyim = rand(noisemod, (Nx, Nx)) .+ true_img
    push!(empsn, DHC_compute_wrapper(noisyim, filter_hash; dhc_args...))
end
coeffsim = hcat(empsn...)'
ncoeffmean = mean(coeffsim, dims=1)

expcoeffmean = my_DHC_compute(apodizer(true_img), filter_hash, filter_hash, fill(sigma, (Nx, Nx)))


p = plot(collect(1:length(ncoeffmean)), ncoeffmean[:], label="Empirical Mean")
plot!(collect(1:length(ncoeffmean)), expcoeffmean[:], label="Analytical Mean")
title!("64x64")

p = plot(collect(1:length(ncoeffmean)), 0.66*ncoeffmean[:], label="Empirical Mean")
plot!(collect(1:length(ncoeffmean)), expcoeffmean[:], label="Analytical Mean Corrected by 0.66")
title!("64x64")

S1mask = falses(1192)
Nf = length(filter_hash["filt_index"])
S1mask[3:Nf+2] .= true
p = plot(collect(1:Nf), ncoeffmean[:][S1mask], label="Empirical Mean")
plot!(collect(1:Nf), expcoeffmean[:][S1mask], label="Analytical Mean")
title!("S1 Coeffs: 64x64")

#Check compatibility
sigma = 0.0
expcoeffmean = my_DHC_compute(true_img, filter_hash, filter_hash, fill(sigma, (Nx, Nx)))
truecoeff =  DHC_compute_wrapper(true_img, filter_hash; dhc_args...)
p = plot(collect(1:length(truecoeff)), truecoeff[:], label="True Img Coeffs using Wrapper")
plot!(collect(1:length(truecoeff)), expcoeffmean[:], label="True Img Coeffs using myDHC")
title!("Coeffs Check: 64x64")
#INCONSISTENT

sigma = 0.0
expcoeffmean = my_DHC_compute(true_img, filter_hash, filter_hash, fill(sigma, (Nx, Nx)); dhc_args...)
truecoeff =  DHC_compute_wrapper(true_img, filter_hash; dhc_args...)
p = plot(collect(1:length(truecoeff)), truecoeff[:], label="True Img Coeffs using DHC-Compute-wrapper")
plot!(collect(1:length(truecoeff)), expcoeffmean[:], label="True Img Coeffs using myDHC")
title!("Coeffs Check: 64x64")
#CONSISTENT

sigma = 0.0
expcoeffmean = my_DHC_compute(true_img, filter_hash, filter_hash, fill(sigma, (Nx, Nx)); dhc_args...)
truecoeff =  DHC_compute(true_img, filter_hash; dhc_args...)
p = plot(collect(1:length(truecoeff)), truecoeff[:], label="True Img Coeffs using DHC-Compute")
plot!(collect(1:length(truecoeff)), expcoeffmean[:], label="True Img Coeffs using myDHC")
title!("Coeffs Check: 64x64")
#CONSISTENT


sigma = 0.0
expcoeffmean = my_DHC_compute(true_img, filter_hash, filter_hash, fill(sigma, (Nx, Nx)); dhc_args...)
truecoeff =  DHC_compute(true_img, filter_hash; dhc_args...)
p = plot(collect(1:length(truecoeff)), truecoeff[:], label="True Img Coeffs using DHC-Compute")
plot!(collect(1:length(truecoeff)), expcoeffmean[:], label="True Img Coeffs using myDHC")
title!("Coeffs Check: 64x64")
#CONSISTENT


#Noisy imgs
Nx=64
logbool=false
apdbool=true
isobool=false
filter_hash = fink_filter_hash(1, 8, nx=Nx, t=1, Omega=true)
Nf = size(filter_hash["filt_index"])[1]

Nr=40000
truefile = load("scratch_NM/StandardizedExp/Nx64/Data_1000.jld2")
true_img = truefile["true_img"]

sigma = 0.2*mean(true_img)
noisemod = Normal(0.0, sigma)

dhc_args = Dict(:doS2=>false, :doS20=>true, :iso=>isobool, :norm=> false) #Iso #CHANGE: Change sig for the sfd data since the noise model is super high and the tiny values make sense
#=
z_noisy =[]
for n=1:Nr
    zlams = eqws_compute_convmap(true_img, filter_hash, filter_hash; dhc_args...)[3:end]
    push!(z_noisy, zlams)
end
=#
#Compare actual and analytical mean of the ZLambdas
#P = plan_ifft(z_noisy[1][1])
#Sim

empsn = []
for n=1:Nr
    noisyim = rand(noisemod, (Nx, Nx)) .+ true_img
    push!(empsn, DHC_compute_wrapper(noisyim, filter_hash; dhc_args...))
end
coeffsim = hcat(empsn...)'
ncoeffmean = mean(coeffsim, dims=1)

expcoeffmean = my_DHC_compute(true_img, filter_hash, filter_hash, fill(sigma, (Nx, Nx)); dhc_args...)
truecoeff = DHC_compute_wrapper(true_img, filter_hash; dhc_args...)

p = plot(collect(1:length(ncoeffmean)), ncoeffmean[:], label="Empirical Mean")
plot!(collect(1:length(ncoeffmean)), expcoeffmean[:], label="Analytical Mean")
title!("64x64")

S1mask = falses(1192)
S1mask[3:2+Nf] .= true
p = plot(collect(1:Nf), (ncoeffmean[:] - truecoeff)[S1mask], label="Empirical Mean Difference")
plot!(collect(1:Nf), (expcoeffmean[:]  - truecoeff)[S1mask], label="Analytical Mean Difference")
title!("64x64 Diff, S1")

#Noisy imgs: with mean subtraction
Nx=64
logbool=false
apdbool=true
isobool=false
filter_hash = fink_filter_hash(1, 8, nx=Nx, t=1, Omega=true)
Nf = size(filter_hash["filt_index"])[1]

Nr=40000
truefile = load("scratch_NM/StandardizedExp/Nx64/Data_1000.jld2")
true_img = truefile["true_img"] .- mean(true_img)

sigma = 2.0*std(true_img)
noisemod = Normal(0.0, sigma)

dhc_args = Dict(:doS2=>false, :doS20=>true, :iso=>isobool, :norm=> false) #Iso #CHANGE: Change sig for the sfd data since the noise model is super high and the tiny values make sense
#=
z_noisy =[]
for n=1:Nr
    zlams = eqws_compute_convmap(true_img, filter_hash, filter_hash; dhc_args...)[3:end]
    push!(z_noisy, zlams)
end
=#
#Compare actual and analytical mean of the ZLambdas
#P = plan_ifft(z_noisy[1][1])
#Sim

empsn = []
for n=1:Nr
    noisyim = rand(noisemod, (Nx, Nx)) .+ true_img
    push!(empsn, DHC_compute_wrapper(noisyim, filter_hash; dhc_args...))
end
coeffsim = hcat(empsn...)'
ncoeffmean = mean(coeffsim, dims=1)

expcoeffmean = my_DHC_compute(true_img, filter_hash, filter_hash, fill(sigma, (Nx, Nx)); dhc_args...)
truecoeff = DHC_compute_wrapper(true_img, filter_hash; dhc_args...)

p = plot(collect(1:length(ncoeffmean)), ncoeffmean[:], label="Empirical Mean")
plot!(collect(1:length(ncoeffmean)), expcoeffmean[:], label="Analytical Mean")
title!("64x64")

S1mask = falses(1192)
S1mask[3:2+Nf] .= true
p = plot(collect(1:Nf), (ncoeffmean[:] - truecoeff)[S1mask], label="Empirical Mean Difference")
plot!(collect(1:Nf), (expcoeffmean[:]  - truecoeff)[S1mask], label="Analytical Mean Difference")
title!("64x64 Diff, S1")

expcoeffmean[:] .- truecoeff
#Without apd
#=
Nx=64
logbool=false
apdbool=false
isobool=false
filter_hash = fink_filter_hash(1, 8, nx=Nx, t=1, Omega=true)
Nf = size(filter_hash["filt_index"])[1]

Nr=40000
truefile = load("scratch_NM/StandardizedExp/Nx64/Data_1000.jld2")
true_img = truefile["true_img"]

sigma = 0.2*mean(true_img)
noisemod = Normal(0.0, sigma)

dhc_args = Dict(:doS2=>false, :doS20=>true, :iso=>isobool, :norm=> false) #Iso #CHANGE: Change sig for the sfd data since the noise model is super high and the tiny values make sense
#=
z_noisy =[]
for n=1:Nr
    zlams = eqws_compute_convmap(true_img, filter_hash, filter_hash; dhc_args...)[3:end]
    push!(z_noisy, zlams)
end
=#
#Compare actual and analytical mean of the ZLambdas
#P = plan_ifft(z_noisy[1][1])
#Sim

empsn = []
for n=1:Nr
    noisyim = rand(noisemod, (Nx, Nx)) .+ true_img
    push!(empsn, DHC_compute_wrapper(noisyim, filter_hash; dhc_args...))
end
coeffsim = hcat(empsn...)'
ncoeffmean = mean(coeffsim, dims=1)

expcoeffmean = my_DHC_compute(true_img, filter_hash, filter_hash, fill(sigma, (Nx, Nx)); dhc_args...)
truecoeff = DHC_compute_wrapper(true_img, filter_hash; dhc_args...)

p = plot(collect(1:length(ncoeffmean)), ncoeffmean[:], label="Empirical Mean")
plot!(collect(1:length(ncoeffmean)), expcoeffmean[:], label="Analytical Mean")
title!("64x64")
=#


#Copying Doug's code to check
Nx=64
truefile = load("scratch_NM/StandardizedExp/Nx64/Data_1000.jld2")
img = truefile["true_img"]
img = img .- mean(img)
doS2 = false
doS20 = true
doiso = false

sigim= fill(0.02, (Nx, Nx))
fhash = fink_filter_hash(1, 8, nx=Nx, t=1, wd=1, Omega=true)
Nf = size(fhash["filt_index"])[1]
i0 = Nf+3
Stry = (my_DHC_compute(img, fhash, fhash, sigim,  doS2=doS2, doS20=doS20, norm=false, iso=doiso))[i0:end]
Nsam = 40000
Smean, Scov = S20_noisecovar(img, fhash, sigim, Nsam, iso=doiso, doS2=doS2, doS20=doS20)
triumask = triu(trues(Nf, Nf))[:]
Strycomp = Stry[triumask]

plot(collect(1:595), Strycomp, label="Analytical")
plot!(collect(1:595), Smean, label="Empirical")

truec = DHC_compute(img, fhash, doS2=doS2, doS20=doS20, norm=false, iso=doiso)[i0:end][triumask]
plot(collect(1:595),0.66*(Strycomp .- truec), label="Analytical Diff")
plot!(collect(1:595), Smean .- truec, label="Empirical Diff")

fullsel = Matrix(I, Nf, Nf)
sel595 = fullsel[triu(trues(Nf, Nf))]
s1inds = findall(sel595 .== 1)

#Code your own noise-biased coefficients
function DHC_compute_noisy(image::Array{Float64,2}, filter_hash::Dict, sigim::Array{Float64,2};
    doS2::Bool=true, doS20::Bool=false, apodize=false, norm=false, iso=false, FFTthreads=1, filter_hash2::Dict=filter_hash, coeff_mask=nothing)
    #Not using coeff_mask here after all. ASSUMES ANY ONE of doS12, doS2 and doS20 are true, when using coeff_mask
    #@assert !iso "Iso not implemented yet"
    Nf = size(filter_hash["filt_index"])[1]

    if apodize
        ap_img = apodizer(image)
        dA = get_dApodizer(image, Dict([(:apodize => apodize)]))
    else
        ap_img = image
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

        # array sizes
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
        if anyrd im_rd_0_1  = Array{Float64, 3}(undef, Nx, Ny, Nf) end
        varim  = sigim.^2

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
        im_fd_0 = fft(norm_im)  # total power=1.0

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
                    psi_pow = sum(f_v.^2)./(Nx*Ny)
                    im_rd_0_1[:,:,f] .= abs2.(P*zarr) + varim.*psi_pow end
                zarr[f_i] .= 0
            end
        end

        append!(out_coeff, iso ? filter_hash["S1_iso_mat"]*S1 : S1)

        if normS1iso
            S1iso = vec(reshape(filter_hash["S1_iso_mat"]*S1,(1,:))*filter_hash["S1_iso_mat"])
        end

        # we stored the abs()^2, so take sqrt (this is faster to do all at once)
        if anyrd im_rd_0_1 .= sqrt.(im_rd_0_1) end

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
                    S2[f1,f2] = sum(abs2.(f_v .* thisim[f_i]))/(Nx*Ny)/normS1pwr
                end
            end
            append!(out_coeff, iso ? Mat2*S2[:] : S2[:])
        end

        # Real domain 2nd order
        if doS20
            Amat = reshape(im_rd_0_1, Nx*Ny, Nf)
            S20  = Amat' * Amat
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
        @assert length(coeff_mask)==length(sall) "The length of the coeff_mask must match the length of the output coefficients"
        return sall[coeff_mask], pred
    else
        return sall
    end

end

Nx=64
logbool=false
apdbool=true
isobool=false
filter_hash = fink_filter_hash(1, 8, nx=Nx, t=1, Omega=true)
Nf = size(filter_hash["filt_index"])[1]

Nr=40000
truefile = load("scratch_NM/StandardizedExp/Nx64/Data_1000.jld2")
true_img = truefile["true_img"]

sigma = 0.2*mean(true_img)
noisemod = Normal(0.0, sigma)

dhc_args = Dict(:doS2=>false, :doS20=>true, :iso=>isobool, :norm=> false) #Iso #CHANGE: Change sig for the sfd data since the noise model is super high and the tiny values make sense
#=
z_noisy =[]
for n=1:Nr
    zlams = eqws_compute_convmap(true_img, filter_hash, filter_hash; dhc_args...)[3:end]
    push!(z_noisy, zlams)
end
=#
#Compare actual and analytical mean of the ZLambdas
#P = plan_ifft(z_noisy[1][1])
#Sim

empsn = []
for n=1:Nr
    noisyim = rand(noisemod, (Nx, Nx)) .+ true_img
    push!(empsn, DHC_compute_wrapper(noisyim, filter_hash; dhc_args...))
end
coeffsim = hcat(empsn...)'
ncoeffmean = mean(coeffsim, dims=1)

expcoeffmean = DHC_compute_noisy(true_img, filter_hash, fill(sigma, (Nx, Nx)); dhc_args...)
truecoeff = DHC_compute_wrapper(true_img, filter_hash; dhc_args...)

p = plot(collect(1:length(ncoeffmean)), ncoeffmean[:], label="Empirical Mean")
plot!(collect(1:length(ncoeffmean)), expcoeffmean[:], label="Analytical Mean")
title!("64x64")

p = plot(collect(1:length(ncoeffmean)), (ncoeffmean[:] - truecoeff), label="Empirical Mean Difference")
plot!(collect(1:length(ncoeffmean)), (expcoeffmean[:]  - truecoeff), label="Analytical Mean Difference")
title!("64x64 Diff")


S1mask = falses(1192)
S2mask = Matrix(I, Nf, Nf)
S1mask[Nf+3:end] .= S2mask[:]
p = plot(collect(1:Nf), (ncoeffmean[:] - truecoeff)[S1mask], label="Empirical Mean Difference")
plot!(collect(1:Nf), (expcoeffmean[:]  - truecoeff)[S1mask], label="Analytical Mean Difference")
title!("64x64 Diff, S1")


another = load("scratch_NM/StandardizedExp/Nx64/Data_10.jld2")
another_img = another["true_img"]
dbn = load("scratch_NM/SavedCovMats/reg_apd_noiso_nologcoeff.jld2")
dbnsfd = dbn["dbncoeffs"]
sfdmean = mean(dbnsfd, dims=1)
ancoeff = DHC_compute_wrapper(another_img, filter_hash; dhc_args...)
S1mask = falses(1192)
S2mask = Matrix(I, Nf, Nf)
S1mask[Nf+3:end] .= S2mask[:]
p = plot(collect(1:Nf), (ncoeffmean[:])[S1mask], label="Empirical Mean", legend=(0.1, 0.9))
plot!(collect(1:Nf), (expcoeffmean[:])[S1mask], label="Analytical Mean", legend=(0.1, 0.9))
plot!(collect(1:Nf), truecoeff[S1mask], label="True Coeffs", legend=(0.1, 0.9))
plot!(collect(1:Nf), ancoeff[S1mask], label="Another Img Coeffs", legend=(0.1, 0.9))
plot!(collect(1:Nf), sfdmean[S1mask], label="SFD Mean", legend=(0.1, 0.9))
title!("64x64, S1")


S1mask = falses(1192)
S2mask = Matrix(I, Nf, Nf)
S2only = trues(Nf, Nf) .- S2mask
S1mask[Nf+3:end] .= triu(S2only)[:]
p = plot(collect(1:count(!iszero, S1mask)), (ncoeffmean[:])[S1mask], label="Empirical Mean", legend=(0.1, 0.9))
plot!(collect(1:count(!iszero, S1mask)), (expcoeffmean[:])[S1mask], label="Analytical Mean", legend=(0.1, 0.9))
plot!(collect(1:count(!iszero, S1mask)), truecoeff[S1mask], label="True Coeffs", legend=(0.1, 0.9))
plot!(collect(1:count(!iszero, S1mask)), ancoeff[S1mask], label="Another Img Coeffs", legend=(0.1, 0.9))
plot!(collect(1:count(!iszero, S1mask)), sfdmean[S1mask], label="SFD Mean", legend=(0.1, 0.9))
title!("64x64, S2Ronly")

p = plot(collect(1:count(!iszero, S1mask)), (ncoeffmean[:] - truecoeff)[S1mask], label="Empirical Mean Difference")
plot!(collect(1:count(!iszero, S1mask)), (expcoeffmean[:]  - truecoeff)[S1mask], label="Analytical Mean Difference")
title!("64x64 Diff, S2R Only")

p = plot(collect(1:count(!iszero, S1mask)), (ncoeffmean[:])[S1mask], label="Empirical Mean")
plot!(collect(1:count(!iszero, S1mask)), (expcoeffmean[:])[S1mask], label="Analytical Mean")
title!("64x64, S2R Only")

S1mask = falses(1192)
S2mask = Matrix(I, Nf, Nf)
S1mask[Nf+3:end] .= S2mask[:]
p = plot(collect(1:Nf), (ncoeffmean[:])[S1mask], label="Empirical Mean")
plot!(collect(1:Nf), (expcoeffmean[:])[S1mask], label="Analytical Mean")
title!("64x64, S1")

S1mask = falses(1192)
S2mask = Matrix(I, Nf, Nf)
S1mask[Nf+3:end] .= S2mask[:]
stdcoeff = std(coeffsim, dims=1)
p = plot(collect(1:Nf), (ncoeffmean[:])[S1mask], color="black", label="Empirical Mean")
plot!(collect(1:Nf), (ncoeffmean[:] - stdcoeff[:])[S1mask], color="black", marker="--")
plot!(collect(1:Nf), (ncoeffmean[:] + stdcoeff[:])[S1mask], color="black", marker="--", label="Std envelope")
plot!(collect(1:Nf), (expcoeffmean[:])[S1mask], label="Analytical Mean")
plot!(collect(1:Nf), (truecoeff[:])[S1mask],  label="True Coefficients")
title!("64x64, S1")



#Compare actual and analytical mean of the noisy coeffs


#Compare actual and analytical covariance of the noisy coeffs


#Compare the Gaussian with the same covariance with the actual empirical distribution
#For coeffs


#Implement it somehow by combining with the prior.
