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
#=
function DHC_compute_noisy(image::Array{Float64,2}, filter_hash::Dict, sigim::Array{Float64,2};
    doS2::Bool=true, doS20::Bool=false, apodize=false, norm=false, iso=false, FFTthreads=1, filter_hash2::Dict=filter_hash, coeff_mask=nothing)
    #Not using coeff_mask here after all. ASSUMES ANY ONE of doS12, doS2 and doS20 are true, when using coeff_mask
    #@assert !iso "Iso not implemented yet"
    Nf = size(filter_hash["filt_index"])[1]
    Nx = size(true_img)[1]

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
        @assert count(!iszero, coeff_mask[1:Nf+2])==0 "This function only handles S20r"
        @assert length(coeff_mask)==length(sall) "Mask must have the same length as the total number of coefficients"
        s20rmask = coeff_mask[Nf+3:end]
        return sall[coeff_mask], cov[s20rmask, s20rmask]
    else
        return sall, cov
    end

end
=#


Nx=64
logbool=false
apdbool=true
isobool=false
filter_hash = fink_filter_hash(1, 8, nx=Nx, t=1, Omega=true)
Nf = size(filter_hash["filt_index"])[1]

Nr=40000
truefile = load("scratch_NM/StandardizedExp/Nx64/Data_1000.jld2")
true_img = truefile["true_img"]
init = truefile["init"]
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

expcoeffmean, _ = DHC_compute_noisy(true_img, filter_hash, fill(sigma, (Nx, Nx)); dhc_args...)
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

##Evidence for SFD Mean being all but useless.
another = load("scratch_NM/StandardizedExp/Nx64/Data_10.jld2")
another_img = another["true_img"]
dbn = load("scratch_NM/SavedCovMats/reg_apd_noiso_nologcoeff.jld2")
dbnsfd = dbn["dbncoeffs"]
sfdmean = mean(dbnsfd, dims=1)
ancoeff = DHC_compute_wrapper(another_img, filter_hash; dhc_args...)
S1mask = falses(1192)
S2mask = Matrix(I, Nf, Nf)
S1mask[Nf+3:end] .= S2mask[:]
dhc_args = Dict(:doS2=>false, :doS20=>true, :iso=>isobool, :norm=> false) #Iso #CHANGE: Change sig for the sfd data since the noise model is super high and the tiny values make sense


initcoeff = DHC_compute_wrapper(init, filter_hash; dhc_args...)
reconcoeff = DHC_compute_noisy(init, filter_hash; dhc_args...)
p = plot(collect(1:Nf), (ncoeffmean[:])[S1mask], label="Empirical Mean", legend=(0.1, 0.9))
plot!(collect(1:Nf), (expcoeffmean[:])[S1mask], label="Analytical Mean", legend=(0.1, 0.9))
plot!(collect(1:Nf), truecoeff[S1mask], label="True Coeffs", legend=(0.1, 0.9))
#plot!(collect(1:Nf), ancoeff[S1mask], label="Another Img Coeffs", legend=(0.1, 0.9))
#plot!(collect(1:Nf), sfdmean[S1mask], label="SFD Mean", legend=(0.1, 0.9))
plot!(collect(1:Nf), initcoeff[S1mask], label="Init", legend=(0.1, 0.9))
title!("64x64, S1")
(initcoeff[S1mask] .- truecoeff[S1mask]) ./ truecoeff[S1mask]
(sfdmean[S1mask] .- truecoeff[S1mask]) ./ truecoeff[S1mask]





S1mask = falses(1192)
S2mask = Matrix(I, Nf, Nf)
S2only = trues(Nf, Nf) .- S2mask
S1mask[Nf+3:end] .= triu(S2only)[:]
#Compare actual and analytical mean of the noisy coeffs
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

coeffmask = falses(1192)
S2mask = trues(Nf, Nf)
coeffmask[Nf+3:end] .= S2mask[:]
dhc_args[:apodize] = true
thcoeffmean, thcov = DHC_compute_noisy(true_img, filter_hash, fill(sigma, (Nx, Nx)); coeff_mask = coeffmask, dhc_args...)

coeffmask = falses(1192)
S2mask = triu(trues(Nf, Nf))
coeffmask[Nf+3:end] .= S2mask[:]
dhc_args[:apodize] = true
thcoeffmean, thcov = DHC_compute_noisy(true_img, filter_hash, fill(sigma, (Nx, Nx)); coeff_mask = coeffmask, dhc_args...)
rsmean = reshape(thcoeffmean, (1, length(thcoeffmean)))
#Compare actual and analytical covariance of the noisy coeffs

empcov = (coeffsim[:, coeffmask] .- rsmean)' * (coeffsim[:, coeffmask] .- rsmean) ./(Nr-1)
size(thcoeffmean)
(thcov .- empcov)./empcov

##Version with apodization everywhere??
Nx=64
logbool=false
apdbool=true
isobool=false
filter_hash = fink_filter_hash(1, 8, nx=Nx, t=1, Omega=true)
Nf = size(filter_hash["filt_index"])[1]

Nr=40000
truefile = load("scratch_NM/StandardizedExp/Nx64/Data_1000.jld2")
true_img = truefile["true_img"]
init = truefile["init"]
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

expcoeffmean, _ = DHC_compute_noisy(true_img, filter_hash, fill(sigma, (Nx, Nx)); dhc_args...)
truecoeff = DHC_compute_wrapper(true_img, filter_hash; dhc_args...)
initexpcoeffmean, _ = DHC_compute_noisy(true_img, filter_hash, fill(sigma, (Nx, Nx)); dhc_args...)

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

##Evidence for SFD Mean being all but useless.
another = load("scratch_NM/StandardizedExp/Nx64/Data_10.jld2")
another_img = another["true_img"]
dbn = load("scratch_NM/SavedCovMats/reg_apd_noiso_nologcoeff.jld2")
dbnsfd = dbn["dbncoeffs"]
sfdmean = mean(dbnsfd, dims=1)
ancoeff = DHC_compute_wrapper(another_img, filter_hash; dhc_args...)
S1mask = falses(1192)
S2mask = Matrix(I, Nf, Nf)
S1mask[Nf+3:end] .= S2mask[:]
dhc_args = Dict(:doS2=>false, :doS20=>true, :iso=>isobool, :norm=> false) #Iso #CHANGE: Change sig for the sfd data since the noise model is super high and the tiny values make sense


initcoeff = DHC_compute_wrapper(init, filter_hash; dhc_args...)
reconcoeff = DHC_compute_noisy(init, filter_hash; dhc_args...)
p = plot(collect(1:Nf), (ncoeffmean[:])[S1mask], label="Empirical Mean", legend=(0.1, 0.9))
plot!(collect(1:Nf), (expcoeffmean[:])[S1mask], label="Analytical Mean", legend=(0.1, 0.9))
plot!(collect(1:Nf), truecoeff[S1mask], label="True Coeffs", legend=(0.1, 0.9))
#plot!(collect(1:Nf), ancoeff[S1mask], label="Another Img Coeffs", legend=(0.1, 0.9))
#plot!(collect(1:Nf), sfdmean[S1mask], label="SFD Mean", legend=(0.1, 0.9))
plot!(collect(1:Nf), initcoeff[S1mask], label="Init", legend=(0.1, 0.9))
title!("64x64, S1")
(initcoeff[S1mask] .- truecoeff[S1mask]) ./ truecoeff[S1mask]
(sfdmean[S1mask] .- truecoeff[S1mask]) ./ truecoeff[S1mask]

##Try denoising assuming you have true image???
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
sigma = loaddf["std"]

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
#sfdall = readsfd(Nx, logbool=logbool)
#dbnocffs = log.(get_dbn_coeffs(sfdall, filter_hash, dhc_args))
dbn = load("scratch_NM/SavedCovMats/reg_apd_noiso_nologcoeff.jld2")
dbnocffs = dbn["dbncoeffs"]
fmean = mean(dbnocffs, dims=1)
size(dbnocffs)

coeffmask = falses(2+Nf+Nf^2)
coeffmask[Nf+3:end] .= Diagonal(trues(Nf))[:] #triu(trues(Nf, Nf))[:]
thcoeffmean, thcov = DHC_compute_noisy(true_img, filter_hash, fill(sigma, (Nx, Nx)); coeff_mask = coeffmask, dhc_args...)
empsn = []
Nr=10000
noisemod = Normal(0.0, sigma)
for n=1:Nr
    noisyim = rand(noisemod, (Nx, Nx)) .+ true_img
    push!(empsn, DHC_compute_wrapper(noisyim, filter_hash; dhc_args...))
end
coeffsim = hcat(empsn...)'
ncoeffmean = mean(coeffsim, dims=1)
empcov = (coeffsim[:, coeffmask] .- reshape(ncoeffmean[coeffmask],(1, count(!iszero, coeffmask))))' * (coeffsim[:, coeffmask] .- reshape(ncoeffmean[coeffmask], (1, count(!iszero, coeffmask)))) ./(Nr-1)

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
recon_settings["coeff_mask_sfd"] = coeffmask
recon_settings["coeff_mask_init"] = coeff_maskinit

fcov = (dbnocffs .- fmean)' * (dbnocffs .- fmean) ./(size(dbnocffs)[1] -1)
fcovinv1 = invert_covmat(thcov, recon_settings["eps_value_sfd"])
fcovinv2 = invert_covmat(fcov[coeff_maskinit, coeff_maskinit], recon_settings["eps_value_init"])

if logbool
    error("Not impl for log im")
end
ftarginit = DHC_compute_wrapper(init, filter_hash; dhc_args...)
ftarginit = ftarginit[coeff_maskinit]

#Weight given to terms
lval2 = 0.0
lval3 = 1.0

println("Regularizer Lambda=", round(lval3, sigdigits=3))
#input::Array{Float64, 2}, filter_hash::Dict, s_targ_mean::Array{Float64, 1}, s_targ_invcov, dhc_args, LossFunc, dLossFunc;
#FFTthreads::Int=1, optim_settings=Dict([("iterations", 10)]), lambda=0.001, func_specific_params=nothing
func_specific_params = Dict([(:reg_input=> init), (:lambda2=> lval2), (:lambda3=> lval3), (:coeff_mask1=> coeffmask), (:target1=>thcoeffmean), (:invcov1=>fcovinv1), (:coeff_mask2=> coeff_maskinit), (:target2=>ftarginit), (:invcov2=>fcovinv2)])


recon_settings["fname_save"] = fname_save * ".jld2"
recon_settings["optim_settings"] = optim_settings

if logbool
    error("Not implemented here")
end

white_noise = randn(Nx, Nx)
res, recon_img = image_recon_derivsum_custom(init, filter_hash, dhc_args, ReconFuncs.Loss3Gaussian,
    ReconFuncs.dLoss3Gaussian!; optim_settings=optim_settings, func_specific_params)
#save(settings["fname_save"], Dict("true_img"=>true_img, "init"=>noisy_init, "recon"=>recon_img, "dict"=>settings, "trace"=>Optim.trace(res), "coeff_mask"=>coeff_mask, "fhash"=>fhash, "dhc_args"=>dhc_args))


function loss1(inp_img)
    s_curr = DHC_compute_wrapper(inp_img, filter_hash, norm=false; dhc_args...)
    s_curr1 = Data_Utils.fnlog(s_curr[coeffmask])
    diff1 = s_curr1 - thcoeffmean
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
plot!(log.(kbins), log.(initps), label="White Noise")
plot!(log.(kbins), log.(smoothps), label="Smoothed Init (CoeffLoss)")
plot!(title="P(k): LogCoeffs 3 Loss")
xlabel!("lnk")
ylabel!("lnP(k)")
p2 = heatmap(apodizer(true_img), title="True Img", clim=clim)
p4= heatmap(apodizer(init), title="Init Img", clim=clim)
p5= heatmap(apodizer(imfilter(init, Kernel.gaussian(0.8))), title="Smoothed init", clim=clim)
p6 = heatmap(apodizer(recon_img)- apodizer(true_img), title="Residual: Recon - True")
p7 = heatmap(apodizer(apdsmoothed)- apodizer(true_img), title="Residual: SmoothedInit - True")

struesel = Data_Utils.fnlog(DHC_compute_wrapper(true_img, filter_hash, norm=false; dhc_args...))
sinitsel = Data_Utils.fnlog(DHC_compute_wrapper(init, filter_hash, norm=false; dhc_args...))
ssmoothsel = Data_Utils.fnlog(DHC_compute_wrapper(imfilter(init, Kernel.gaussian(0.8)), filter_hash, norm=false; dhc_args...))
sreconsel = Data_Utils.fnlog(DHC_compute_wrapper(recon_img, filter_hash, norm=false; dhc_args...))
slims = (minimum(struesel[JS1ind]), maximum(struesel[JS1ind]))
cg = cgrad([:blue, :white, :red])
truephi, trueomg = round(struesel[2+filter_hash["phi_index"]], sigdigits=3), round(struesel[2+filter_hash["Omega_index"]], sigdigits=3)
reconphi, reconomg = round(sreconsel[2+filter_hash["phi_index"]], sigdigits=3), round(sreconsel[2+filter_hash["Omega_index"]], sigdigits=3)
smoothphi, smoothomg = round(sinitsel[2+filter_hash["phi_index"]], sigdigits=3), round(sinitsel[2+filter_hash["Omega_index"]], sigdigits=3)

p8 = heatmap(struesel[JS1ind], title="True Coeffs ϕ=" * string(truephi) * "Ω=" * string(trueomg) , clims=slims, c=cg)
p9 = heatmap(sreconsel[JS1ind], title="Recon Coeffs ϕ=" * string(reconphi) * "Ω=" * string(reconomg), clims=slims, c=cg)
p10 = heatmap(ssmoothsel[JS1ind], title="Init Coeffs ϕ=" * string(smoothphi) * "Ω=" * string(smoothomg), clims=slims, c=cg)
p = plot(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, layout=(5, 2), size=(1800, 2400))
savefig(p, "scratch_NM/NewWrapper/5-23/Case1.png")
save("scratch_NM/NewWrapper/4-28/1000_C5_S2sfd_highjinit.jld2", Dict("true_img"=>true_img, "init"=>init, "recon"=>recon_img, "dict"=>recon_settings, "trace"=>Optim.trace(res), "coeff_mask"=>[coeff_masksfd, coeff_maskinit], "fhash"=>filter_hash, "dhc_args"=>dhc_args, "func_specific_params"=>func_specific_params))



#Compare the Gaussian with the same covariance with the actual empirical distribution
#For coeffs


#Implement it somehow by combining with the prior.





##5-24: Bootstrapping procedure:
##Assumption 1: Using the real ZT
#Is empcov just GG'?
numfile = 1000
direc = "scratch_NM/StandardizedExp/Nx64/"
datfile = direc * "Data_" * string(numfile) * ".jld2" #Replace w SLURM array
loaddf = load(datfile)
true_img = loaddf["true_img"]
sigma = loaddf["std"]
Nx=64
Nr=10000
noisemod = Normal(0.0, sigma)
empsn = []
isobool = false
apdbool = false
dhc_args = Dict(:doS2=>false, :doS20=>true, :apodize=>apdbool, :iso=>isobool)
filter_hash = fink_filter_hash(1, 8, nx=Nx, t=1, wd=1, Omega=true)
for n=1:Nr
    noisyim = rand(noisemod, (Nx, Nx)) .+ true_img
    push!(empsn, DHC_compute_wrapper(noisyim, filter_hash; dhc_args...))
end
coeffsim = hcat(empsn...)'
ncoeffmean = mean(coeffsim, dims=1)
empcov = (coeffsim .- ncoeffmean)' * (coeffsim .- ncoeffmean) / (Nr-1)

heatmap(log.((x->maximum([x, 0])).(empcov)))

dS20dp = wst_S20_deriv(true_img, filter_hash)
Nf = length(filter_hash["filt_index"])
G = reshape(dS20dp, Nx*Nx, Nf*Nf) .* fill(sigma, (Nx*Nx, 1))
pred = G'*G
s20rall = falses(2+Nf+Nf^2)
s20rall[Nf+3:end] .= true
fracerr = (pred .- empcov[s20rall, s20rall]) ./ empcov[s20rall, s20rall]


#With mean subtraction
numfile = 1000
direc = "scratch_NM/StandardizedExp/Nx64/"
datfile = direc * "Data_" * string(numfile) * ".jld2" #Replace w SLURM array
loaddf = load(datfile)
true_img = loaddf["true_img"]
true_img = true_img .- mean(true_img)
sigma = loaddf["std"]
Nx=64
Nr=10000
noisemod = Normal(0.0, sigma)
empsn = []
isobool = false
apdbool = false
dhc_args = Dict(:doS2=>false, :doS20=>true, :apodize=>apdbool, :iso=>isobool)
filter_hash = fink_filter_hash(1, 8, nx=Nx, t=1, wd=1, Omega=true)
for n=1:Nr
    noisyim = rand(noisemod, (Nx, Nx)) .+ true_img
    push!(empsn, DHC_compute_wrapper(noisyim, filter_hash; dhc_args...))
end
coeffsim = hcat(empsn...)'
ncoeffmean = mean(coeffsim, dims=1)
empcov = (coeffsim .- ncoeffmean)' * (coeffsim .- ncoeffmean) / (Nr-1)

heatmap(log.((x->maximum([x, 0])).(empcov)))

dS20dp = wst_S20_deriv(true_img, filter_hash)
Nf = length(filter_hash["filt_index"])
G = reshape(dS20dp, Nx*Nx, Nf*Nf) .* fill(sigma, (Nx*Nx, 1))
pred = G'*G
s20rall = falses(2+Nf+Nf^2)
s20rall[Nf+3:end] .= true
fracerr = (pred .- empcov[s20rall, s20rall]) ./ empcov[s20rall, s20rall]

#With mean subtraction and rnad*sigma code
numfile = 1000
direc = "scratch_NM/StandardizedExp/Nx64/"
datfile = direc * "Data_" * string(numfile) * ".jld2" #Replace w SLURM array
loaddf = load(datfile)
true_img = loaddf["true_img"]
true_img = true_img .- mean(true_img)
sigma = loaddf["std"]
Nx=64
Nr=10000
empsn = []
isobool = false
apdbool = false
dhc_args = Dict(:doS2=>false, :doS20=>true, :apodize=>apdbool, :iso=>isobool)
filter_hash = fink_filter_hash(1, 8, nx=Nx, t=1, wd=1, Omega=true)
for n=1:Nr
    noisyim = randn((Nx, Nx)).*sigma .+ true_img
    push!(empsn, DHC_compute_wrapper(noisyim, filter_hash; dhc_args...))
end
coeffsim = hcat(empsn...)'
ncoeffmean = mean(coeffsim, dims=1)
empcov = (coeffsim .- ncoeffmean)' * (coeffsim .- ncoeffmean) / (Nr-1)

heatmap(log.((x->maximum([x, 0])).(empcov)))

dS20dp = wst_S20_deriv(true_img, filter_hash)
Nf = length(filter_hash["filt_index"])
G = reshape(dS20dp, Nx*Nx, Nf*Nf) .* fill(sigma, (Nx*Nx, 1))
pred = G'*G
s20rall = falses(2+Nf+Nf^2)
s20rall[Nf+3:end] .= true
fracerr = (pred .- empcov[s20rall, s20rall]) ./ empcov[s20rall, s20rall]
empcov[s20rall, s20rall]

Random.seed!(34)
list = []
for u=1:2
    push!(list, randn(3))
end
coeffsim ./reshape(DHC_compute_wrapper(true_img, filter_hash; dhc_args...), (1, 1192))
coeffmask = falses(2+Nf+Nf^2)
coeffmask[Nf+3:end] .= triu(trues(Nf, Nf))[:]
thcoeffmean, thcov = DHC_compute_noisy(true_img, filter_hash, fill(sigma, (Nx, Nx)); coeff_mask = coeffmask, dhc_args...)

ratio = coeffsim[:, coeffmask] ./ reshape(DHC_compute_wrapper(true_img, filter_hash; dhc_args...), (1, 1192))[:, coeffmask]
ratioth = thcoeffmean ./ DHC_compute_wrapper(true_img, filter_hash; dhc_args...)[coeffmask]

maximum(abs.((thcoeffmean .- mean(coeffsim[:, coeffmask], dims=1)[:])./ mean(coeffsim[:, coeffmask], dims=1)[:]))
mean(abs.((thcoeffmean .- mean(coeffsim[:, coeffmask], dims=1)[:])./ mean(coeffsim[:, coeffmask], dims=1)[:]))


fracerr = (thcov .- empcov[coeffmask, coeffmask]) ./empcov[coeffmask, coeffmask]
maximum(abs.(fracerr))
mean(abs.(fracerr))

heatmap(fracerr, title="Frac Deviation of Covariance")


#Without mean subtraction
numfile = 1000
direc = "scratch_NM/StandardizedExp/Nx64/"
datfile = direc * "Data_" * string(numfile) * ".jld2" #Replace w SLURM array
loaddf = load(datfile)
true_img = loaddf["true_img"]
true_img = true_img
sigma = loaddf["std"]
Nx=64
Nr=10000
empsn = []
isobool = false
apdbool = false
dhc_args = Dict(:doS2=>false, :doS20=>true, :apodize=>apdbool, :iso=>isobool)
filter_hash = fink_filter_hash(1, 8, nx=Nx, t=1, wd=1, Omega=true)
for n=1:Nr
    noisyim = randn((Nx, Nx)).*sigma .+ true_img
    push!(empsn, DHC_compute_wrapper(noisyim, filter_hash; dhc_args...))
end
coeffsim = hcat(empsn...)'
ncoeffmean = mean(coeffsim, dims=1)
empcov = (coeffsim .- ncoeffmean)' * (coeffsim .- ncoeffmean) / (Nr-1)

heatmap(log.((x->maximum([x, 0])).(empcov)))

dS20dp = wst_S20_deriv(true_img, filter_hash)
Nf = length(filter_hash["filt_index"])
G = reshape(dS20dp, Nx*Nx, Nf*Nf) .* fill(sigma, (Nx*Nx, 1))
pred = G'*G
s20rall = falses(2+Nf+Nf^2)
s20rall[Nf+3:end] .= true
fracerr = (pred .- empcov[s20rall, s20rall]) ./ empcov[s20rall, s20rall]
empcov[s20rall, s20rall]

Random.seed!(34)
list = []
for u=1:2
    push!(list, randn(3))
end
coeffsim ./reshape(DHC_compute_wrapper(true_img, filter_hash; dhc_args...), (1, 1192))
coeffmask = falses(2+Nf+Nf^2)
coeffmask[Nf+3:end] .= triu(trues(Nf, Nf))[:]
thcoeffmean, thcov = DHC_compute_noisy(true_img, filter_hash, fill(sigma, (Nx, Nx)); coeff_mask = coeffmask, dhc_args...)

ratio = coeffsim[:, coeffmask] ./ reshape(DHC_compute_wrapper(true_img, filter_hash; dhc_args...), (1, 1192))[:, coeffmask]
ratioth = thcoeffmean ./ DHC_compute_wrapper(true_img, filter_hash; dhc_args...)[coeffmask]

maximum(abs.((thcoeffmean .- mean(coeffsim[:, coeffmask], dims=1)[:])./ mean(coeffsim[:, coeffmask], dims=1)[:]))
mean(abs.((thcoeffmean .- mean(coeffsim[:, coeffmask], dims=1)[:])./ mean(coeffsim[:, coeffmask], dims=1)[:]))


fracerr = (thcov .- empcov[coeffmask, coeffmask]) ./empcov[coeffmask, coeffmask]
maximum(abs.(fracerr))
mean(abs.(fracerr))

#With the noisy image as starting point
numfile = 1000
direc = "scratch_NM/StandardizedExp/Nx64/"
datfile = direc * "Data_" * string(numfile) * ".jld2" #Replace w SLURM array
loaddf = load(datfile)
true_img = loaddf["true_img"]
init = loaddf["init"]
true_img = true_img
sigma = loaddf["std"]
Nx=64
Nr=10000
empsn = []
isobool = false
apdbool = false
dhc_args = Dict(:doS2=>false, :doS20=>true, :apodize=>apdbool, :iso=>isobool)
filter_hash = fink_filter_hash(1, 8, nx=Nx, t=1, wd=1, Omega=true)
for n=1:Nr
    noisyim = randn((Nx, Nx)).*sigma .+ true_img
    push!(empsn, DHC_compute_wrapper(noisyim, filter_hash; dhc_args...))
end
coeffsim = hcat(empsn...)'
ncoeffmean = mean(coeffsim, dims=1)
empcov = (coeffsim .- ncoeffmean)' * (coeffsim .- ncoeffmean) / (Nr-1)

heatmap(log.((x->maximum([x, 0])).(empcov)))

dS20dp = wst_S20_deriv(true_img, filter_hash)
Nf = length(filter_hash["filt_index"])
G = reshape(dS20dp, Nx*Nx, Nf*Nf) .* fill(sigma, (Nx*Nx, 1))
pred = G'*G
s20rall = falses(2+Nf+Nf^2)
s20rall[Nf+3:end] .= true
fracerr = (pred .- empcov[s20rall, s20rall]) ./ empcov[s20rall, s20rall]
empcov[s20rall, s20rall]

Random.seed!(34)
list = []
for u=1:2
    push!(list, randn(3))
end
coeffsim ./reshape(DHC_compute_wrapper(true_img, filter_hash; dhc_args...), (1, 1192))
coeffmask = falses(2+Nf+Nf^2)
coeffmask[Nf+3:end] .= triu(trues(Nf, Nf))[:]
thcoeffmean, thcov = DHC_compute_noisy(init, filter_hash, fill(sigma, (Nx, Nx)); coeff_mask = coeffmask, dhc_args...)

ratio = coeffsim[:, coeffmask] ./ reshape(DHC_compute_wrapper(true_img, filter_hash; dhc_args...), (1, 1192))[:, coeffmask]
ratioth = thcoeffmean ./ DHC_compute_wrapper(true_img, filter_hash; dhc_args...)[coeffmask]

maximum(abs.((thcoeffmean .- mean(coeffsim[:, coeffmask], dims=1)[:])./ mean(coeffsim[:, coeffmask], dims=1)[:]))
mean(abs.((thcoeffmean .- mean(coeffsim[:, coeffmask], dims=1)[:])./ mean(coeffsim[:, coeffmask], dims=1)[:]))


fracerr = (thcov .- empcov[coeffmask, coeffmask]) ./empcov[coeffmask, coeffmask]
maximum(abs.(fracerr))
mean(abs.(fracerr))


##Case2
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
dbfile = load("scratch_NM/SavedCovMats/reg_apd_noiso_nologcoeff.jld2")
dbnocffs = log.(dbfile["dbncoeffs"])
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
lval3= 0.00

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

#save("scratch_NM/NewWrapper/4-28/1000_C5_S2sfd_highjinit.jld2", Dict("true_img"=>true_img, "init"=>init, "recon"=>recon_img, "dict"=>recon_settings, "trace"=>Optim.trace(res), "coeff_mask"=>[coeff_masksfd, coeff_maskinit], "fhash"=>filter_hash, "dhc_args"=>dhc_args, "func_specific_params"=>func_specific_params))



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
savefig(p, "scratch_NM/NewWrapper/5-23/Case2_It0.png")

#STEP 2
sigma= loaddf["std"]
coeffmask = falses(2+Nf+Nf^2)
coeffmask[Nf+3:end] .= triu(trues(Nf, Nf))[:]
thcoeffmean, thcov = DHC_compute_noisy(recon_img, filter_hash, fill(sigma, (Nx, Nx)); coeff_mask = coeffmask, dhc_args...)

empsn = []
Nr=10000
noisemod = Normal(0.0, sigma)
for n=1:Nr
    noisyim = rand(noisemod, (Nx, Nx)) .+ true_img
    push!(empsn, DHC_compute_wrapper(noisyim, filter_hash; dhc_args...))
end
coeffsim = hcat(empsn...)'
ncoeffmean = mean(coeffsim, dims=1)
empcov = (coeffsim[:, coeffmask] .- reshape(ncoeffmean[coeffmask],(1, count(!iszero, coeffmask))))' * (coeffsim[:, coeffmask] .- reshape(ncoeffmean[coeffmask], (1, count(!iszero, coeffmask)))) ./(Nr-1)


JS1ind = J_S1indices([0, 1, 2, 3], filter_hash)

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
recon_settings["coeff_mask_sfd"] = coeffmask
recon_settings["coeff_mask_init"] = coeff_maskinit

fcov = (dbnocffs .- fmean)' * (dbnocffs .- fmean) ./(size(dbnocffs)[1] -1)
fcovinv1 = invert_covmat(thcov, recon_settings["eps_value_sfd"])
fcovinv2 = invert_covmat(fcov[coeff_maskinit, coeff_maskinit], recon_settings["eps_value_init"])

if logbool
    error("Not impl for log im")
end
ftarginit = DHC_compute_wrapper(init, filter_hash; dhc_args...)
ftarginit = ftarginit[coeff_maskinit]

#Weight given to terms
lval2 = 0.0
lval3 = 1.0

println("Regularizer Lambda=", round(lval3, sigdigits=3))
#input::Array{Float64, 2}, filter_hash::Dict, s_targ_mean::Array{Float64, 1}, s_targ_invcov, dhc_args, LossFunc, dLossFunc;
#FFTthreads::Int=1, optim_settings=Dict([("iterations", 10)]), lambda=0.001, func_specific_params=nothing
func_specific_params = Dict([(:reg_input=> init), (:lambda2=> lval2), (:lambda3=> lval3), (:coeff_mask1=> coeffmask), (:target1=>thcoeffmean), (:invcov1=>fcovinv1), (:coeff_mask2=> coeff_maskinit), (:target2=>ftarginit), (:invcov2=>fcovinv2)])


recon_settings["fname_save"] = fname_save * ".jld2"
recon_settings["optim_settings"] = optim_settings

if logbool
    error("Not implemented here")
end

white_noise = randn(Nx, Nx)
res, recon_img2 = image_recon_derivsum_custom(recon_img, filter_hash, dhc_args, ReconFuncs.Loss3Gaussian,
    ReconFuncs.dLoss3Gaussian!; optim_settings=optim_settings, func_specific_params)
#save(settings["fname_save"], Dict("true_img"=>true_img, "init"=>noisy_init, "recon"=>recon_img, "dict"=>settings, "trace"=>Optim.trace(res), "coeff_mask"=>coeff_mask, "fhash"=>fhash, "dhc_args"=>dhc_args))


function loss1(inp_img)
    s_curr = DHC_compute_wrapper(inp_img, filter_hash, norm=false; dhc_args...)
    s_curr1 = Data_Utils.fnlog(s_curr[coeffmask])
    diff1 = s_curr1 - thcoeffmean
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

init = recon_img
println("Loss Term 1: SFDTarg")
println("True", loss1(true_img))
println("Init", loss1(init))
println("Recon", loss1(recon_img2))

println("Loss Term 2: SmoothedInitTarg")
println("True", loss2(true_img))
println("Init", loss2(init))
println("Recon", loss2(recon_img2))

println("Loss Term 3: Reg")
println("True", loss3(true_img))
println("Init", loss3(init))
println("Recon", loss3(recon_img2))

apdsmoothed = apodizer(imfilter(init, Kernel.gaussian(0.8)))
kbins= convert(Array{Float64}, collect(1:32))
clim=(minimum(apodizer(true_img)), maximum(apodizer(true_img)))
true_ps = Data_Utils.calc_1dps(apodizer(true_img), kbins)
initps = Data_Utils.calc_1dps(apodizer(init), kbins)
recps = Data_Utils.calc_1dps(apodizer(recon_img2), kbins)
smoothps = Data_Utils.calc_1dps(apdsmoothed, kbins)
p1 = heatmap(apodizer(recon_img2), title="Recon", clim=clim)
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
residual = apodizer(recon_img2)- apodizer(true_img)
rlims = (minimum(residual), maximum(residual))
#symmax = maximum([abs(minimum(residual)), maximum(residual)])
#rg = cgrad(:bwr, [-symmax/2.0, symmax/2.0])
p6 = heatmap(residual, title="Residual: Recon - True", clims=rlims, c=:bwr)
p7 = heatmap(apdsmoothed- apodizer(true_img), title="Residual: SmoothedInit - True", clims=rlims, c=:bwr)

struesel = Data_Utils.fnlog(DHC_compute_wrapper(true_img, filter_hash, norm=false; dhc_args...))
sinitsel = Data_Utils.fnlog(DHC_compute_wrapper(init, filter_hash, norm=false; dhc_args...))
ssmoothsel = Data_Utils.fnlog(DHC_compute_wrapper(imfilter(init, Kernel.gaussian(0.8)), filter_hash, norm=false; dhc_args...))
sreconsel = Data_Utils.fnlog(DHC_compute_wrapper(recon_img2, filter_hash, norm=false; dhc_args...))
slims = (minimum(struesel[JS1ind]), maximum(struesel[JS1ind]))
cg = cgrad([:blue, :white, :red])
truephi, trueomg = round(struesel[2+filter_hash["phi_index"]], sigdigits=3), round(struesel[2+filter_hash["Omega_index"]], sigdigits=3)
reconphi, reconomg = round(sreconsel[2+filter_hash["phi_index"]], sigdigits=3), round(sreconsel[2+filter_hash["Omega_index"]], sigdigits=3)
smoothphi, smoothomg = round(ssmoothsel[2+filter_hash["phi_index"]], sigdigits=3), round(ssmoothsel[2+filter_hash["Omega_index"]], sigdigits=3)

p8 = heatmap(struesel[JS1ind], title="True Coeffs ϕ=" * string(truephi) * "Ω=" * string(trueomg) , clims=slims, c=cg)
p9 = heatmap(sreconsel[JS1ind], title="Recon Coeffs ϕ=" * string(reconphi) * "Ω=" * string(reconomg), clims=slims, c=cg)
p10 = heatmap(ssmoothsel[JS1ind], title="Smooth Init ϕ=" * string(smoothphi) * "Ω=" * string(smoothomg), clims=slims, c=cg)
p = plot(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, layout=(5, 2), size=(1800, 2400))
savefig(p, "scratch_NM/NewWrapper/5-23/Case2_It1.png")



##5-23
function sqrtplot()
    σ = 0.05 .* collect(0:20)

    # real randoms
    ran = randn(1000000)
    diff = [mean(sqrt.(abs.(1 .+ ran*fac))) for fac in σ]

    # complex randoms (variance 1, so 0.5 real and 0.5 imaginary)
    cran = randn(ComplexF64,1000000)
    cdiff = [mean((abs.(1 .+ cran*fac))) for fac in σ]

    p=plot(σ ,diff)
    plot!(p, σ, cdiff, label="cdiff")

    # sqrt(1+d) = 1 + (1/2)d - (1/8)d^2 + (1/16)d^3 -(5/128)d^4 + ...
    plot!(p, σ, 1 .-(σ.^2/8), label="real Taylor")

    # variance of the complex power -- this is needed to correct mean power
    # 2 from expansion of (1+d)^2, then squared for variance, but /2 bc half of power in real direction
    myvar = 1^2 * 4/2 .* (σ.^2)

    # The 1/48 is by eye...
    plot!(p, σ, sqrt.(1 .+(σ.^2)) -myvar/8 +(1/48).*myvar.^1.5, label="complex Taylor")


    display(p)
    return
end


sqrtplot()


#Actual square root correction check

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


function DHC_compute_S20r_noisy_so2(image::Array{Float64,2}, filter_hash::Dict, sigim::Array{Float64,2};
    doS2::Bool=true, doS20::Bool=false, apodize=false, norm=false, iso=false, FFTthreads=1, filter_hash2::Dict=filter_hash, coeff_mask=nothing)
    #Not using coeff_mask here after all. ASSUMES ANY ONE of doS12, doS2 and doS20 are true, when using coeff_mask
    #@assert !iso "Iso not implemented yet"
    Nf = size(filter_hash["filt_index"])[1]
    Nx = size(true_img)[1]

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
                    psi_pow = sum(f_v.^2)./(Nx*Ny)
                    im_rd_0_1[:,:,f] .= abs2.(P*zarr)
                    rsψmat[f, :, :] = realspace_filter(Nx, f_i, f_v)
                    frealψ = Pf*(real.(rsψmat[f, :, :]))
                    fimψ = Pf*(imag.(rsψmat[f, :, :]))
                    ψsqfac[f, :, :, 1] = real.(rsψmat[f, :, :]) #check that real
                    ψsqfac[f, :, :, 2] = imag.(rsψmat[f, :, :])

                    Iψfac[f, :, :, 1] = P*(im_fd_0 .* frealψ) #(I ✪ ψR)
                    Iψfac[f, :, :, 2] = P*(im_fd_0 .* fimψ)   #(I ✪ ψI)
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
                sozoterms[f1, :, :, 1] = P*(fvar .* powrs) ./ im_rd_0_1[:, :, f1]                   #(σ2 ✪ |ψ|^2)/|I ✪ ψ|
                #sozoterms[f1, :, :, 2] = real.((Iψfac[f1, :, :, 1].^2) .* (P*(fvar .* fsqrsψ)))     #(I ✪ ψR)^2 . (σ2 ✪ ψR^2)
                #sozoterms[f1, :, :, 2] += real.((Iψfac[f1, :, :, 2].^2) .* (P*(fvar .* fsqimψ)))    #(I ✪ ψI)^2 . (σ2 ✪ ψI^2)
                #sozoterms[f1, :, :, 2] += real.(2*Iψfac[f1, :, :, 1].*Iψfac[f1, :, :, 2].* (P*(fvar .* (Pf*(imag.(rsψmat[f1, :, :]).*real.(rsψmat[f1, :, :]))))))  #2(I ✪ ψR)(I ✪ ψI) . (σ2 ✪ ψRψI)
                #sozoterms[f1, :, :, 2] = sozoterms[f1, :, :, 2]./(im_rd_0_1[:, :, f1].^3)
                #sozoterms[f1, :, :, 1] -= sozoterms[f1, :, :, 2]
            end
            for f1=1:Nf
                for f2=1:Nf
                    #println("f2", f2)
                    val1 = Pf*(ψsqfac[f1, :, :, 1] .* ψsqfac[f2, :, :, 1])
                    #println(size(val1), size(fvar))
                    term1 = (P*(fvar .* val1) .* Iψfac[f1, :, :, 1]) .* Iψfac[f2, :, :, 1]
                    term2 = (P*(fvar .* (Pf*(ψsqfac[f1, :, :, 1] .* ψsqfac[f2, :, :, 2]))).* Iψfac[f1, :, :, 1]) .* Iψfac[f2, :, :, 2]
                    term3 = (P*(fvar .* (Pf*(ψsqfac[f1, :, :, 2] .* ψsqfac[f2, :, :, 1]))).* Iψfac[f1, :, :, 2]) .* Iψfac[f2, :, :, 1]
                    term4 = (P*(fvar .* (Pf*(ψsqfac[f1, :, :, 2] .* ψsqfac[f2, :, :, 2]))).* Iψfac[f1, :, :, 2]) .* Iψfac[f2, :, :, 2]
                    so1zo2 = 0.5 .* im_rd_0_1[:, :, f2] .* sozoterms[f1, :, :, 1]
                    so2zo1 = 0.5 .* im_rd_0_1[:, :, f1] .* sozoterms[f2, :, :, 1]
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

#SO
Nx=64
filter_hash = fink_filter_hash(1, 8, nx=Nx, t=1, Omega=true)
psir = realspace_filter(Nx, filter_hash["filt_index"][4], filter_hash["filt_value"][4])
maximum(real.(psir))
maximum(imag.(psir))
#Check s20r theoretical
numfile = 1000
direc = "scratch_NM/StandardizedExp/Nx64/"
datfile = direc * "Data_" * string(numfile) * ".jld2" #Replace w SLURM array
loaddf = load(datfile)
true_img = loaddf["true_img"]
true_img = true_img .- mean(true_img)
sigma = loaddf["std"]
Nx=64
Nr=10000
empsn = []
isobool = false
apdbool = false
dhc_args = Dict(:doS2=>false, :doS20=>true, :apodize=>apdbool, :iso=>isobool)
filter_hash = fink_filter_hash(1, 8, nx=Nx, t=1, wd=1, Omega=true)
for n=1:Nr
    noisyim = randn((Nx, Nx)).*sigma .+ true_img
    push!(empsn, DHC_compute_wrapper(noisyim, filter_hash; dhc_args...))
end
coeffsim = hcat(empsn...)'
ncoeffmean = mean(coeffsim, dims=1)
Nf = size(filter_hash["filt_index"])[1]
coeffmask = falses(2+Nf+Nf^2)
coeffmask[Nf+3:end] .= triu(trues(Nf, Nf))[:]
#thcoeffmean, thcov = DHC_compute_S20r_noisy(true_img, filter_hash, fill(sigma, (Nx, Nx)); coeff_mask = coeffmask, dhc_args...)
thcoeffmean, thcov = DHC_compute_S20r_noisy_so(true_img, filter_hash, fill(sigma, (Nx, Nx)); coeff_mask = coeffmask, dhc_args...)
thcoeffmean2, thcov2 = DHC_compute_S20r_noisy_so2(true_img, filter_hash, fill(sigma, (Nx, Nx)); coeff_mask = coeffmask, dhc_args...)

plot(collect(1:595), thcoeffmean, label="th")
plot!(collect(1:595), mean(coeffsim[:, coeffmask], dims=1)[:], label="emp")
plot!(collect(1:595), thcoeffmean2, label="th2")

discmat = zeros(Nf, Nf)
triumask = triu(trues(Nf, Nf))
discmat[triumask] .= (thcoeffmean ./ mean(coeffsim[:, coeffmask], dims=1)[:])
heatmap(discmat, title="S20R")
heatmap(discmat, clim=(1.0, 10.0))
discmat
mask = Diagonal(trues(Nf, Nf))
discmat[mask]
ncoeffmean[:][coeffmask]


Nx=64
filter_hash = fink_filter_hash(1, 8, nx=Nx, t=1, Omega=true)
psir = realspace_filter(Nx, filter_hash["filt_index"][4], filter_hash["filt_value"][4])
maximum(real.(psir))
maximum(imag.(psir))
#Check s20r theoretical
numfile = 1000
direc = "scratch_NM/StandardizedExp/Nx64/"
datfile = direc * "Data_" * string(numfile) * ".jld2" #Replace w SLURM array
loaddf = load(datfile)
true_img = loaddf["true_img"]
true_img = true_img .- mean(true_img)
sigma = loaddf["std"]
Nx=64
Nr=10000
empsn = []
isobool = false
apdbool = false
dhc_args = Dict(:doS2=>false, :doS20=>true, :apodize=>apdbool, :iso=>isobool)
filter_hash = fink_filter_hash(1, 8, nx=Nx, t=1, wd=1, Omega=true)
for n=1:Nr
    noisyim = randn((Nx, Nx)).*sigma .+ true_img
    push!(empsn, DHC_compute_wrapper(noisyim, filter_hash; dhc_args...))
end
coeffsim = hcat(empsn...)'
ncoeffmean = mean(coeffsim, dims=1)
Nf = size(filter_hash["filt_index"])[1]
coeffmask = falses(2+Nf+Nf^2)
coeffmask[Nf+3:end] .= triu(trues(Nf, Nf))[:]
thcoeffmean, thcov = DHC_compute_S20r_noisy(true_img, filter_hash, fill(sigma, (Nx, Nx)); coeff_mask = coeffmask, dhc_args...)
ratio = coeffsim[:, coeffmask] ./ reshape(DHC_compute_wrapper(true_img, filter_hash; dhc_args...), (1, 1192))[:, coeffmask]
ratioth = thcoeffmean ./ DHC_compute_wrapper(true_img, filter_hash; dhc_args...)[coeffmask]

maximum(abs.((thcoeffmean .- mean(coeffsim[:, coeffmask], dims=1)[:])./ mean(coeffsim[:, coeffmask], dims=1)[:]))
mean(abs.((thcoeffmean .- mean(coeffsim[:, coeffmask], dims=1)[:])./ mean(coeffsim[:, coeffmask], dims=1)[:]))
thcoeffmean ./mean(coeffsim[:, coeffmask], dims=1)[:]

plot(collect(1:595), (thcoeffmean .- mean(coeffsim[:, coeffmask], dims=1)[:])./ mean(coeffsim[:, coeffmask], dims=1)[:])

plot(collect(1:595), thcoeffmean, label="th")
plot!(collect(1:595), mean(coeffsim[:, coeffmask], dims=1)[:], label="emp")


empcov = (coeffsim .- ncoeffmean)' * (coeffsim .- ncoeffmean) / (Nr-1)

heatmap(log.((x->maximum([x, 0])).(empcov)))

dS20dp = wst_S20_deriv(true_img, filter_hash)
Nf = length(filter_hash["filt_index"])
G = reshape(dS20dp, Nx*Nx, Nf*Nf) .* fill(sigma, (Nx*Nx, 1))
pred = G'*G
s20rall = falses(2+Nf+Nf^2)
s20rall[Nf+3:end] .= true
fracerr = (pred .- empcov[s20rall, s20rall]) ./ empcov[s20rall, s20rall]
empcov[s20rall, s20rall]

coeffsim ./reshape(DHC_compute_wrapper(true_img, filter_hash; dhc_args...), (1, 1192))
coeffmask = falses(2+Nf+Nf^2)
coeffmask[Nf+3:end] .= triu(trues(Nf, Nf))[:]


rsψ = realspace_filter(Nx, filter_hash["filt_index"][5], filter_hash["filt_value"][5])
ψf = fft(real.(rsψ) + imag.(rsψ))

## Distribution intersection??
logsfddbn = load("scratch_NM/SavedCovMats/reg_apd_noiso_logcoeff.jld2")
Nf = length(logsfddbn["filter_hash"]["filt_index"])
coeffmask = falses(2+Nf+Nf^2)
coeffmask[Nf+3:end] .= triu(trues(Nf, Nf))[:]
covreg = logsfddbn["sampcov"][coeffmask, coeffmask] + I*1e-10
sfddbn = MultivariateNormal(logsfddbn["sampmean"][:][coeffmask], covreg)
sampssfd = rand(sfddbn, 10000)
sampssfd = sampssfd'


loaddf = load("scratch_NM/StandardizedExp/Nx64/Data_1000.jld2")
dhc_args = logsfddbn["dhc_args"]
truec = log.(DHC_compute_wrapper(loaddf["true_img"], logsfddbn["filter_hash"], norm=false; dhc_args...)[coeffmask])
initc = log.(DHC_compute_wrapper(loaddf["init"], logsfddbn["filter_hash"], norm=false; dhc_args...)[coeffmask])
reconc = log.(DHC_compute_wrapper(recon_img, logsfddbn["filter_hash"], norm=false; dhc_args...)[coeffmask])
ind1, ind2 = 1, 3
p = scatter(sampssfd[:, ind1], sampssfd[:, ind2], label="Samples_SFD_Prior", legend=(0.1, 0.1))
scatter!([truec[ind1]], [truec[ind2]], label="True Coeff", legend=(0.1, 0.1))
scatter!([initc[ind1]], [initc[ind2]], label="Init Coeff", legend=(0.1, 0.1))
xlabel!("J=1, L=1")
ylabel!("J=2, L=1")
noisymean, noisycov = DHC_compute_noisy(loaddf["init"], logsfddbn["filter_hash"], fill(loaddf["std"], (Nx, Nx)); coeff_mask = coeffmask, dhc_args...)
ncovreg = noisycov + I*1e-10
cond(ncovreg)
noisydbn = MultivariateNormal(convert(Array{Float64, 1}, noisymean), ncovreg)
noisysamp = rand(noisydbn, 10000)
noisysamp = log.(noisysamp')
p = scatter(sampssfd[:, ind1], sampssfd[:, ind2], label="Samples_SFD_Prior", legend=(0.1, 0.1))
scatter!(noisysamp[:, ind1], noisysamp[:, ind2], label="Noise-Prior Init Proxy", legend=(0.1, 0.1))
scatter!([truec[ind1]], [truec[ind2]], label="True Coeff", legend=(0.1, 0.1), c="black")
scatter!([initc[ind1]], [initc[ind2]], label="Init Coeff", legend=(0.1, 0.1))
xlabel!("J=1, L=1")
ylabel!("J=2, L=1")

noisymean, noisycov = DHC_compute_noisy(loaddf["true_img"], logsfddbn["filter_hash"], fill(loaddf["std"], (Nx, Nx)); coeff_mask = coeffmask, dhc_args...)
ncovreg = noisycov + I*1e-10
noisydbn = MultivariateNormal(convert(Array{Float64, 1}, noisymean), ncovreg)
noisysamp = rand(noisydbn, 10000)
noisysamp = log.(noisysamp')
p = scatter(sampssfd[:, ind1], sampssfd[:, ind2], label="Samples_SFD_Prior", legend=(0.1, 0.1))
scatter!(noisysamp[:, ind1], noisysamp[:, ind2], label="Theoretical Noise-Dbn Using True", legend=(0.1, 0.1))
scatter!([truec[ind1]], [truec[ind2]], label="True Coeff", legend=(0.1, 0.1), c="black")
scatter!([initc[ind1]], [initc[ind2]], label="Init Coeff", legend=(0.1, 0.1))
scatter!([reconc[ind1]], [reconc[ind2]], label="Recon Coeff", legend=(0.1, 0.1))
xlabel!("J=1, L=1")
ylabel!("J=2, L=1")


noisymean, noisycov = DHC_compute_noisy(loaddf["true_img"], logsfddbn["filter_hash"], fill(0.0, (Nx, Nx)); coeff_mask = coeffmask, dhc_args...)
ncovreg = noisycov + I*1e-10
noisydbn = MultivariateNormal(convert(Array{Float64, 1}, noisymean), ncovreg)
noisysamp = rand(noisydbn, 10000)
noisysamp = log.((x->maximum([x, 0])).(noisysamp'))
p = scatter(sampssfd[:, ind1], sampssfd[:, ind2], label="Samples_SFD_Prior", legend=(0.1, 0.1))
scatter!(noisysamp[:, ind1], noisysamp[:, ind2], label="Test Theoretical Noise-Dbn Using True, σ=0", legend=(0.1, 0.1))
scatter!([truec[ind1]], [truec[ind2]], label="True Coeff", legend=(0.1, 0.1), c="black")
scatter!([initc[ind1]], [initc[ind2]], label="Init Coeff", legend=(0.1, 0.1))
scatter!([reconc[ind1]], [reconc[ind2]], label="Recon Coeff", legend=(0.1, 0.1))
xlabel!("J=1, L=1")
ylabel!("J=2, L=1")

#Check if the empirical dbn also has an offset
logsfddbn = load("scratch_NM/SavedCovMats/reg_apd_noiso_logcoeff.jld2")
Nf = length(logsfddbn["filter_hash"]["filt_index"])
coeffmask = falses(2+Nf+Nf^2)
coeffmask[Nf+3:end] .= triu(trues(Nf, Nf))[:]
covreg = logsfddbn["sampcov"][coeffmask, coeffmask] + I*1e-10
sfddbn = MultivariateNormal(logsfddbn["sampmean"][:][coeffmask], covreg)
sampssfd = rand(sfddbn, 10000)
sampssfd = sampssfd'

empsn = []
dhc_args = logsfddbn["dhc_args"]
filter_hash = logsfddbn["filter_hash"]
loaddf = load("scratch_NM/StandardizedExp/Nx64/Data_1000.jld2")
true_img = loaddf["true_img"]
sigma = loaddf["std"]
for n=1:Nr
    noisyim = randn((Nx, Nx)).*sigma .+ true_img
    push!(empsn, DHC_compute_wrapper(noisyim, filter_hash; dhc_args...))
end
coeffsim = hcat(empsn...)'
ncoeffmean = mean(coeffsim, dims=1)[:, coeffmask]
empcov = (coeffsim[:, coeffmask] .-ncoeffmean)' * (coeffsim[:, coeffmask] .- ncoeffmean) ./(Nr-1)
covreg = Symmetric(empcov + I*1e-10)
cond(covreg)
empdbn = MultivariateNormal(ncoeffmean[:], covreg)
empsamps = log.((x->maximum([x, 0])).(rand(empdbn, 10000)'))
#Analytical
noisymean, noisycov = DHC_compute_noisy(loaddf["true_img"], logsfddbn["filter_hash"], fill(loaddf["std"], (Nx, Nx)); coeff_mask = coeffmask, dhc_args...)
ncovreg = noisycov + I*1e-10
noisydbn = MultivariateNormal(convert(Array{Float64, 1}, noisymean), ncovreg)
noisysamp = rand(noisydbn, 10000)
noisysamp = log.(noisysamp')
truec = log.(DHC_compute_wrapper(loaddf["true_img"], logsfddbn["filter_hash"], norm=false; dhc_args...)[coeffmask])
initc = log.(DHC_compute_wrapper(loaddf["init"], logsfddbn["filter_hash"], norm=false; dhc_args...)[coeffmask])

p = scatter(sampssfd[:, ind1], sampssfd[:, ind2], label="Samples_SFD_Prior", legend=(0.1, 0.1))
scatter!(empsamps[:, ind1], empsamps[:, ind2], label="Empirical Noisy Dbn", legend=(0.1, 0.1))
scatter!(noisysamp[:, ind1], noisysamp[:, ind2], label="Theoretical Noise-Dbn Using True, σ", legend=(0.1, 0.1))
scatter!([truec[ind1]], [truec[ind2]], label="True Coeff", legend=(0.1, 0.1), c="black")
scatter!([initc[ind1]], [initc[ind2]], label="Init Coeff", legend=(0.1, 0.1), c="red")
#scatter!([reconc[ind1]], [reconc[ind2]], label="Recon Coeff", legend=(0.1, 0.1), c="green")
title!("Another Image: 1000")
xlabel!("J=1, L=1")
ylabel!("J=2, L=1")

#Where does the boosted(init) dbn lie?
logsfddbn = load("scratch_NM/SavedCovMats/reg_apd_noiso_logcoeff.jld2")
Nf = length(logsfddbn["filter_hash"]["filt_index"])
coeffmask = falses(2+Nf+Nf^2)
coeffmask[Nf+3:end] .= triu(trues(Nf, Nf))[:]
covreg = logsfddbn["sampcov"][coeffmask, coeffmask] + I*1e-10
sfddbn = MultivariateNormal(logsfddbn["sampmean"][:][coeffmask], covreg)
sampssfd = rand(sfddbn, 10000)
sampssfd = sampssfd'

empsn = []
dhc_args = logsfddbn["dhc_args"]
filter_hash = logsfddbn["filter_hash"]
loaddf = load("scratch_NM/StandardizedExp/Nx64/Data_1000.jld2")
true_img = loaddf["true_img"]
sigma = loaddf["std"]
for n=1:Nr
    noisyim = randn((Nx, Nx)).*sigma .+ true_img
    push!(empsn, DHC_compute_wrapper(noisyim, filter_hash; dhc_args...))
end
coeffsim = hcat(empsn...)'
ncoeffmean = mean(coeffsim, dims=1)[:, coeffmask]
empcov = (coeffsim[:, coeffmask] .-ncoeffmean)' * (coeffsim[:, coeffmask] .- ncoeffmean) ./(Nr-1)
covreg = Symmetric(empcov + I*1e-10)
cond(covreg)
empdbn = MultivariateNormal(ncoeffmean[:], covreg)
empsamps = log.((x->maximum([x, 0])).(rand(empdbn, 10000)'))
#Analytical
noisymean, noisycov = DHC_compute_noisy(loaddf["true_img"], logsfddbn["filter_hash"], fill(loaddf["std"], (Nx, Nx)); coeff_mask = coeffmask, dhc_args...)
ncovreg = noisycov + I*1e-10
noisydbn = MultivariateNormal(convert(Array{Float64, 1}, noisymean), ncovreg)
noisysamp = rand(noisydbn, 10000)
noisysamp = log.(noisysamp')
truec = log.(DHC_compute_wrapper(loaddf["true_img"], logsfddbn["filter_hash"], norm=false; dhc_args...)[coeffmask])
initc = log.(DHC_compute_wrapper(loaddf["init"], logsfddbn["filter_hash"], norm=false; dhc_args...)[coeffmask])


empsn = []
for n=1:Nr
    noisyim = randn((Nx, Nx)).*sigma .+ loaddf["init"]
    push!(empsn, DHC_compute_wrapper(noisyim, filter_hash; dhc_args...))
end
coeffsim = hcat(empsn...)'
ncoeffmean = mean(coeffsim, dims=1)[:, coeffmask]
empcov = (coeffsim[:, coeffmask] .-ncoeffmean)' * (coeffsim[:, coeffmask] .- ncoeffmean) ./(Nr-1)
covreg = Symmetric(empcov + I*1e-10)
cond(covreg)
empdbninit = MultivariateNormal(ncoeffmean[:], covreg)
empsampsinit = log.((x->maximum([x, 0])).(rand(empdbninit, 10000)'))
p = scatter(sampssfd[:, ind1], sampssfd[:, ind2], label="Samples_SFD_Prior", legend=(0.1, 0.1))
scatter!(empsamps[:, ind1], empsamps[:, ind2], label="Empirical Noisy Dbn", legend=(0.1, 0.1))
scatter!(empsampsinit[:, ind1], empsampsinit[:, ind2], label="Empirical Noisy Dbn (added noise to init)", legend=(0.1, 0.1))
scatter!(noisysamp[:, ind1], noisysamp[:, ind2], label="Theoretical Noise-Dbn Using True, σ", legend=(0.1, 0.1))
scatter!([truec[ind1]], [truec[ind2]], label="True Coeff", legend=(0.1, 0.1), c="black")
scatter!([initc[ind1]], [initc[ind2]], label="Init Coeff", legend=(0.1, 0.1), c="red")
#scatter!([reconc[ind1]], [reconc[ind2]], label="Recon Coeff", legend=(0.1, 0.1), c="green")
title!("Another Image: 1000")
xlabel!("J=1, L=1")
ylabel!("J=2, L=1")


#Using S20R noisy
logsfddbn = load("scratch_NM/SavedCovMats/reg_apd_noiso_logcoeff.jld2")
Nf = length(logsfddbn["filter_hash"]["filt_index"])
coeffmask = falses(2+Nf+Nf^2)
coeffmask[Nf+3:end] .= triu(trues(Nf, Nf))[:]
covreg = logsfddbn["sampcov"][coeffmask, coeffmask] + I*1e-10
sfddbn = MultivariateNormal(logsfddbn["sampmean"][:][coeffmask], covreg)
sampssfd = rand(sfddbn, 10000)
sampssfd = sampssfd'

empsn = []
dhc_args = logsfddbn["dhc_args"]
filter_hash = logsfddbn["filter_hash"]
loaddf = load("scratch_NM/StandardizedExp/Nx64/Data_1000.jld2")
true_img = loaddf["true_img"]
sigma = loaddf["std"]
Nr=10000
Nx=64
for n=1:Nr
    noisyim = randn((Nx, Nx)).*sigma .+ true_img
    push!(empsn, DHC_compute_wrapper(noisyim, filter_hash; dhc_args...))
end
coeffsim = hcat(empsn...)'
ncoeffmean = mean(coeffsim, dims=1)[:, coeffmask]
empcov = (coeffsim[:, coeffmask] .-ncoeffmean)' * (coeffsim[:, coeffmask] .- ncoeffmean) ./(Nr-1)
covreg = Symmetric(empcov + I*1e-10)
cond(covreg)
empdbn = MultivariateNormal(ncoeffmean[:], covreg)
empsamps = log.((x->maximum([x, 0])).(rand(empdbn, 10000)'))
#Analytical
noisymean, noisycov = DHC_compute_S20r_noisy(loaddf["true_img"], logsfddbn["filter_hash"], fill(loaddf["std"], (Nx, Nx)); coeff_mask = coeffmask, dhc_args...)
ncovreg = noisycov + I*1e-10
noisydbn = MultivariateNormal(convert(Array{Float64, 1}, noisymean), ncovreg)
noisysamp = rand(noisydbn, 10000)
noisysamp = log.((x->maximum([x, 0])).(noisysamp'))
truec = log.(DHC_compute_wrapper(loaddf["true_img"], logsfddbn["filter_hash"], norm=false; dhc_args...)[coeffmask])
initc = log.(DHC_compute_wrapper(loaddf["init"], logsfddbn["filter_hash"], norm=false; dhc_args...)[coeffmask])
reconc = log.(DHC_compute_wrapper(recon_img, logsfddbn["filter_hash"], norm=false; dhc_args...)[coeffmask])

#=
empsn = []
for n=1:Nr
    noisyim = randn((Nx, Nx)).*sigma .+ loaddf["init"]
    push!(empsn, DHC_compute_wrapper(noisyim, filter_hash; dhc_args...))
end
coeffsim = hcat(empsn...)'
ncoeffmean = mean(coeffsim, dims=1)[:, coeffmask]
empcov = (coeffsim[:, coeffmask] .-ncoeffmean)' * (coeffsim[:, coeffmask] .- ncoeffmean) ./(Nr-1)
covreg = Symmetric(empcov + I*1e-10)
cond(covreg)
=#
#empdbninit = MultivariateNormal(ncoeffmean[:], covreg)
#empsampsinit = log.((x->maximum([x, 0])).(rand(empdbninit, 10000)'))
ind1, ind2= 1, 3
p = scatter(sampssfd[:, ind1], sampssfd[:, ind2], label="Samples_SFD_Prior", legend=(0.1, 0.1))
scatter!(empsamps[:, ind1], empsamps[:, ind2], label="Empirical Noisy Dbn", legend=(0.1, 0.1))
#scatter!(empsampsinit[:, ind1], empsampsinit[:, ind2], label="Empirical Noisy Dbn (added noise to init)", legend=(0.1, 0.1))
scatter!(noisysamp[:, ind1], noisysamp[:, ind2], label="Theoretical Noise-Dbn Using True, σ", legend=(0.1, 0.1))
scatter!([truec[ind1]], [truec[ind2]], label="True Coeff", legend=(0.1, 0.1), c="black")
scatter!([initc[ind1]], [initc[ind2]], label="Init Coeff", legend=(0.1, 0.1), c="red")
scatter!([reconc[ind1]], [reconc[ind2]], label="Recon Coeff", legend=(0.1, 0.1), c="green")
title!("Another Image: 1000, using the complex S20r correction")
xlabel!("J=1, L=1")
ylabel!("J=2, L=1")

##Case3

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
dbfile = load("scratch_NM/SavedCovMats/reg_apd_noiso_nologcoeff.jld2")
dbnocffs = log.(dbfile["dbncoeffs"])
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
lval3= 0.00

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

#save("scratch_NM/NewWrapper/4-28/1000_C5_S2sfd_highjinit.jld2", Dict("true_img"=>true_img, "init"=>init, "recon"=>recon_img, "dict"=>recon_settings, "trace"=>Optim.trace(res), "coeff_mask"=>[coeff_masksfd, coeff_maskinit], "fhash"=>filter_hash, "dhc_args"=>dhc_args, "func_specific_params"=>func_specific_params))

function Loss3noisy(img_curr, filter_hash, dhc_args; coeff_mask1=nothing, target1=nothing, invcov1=nothing, reg_input=nothing, coeff_mask2=nothing, target2=nothing, invcov2=nothing, lambda2=nothing, lambda3=nothing, noisy_img=nothing, sigma=nothing)
    #func_specific_params should contain: coeff_mask2, target2, invcov2
    s_curr = DHC_compute_wrapper(img_curr, filter_hash, norm=false; dhc_args...)
    s_curr1 = s_curr[coeff_mask1]
    noisycoeff = DHC_compute_wrapper(noisy_img, filter_hash, norm=false; dhc_args...)[coeff_mask2]
    regterm =  0.5*lambda3*sum((adaptive_apodizer(img_curr, dhc_args) - adaptive_apodizer(reg_input, dhc_args)).^2)
    lnlik_sfd = ( 0.5 .* (s_curr1 - target1)' * invcov1 * (s_curr1 - target1))
    target2 = DHC_compute_noisy(img_curr, filter_hash, fill(sigma, (Nx, Nx)); coeff_mask = coeff_mask2, dhc_args...) #s_curr[coeff_mask2
    lnlik_init = ( 0.5*lambda2 .* (noisycoeff - target2)' * invcov2 * (noisycoeff - target2))
    neglogloss = lnlik_sfd[1] + lnlik_init[1] + regterm
end

function dLoss3noisy!(storage_grad, img_curr, filter_hash, dhc_args; coeff_mask1=nothing, target1=nothing, invcov1=nothing, reg_input=nothing, coeff_mask2=nothing, target2=nothing, invcov2=nothing, dA=nothing, lambda2=nothing, lambda3=nothing, noisy_img=nothing)
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
savefig(p, "scratch_NM/NewWrapper/5-23/Case2_It0.png")

#STEP 2
sigma= loaddf["std"]
coeffmask = falses(2+Nf+Nf^2)
coeffmask[Nf+3:end] .= triu(trues(Nf, Nf))[:]
thcoeffmean, thcov = DHC_compute_noisy(recon_img, filter_hash, fill(sigma, (Nx, Nx)); coeff_mask = coeffmask, dhc_args...)

empsn = []
Nr=10000
noisemod = Normal(0.0, sigma)
for n=1:Nr
    noisyim = rand(noisemod, (Nx, Nx)) .+ true_img
    push!(empsn, DHC_compute_wrapper(noisyim, filter_hash; dhc_args...))
end
coeffsim = hcat(empsn...)'
ncoeffmean = mean(coeffsim, dims=1)
empcov = (coeffsim[:, coeffmask] .- reshape(ncoeffmean[coeffmask],(1, count(!iszero, coeffmask))))' * (coeffsim[:, coeffmask] .- reshape(ncoeffmean[coeffmask], (1, count(!iszero, coeffmask)))) ./(Nr-1)


JS1ind = J_S1indices([0, 1, 2, 3], filter_hash)

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
recon_settings["coeff_mask_sfd"] = coeffmask
recon_settings["coeff_mask_init"] = coeff_maskinit

fcov = (dbnocffs .- fmean)' * (dbnocffs .- fmean) ./(size(dbnocffs)[1] -1)
fcovinv1 = invert_covmat(thcov, recon_settings["eps_value_sfd"])
fcovinv2 = invert_covmat(fcov[coeff_maskinit, coeff_maskinit], recon_settings["eps_value_init"])

if logbool
    error("Not impl for log im")
end
ftarginit = DHC_compute_wrapper(init, filter_hash; dhc_args...)
ftarginit = ftarginit[coeff_maskinit]

#Weight given to terms
lval2 = 0.0
lval3 = 1.0

println("Regularizer Lambda=", round(lval3, sigdigits=3))
#input::Array{Float64, 2}, filter_hash::Dict, s_targ_mean::Array{Float64, 1}, s_targ_invcov, dhc_args, LossFunc, dLossFunc;
#FFTthreads::Int=1, optim_settings=Dict([("iterations", 10)]), lambda=0.001, func_specific_params=nothing
func_specific_params = Dict([(:reg_input=> init), (:lambda2=> lval2), (:lambda3=> lval3), (:coeff_mask1=> coeffmask), (:target1=>thcoeffmean), (:invcov1=>fcovinv1), (:coeff_mask2=> coeff_maskinit), (:target2=>ftarginit), (:invcov2=>fcovinv2)])


recon_settings["fname_save"] = fname_save * ".jld2"
recon_settings["optim_settings"] = optim_settings

if logbool
    error("Not implemented here")
end

white_noise = randn(Nx, Nx)
res, recon_img2 = image_recon_derivsum_custom(recon_img, filter_hash, dhc_args, ReconFuncs.Loss3Gaussian,
    ReconFuncs.dLoss3Gaussian!; optim_settings=optim_settings, func_specific_params)
#save(settings["fname_save"], Dict("true_img"=>true_img, "init"=>noisy_init, "recon"=>recon_img, "dict"=>settings, "trace"=>Optim.trace(res), "coeff_mask"=>coeff_mask, "fhash"=>fhash, "dhc_args"=>dhc_args))


function loss1(inp_img)
    s_curr = DHC_compute_wrapper(inp_img, filter_hash, norm=false; dhc_args...)
    s_curr1 = Data_Utils.fnlog(s_curr[coeffmask])
    diff1 = s_curr1 - thcoeffmean
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

init = recon_img
println("Loss Term 1: SFDTarg")
println("True", loss1(true_img))
println("Init", loss1(init))
println("Recon", loss1(recon_img2))

println("Loss Term 2: SmoothedInitTarg")
println("True", loss2(true_img))
println("Init", loss2(init))
println("Recon", loss2(recon_img2))

println("Loss Term 3: Reg")
println("True", loss3(true_img))
println("Init", loss3(init))
println("Recon", loss3(recon_img2))

apdsmoothed = apodizer(imfilter(init, Kernel.gaussian(0.8)))
kbins= convert(Array{Float64}, collect(1:32))
clim=(minimum(apodizer(true_img)), maximum(apodizer(true_img)))
true_ps = Data_Utils.calc_1dps(apodizer(true_img), kbins)
initps = Data_Utils.calc_1dps(apodizer(init), kbins)
recps = Data_Utils.calc_1dps(apodizer(recon_img2), kbins)
smoothps = Data_Utils.calc_1dps(apdsmoothed, kbins)
p1 = heatmap(apodizer(recon_img2), title="Recon", clim=clim)
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
residual = apodizer(recon_img2)- apodizer(true_img)
rlims = (minimum(residual), maximum(residual))
#symmax = maximum([abs(minimum(residual)), maximum(residual)])
#rg = cgrad(:bwr, [-symmax/2.0, symmax/2.0])
p6 = heatmap(residual, title="Residual: Recon - True", clims=rlims, c=:bwr)
p7 = heatmap(apdsmoothed- apodizer(true_img), title="Residual: SmoothedInit - True", clims=rlims, c=:bwr)

struesel = Data_Utils.fnlog(DHC_compute_wrapper(true_img, filter_hash, norm=false; dhc_args...))
sinitsel = Data_Utils.fnlog(DHC_compute_wrapper(init, filter_hash, norm=false; dhc_args...))
ssmoothsel = Data_Utils.fnlog(DHC_compute_wrapper(imfilter(init, Kernel.gaussian(0.8)), filter_hash, norm=false; dhc_args...))
sreconsel = Data_Utils.fnlog(DHC_compute_wrapper(recon_img2, filter_hash, norm=false; dhc_args...))
slims = (minimum(struesel[JS1ind]), maximum(struesel[JS1ind]))
cg = cgrad([:blue, :white, :red])
truephi, trueomg = round(struesel[2+filter_hash["phi_index"]], sigdigits=3), round(struesel[2+filter_hash["Omega_index"]], sigdigits=3)
reconphi, reconomg = round(sreconsel[2+filter_hash["phi_index"]], sigdigits=3), round(sreconsel[2+filter_hash["Omega_index"]], sigdigits=3)
smoothphi, smoothomg = round(ssmoothsel[2+filter_hash["phi_index"]], sigdigits=3), round(ssmoothsel[2+filter_hash["Omega_index"]], sigdigits=3)

p8 = heatmap(struesel[JS1ind], title="True Coeffs ϕ=" * string(truephi) * "Ω=" * string(trueomg) , clims=slims, c=cg)
p9 = heatmap(sreconsel[JS1ind], title="Recon Coeffs ϕ=" * string(reconphi) * "Ω=" * string(reconomg), clims=slims, c=cg)
p10 = heatmap(ssmoothsel[JS1ind], title="Smooth Init ϕ=" * string(smoothphi) * "Ω=" * string(smoothomg), clims=slims, c=cg)
p = plot(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, layout=(5, 2), size=(1800, 2400))
savefig(p, "scratch_NM/NewWrapper/5-23/Case2_It1.png")


##5-31
#Iterative for only S1
Nx=64
filter_hash = fink_filter_hash(1, 8, nx=Nx, t=1, Omega=true)
psir = realspace_filter(Nx, filter_hash["filt_index"][4], filter_hash["filt_value"][4])

#Check s20r theoretical
numfile = 1000
direc = "scratch_NM/StandardizedExp/Nx64/"
datfile = direc * "Data_" * string(numfile) * ".jld2" #Replace w SLURM array
loaddf = load(datfile)
true_img = loaddf["true_img"]
true_img = true_img .- mean(true_img)
sigma = loaddf["std"
Nx=64
Nr=10000
empsn = []
isobool = false
apdbool = false
dhc_args = Dict(:doS2=>false, :doS20=>true, :apodize=>apdbool, :iso=>isobool)
filter_hash = fink_filter_hash(1, 8, nx=Nx, t=1, wd=1, Omega=true)
for n=1:Nr
    noisyim = randn((Nx, Nx)).*sigma .+ true_img
    push!(empsn, DHC_compute_wrapper(noisyim, filter_hash; dhc_args...))
end
coeffsim = hcat(empsn...)'
ncoeffmean = mean(coeffsim, dims=1)
Nf = size(filter_hash["filt_index"])[1]
coeffmask = falses(2+Nf+Nf^2)
coeffmask[Nf+3:end] .= triu(trues(Nf, Nf))[:]
thcoeffmean, thcov = DHC_compute_S20r_noisy_so(true_img, filter_hash, fill(sigma, (Nx, Nx)); coeff_mask = coeffmask, dhc_args...)

disc = thcoeffmean[:] ./ ncoeffmean[:][coeffmask]
discmat = zeros(Nf, Nf)
triumask = triu(trues(Nf, Nf))
discmat[triumask] .= (thcoeffmean ./ mean(coeffsim[:, coeffmask], dims=1)[:])
#heatmap(discmat, title="S20R")
#heatmap(discmat, clim=(1.0, 10.0))
discmat
mask = Diagonal(trues(Nf, Nf))
discmat[mask]


# Distribution intersection??
#Log SFD DBN
logsfddbn = load("scratch_NM/SavedCovMats/reg_apd_noiso_logcoeff.jld2")
Nf = length(logsfddbn["filter_hash"]["filt_index"])
coeffmask = falses(2+Nf+Nf^2)
coeffmask[Nf+3:end] .= triu(trues(Nf, Nf))[:]
covreg = logsfddbn["sampcov"][coeffmask, coeffmask] + I*1e-10
sfddbn = MultivariateNormal(logsfddbn["sampmean"][:][coeffmask], covreg)
sampssfd = rand(sfddbn, 10000)
sampssfd = sampssfd'


loaddf = load("scratch_NM/StandardizedExp/Nx64/Data_1000.jld2")
dhc_args = logsfddbn["dhc_args"]
true_img = loaddf["true_img"]
truec = log.(DHC_compute_wrapper(loaddf["true_img"], logsfddbn["filter_hash"], norm=false; dhc_args...)[coeffmask])
initc = log.(DHC_compute_wrapper(loaddf["init"], logsfddbn["filter_hash"], norm=false; dhc_args...)[coeffmask])
#reconc = log.(DHC_compute_wrapper(recon_img, logsfddbn["filter_hash"], norm=false; dhc_args...)[coeffmask])
ind1, ind2 = 1, 3

#Theoretical noisy
noisymean, noisycov = DHC_compute_S20r_noisy_so(loaddf["true_img"], logsfddbn["filter_hash"], fill(loaddf["std"], (Nx, Nx)); coeff_mask = coeffmask, dhc_args...)
ncovreg = noisycov + I*1e-10
noisydbn = MultivariateNormal(convert(Array{Float64, 1}, noisymean), ncovreg)
noisysamp = rand(noisydbn, 10000)
#Log theoretical noisy
noisysamp = log.((x->maximum([x, 0])).(noisysamp'))
p = scatter(sampssfd[:, ind1], sampssfd[:, ind2], label="Samples_SFD_Prior", legend=(0.1, 0.1))
scatter!(noisysamp[:, ind1], noisysamp[:, ind2], label="Theoretical Noise-Dbn Using True", legend=(0.1, 0.1))
scatter!([truec[ind1]], [truec[ind2]], label="True Coeff", legend=(0.1, 0.1), c="black")
scatter!([initc[ind1]], [initc[ind2]], label="Init Coeff", legend=(0.1, 0.1))
xlabel!("J=1, L=1")
ylabel!("J=2, L=1")

#Why dont init and the analytical noisy dbn match?
#Empirical noisy
empsn = []
for n=1:Nr
    noisyim = randn((Nx, Nx)).*loaddf["std"] .+ true_img
    push!(empsn, DHC_compute_wrapper(noisyim, logsfddbn["filter_hash"];  dhc_args...))
end
noisymean
coeffsim = hcat(empsn...)'
ncoeffmean = mean(coeffsim, dims=1)
ncoeffmean[:][coeffmask]
#ncoeffmean: empirical mean of noisy coeffs 0.007, 0.003...
#noisymean: analytical mean of noisy coeffs 0.01, 0.



#CONSISTENT code with mean subtraction
numfile = 1000
direc = "scratch_NM/StandardizedExp/Nx64/"
datfile = direc * "Data_" * string(numfile) * ".jld2" #Replace w SLURM array
loaddf = load(datfile)
true_img = loaddf["true_img"]
true_img = true_img .- mean(true_img)
sigma = loaddf["std"]
Nx=64
Nr=10000
empsn = []
isobool = false
apdbool = false
dhc_args = Dict(:doS2=>false, :doS20=>true, :apodize=>apdbool, :iso=>isobool)
filter_hash = fink_filter_hash(1, 8, nx=Nx, t=1, wd=1, Omega=true)
for n=1:Nr
    noisyim = randn((Nx, Nx)).*sigma .+ true_img
    push!(empsn, DHC_compute_wrapper(noisyim, filter_hash; dhc_args...))
end
coeffsim = hcat(empsn...)'
ncoeffmean = mean(coeffsim, dims=1)
Nf = size(filter_hash["filt_index"])[1]
coeffmask = falses(2+Nf+Nf^2)
coeffmask[Nf+3:end] .= triu(trues(Nf, Nf))[:]
#thcoeffmean, thcov = DHC_compute_S20r_noisy(true_img, filter_hash, fill(sigma, (Nx, Nx)); coeff_mask = coeffmask, dhc_args...)
thcoeffmean, thcov = DHC_compute_S20r_noisy_so(true_img, filter_hash, fill(sigma, (Nx, Nx)); coeff_mask = coeffmask, dhc_args...)
thcoeffmean2, thcov2 = DHC_compute_S20r_noisy_so2(true_img, filter_hash, fill(sigma, (Nx, Nx)); coeff_mask = coeffmask, dhc_args...)

plot(collect(1:595), thcoeffmean, label="th")
plot!(collect(1:595), mean(coeffsim[:, coeffmask], dims=1)[:], label="emp")
plot!(collect(1:595), thcoeffmean2, label="th2")

discmat = zeros(Nf, Nf)
triumask = triu(trues(Nf, Nf))
discmat[triumask] .= (thcoeffmean ./ mean(coeffsim[:, coeffmask], dims=1)[:])
heatmap(discmat, title="S20R")
heatmap(discmat, clim=(1.0, 10.0))
discmat
mask = Diagonal(trues(Nf, Nf))
discmat[mask]
ncoeffmean[:][coeffmask]


#Without mean subtraction? Still works here
numfile = 1000
direc = "scratch_NM/StandardizedExp/Nx64/"
datfile = direc * "Data_" * string(numfile) * ".jld2" #Replace w SLURM array
loaddf = load(datfile)
true_img = loaddf["true_img"]
#true_img = true_img .- mean(true_img)
sigma = loaddf["std"]
Nx=64
Nr=10000
empsn = []
isobool = false
apdbool = false
dhc_args = Dict(:doS2=>false, :doS20=>true, :apodize=>apdbool, :iso=>isobool)
filter_hash = fink_filter_hash(1, 8, nx=Nx, t=1, wd=1, Omega=true)
for n=1:Nr
    noisyim = randn((Nx, Nx)).*sigma .+ true_img
    push!(empsn, DHC_compute_wrapper(noisyim, filter_hash; dhc_args...))
end
coeffsim = hcat(empsn...)'
ncoeffmean = mean(coeffsim, dims=1)
Nf = size(filter_hash["filt_index"])[1]
coeffmask = falses(2+Nf+Nf^2)
coeffmask[Nf+3:end] .= triu(trues(Nf, Nf))[:]
#thcoeffmean, thcov = DHC_compute_S20r_noisy(true_img, filter_hash, fill(sigma, (Nx, Nx)); coeff_mask = coeffmask, dhc_args...)
thcoeffmean, thcov = DHC_compute_S20r_noisy_so(true_img, filter_hash, fill(sigma, (Nx, Nx)); coeff_mask = coeffmask, dhc_args...)
thcoeffmean2, thcov2 = DHC_compute_S20r_noisy_so2(true_img, filter_hash, fill(sigma, (Nx, Nx)); coeff_mask = coeffmask, dhc_args...)

plot(collect(1:595), thcoeffmean, label="th")
plot!(collect(1:595), mean(coeffsim[:, coeffmask], dims=1)[:], label="emp")
plot!(collect(1:595), thcoeffmean2, label="th2")

discmat = zeros(Nf, Nf)
triumask = triu(trues(Nf, Nf))
discmat[triumask] .= (thcoeffmean ./ mean(coeffsim[:, coeffmask], dims=1)[:])
heatmap(discmat, title="S20R")
heatmap(discmat, clim=(1.0, 10.0))
discmat
mask = Diagonal(trues(Nf, Nf))
discmat[mask]

#Without mean subtraction and without apodization? Still works here
numfile = 1000
direc = "scratch_NM/StandardizedExp/Nx64/"
datfile = direc * "Data_" * string(numfile) * ".jld2" #Replace w SLURM array
loaddf = load(datfile)
true_img = loaddf["true_img"]
#true_img = true_img .- mean(true_img)
sigma = loaddf["std"]
Nx=64
Nr=10000
empsn = []
isobool = false
apdbool = true
dhc_args = Dict(:doS2=>false, :doS20=>true, :apodize=>apdbool, :iso=>isobool)
filter_hash = fink_filter_hash(1, 8, nx=Nx, t=1, wd=1, Omega=true)
for n=1:Nr
    noisyim = randn((Nx, Nx)).*sigma .+ true_img
    push!(empsn, DHC_compute_wrapper(noisyim, filter_hash; dhc_args...))
end
coeffsim = hcat(empsn...)'
ncoeffmean = mean(coeffsim, dims=1)
Nf = size(filter_hash["filt_index"])[1]
coeffmask = falses(2+Nf+Nf^2)
coeffmask[Nf+3:end] .= triu(trues(Nf, Nf))[:]
#thcoeffmean, thcov = DHC_compute_S20r_noisy(true_img, filter_hash, fill(sigma, (Nx, Nx)); coeff_mask = coeffmask, dhc_args...)
thcoeffmean, thcov = DHC_compute_S20r_noisy_so(true_img, filter_hash, fill(sigma, (Nx, Nx)); coeff_mask = coeffmask, dhc_args...)
#thcoeffmean2, thcov2 = DHC_compute_S20r_noisy_so2(true_img, filter_hash, fill(sigma, (Nx, Nx)); coeff_mask = coeffmask, dhc_args...)

plot(collect(1:595), thcoeffmean, label="th")
plot!(collect(1:595), mean(coeffsim[:, coeffmask], dims=1)[:], label="emp")
plot!(collect(1:595), thcoeffmean2, label="th2")

discmat = zeros(Nf, Nf)
triumask = triu(trues(Nf, Nf))
discmat[triumask] .= (thcoeffmean ./ mean(coeffsim[:, coeffmask], dims=1)[:])
p = heatmap(discmat, title="S20R with apodization")
savefig(p, "scratch_NM/NewWrapper/5-30/apd_discrepancy.png")
heatmap(discmat, clim=(1.0, 10.0))
discmat
mask = Diagonal(trues(Nf, Nf))
discmat[mask]



#Code using loaddf
loaddf = load("scratch_NM/StandardizedExp/Nx64/Data_1000.jld2")
dhc_args = Dict(:doS2=>false, :doS20=>true, :apodize=>apdbool, :iso=>isobool)
true_img = loaddf["true_img"]
filter_hash = fink_filter_hash(1, 8, nx=Nx, t=1, wd=1, Omega=true)
truec = log.(DHC_compute_wrapper(loaddf["true_img"],filter_hash, norm=false; dhc_args...)[coeffmask])
initc = log.(DHC_compute_wrapper(loaddf["init"], filter_hash, norm=false; dhc_args...)[coeffmask])
#reconc = log.(DHC_compute_wrapper(recon_img, logsfddbn["filter_hash"], norm=false; dhc_args...)[coeffmask])
ind1, ind2 = 1, 3

#Theoretical noisy
noisymean, noisycov = DHC_compute_S20r_noisy_so(loaddf["true_img"], filter_hash, fill(loaddf["std"], (Nx, Nx)); coeff_mask = coeffmask, dhc_args...)
ncovreg = noisycov + I*1e-10
noisydbn = MultivariateNormal(convert(Array{Float64, 1}, noisymean), ncovreg)
noisysamp = rand(noisydbn, 10000)
#Log theoretical noisy
noisysamp = log.((x->maximum([x, 0])).(noisysamp'))
p = scatter(sampssfd[:, ind1], sampssfd[:, ind2], label="Samples_SFD_Prior", legend=(0.1, 0.1))
scatter!(noisysamp[:, ind1], noisysamp[:, ind2], label="Theoretical Noise-Dbn Using True", legend=(0.1, 0.1))
scatter!([truec[ind1]], [truec[ind2]], label="True Coeff", legend=(0.1, 0.1), c="black")
scatter!([initc[ind1]], [initc[ind2]], label="Init Coeff", legend=(0.1, 0.1))
xlabel!("J=1, L=1")
ylabel!("J=2, L=1")

#Why dont init and the analytical noisy dbn match?
#Empirical noisy
empsn = []
for n=1:Nr
    noisyim = randn((Nx, Nx)).*loaddf["std"] .+ true_img
    push!(empsn, DHC_compute_wrapper(noisyim, filter_hash;  dhc_args...))
end
noisymean
coeffsim = hcat(empsn...)'
ncoeffmean = mean(coeffsim, dims=1)
ncoeffmean[:][coeffmask]


#ncoeffmean: empirical mean of noisy coeffs 0.007, 0.003...
#noisymean: analytical mean of noisy coeffs 0.01, 0.


#Without apodization comparing theoretical / empriical
Nx=64
#Samples from logSFD prior
logbool = false
apdbool=false
isobool = false
filter_hash = fink_filter_hash(1, 8, nx=Nx, t=1, wd=1, Omega=true)
Nf = length(filter_hash["filt_index"])
sfdall = readsfd(Nx, logbool=logbool)
dhc_args = Dict(:doS2=>false, :doS20=>true, :apodize=>apdbool, :iso=>isobool)
truesamps = get_dbn_coeffs(sfdall, filter_hash, dhc_args)
truesamps = log.(truesamps)
sampmean = mean(truesamps, dims=1)
sampcov = (truesamps .- sampmean)' * (truesamps .- sampmean) ./ (size(truesamps)[1] - 1)
coeffmask = falses(2+Nf+Nf^2)
coeffmask[Nf+3:end] .= triu(trues(Nf, Nf))[:]
sampmean = sampmean[:][coeffmask]
covreg = sampcov[coeffmask, coeffmask]
cond(covreg)
sfddbn = MultivariateNormal(sampmean, covreg)
sampssfd = rand(sfddbn, 10000)
sampssfd = sampssfd'

loaddf = load("scratch_NM/StandardizedExp/Nx64/Data_1000.jld2")
true_img = loaddf["true_img"]
truec = log.(DHC_compute_wrapper(loaddf["true_img"], filter_hash, norm=false; dhc_args...)[coeffmask])
initc = log.(DHC_compute_wrapper(loaddf["init"], filter_hash, norm=false; dhc_args...)[coeffmask])
#reconc = log.(DHC_compute_wrapper(recon_img, logsfddbn["filter_hash"], norm=false; dhc_args...)[coeffmask])
ind1, ind2 = 1, 3

#Theoretical noisy
ncovreg = noisycov + I*1e-10
noisymean, noisycov = DHC_compute_S20r_noisy_so(loaddf["true_img"], filter_hash, fill(loaddf["std"], (Nx, Nx)); coeff_mask = coeffmask, dhc_args...)
noisydbn = MultivariateNormal(convert(Array{Float64, 1}, noisymean), ncovreg)
noisysamp = rand(noisydbn, 10000)
#Log theoretical noisy
noisysamp = log.((x->maximum([x, 0])).(noisysamp'))
p = scatter(sampssfd[:, ind1], sampssfd[:, ind2], label="Samples_SFD_Prior", legend=(0.1, 0.1))
scatter!(noisysamp[:, ind1], noisysamp[:, ind2], label="Theoretical Noise-Dbn Using True", legend=(0.1, 0.1))
scatter!([truec[ind1]], [truec[ind2]], label="True Coeff", legend=(0.1, 0.1), c="black")
scatter!([initc[ind1]], [initc[ind2]], label="Init Coeff", legend=(0.1, 0.1))
xlabel!("J=1, L=1")
ylabel!("J=2, L=1")
savefig(p, "scratch_NM/NewWrapper/5-30/Non_apd_logS1.png")
#Why dont init and the analytical noisy dbn match?
#Empirical noisy
Nr=10000
empsn = []
for n=1:Nr
    noisyim = randn((Nx, Nx)).*loaddf["std"] .+ true_img
    push!(empsn, DHC_compute_wrapper(noisyim, filter_hash;  dhc_args...))
end
coeffsim = hcat(empsn...)'
coeffsim = coeffsim[:, coeffmask]
coeffsim = log.(coeffsim)

p = scatter(sampssfd[:, ind1], sampssfd[:, ind2], label="Samples_SFD_Prior", legend=(0.1, 0.1))
scatter!(coeffsim[:, ind1], coeffsim[:, ind2], label="Empirical Noise-Dbn Using True", legend=(0.1, 0.1))
scatter!(noisysamp[:, ind1], noisysamp[:, ind2], label="Theoretical Noise-Dbn Using True", legend=(0.1, 0.1))
scatter!([truec[ind1]], [truec[ind2]], label="True Coeff", legend=(0.1, 0.1), c="black")
scatter!([initc[ind1]], [initc[ind2]], label="Init Coeff", legend=(0.1, 0.1))
xlabel!("J=1, L=1")
ylabel!("J=2, L=1")
savefig(p, "scratch_NM/NewWrapper/5-30/Non_apd_logS1_empiricalcomparison2.png")

proxymean, proxycov = DHC_compute_S20r_noisy_so(loaddf["init"], filter_hash, fill(loaddf["std"], (Nx, Nx)); coeff_mask = coeffmask, dhc_args...)
pcovreg = proxycov + 1e-10*I
proxydbn = MultivariateNormal(convert(Array{Float64, 1}, proxymean), pcovreg)
proxysamp = rand(proxydbn, 10000)
#Log theoretical noisy
proxysamp = log.((x->maximum([x, 0])).(proxysamp'))

p = scatter(sampssfd[:, ind1], sampssfd[:, ind2], label="Samples_SFD_Prior", legend=(0.1, 0.1))
scatter!(coeffsim[:, ind1], coeffsim[:, ind2], label="Empirical Noise-Dbn Using True", legend=(0.1, 0.1))
scatter!(noisysamp[:, ind1], noisysamp[:, ind2], label="Theoretical Noise-Dbn Using True", legend=(0.1, 0.1))
scatter!(proxysamp[:, ind1], proxysamp[:, ind2], label="Theoretical Noise-Dbn Using Init", legend=(0.1, 0.1))
scatter!([truec[ind1]], [truec[ind2]], label="True Coeff", legend=(0.1, 0.1), c="black")
scatter!([initc[ind1]], [initc[ind2]], label="Init Coeff", legend=(0.1, 0.1))
xlabel!("J=1, L=1")
ylabel!("J=2, L=1")
savefig(p, "scratch_NM/NewWrapper/5-30/Non_apd_logS1_empiricalcomparisonproxy.png")

#How different are shift(true) and shift(noisy)?
shift_noisy = proxymean - DHC_compute_wrapper(loaddf["init"], filter_hash, norm=false; dhc_args...)[coeffmask]
shift_true = noisymean - DHC_compute_wrapper(loaddf["true_img"], filter_hash, norm=false; dhc_args...)[coeffmask]

proxyres = zeros(Nf, Nf)
triumask = triu(trues(Nf, Nf))
proxyres[triumask] .= ((shift_noisy .- shift_true)./shift_true)[:]
ph = heatmap(proxyres, title="(Shift(Init) - Shift(True))/Shift(True)")
savefig(ph, "scratch_NM/NewWrapper/5-30/frac_devn_shifts.png")

ph = heatmap(proxyres, title="(Shift(Init) - Shift(True))/Shift(True)", clim=(-2.0, 2.0), c=cgrad([:blue, :white, :red]))
savefig(ph, "scratch_NM/NewWrapper/5-30/frac_devn_shifts_sym.png")

proxyres = zeros(Nf, Nf)
triumask = triu(trues(Nf, Nf))
proxyres[triumask] .= ((shift_noisy .- shift_true)./shift_true)[:]

#How different are shift(true, noise in quad) and shift(noisy)?
noisymeanquad, noisycovquad = DHC_compute_S20r_noisy_so(loaddf["true_img"], filter_hash, fill(sqrt(2)*loaddf["std"], (Nx, Nx)); coeff_mask = coeffmask, dhc_args...)
shift_truequad = noisymeanquad - DHC_compute_wrapper(loaddf["true_img"], filter_hash, norm=false; dhc_args...)[coeffmask]

proxyresq = zeros(Nf, Nf)
triumask = triu(trues(Nf, Nf))
proxyresq[triumask] .= ((shift_noisy .- shift_truequad)./shift_truequad)[:]
ph = heatmap(proxyresq, title="(Shift(Init) - Shift(True, rt(2)sig))/Shift(True, rt(2)sig)")
savefig(ph, "scratch_NM/NewWrapper/5-30/frac_devn_shifts_quad.png")

ph = heatmap(proxyresq, title="(Shift(Init) - Shift(True))/Shift(True)", clim=(-2.0, 2.0), c=cgrad([:blue, :white, :red]))
savefig(ph, "scratch_NM/NewWrapper/5-30/frac_devn_shifts_quadsym.png")

proxyres = zeros(Nf, Nf)
triumask = triu(trues(Nf, Nf))
proxyres[triumask] .= ((shift_noisy .- shift_true)./shift_true)[:]

Diagonal(proxyresq)


DHC_compute_wrapper(loaddf["init"], filter_hash, norm=false; dhc_args...)[coeffmask]

DHC_compute_wrapper(loaddf["true_img"], filter_hash, norm=false; dhc_args...)[coeffmask]

#Iterative without GD

#iterative
#INITIAL SETTINGS
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
#input::Array{Float64, 2}, filter_hash::Dict, s_targ_mean::Array{Float64, 1}, s_targ_invcov, dhc_args, LossFunc, dLossFunc;
#FFTthreads::Int=1, optim_settings=Dict([("iterations", 10)]), lambda=0.001, func_specific_params=nothing
s_noisy = DHC_compute_wrapper(loaddf["init"], filter_hash, norm=false; dhc_args...)[coeffmask]
img_guess = init
recon_img = img_guess

img_list = []
push!(img_list, img_guess)
num_rounds=5

#if YOU HAD THE TRUE IMAGE
sproxymean, scov = DHC_compute_S20r_noisy_so(loaddf["true_img"], filter_hash, fill(loaddf["std"], (Nx, Nx)); coeff_mask = coeffmask, dhc_args...) #Cant do this because function of image
strue = DHC_compute_wrapper(loaddf["true_img"], filter_hash, norm=false; dhc_args...)[coeffmask]
starget = s_noisy - (sproxymean - strue) #DHC_compute_wrapper(loaddf["true_img"], filter_hash, norm=false; dhc_args...)[coeffmask] #
scovinv = invert_covmat(scov, 1e-10)

func_specific_params = Dict([(:reg_input=> init), (:coeff_mask1=> coeffmask), (:target1=>starget), (:invcov1=>scovinv), (:coeff_mask2=> coeffmask), (:target2=>starget), (:invcov2=>scovinv), (:lambda2=>0.0), (:lambda3=>0.0)])

recon_settings["fname_save"] = fname_save * ".jld2"
recon_settings["optim_settings"] = optim_settings

if logbool
    error("Not implemented here")
end

res, recon_img = image_recon_derivsum_custom(img_guess, filter_hash, dhc_args, ReconFuncs.Loss3Gaussian, ReconFuncs.dLoss3Gaussian!; optim_settings=optim_settings, func_specific_params)
push!(img_list, recon_img)
heatmap(init)
heatmap(recon_img, title="Using true shift")
heatmap(true_img)
#Using proxy
s_noisy = DHC_compute_wrapper(loaddf["init"], filter_hash, norm=false; dhc_args...)[coeffmask]
img_guess = init
recon_img = img_guess


img_list = []
push!(img_list, img_guess)
num_rounds=5
s_noisy = DHC_compute_wrapper(loaddf["init"], filter_hash, norm=false; dhc_args...)[coeffmask]
img_guess = init
recon_img = img_guess

img_list = []
push!(img_list, img_guess)
num_rounds=5

#alTERNATIVE
sproxymean, scov = DHC_compute_S20r_noisy_so(loaddf["init"], filter_hash, fill(loaddf["std"], (Nx, Nx)); coeff_mask = coeffmask, dhc_args...) #Cant do this because function of image
sinit = DHC_compute_wrapper(loaddf["init"], filter_hash, norm=false; dhc_args...)[coeffmask]
starget = s_noisy - (sproxymean - sinit) #DHC_compute_wrapper(loaddf["true_img"], filter_hash, norm=false; dhc_args...)[coeffmask] #
scovinv = invert_covmat(scov, 1e-10)

func_specific_params = Dict([(:reg_input=> init), (:coeff_mask1=> coeffmask), (:target1=>starget), (:invcov1=>scovinv), (:coeff_mask2=> coeffmask), (:target2=>starget), (:invcov2=>scovinv), (:lambda2=>0.0), (:lambda3=>0.0)])

recon_settings["fname_save"] = fname_save * ".jld2"
recon_settings["optim_settings"] = optim_settings

if logbool
    error("Not implemented here")
end

res, recon_img = image_recon_derivsum_custom(init, filter_hash, dhc_args, ReconFuncs.Loss3Gaussian, ReconFuncs.dLoss3Gaussian!; optim_settings=optim_settings, func_specific_params)

heatmap(init)
p=heatmap(recon_img, title="Using proxy shift")
heatmap(true_img)

#Plotting panel
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
apdsmoothed = imfilter(init, Kernel.gaussian(0.8))
true_ps = calc_1dps_local(true_img, kbins)
initps = calc_1dps_local(init, kbins)
recps = calc_1dps_local(recon_img, kbins)
smoothps = calc_1dps_local(apdsmoothed, kbins)
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
savefig(p, "scratch_NM/NewWrapper/5-30/init_shift_denoised.png")


savefig(p, "scratch_NM/NewWrapper/5-30/iterative_proxyshift.png")
for i=1:num_rounds
    sproxymean, scov = DHC_compute_S20r_noisy_so(recon_img, filter_hash, fill(loaddf["std"], (Nx, Nx)); coeff_mask = coeffmask, dhc_args...) #Cant do this because function of image
    starget = s_noisy - sproxymean
    scovinv = invert_covmat(scov, 1e-10)

    func_specific_params = Dict([(:reg_input=> init), (:coeff_mask1=> coeffmask), (:target1=>starget), (:invcov1=>scovinv), (:coeff_mask2=> coeffmask), (:target2=>starget), (:invcov2=>scovinv)])


    recon_settings["fname_save"] = fname_save * ".jld2"
    recon_settings["optim_settings"] = optim_settings

    if logbool
        error("Not implemented here")
    end

    res, recon_img = image_recon_derivsum_custom(recon_img, filter_hash, dhc_args, ReconFuncs.Loss3Gaussian, ReconFuncs.dLoss3Gaussian!; optim_settings=optim_settings, func_specific_params)
    push!(img_list, recon_img)
end

heatmap(recon_img)



datfile = direc * "Data_" * string(numfile) * ".jld2" #Replace w SLURM array
loaddf = load(datfile)
true_img = loaddf["true_img"]
loaddf["std"]/mean(true_img)


##Reducin noise
Nx=64
filter_hash = fink_filter_hash(1, 8, nx=Nx, t=1, Omega=true)
psir = realspace_filter(Nx, filter_hash["filt_index"][4], filter_hash["filt_value"][4])

#Check s20r theoretical
numfile = 1000
direc = "scratch_NM/StandardizedExp/Nx64/"
datfile = direc * "Data_" * string(numfile) * ".jld2" #Replace w SLURM array
loaddf = load(datfile)
true_img = loaddf["true_img"]
true_img = true_img .- mean(true_img)
sigma = loaddf["std"]
Nx=64
Nr=10000
empsn = []
isobool = false
apdbool = false
dhc_args = Dict(:doS2=>false, :doS20=>true, :apodize=>apdbool, :iso=>isobool)
filter_hash = fink_filter_hash(1, 8, nx=Nx, t=1, wd=1, Omega=true)
for n=1:Nr
    noisyim = randn((Nx, Nx)).*sigma .+ true_img
    push!(empsn, DHC_compute_wrapper(noisyim, filter_hash; dhc_args...))
end
coeffsim = hcat(empsn...)'
ncoeffmean = mean(coeffsim, dims=1)
Nf = size(filter_hash["filt_index"])[1]
coeffmask = falses(2+Nf+Nf^2)
coeffmask[Nf+3:end] .= triu(trues(Nf, Nf))[:]
thcoeffmean, thcov = DHC_compute_S20r_noisy_so(true_img, filter_hash, fill(sigma, (Nx, Nx)); coeff_mask = coeffmask, dhc_args...)

disc = thcoeffmean[:] ./ ncoeffmean[:][coeffmask]
discmat = zeros(Nf, Nf)
triumask = triu(trues(Nf, Nf))
discmat[triumask] .= (thcoeffmean ./ mean(coeffsim[:, coeffmask], dims=1)[:])

fracmat = zeros(Nf, Nf)
fracmat[triumask] .= ncoeffmean[:][coeffmask] ./ DHC_compute_wrapper(true_img, filter_hash; dhc_args...)[coeffmask]
heatmap(fracmat)
#heatmap(discmat, title="S20R")
#heatmap(discmat, clim=(1.0, 10.0))
discmat
mask = Diagonal(trues(Nf, Nf))
discmat[mask]
heatmap(discmat, title="sigma=40% mean")


res = (thcoeffmean[:] .- ncoeffmean[:][coeffmask])./DHC_compute_wrapper(true_img, filter_hash; dhc_args...)[coeffmask]
resmat = zeros(Nf, Nf)
triumask = triu(trues(Nf, Nf))
resmat[triumask] .= (thcoeffmean .- mean(coeffsim[:, coeffmask], dims=1)[:])./DHC_compute_wrapper(true_img, filter_hash; dhc_args...)[coeffmask]
#heatmap(discmat, title="S20R")
#heatmap(discmat, clim=(1.0, 10.0))
resmat
mask = Diagonal(trues(Nf, Nf))
resmat[mask]
heatmap(resmat, title="(ThNoisyS - EmpNoisyS)/Strue sigma=40% mean")


#LESS NOISE
Nx=64
filter_hash = fink_filter_hash(1, 8, nx=Nx, t=1, Omega=true)
psir = realspace_filter(Nx, filter_hash["filt_index"][4], filter_hash["filt_value"][4])

#Check s20r theoretical
numfile = 1000
direc = "scratch_NM/StandardizedExp/Nx64/"
datfile = direc * "Data_" * string(numfile) * ".jld2" #Replace w SLURM array
loaddf = load(datfile)
true_img = loaddf["true_img"]
#true_img = true_img .- mean(true_img)
sigma = loaddf["std"]
Nx=64
Nr=10000
empsn = []
isobool = false
apdbool = false
dhc_args = Dict(:doS2=>false, :doS20=>true, :apodize=>apdbool, :iso=>isobool)
filter_hash = fink_filter_hash(1, 8, nx=Nx, t=1, wd=1, Omega=true)
for n=1:Nr
    noisyim = randn((Nx, Nx)).*sigma .+ true_img
    push!(empsn, DHC_compute_wrapper(noisyim, filter_hash; dhc_args...))
end
coeffsim = hcat(empsn...)'
ncoeffmean = mean(coeffsim, dims=1)
Nf = size(filter_hash["filt_index"])[1]
coeffmask = falses(2+Nf+Nf^2)
coeffmask[Nf+3:end] .= triu(trues(Nf, Nf))[:]
thcoeffmean, thcov = DHC_compute_S20r_noisy_so(true_img, filter_hash, fill(sigma, (Nx, Nx)); coeff_mask = coeffmask, dhc_args...)

disc = thcoeffmean[:] ./ ncoeffmean[:][coeffmask]
discmat = zeros(Nf, Nf)
triumask = triu(trues(Nf, Nf))
discmat[triumask] .= (thcoeffmean ./ mean(coeffsim[:, coeffmask], dims=1)[:])
#heatmap(discmat, title="S20R")
#heatmap(discmat, clim=(1.0, 10.0))
discmat
mask = Diagonal(trues(Nf, Nf))
discmat[mask]
heatmap(discmat, title="sigma=40% mean, no meansub")
heatmap(true_img)
res = (thcoeffmean[:] .- ncoeffmean[:][coeffmask])./DHC_compute_wrapper(true_img, filter_hash; dhc_args...)[coeffmask]
resmat = zeros(Nf, Nf)
triumask = triu(trues(Nf, Nf))
resmat[triumask] .= (thcoeffmean .- mean(coeffsim[:, coeffmask], dims=1)[:])./DHC_compute_wrapper(true_img, filter_hash; dhc_args...)[coeffmask]
#heatmap(discmat, title="S20R")
#heatmap(discmat, clim=(1.0, 10.0))
resmat
mask = Diagonal(trues(Nf, Nf))
resmat[mask]
heatmap(resmat, title="(ThNoisyS - EmpNoisyS)/Strue, sigma=4% mean")


smat = zeros(Nf, Nf)
triumask = triu(trues(Nf, Nf))
smat[triumask] .= DHC_compute_wrapper(true_img, filter_hash; dhc_args...)[coeffmask]
heatmap(log.(smat))

rmat = zeros(Nf, Nf)
rmat[triumask] .= (ncoeffmean[:][coeffmask]) ./ DHC_compute_wrapper(true_img, filter_hash; dhc_args...)[coeffmask]
heatmap(rmat)

rmat


#Does the discrepancy scale with sigma?
function scaling_loop(sigma_perc, idx=594)
    Nx=64
    filter_hash = fink_filter_hash(1, 8, nx=Nx, t=1, Omega=true)
    psir = realspace_filter(Nx, filter_hash["filt_index"][4], filter_hash["filt_value"][4])

    #Check s20r theoretical
    numfile = 1000
    direc = "scratch_NM/StandardizedExp/Nx64/"
    datfile = direc * "Data_" * string(numfile) * ".jld2" #Replace w SLURM array
    loaddf = load(datfile)


    Nx=64
    Nr=10000

    isobool = false
    apdbool = false
    dhc_args = Dict(:doS2=>false, :doS20=>true, :apodize=>apdbool, :iso=>isobool)
    filter_hash = fink_filter_hash(1, 8, nx=Nx, t=1, wd=1, Omega=true)
    p4rat = []
    p2rat = []
    thlist = []
    emplist = []
    for sp in sigma_perc
        true_img = loaddf["true_img"]
        empsn = []
        sigma = mean(true_img)*sp
        for n=1:Nr
            noisyim = randn((Nx, Nx)).*sigma .+ true_img
            push!(empsn, DHC_compute_wrapper(noisyim, filter_hash; dhc_args...))
        end
        coeffsim = hcat(empsn...)'
        ncoeffmean = mean(coeffsim, dims=1)
        Nf = size(filter_hash["filt_index"])[1]
        coeffmask = falses(2+Nf+Nf^2)
        coeffmask[Nf+3:end] .= triu(trues(Nf, Nf))[:]
        true_img = loaddf["true_img"]
        println(mean(true_img))
        thcoeffmean, thcov = DHC_compute_S20r_noisy_so(true_img, filter_hash, fill(sigma, (Nx, Nx)); coeff_mask = coeffmask, dhc_args...)

        residual = thcoeffmean[:] .- ncoeffmean[:][coeffmask]
        push!(emplist, ncoeffmean[:][coeffmask])
        push!(thlist, thcoeffmean[:])
        push!(p4rat, residual[idx]/(sigma^4))
        push!(p2rat, residual[idx]/(sigma^2))
    end
    p=display(plot(sigma_perc, p2rat, title="Residual/sigma^2"))
    p=display(plot(sigma_perc, p4rat, title="Residual/sigma^4"))
    return thlist, emplist, p2rat, p4rat

end

function scaling(sigma_perc, idx=594)
    Nx=64
    filter_hash = fink_filter_hash(1, 8, nx=Nx, t=1, Omega=true)
    psir = realspace_filter(Nx, filter_hash["filt_index"][4], filter_hash["filt_value"][4])

    #Check s20r theoretical
    numfile = 1000
    direc = "scratch_NM/StandardizedExp/Nx64/"
    datfile = direc * "Data_" * string(numfile) * ".jld2" #Replace w SLURM array
    loaddf = load(datfile)

    Nr=1000
    isobool = false
    apdbool = false
    dhc_args = Dict(:doS2=>false, :doS20=>true, :apodize=>apdbool, :iso=>isobool)
    filter_hash = fink_filter_hash(1, 8, nx=Nx, t=1, wd=1, Omega=true)

    true_img = loaddf["true_img"]
    empsn = []
    sigma = mean(true_img)*sigma_perc
    for n=1:Nr
        noisyim = randn((Nx, Nx)).*sigma .+ true_img
        push!(empsn, DHC_compute_wrapper(noisyim, filter_hash; dhc_args...))
    end
    coeffsim = hcat(empsn...)'
    ncoeffmean = mean(coeffsim, dims=1)
    Nf = size(filter_hash["filt_index"])[1]
    coeffmask = falses(2+Nf+Nf^2)
    coeffmask[Nf+3:end] .= triu(trues(Nf, Nf))[:]
    thcoeffmean, thcov = DHC_compute_S20r_noisy_so(true_img, filter_hash, fill(sigma, (Nx, Nx)); coeff_mask = coeffmask, dhc_args...)

    residual = thcoeffmean[:] .- ncoeffmean[:][coeffmask]

    return thcoeffmean[:], ncoeffmean[:][coeffmask], residual[idx]/(sigma^2), residual[idx]/(sigma^4)
end

emplist = []
thlist = []
p2rat = []
p4rat = []
siglist = [0.025, 0.05, 0.075, 0.1, 0.2, 0.4, 0.5]
for sigperc in siglist
    thvec, empvec, p2vec, p4vec = scaling(sigperc)
    push!(thlist, thvec)
    push!(emplist, empvec)
    push!(p2rat, p2vec)
    push!(p4rat, p4vec)
end

p= plot(siglist, p2rat, label="Error/Sigma^2")
plot!(siglist, p4rat, label="Error/Sigma^4")

numfile=1000
direc = "scratch_NM/StandardizedExp/Nx64/"
datfile = direc * "Data_" * string(numfile) * ".jld2" #Replace w SLURM array
loaddf = load(datfile)
Nf = length(filter_hash["filt_index"])
coeffmask = falses(2+Nf+Nf^2)
coeffmask[Nf+3:end] .= triu(trues(Nf, Nf))[:]

isobool = false
apdbool = false
dhc_args = Dict(:doS2=>false, :doS20=>true, :apodize=>apdbool, :iso=>isobool)
Nx=64
filter_hash = fink_filter_hash(1, 8, nx=Nx, t=1, wd=1, Omega=true)
strue = DHC_compute_wrapper(loaddf["true_img"], filter_hash; norm=false, dhc_args...)[coeffmask]


ratio_noisy_true = zeros(Nf, Nf)
triumask = triu(trues(Nf, Nf))
ratio_noisy_true[triumask] .= emplist[4]./strue
heatmap(ratio_noisy_true, label="Empirical Noisy Mean/True Coeffs")


##6-4
#Iterative: using empirical noisy dbn and true
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
#input::Array{Float64, 2}, filter_hash::Dict, s_targ_mean::Array{Float64, 1}, s_targ_invcov, dhc_args, LossFunc, dLossFunc;
#FFTthreads::Int=1, optim_settings=Dict([("iterations", 10)]), lambda=0.001, func_specific_params=nothing
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

func_specific_params = Dict([(:reg_input=> init), (:coeff_mask1=> coeffmask), (:target1=>starget), (:invcov1=>scovinv), (:coeff_mask2=> coeffmask), (:target2=>starget), (:invcov2=>scovinv), (:lambda2=>0.0), (:lambda3=>0.0)])

recon_settings["fname_save"] = fname_save * ".jld2"
recon_settings["optim_settings"] = optim_settings

if logbool
    error("Not implemented here")
end

res, recon_img = image_recon_derivsum_custom(img_guess, filter_hash, dhc_args, ReconFuncs.Loss3Gaussian, ReconFuncs.dLoss3Gaussian!; optim_settings=optim_settings, func_specific_params)
heatmap(init)
heatmap(recon_img, title="Using true shift")
heatmap(true_img)

#Plotting code
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
apdsmoothed = imfilter(init, Kernel.gaussian(0.8))
true_ps = calc_1dps_local(true_img, kbins)
initps = calc_1dps_local(init, kbins)
recps = calc_1dps_local(recon_img, kbins)
smoothps = calc_1dps_local(apdsmoothed, kbins)
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
savefig(p, "scratch_NM/NewWrapper/5-30/denoising_trueempnoisydbn.png")


##Iterative: using empirical noisy dbn and true: with pixelwise regularization
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
#input::Array{Float64, 2}, filter_hash::Dict, s_targ_mean::Array{Float64, 1}, s_targ_invcov, dhc_args, LossFunc, dLossFunc;
#FFTthreads::Int=1, optim_settings=Dict([("iterations", 10)]), lambda=0.001, func_specific_params=nothing
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

func_specific_params = Dict([(:reg_input=> init), (:coeff_mask1=> coeffmask), (:target1=>starget), (:invcov1=>scovinv), (:coeff_mask2=> coeffmask), (:target2=>starget), (:invcov2=>scovinv), (:lambda2=>0.0), (:lambda3=>0.0)])

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
apdsmoothed = imfilter(init, Kernel.gaussian(0.8))
true_ps = calc_1dps_local(true_img, kbins)
initps = calc_1dps_local(init, kbins)
recps = calc_1dps_local(recon_img, kbins)
smoothps = calc_1dps_local(apdsmoothed, kbins)
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
savefig(p, "scratch_NM/NewWrapper/5-30/denoisingwpixreg_trueempnoisydbn.png")

##Iterative: using empirical noisy dbn and true: with pixelwise regularization
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
#input::Array{Float64, 2}, filter_hash::Dict, s_targ_mean::Array{Float64, 1}, s_targ_invcov, dhc_args, LossFunc, dLossFunc;
#FFTthreads::Int=1, optim_settings=Dict([("iterations", 10)]), lambda=0.001, func_specific_params=nothing
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
starget = strue #DHC_compute_wrapper(loaddf["true_img"], filter_hash, norm=false; dhc_args...)[coeffmask] #
scovinv = invert_covmat(scov, 1e-10)

func_specific_params = Dict([(:reg_input=> init), (:coeff_mask1=> coeffmask), (:target1=>starget), (:invcov1=>scovinv), (:coeff_mask2=> coeffmask), (:target2=>starget), (:invcov2=>scovinv), (:lambda2=>0.0), (:lambda3=>0.0)])

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
savefig(p, "scratch_NM/NewWrapper/5-30/denoisingwstrue_trueempnoisydbn.png")


#S2R
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

optim_settings = Dict([("iterations", 1000), ("norm", false), ("minmethod", ConjugateGradient())])
recon_settings = Dict([("log", logbool), ("Invcov_matrix", ARGS_buffer[6]), ("optim_settings", optim_settings), ("eps_value_sfd", 1e-5), ("eps_value_init", 1e-10)]) #Add constraints

recon_settings["datafile"] = datfile

if dhc_args[:iso]
    error("Not constructed for iso")
else #Not iso
    coeffmask = falses(2+Nf+Nf^2)
    coeffmask[Nf+3:end] .= Diagonal(triu(trues(Nf, Nf)))[:]
end


#Weight given to terms
lval2 = 0.0
lval3 = 1.0

println("Regularizer Lambda=", round(lval3, sigdigits=3))
#input::Array{Float64, 2}, filter_hash::Dict, s_targ_mean::Array{Float64, 1}, s_targ_invcov, dhc_args, LossFunc, dLossFunc;
#FFTthreads::Int=1, optim_settings=Dict([("iterations", 10)]), lambda=0.001, func_specific_params=nothing
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
starget = strue #DHC_compute_wrapper(loaddf["true_img"], filter_hash, norm=false; dhc_args...)[coeffmask] #
scovinv = invert_covmat(scov, 1e-10)

func_specific_params = Dict([(:reg_input=> init), (:coeff_mask1=> coeffmask), (:target1=>starget), (:invcov1=>scovinv), (:coeff_mask2=> coeffmask), (:target2=>starget), (:invcov2=>scovinv), (:lambda2=>0.0), (:lambda3=>0.0)])

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
apdsmoothed = imfilter(init, Kernel.gaussian(0.9))
true_ps = calc_1dps_local(true_img, kbins)
initps = calc_1dps_local(init, kbins)
recps = calc_1dps_local(recon_img, kbins)
smoothps = calc_1dps_local(apdsmoothed, kbins)
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
savefig(p, "scratch_NM/NewWrapper/5-30/denoisingwstrueS2R_trueempnoisydbn.png")

##
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


#Code using loaddf
loaddf = load("scratch_NM/StandardizedExp/Nx64/Data_1000.jld2")
dhc_args = Dict(:doS2=>false, :doS20=>true, :apodize=>apdbool, :iso=>isobool)
true_img = loaddf["true_img"]
filter_hash = fink_filter_hash(1, 8, nx=Nx, t=1, wd=1, Omega=true)
truec = log.(DHC_compute_wrapper(loaddf["true_img"],filter_hash, norm=false; dhc_args...)[coeffmask])
initc = log.(DHC_compute_wrapper(loaddf["init"], filter_hash, norm=false; dhc_args...)[coeffmask])
#reconc = log.(DHC_compute_wrapper(recon_img, logsfddbn["filter_hash"], norm=false; dhc_args...)[coeffmask])
ind1, ind2 = 1, 3

#Theoretical noisy
noisymean, noisycov = DHC_compute_S20r_noisy_so(loaddf["true_img"], filter_hash, fill(loaddf["std"], (Nx, Nx)); coeff_mask = coeffmask, dhc_args...)
ncovreg = noisycov + I*1e-10
noisydbn = MultivariateNormal(convert(Array{Float64, 1}, noisymean), ncovreg)
noisysamp = rand(noisydbn, 10000)
#Log theoretical noisy
noisysamp = log.((x->maximum([x, 0])).(noisysamp'))
p = scatter(sampssfd[:, ind1], sampssfd[:, ind2], label="Samples_SFD_Prior", legend=(0.1, 0.1))
scatter!(noisysamp[:, ind1], noisysamp[:, ind2], label="Theoretical Noise-Dbn Using True", legend=(0.1, 0.1))
scatter!([truec[ind1]], [truec[ind2]], label="True Coeff", legend=(0.1, 0.1), c="black")
scatter!([initc[ind1]], [initc[ind2]], label="Init Coeff", legend=(0.1, 0.1))
xlabel!("J=1, L=1")
ylabel!("J=2, L=1")

#Why dont init and the analytical noisy dbn match?
#Empirical noisy
empsn = []
for n=1:Nr
    noisyim = randn((Nx, Nx)).*loaddf["std"] .+ true_img
    push!(empsn, DHC_compute_wrapper(noisyim, filter_hash;  dhc_args...))
end
noisymean
coeffsim = hcat(empsn...)'
ncoeffmean = mean(coeffsim, dims=1)
ncoeffmean[:][coeffmask]


#ncoeffmean: empirical mean of noisy coeffs 0.007, 0.003...
#noisymean: analytical mean of noisy coeffs 0.01, 0.


#Without apodization comparing theoretical / empriical
Nx=64
#Samples from logSFD prior
logbool = false
apdbool=false
isobool = false
filter_hash = fink_filter_hash(1, 8, nx=Nx, t=1, wd=1, Omega=true)
Nf = length(filter_hash["filt_index"])
sfdall = readsfd(Nx, logbool=logbool)
dhc_args = Dict(:doS2=>false, :doS20=>true, :apodize=>apdbool, :iso=>isobool)
truesamps = get_dbn_coeffs(sfdall, filter_hash, dhc_args)
truesamps = log.(truesamps)
sampmean = mean(truesamps, dims=1)
sampcov = (truesamps .- sampmean)' * (truesamps .- sampmean) ./ (size(truesamps)[1] - 1)
coeffmask = falses(2+Nf+Nf^2)
coeffmask[Nf+3:end] .= triu(trues(Nf, Nf))[:]
sampmean = sampmean[:][coeffmask]
covreg = sampcov[coeffmask, coeffmask]
cond(covreg)
sfddbn = MultivariateNormal(sampmean, covreg)
sampssfd = rand(sfddbn, 10000)
sampssfd = sampssfd'

loaddf = load("scratch_NM/StandardizedExp/Nx64/Data_1000.jld2")
true_img = loaddf["true_img"]
truec = log.(DHC_compute_wrapper(loaddf["true_img"], filter_hash, norm=false; dhc_args...)[coeffmask])
initc = log.(DHC_compute_wrapper(loaddf["init"], filter_hash, norm=false; dhc_args...)[coeffmask])
#reconc = log.(DHC_compute_wrapper(recon_img, logsfddbn["filter_hash"], norm=false; dhc_args...)[coeffmask])
ind1, ind2 = 1, 595

#Theoretical noisy
ncovreg = noisycov + I*1e-10
noisymean, noisycov = DHC_compute_S20r_noisy_so(loaddf["true_img"], filter_hash, fill(loaddf["std"], (Nx, Nx)); coeff_mask = coeffmask, dhc_args...)
noisydbn = MultivariateNormal(convert(Array{Float64, 1}, noisymean), ncovreg)
noisysamp = rand(noisydbn, 10000)
#Log theoretical noisy
noisysamp = log.((x->maximum([x, 0])).(noisysamp'))
Nr=10000
empsn = []
for n=1:Nr
    noisyim = randn((Nx, Nx)).*loaddf["std"] .+ true_img
    push!(empsn, DHC_compute_wrapper(noisyim, filter_hash;  dhc_args...))
end
coeffsim = hcat(empsn...)'
coeffsim = coeffsim[:, coeffmask]
coeffsim = log.(coeffsim)
ind1, ind2 = 1, 594
p = scatter(sampssfd[:, ind1], sampssfd[:, ind2], label="Samples_SFD_Prior", legend=(0.1, 0.1))
scatter!(coeffsim[:, ind1], coeffsim[:, ind2], label="Empirical Noise-Dbn Using True", legend=(0.1, 0.1))
scatter!(noisysamp[:, ind1], noisysamp[:, ind2], label="Theoretical Noise-Dbn Using True", legend=(0.1, 0.1))
scatter!([truec[ind1]], [truec[ind2]], label="True Coeff", legend=(0.1, 0.1), c="black")
scatter!([initc[ind1]], [initc[ind2]], label="Init Coeff", legend=(0.1, 0.1))
xlabel!("J=1, L=1")
ylabel!("J=4,L=8-Omega")



#=
##Iterative: using theoretical noisy dbn and true
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
#input::Array{Float64, 2}, filter_hash::Dict, s_targ_mean::Array{Float64, 1}, s_targ_invcov, dhc_args, LossFunc, dLossFunc;
#FFTthreads::Int=1, optim_settings=Dict([("iterations", 10)]), lambda=0.001, func_specific_params=nothing
img_guess = init
recon_img = img_guess

img_list = []
push!(img_list, img_guess)
num_rounds=5

#if YOU HAD THE TRUE IMAGE, calculating the shift empirically
sproxymean, scov = DHC_compute_S20r_noisy_so(loaddf["true_img"], filter_hash, fill(loaddf["std"], (Nx, Nx)); coeff_mask = coeffmask, dhc_args...) #Cant do this because function of image
strue = DHC_compute_wrapper(loaddf["true_img"], filter_hash, norm=false; dhc_args...)[coeffmask]
shift_true = sproxymean .- strue
s_noisy = DHC_compute_wrapper(loaddf["init"], filter_hash, norm=false; dhc_args...)[coeffmask]
starget = s_noisy - shift_true #DHC_compute_wrapper(loaddf["true_img"], filter_hash, norm=false; dhc_args...)[coeffmask] #
scovinv = invert_covmat(scov, 1e-10)

func_specific_params = Dict([(:reg_input=> init), (:coeff_mask1=> coeffmask), (:target1=>starget), (:invcov1=>scovinv), (:coeff_mask2=> coeffmask), (:target2=>starget), (:invcov2=>scovinv), (:lambda2=>0.0), (:lambda3=>0.0)])

recon_settings["fname_save"] = fname_save * ".jld2"
recon_settings["optim_settings"] = optim_settings

if logbool
    error("Not implemented here")
end

res, recon_img = image_recon_derivsum_custom(img_guess, filter_hash, dhc_args, ReconFuncs.Loss3Gaussian, ReconFuncs.dLoss3Gaussian!; optim_settings=optim_settings, func_specific_params)


#alTERNATIVE
sproxymean, scov = DHC_compute_S20r_noisy_so(loaddf["init"], filter_hash, fill(loaddf["std"], (Nx, Nx)); coeff_mask = coeffmask, dhc_args...) #Cant do this because function of image
sinit = DHC_compute_wrapper(loaddf["init"], filter_hash, norm=false; dhc_args...)[coeffmask]
starget = s_noisy - (sproxymean - sinit) #DHC_compute_wrapper(loaddf["true_img"], filter_hash, norm=false; dhc_args...)[coeffmask] #
scovinv = invert_covmat(scov, 1e-10)

func_specific_params = Dict([(:reg_input=> init), (:coeff_mask1=> coeffmask), (:target1=>starget), (:invcov1=>scovinv), (:coeff_mask2=> coeffmask), (:target2=>starget), (:invcov2=>scovinv), (:lambda2=>0.0), (:lambda3=>0.0)])

recon_settings["fname_save"] = fname_save * ".jld2"
recon_settings["optim_settings"] = optim_settings

if logbool
    error("Not implemented here")
end

res, recon_img = image_recon_derivsum_custom(init, filter_hash, dhc_args, ReconFuncs.Loss3Gaussian, ReconFuncs.dLoss3Gaussian!; optim_settings=optim_settings, func_specific_params)
push!(img_list, recon_img)
heatmap(init)
heatmap(recon_img, title="Using true shift")
heatmap(true_img)
=#


##Iterative: using empirical noisy dbn and true: with pixelwise regularization
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
#input::Array{Float64, 2}, filter_hash::Dict, s_targ_mean::Array{Float64, 1}, s_targ_invcov, dhc_args, LossFunc, dLossFunc;
#FFTthreads::Int=1, optim_settings=Dict([("iterations", 10)]), lambda=0.001, func_specific_params=nothing
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
starget = strue #DHC_compute_wrapper(loaddf["true_img"], filter_hash, norm=false; dhc_args...)[coeffmask] #
scovinv = invert_covmat(scov, 1e-10)

func_specific_params = Dict([(:reg_input=> init), (:coeff_mask1=> coeffmask), (:target1=>starget), (:invcov1=>scovinv), (:coeff_mask2=> coeffmask), (:target2=>starget), (:invcov2=>scovinv), (:lambda2=>0.0), (:lambda3=>0.0)])

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
savefig(p, "scratch_NM/NewWrapper/5-30/denoisingwstrue_trueempnoisydbn.png")
fracres = (init .- true_img)./true_img
fps = (initps .- true_ps)./true_ps
println("Mean Abs Frac, Init = ", round(mean(abs.(fracres)), digits=3), "Smoothed = ", round(mean(abs.((apdsmoothed .- true_img)./true_img)), digits=3), "Recon = ", round(mean(abs.((recon_img .- true_img)./true_img)), digits=3))
println("MSE, Init = ", round(mean((init .- true_img).^2), digits=5), "Smoothed = ", round(mean((apdsmoothed .- true_img).^2), digits=5), "Recon = ", round(mean(apdsmoothed), digits=5))
println("Power Spec Frac Res, Init = ", round(mean(abs.(fps)), digits=3), "Smoothed = ", round(mean(abs.(smoothps .- true_ps)), digits=3), "Recon = ", round(mean(abs.(recps .- true_ps)), digits=3))



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
lval3=1.0

println("Regularizer Lambda=", round(lval3, sigdigits=3))
#input::Array{Float64, 2}, filter_hash::Dict, s_targ_mean::Array{Float64, 1}, s_targ_invcov, dhc_args, LossFunc, dLossFunc;
#FFTthreads::Int=1, optim_settings=Dict([("iterations", 10)]), lambda=0.001, func_specific_params=nothing
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

#4) Why oversmoothed
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
savefig(p, "scratch_NM/NewWrapper/5-30/denoisingwstrue_trueempnoisydbn_lam100.png")
fracres = (init .- true_img)./true_img
fps = (initps .- true_ps)./true_ps
println("Mean Abs Frac, Init = ", round(mean(abs.(fracres)), digits=3), "Smoothed = ", round(mean(abs.((apdsmoothed .- true_img)./true_img)), digits=3), "Recon = ", round(mean(abs.((recon_img .- true_img)./true_img)), digits=3))
println("MSE, Init = ", round(mean((init .- true_img).^2), digits=5), "Smoothed = ", round(mean((apdsmoothed .- true_img).^2), digits=5), "Recon = ", round(mean(apdsmoothed), digits=5))
println("Power Spec Frac Res, Init = ", round(mean(abs.(fps)), digits=3), "Smoothed = ", round(mean(abs.(smoothps .- true_ps)), digits=3), "Recon = ", round(mean(abs.(recps .- true_ps)), digits=3))
