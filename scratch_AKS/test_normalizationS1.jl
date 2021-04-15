using Statistics
using FFTW
using Plots
using BenchmarkTools
using Profile
using LinearAlgebra
using Measures
using ProgressMeter
push!(LOAD_PATH, pwd()*"/main")
using DHC_2DUtils

function DHC_compute_temp(image::Array{Float64,2}, filter_hash::Dict, filter_hash2::Dict=filter_hash;
    doS2::Bool=true, doS20::Bool=false, norm=true, iso=false, FFTthreads=2, normS1::Bool=false)
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
                im_rd_0_1[:,:,f] .= abs2.(P*zarr) end
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
            # Loop over f2 and do second-order convolution
            if normS1
                normS1pwr = S1[f1]
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

filter_hash = fink_filter_hash(1,8,nx=256,wd=2)

DHC_compute_temp(rand(256,256), filter_hash, iso=false, normS1=true)
