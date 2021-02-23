##Home to deprecated functions for future reference
# AKS 2021_02_22

function DHC_compute_old(image::Array{Float64,2}, filter_hash, filter_hash2)
    # Use 2 threads for FFT
    FFTW.set_num_threads(2)

    # array sizes
    (Nx, Ny)  = size(image)
    (Nf, )    = size(filter_hash["filt_index"])

    # allocate coeff arrays
    out_coeff = []
    S0  = zeros(Float64, 2)
    S1  = zeros(Float64, Nf)
    S20 = zeros(Float64, Nf, Nf)
    S12 = zeros(Float64, Nf, Nf)
    S2  = zeros(Float64, Nf, Nf)  # traditional 2nd order

    # allocate image arrays for internal use
    im_fdf_0_1 = zeros(Float64,           Nx, Ny, Nf)   # this must be zeroed!
    im_rd_0_1  = Array{Float64, 3}(undef, Nx, Ny, Nf)

    ## 0th Order
    S0[1]   = mean(image)
    norm_im = image.-S0[1]
    S0[2]   = sum(norm_im .* norm_im)/(Nx*Ny)
    norm_im ./= sqrt(Nx*Ny*S0[2])

    append!(out_coeff,S0[:])

    ## 1st Order
    im_fd_0 = fft(norm_im)  # total power=1.0

    # unpack filter_hash
    f_ind   = filter_hash["filt_index"]  # (J, L) array of filters represented as index value pairs
    f_val   = filter_hash["filt_value"]

    f_ind2   = filter_hash2["filt_index"]  # (J, L) array of filters represented as index value pairs
    f_val2   = filter_hash2["filt_value"]

    zarr = zeros(ComplexF64, Nx, Ny)  # temporary array to fill with zvals

    # make a FFTW "plan" for an array of the given size and type
    P   = plan_ifft(im_fd_0)   # P is an operator, P*im is ifft(im)

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
                zarr[ind] = zval
                im_fdf_0_1[ind,f] = abs(zval)
            end
            S1[f] = S1tot/(Nx*Ny)  # image power
            im_rd_0_1[:,:,f] .= abs2.(P*zarr)
            zarr[f_i] .= 0
        end
    end
    append!(out_coeff, S1[:])

    # we stored the abs()^2, so take sqrt (this is faster to do all at once)
    im_rd_0_1 .= sqrt.(im_rd_0_1)

    ## 2nd Order
    #Amat = reshape(im_fdf_0_1, Nx*Ny, Nf)
    #S12  = Amat' * Amat
    #Amat = reshape(im_rd_0_1, Nx*Ny, Nf)
    #S20  = Amat' * Amat

    #append!(out_coeff, S20)
    #append!(out_coeff, S12)

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
    append!(out_coeff, S2)

    return out_coeff
end
