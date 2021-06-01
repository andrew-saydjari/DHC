## Core compute function

function eqws_compute(image::Array{Float64,2}, filter_hash::Dict, filter_hash2::Dict=filter_hash;
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

function eqws_compute_convmap(image::Array{Float64,2}, filter_hash::Dict, filter_hash2::Dict=filter_hash;
    doS2::Bool=true, doS20::Bool=false, norm=true, iso=false, FFTthreads=2, normS1::Bool=false, normS1iso::Bool=false,
    p=2)
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

    # allocate image arrays for internal use
    im_rd_0_1  = Array{Float64, 3}(undef, Nx, Ny, Nf)
    im_rd_0_2  = Array{Float64, 4}(undef, Nx, Ny, Nf, Nf)

    ## 0th Order
    S0[1]   = mean(image)
    norm_im = image.-S0[1]
    S0[2]   = sum(abs.(norm_im).^p)/(Nx*Ny)
    if norm
        norm_im ./= (Nx*Ny*S0[2]).^(1/p)
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
    P = plan_ifft(im_fd_0)  # P is an operator, P*im is ifft(im)

    ## Main 1st Order and Precompute 2nd Order
    for f = 1:Nf
        f_i = f_ind[f]  # CartesianIndex list for filter
        f_v = f_val[f]  # Values for f_i
        if length(f_i) > 0
            zarr[f_i] = f_v.*im_fd_0[f_i]
            im_rd_0_1[:,:,f] .= abs.(P*zarr)
            S1[f] = sum(im_rd_0_1[:,:,f].^(p))  # image power
            #zero out the intermediate arrays
            zarr[f_i] .= 0
        end
    end

    #append!(out_coeff, iso ? filter_hash["S1_iso_mat"].*S1 : S1)
    append!(out_coeff, [im_rd_0_1[:,:,i].^(p)  for i in 1:Nf])

    if normS1iso
        S1iso = vec(reshape(filter_hash["S1_iso_mat"]*S1,(1,:))*filter_hash["S1_iso_mat"])
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
                zarr[f_i] = f_v .* thisim[f_i]
                im_rd_0_2[:,:,f1,f2] .= abs.(P*zarr).^(p)
                S2[f1,f2] = sum(im_rd_0_2[:,:,f1,f2])/normS1pwr
                #zero out the intermediate arrays
                zarr[f_i] .= 0
            end
        end
        #append!(out_coeff, iso ? Mat2*S2[:] : S2[:])
        append!(out_coeff, [im_rd_0_2[:,:,f1,f2] for f2 in 1:Nf for f1 in 1:Nf])
    end

    # Real domain 2nd order
    if doS20
        Amat = reshape(im_rd_0_1, Nx*Ny, Nf)
        S20  = Amat' * Amat
        append!(out_coeff, iso ? Mat2*S20[:] : S20[:])
    end

    return out_coeff
end

function eqws_compute_p(image::Array{Float64,2}, filter_hash::Dict, filter_hash2::Dict=filter_hash;
    doS2::Bool=true, doS20::Bool=false, norm=true, iso=false, FFTthreads=2, normS1::Bool=false, normS1iso::Bool=false,
    p=2)
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

    # allocate image arrays for internal use
    im_rd_0_1  = Array{Float64, 3}(undef, Nx, Ny, Nf)

    ## 0th Order
    S0[1]   = mean(image)
    norm_im = image.-S0[1]
    S0[2]   = sum(abs.(norm_im).^p)/(Nx*Ny)
    if norm
        norm_im ./= (Nx*Ny*S0[2]).^(1/p)
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
    P = plan_ifft(im_fd_0)  # P is an operator, P*im is ifft(im)

    ## Main 1st Order and Precompute 2nd Order
    for f = 1:Nf
        f_i = f_ind[f]  # CartesianIndex list for filter
        f_v = f_val[f]  # Values for f_i
        if length(f_i) > 0
            zarr[f_i] = f_v.*im_fd_0[f_i]
            im_rd_0_1[:,:,f] .= abs.(P*zarr)
            S1[f] = sum(im_rd_0_1[:,:,f].^(p))  # image power
            #zero out the intermediate arrays
            zarr[f_i] .= 0
        end
    end

    append!(out_coeff, iso ? filter_hash["S1_iso_mat"]*S1 : S1)

    if normS1iso
        S1iso = vec(reshape(filter_hash["S1_iso_mat"]*S1,(1,:))*filter_hash["S1_iso_mat"])
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
                zarr[f_i] = f_v .* thisim[f_i]
                S2[f1,f2] = sum(abs.(P*zarr).^(p))/normS1pwr
                #zero out the intermediate arrays
                zarr[f_i] .= 0
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

function eqws_compute_RGB(image::Array{Float64}, filter_hash::Dict, filter_hash2::Dict=filter_hash;
    doS20::Bool=true, norm=true, iso=false, FFTthreads=2)
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

    # array sizes
    (Nx, Ny, Nc)  = size(image)
    if Nx != Ny error("Input image must be square") end
    (Nf, )    = size(filter_hash["filt_index"])
    if Nf == 0  error("filter hash corrupted") end
    @assert Nx==filter_hash["npix"] "Filter size should match npix"
    @assert Nx==filter_hash2["npix"] "Filter2 size should match npix"

    # allocate coeff arrays
    out_coeff = []
    S0  = zeros(Float64, Nc*2)
    S1  = zeros(Float64, Nc*Nf)

    if doS20 S20 = zeros(Float64, Nf, Nf) end  # real space correlatio
    # anyM2 = doS2 | doS12 | doS20
    anyrd = doS20 #| doS2             # compute real domain with iFFT

    # allocate image arrays for internal use
    mean_im = zeros(Float64,1,1,Nc)
    pwr_im = zeros(Float64,1,1,Nc)
    norm_im = zeros(Float64,Nx,Ny,Nc)
    im_fd_0 = zeros(ComplexF64, Nx, Ny, Nc)
    im_fd_0_sl = zeros(ComplexF64, Nx, Ny)

    if doS20
        Amat1 = zeros(Nx*Ny, Nf)
        Amat2 = zeros(Nx*Ny, Nf)
    end

    if anyrd im_rd_0_1  = Array{Float64, 4}(undef, Nx, Ny, Nf, Nc) end

    ## 0th Order
    mean_im = mean(image, dims=(1,2))
    S0[1:Nc]   = dropdims(mean_im,dims=(1,2))
    norm_im = image.-mean_im
    pwr_im = sum(norm_im .* norm_im,dims=(1,2))
    S0[1+Nc:end]   = dropdims(pwr_im,dims=(1,2))./(Nx*Ny)
    if norm
        norm_im ./= sqrt.(pwr_im)
    else
        norm_im = copy(image)
    end

    append!(out_coeff,S0[:])

    ## 1st Order
    im_fd_0 .= fft(norm_im,(1,2))  # total power=1.0

    # unpack filter_hash
    f_ind   = filter_hash["filt_index"]  # (J, L) array of filters represented as index value pairs
    f_val   = filter_hash["filt_value"]

    zarr = zeros(ComplexF64, Nx, Ny)  # temporary array to fill with zvals

    # make a FFTW "plan" for an array of the given size and type
    if anyrd
        P = plan_ifft(zarr) end  # P is an operator, P*im is ifft(im)

    ## Main 1st Order and Precompute 2nd Order
    for f = 1:Nf
        S1tot = 0.0
        f_i = f_ind[f]  # CartesianIndex list for filter
        f_v = f_val[f]  # Values for f_i
        # for (ind, val) in zip(f_i, f_v)   # this is slower!
        for chan = 1:Nc
            im_fd_0_sl .= im_fd_0[:,:,chan]
            if length(f_i) > 0
                for i = 1:length(f_i)
                    ind       = f_i[i]
                    zval      = f_v[i] * im_fd_0_sl[ind]
                    S1tot    += abs2(zval)
                    zarr[ind] = zval        # filter*image in Fourier domain
                end
                S1[f+(chan-1)*Nf] = S1tot/(Nx*Ny)  # image power
                if anyrd
                    im_rd_0_1[:,:,f,chan] .= abs2.(P*zarr) end
                zarr[f_i] .= 0
            end
        end
    end

    if iso
        S1M = filter_hash["S1_iso_mat"]
        M1 = blockdiag(S1M,S1M,S1M)
    end
    append!(out_coeff, iso ? M1*S1 : S1)

    # we stored the abs()^2, so take sqrt (this is faster to do all at once)
    if anyrd im_rd_0_1 .= sqrt.(im_rd_0_1) end

    # Real domain 2nd order
    if doS20
        for chan1 = 1:Nc
            for chan2 = 1:Nc
                Amat1 = reshape(im_rd_0_1[:,:,:,chan1], Nx*Ny, Nf)
                Amat2 = reshape(im_rd_0_1[:,:,:,chan2], Nx*Ny, Nf)
                S20  = Amat1' * Amat2
                append!(out_coeff, iso ? filter_hash["S2_iso_mat"]*S20[:] : S20[:])
            end
        end
    end

    return out_coeff
end

function eqws_compute_wrapper(image::Array{Float64,2}, filter_hash::Dict;
    doS2::Bool=true, doS20::Bool=false, apodize=false, norm=false, iso=false, FFTthreads=1, filter_hash2::Dict=filter_hash, coeff_mask=nothing)
    Nf = size(filter_hash["filt_index"])[1]
    #Not using coeff_mask here after all. ASSUMES ANY ONE of doS12, doS2 and doS20 are true, when using coeff_mask
    #@assert !iso "Iso not implemented yet"

    if apodize
        ap_img = apodizer(image)
    else
        ap_img = image
    end

    sall = eqws_compute(ap_img, filter_hash, filter_hash2, doS2=doS2, doS20=doS20, norm=norm, iso=iso, FFTthreads=FFTthreads)
    if coeff_mask!=nothing
        @assert length(coeff_mask)==length(sall) "The length of the coeff_mask must match the length of the output coefficients"
        return sall[coeff_mask]
    else
        return sall
    end
end

function eqws_compute_gpu_fake(image::Array{Float64,2}, filter_hash::Dict, filter_hash2::Dict=filter_hash;
    doS2::Bool=true, doS12::Bool=false, doS20::Bool=false, norm=true, iso=false,batch_mode::Bool=false,opt_memory::Bool=false,max_memory::Int64=0)

    @assert !iso "Isotropic is not implemented on GPU"
    @assert !doS20 "S20 is not supported on GPU"
    # image        - input for WST
    # filter_hash  - filter hash from fink_filter_hash
    # filter_hash2 - filters for second order.  Default to same as first order.
    # doS2         - compute S2 coeffs Similar to Mallat(2014), Abs in real space
    # doS12        - compute S2 coeffs Abs in Fourier space
    # doS20        - compute S2 coeffs Square in real space
    # norm         - scale to mean zero, unit variance
    # iso          - sum over angles to obtain isotropic coeffs


    # array sizes
    (Nx, Ny)  = size(image)
    if Nx != Ny error("Input image must be square") end
    (Nf, )    = size(filter_hash["filt_vals"])
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
    if doS12 im_fdf_0_1 = zeros(Float64,           Nx, Ny) end   # this must be zeroed!
    if anyrd im_rd_0_1  = Array{Float64, 2}(undef, Nx, Ny) end

    image= image
    ## 0th Order
    S0[1]   = mean(image)
    norm_im = image.-S0[1]
    S0[2]   = sum(norm_im .* norm_im)/(Nx*Ny)
    if norm
        norm_im./= sqrt(Nx*Ny*S0[2])
    else
        norm_im = image
    end


    ## 1st Order
    #REview: We fft shift!
    im_fd_0 = FFTW.fftshift(FFTW.fft(norm_im))  # total power=1.0

    # unpack filter_hash
    f_mms   = filter_hash["filt_mms"]  # (J, L) array of filters represented as index value pairs
    f_vals   = filter_hash["filt_vals"]

    zarr = zeros(ComplexF64, Nx, Ny)  # temporary array to fill with zvals

    # make a FFTW "plan" for an array of the given size and type
    if anyrd
        P = FFTW.plan_ifft(im_fd_0) end  # P is an operator, P*im is ifft(im)

    ## Main 1st Order and Precompute 2nd Order
    for f1 = 1:Nf
        f_mm1 = f_mms[:,:,f1]  # Min,Max
        f_val1 = f_vals[f1]  # values

        #Review: Removes length check-> empty filter case???, I think this should be prevented beforehands
        zv=f_val1.*im_fd_0[f_mm1[1,1]:f_mm1[1,2],f_mm1[2,1]:f_mm1[2,2]] #dense multiplication for the filter region
        zarr[f_mm1[1,1]:f_mm1[1,2],f_mm1[2,1]:f_mm1[2,2]]=zv

        if doS12 im_fdf_0_1[f_mm1[1,1]:f_mm1[1,2],f_mm1[2,1]:f_mm1[2,2]] .= abs.(zv) end

        S1[f1] = sum(abs2.(zv))/(Nx*Ny)  # image power
        if anyrd im_rd_0_1.= abs2.(P*FFTW.ifftshift(zarr)) end
        zarr[f_mm1[1,1]:f_mm1[1,2],f_mm1[2,1]:f_mm1[2,2]] .= 0


        im_rd_0_1 .= sqrt.(im_rd_0_1)
        im_fdf_1_1 = FFTW.fftshift(FFTW.fft(im_rd_0_1))
        for f2 = 1:Nf
            f_mm2 = f_mms[:,:,f2]
            f_val2 = f_vals[f2]

            S2[f1,f2] = sum(abs2.(f_val2 .* im_fdf_1_1[f_mm2[1,1]:f_mm2[1,2],f_mm2[2,1]:f_mm2[2,2]]))/(Nx*Ny)
            if !doS12
                continue
            end

            if f2>f1
                S12[f1,f2]=NaN
                continue
            end

            if f_mm1[1,1]>f_mm2[1,2] || f_mm2[1,1]>f_mm1[1,2] || f_mm1[2,1]>f_mm2[2,2] || f_mm2[2,1]>f_mm1[2,2]
                S12[f1,f2]=NaN
                continue
            end

            mx=max(f_mm1[1,1],f_mm2[1,1])
            my=max(f_mm1[2,1],f_mm2[2,1])
            Mx=min(f_mm1[1,2],f_mm2[1,2])
            My=min(f_mm1[2,2],f_mm2[2,2])
            sfx=Mx-mx
            sfy=My-my
            st1x=mx-f_mm1[1,1]+1
            st1y=my-f_mm1[2,1]+1
            st2x=mx-f_mm2[1,1]+1
            st2y=my-f_mm2[2,1]+1

            S12[f1,f2]=sum(f_val1[st1x:st1x+sfx,st1y:st1y+sfy].*f_val2[st2x:st2x+sfx,st2y:st2y+sfy].*abs2.(im_fd_0[mx:Mx,my:My]))
        end
    end
    append!(out_coeff,S0)
    append!(out_coeff, iso ? filter_hash["S1_iso_mat"]*S1 : S1)
    if doS2 append!(out_coeff, iso ? filter_hash["S1_iso_mat"]*S2 : S2) end
    if doS12 append!(out_coeff, iso ? filter_hash["S1_iso_mat"]*S12 : S12) end
    return out_coeff
end

function eqws_compute_gpu(image::Array{Float64,2}, filter_hash::Dict, filter_hash2::Dict=filter_hash;
    doS2::Bool=true, doS12::Bool=false, doS20::Bool=false, norm=true, iso=false,batch_mode::Bool=false,opt_memory::Bool=false,max_memory::Int64=0)

    @assert !iso "Isotropic is not implemented on GPU"
    @assert !doS20 "S20 is not supported on GPU"
    # image        - input for WST
    # filter_hash  - filter hash from fink_filter_hash
    # filter_hash2 - filters for second order.  Default to same as first order.
    # doS2         - compute S2 coeffs Similar to Mallat(2014), Abs in real space
    # doS12        - compute S2 coeffs Abs in Fourier space
    # doS20        - compute S2 coeffs Square in real space
    # norm         - scale to mean zero, unit variance
    # iso          - sum over angles to obtain isotropic coeffs


    # array sizes
    (Nx, Ny)  = size(image)
    if Nx != Ny error("Input image must be square") end
    (Nf, )    = size(filter_hash["filt_vals"])
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
    if doS12 im_fdf_0_1 = CUDA.zeros(Float64, Nx, Ny) end   # this must be zeroed!
    if anyrd im_rd_0_1  = CUDA.CuArray(Array{Float64, 2}(undef, Nx, Ny)) end

    image= CUDA.CuArray(image)
    ## 0th Order
    S0[1]   = mean(image)
    norm_im = image.-S0[1]
    S0[2]   = sum(norm_im .* norm_im)/(Nx*Ny)
    if norm
        norm_im./= sqrt(Nx*Ny*S0[2])
    else
        norm_im = image
    end


    ## 1st Order
    #REview: We fft shift!
    im_fd_0 = CUDA.CUFFT.fftshift(CUDA.CUFFT.fft(norm_im))  # total power=1.0

    # unpack filter_hash
    f_mms   = filter_hash["filt_mms"]  # (J, L) array of filters represented as index value pairs
    f_vals   = filter_hash["filt_vals"]

    zarr = CUDA.zeros(ComplexF64, Nx, Ny)  # temporary array to fill with zvals

    # make a FFTW "plan" for an array of the given size and type
    if anyrd
        P = CUDA.CUFFT.plan_ifft(im_fd_0) end  # P is an operator, P*im is ifft(im)

    ## Main 1st Order and Precompute 2nd Order
    for f1 = 1:Nf
        f_mm1 = f_mms[:,:,f1]  # Min,Max
        f_val1 = f_vals[f1]  # values

        #Review: Removes length check-> empty filter case???, I think this should be prevented beforehands
        zv=f_val1.*im_fd_0[f_mm1[1,1]:f_mm1[1,2],f_mm1[2,1]:f_mm1[2,2]] #dense multiplication for the filter region
        zarr[f_mm1[1,1]:f_mm1[1,2],f_mm1[2,1]:f_mm1[2,2]]=zv

        if doS12 im_fdf_0_1[f_mm1[1,1]:f_mm1[1,2],f_mm1[2,1]:f_mm1[2,2]] .= abs.(zv) end

        S1[f1] = sum(abs2.(zv))/(Nx*Ny)  # image power
        if anyrd im_rd_0_1.= abs2.(P*CUDA.CUFFT.ifftshift(zarr)) end
        zarr[f_mm1[1,1]:f_mm1[1,2],f_mm1[2,1]:f_mm1[2,2]] .= 0


        im_rd_0_1 .= sqrt.(im_rd_0_1)
        im_fdf_1_1 = CUDA.CUFFT.fftshift(CUDA.CUFFT.fft(im_rd_0_1))
        for f2 = 1:Nf
            f_mm2 = f_mms[:,:,f2]
            f_val2 = f_vals[f2]

            S2[f1,f2] = sum(abs2.(f_val2 .* im_fdf_1_1[f_mm2[1,1]:f_mm2[1,2],f_mm2[2,1]:f_mm2[2,2]]))/(Nx*Ny)
            if !doS12
                continue
            end

            if f2>f1
                S12[f1,f2]=NaN
                continue
            end

            if f_mm1[1,1]>f_mm2[1,2] || f_mm2[1,1]>f_mm1[1,2] || f_mm1[2,1]>f_mm2[2,2] || f_mm2[2,1]>f_mm1[2,2]
                S12[f1,f2]=NaN
                continue
            end

            mx=max(f_mm1[1,1],f_mm2[1,1])
            my=max(f_mm1[2,1],f_mm2[2,1])
            Mx=min(f_mm1[1,2],f_mm2[1,2])
            My=min(f_mm1[2,2],f_mm2[2,2])
            sfx=Mx-mx
            sfy=My-my
            st1x=mx-f_mm1[1,1]+1
            st1y=my-f_mm1[2,1]+1
            st2x=mx-f_mm2[1,1]+1
            st2y=my-f_mm2[2,1]+1

            S12[f1,f2]=sum(f_val1[st1x:st1x+sfx,st1y:st1y+sfy].*f_val2[st2x:st2x+sfx,st2y:st2y+sfy].*abs2.(im_fd_0[mx:Mx,my:My]))
        end
    end
    append!(out_coeff,S0)
    append!(out_coeff, iso ? filter_hash["S1_iso_mat"]*S1 : S1)
    if doS2 append!(out_coeff, iso ? filter_hash["S1_iso_mat"]*S2 : S2) end
    if doS12 append!(out_coeff, iso ? filter_hash["S1_iso_mat"]*S12 : S12) end
    return out_coeff
end

function eqws_compute_batch_gpu(image::Union{Array{Float64,3},Array{Float32,3}}, filter_hash::Dict, filter_hash2::Dict=filter_hash;
    doS2::Bool=true, doS20::Bool=false, norm=true, iso=false, normS1::Bool=false, normS1iso::Bool=false,prec=Float64)
    # image        - input for WST
    # filter_hash  - filter hash from fink_filter_hash
    # filter_hash2 - filters for second order.  Default to same as first order.
    # doS2         - compute S2 coeffs
    # doS20        - compute S2 coeffs
    # norm         - scale to mean zero, unit variance
    # iso          - sum over angles to obtain isotropic coeffs

    # array sizes
    (Nx, Ny, Nb)  = size(image)
    if Nx != Ny error("Input image must be square") end
    Nf    = filter_hash["Nf"]
    if Nf == 0  error("filter hash corrupted") end
    @assert Nx==filter_hash["npix"] "Filter size should match npix"
    @assert Nx==filter_hash2["npix"] "Filter2 size should match npix"
    @assert (normS1 && normS1iso) != 1 "normS1 and normS1iso are mutually exclusive"
    if prec == Float64
        cprec = ComplexF64
    elseif prec == Float32
        cprec = ComplexF32
    else
        error("precision is not acceptable, please choose Float32 or Float64")
    end

    # allocate coeff arrays
    anyrd = doS2 | doS20             # compute real domain with iFFT

    if iso
        ind=1
    else
        ind=2
    end

    if !anyrd
        S1s = size(filter_hash["S1_iso_mat"])[ind]
        out_coeff = zeros(S1s+2,Nb)
    elseif doS2 ⊻ doS20
        S1s = size(filter_hash["S1_iso_mat"])[ind]
        S2s = size(filter_hash["S2_iso_mat"])[ind]
        out_coeff = zeros(S1s + S2s + 2,Nb)
    else
        S1s = size(filter_hash["S1_iso_mat"])[ind]
        S2s = size(filter_hash["S2_iso_mat"])[ind]
        out_coeff = zeros(S1s + 2*S2s + 2,Nb)
    end

    S0  = zeros(prec, 2, Nb)
    S1  = zeros(prec, Nf, Nb)
    if doS2  S2  = zeros(prec, Nf, Nf, Nb) end  # traditional 2nd order
    if doS20 S20 = zeros(prec, Nf, Nf, Nb) end  # real space correlation

    # allocate image arrays for internal use
    if anyrd im_rd_0_1  = CUDA.CuArray{prec}(undef, Nx, Ny, Nf, Nb) end

    #image handling
    if typeof(image[1]) != prec
        image = convert(Array{prec,3},image)
    end

    ## 0th Order
    μ = mean(image, dims=(1,2))
    S0[1,:] = dropdims(μ, dims=(1,2))
    norm_im = image.-μ
    σ       = sum(norm_im .* norm_im, dims=(1,2))/(Nx*Ny)
    S0[2,:] = dropdims(σ, dims=(1,2))
    if norm
        norm_im ./= sqrt.(Nx*Ny*σ)
    else
        norm_im = image
    end

    out_coeff[1:2,:] .= S0

    image_in = CUDA.CuArray(norm_im)

    if doS2
        P2 = CUDA.CUFFT.plan_fft(image_in,[1,2])
        im_fd_0 = P2*image_in
    else
        im_fd_0 = CUDA.CUFFT.fft(image_in,[1,2])  # total power=1.0
    end

    CUDA.unsafe_free!(image_in)

    # unpack filter_hash
    filts   = filter_hash["gpu_filt"]  # dense gpu array

    zarr = CUDA.zeros(cprec, Nx, Ny, Nb)  # temporary array to fill with zvals

    # make a FFTW "plan" for an array of the given size and type
    if anyrd
        P1 = CUDA.CUFFT.plan_ifft(im_fd_0,[1,2]) # P is an operator, P*im is ifft(im)
    end

    ## Main 1st Order and Precompute 2nd Order
    for f = 1:Nf
        zarr = filts[:,:,f].*im_fd_0

        if anyrd
            im_rd_0_1[:,:,f,:] = abs2.(P1*zarr)
            S1[f,:] = sum(im_rd_0_1[:,:,f,:],dims=(1,2))  # image power
        else
            S1[f,:] = sum(abs2.(zarr),dims=(1,2))/(Nx*Ny)
        end
    end

    CUDA.unsafe_free!(zarr)

    if iso
        out_coeff[3:3+S1s-1,:] .= filter_hash["S1_iso_mat"]*S1
    else
        out_coeff[3:3+S1s-1,:] .= S1
    end

    if normS1iso
        S1iso = vec(reshape(filter_hash["S1_iso_mat"]*S1,(1,:))*filter_hash["S1_iso_mat"]) #FIXME CuVec?
    end

    # we stored the abs()^2, so take sqrt (this is faster to do all at once)
    if anyrd im_rd_0_1 = sqrt.(im_rd_0_1) end

    if (doS2 | doS20) & iso
        Mat2 = filter_hash["S2_iso_mat"]
    end

    # Real domain 2nd order
    if doS20
        Amat = reshape(im_rd_0_1, Nx*Ny, Nf, Nb)
        S20  = permutedims(Amat,(2,1,3)) ⊠ Amat
        if iso
            out_coeff[3+S1s:3+S1s+S2s-1,:] .= Mat2*reshape(S20,Nf*Nf,Nb)
        else
            out_coeff[3+S1s:3+S1s+S2s-1,:] .= reshape(S20,Nf*Nf,Nb)
        end
    end

    if doS2
        ## Traditional second order
        for f1 = 1:Nf
            thisim = P2*im_rd_0_1[:,:,f1,:]
            # Loop over f2 and do second-order convolution
            if normS1
                normS1pwr = S1[f1,:]
            elseif normS1iso
                normS1pwr = S1iso[f1,:]
            else
                normS1pwr = 1
            end
            for f2 = 1:Nf
                # sum im^2 = sum(|fft|^2/npix)
                S2[f1,f2,:] .= sum(abs2.(filts[:,:,f2].*thisim))/(Nx*Ny)/normS1pwr
            end
        end

        if iso
            out_coeff[end-S2s+1:end,:] .= Mat2*reshape(S2,Nf*Nf,Nb)
        else
            out_coeff[end-S2s+1:end,:] .= reshape(S2,Nf*Nf,Nb)
        end
    end

    return out_coeff
end

function eqws_compute_3d(image::Array{Float64,3}, filter_hash; FFTthreads=2, iso=false)
    # Use 2 threads for FFT
    FFTW.set_num_threads(FFTthreads)

    # array sizes
    (Nx, Ny, Nz)  = size(image)
    (Nf, )    = size(filter_hash["filt_index"])

    # allocate coeff arrays
    out_coeff = []
    S0  = zeros(Float64, 2)
    S1  = zeros(Float64, Nf)
    S20 = zeros(Float64, Nf, Nf)
    S12 = zeros(Float64, Nf, Nf)
    S2  = zeros(Float64, Nf, Nf)  # traditional 2nd order

    # allocate image arrays for internal use
    im_fdf_0_1 = zeros(Float64,           Nx, Ny, Nz, Nf)   # this must be zeroed!
    im_rd_0_1  = Array{Float64, 4}(undef, Nx, Ny, Nz, Nf)

    ## 0th Order
    S0[1]   = mean(image)
    norm_im = image.-S0[1]
    S0[2]   = sum(norm_im .* norm_im)/(Nx*Ny*Nz)
    norm_im ./= sqrt(Nx*Ny*Nz*S0[2])

    append!(out_coeff,S0[:])

    ## 1st Order
    im_fd_0 = fft(norm_im)  # total power=1.0

    # unpack filter_hash
    f_ind   = filter_hash["filt_index"]  # (J, L) array of filters represented as index value pairs
    f_val   = filter_hash["filt_value"]

    #f_ind2   = filter_hash2["filt_index"]  # (J, L) array of filters represented as index value pairs
    #f_val2   = filter_hash2["filt_value"]

    zarr = zeros(ComplexF64, Nx, Ny, Nz)  # temporary array to fill with zvals

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
            S1[f] = S1tot/(Nx*Ny*Nz)  # image power ###why do we need to normalize again?
            im_rd_0_1[:,:,:,f] .= abs2.(P*zarr)
            zarr[f_i] .= 0
        end
    end
    append!(out_coeff, iso ? filter_hash["S1_iso_mat"]*S1 : S1)

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
    if iso
        Mat2 = filter_hash["S2_iso_mat"]
    end
    for f1 = 1:Nf
        thisim = fft(im_rd_0_1[:,:,:,f1])  # Could try rfft here
        # println("  f1",f1,"  sum(fft):",sum(abs2.(thisim))/Nx^2, "  sum(im): ",sum(abs2.(im_rd_0_1[:,:,f1])))
        # Loop over f2 and do second-order convolution
        for f2 = 1:Nf
            f_i = f_ind[f2]  # CartesianIndex list for filter
            f_v = f_val[f2]  # Values for f_i
            # sum im^2 = sum(|fft|^2/npix)
            S2[f1,f2] = sum(abs2.(f_v .* thisim[f_i]))/(Nx*Ny*Nz)
        end
    end
    append!(out_coeff, iso ? Mat2*S2[:] : S2[:])

    return out_coeff
end

function eqws_compute_3d_gpu(image::Array{Float64,3}, filter_hash;doS2::Bool=true, doS12::Bool=false, norm=true)
    CUDA.reclaim()
    # array sizes
    (Nx, Ny, Nz)  = size(image)
    (Nf, )    = size(filter_hash["filt_vals"])

    # allocate coeff arrays
    out_coeff = []
    S0  = zeros(Float64, 2)
    S1  = zeros(Float64, Nf)
    #S20 = zeros(Float64, Nf, Nf)
    if doS12 S12 = zeros(Float64, Nf, Nf) end
    if doS2 S2  = zeros(Float64, Nf, Nf)  end # traditional 2nd order

    # allocate image arrays for internal use
    im_fdf_0_1 = CUDA.CuArray(Array{Float64, 3}(undef, Nx, Ny, Nz))
    im_rd_0_1  = CUDA.CuArray(Array{Float64, 3}(undef, Nx, Ny, Nz))

    image=CUDA.CuArray(image)

    ## 0th Order
    S0[1]   = mean(image)
    norm_im = image.-S0[1]
    S0[2]   = sum(norm_im .* norm_im)/(Nx*Ny*Nz)
    if norm
        norm_im ./= sqrt(Nx*Ny*Nz*S0[2])
    else
        norm_im=image
    end

    ## 1st Order
    im_fd_0 = CUDA.CUFFT.fftshift(CUDA.CUFFT.fft(norm_im))  # total power=1.0

    # unpack filter_hash
    f_mms   = filter_hash["filt_mms"]  # (J, L) array of filters represented as index value pairs
    f_vals   = filter_hash["filt_vals"]

    #f_ind2   = filter_hash2["filt_index"]  # (J, L) array of filters represented as index value pairs
    #f_val2   = filter_hash2["filt_value"]

    zarr = CUDA.zeros(ComplexF64, Nx, Ny, Nz)  # temporary array to fill with zvals

    # make a FFTW "plan" for an array of the given size and type
    P   = CUDA.CUFFT.plan_ifft(im_fd_0)   # P is an operator, P*im is ifft(im)

    #depth first search
    for f1 = 1:Nf
        im_fdf_0_1.=0.
        f_mm1 = f_mms[:,:,f1]  # CartesianIndex list for filter
        f_val1 = f_vals[f1]  # Values for f_i
        # for (ind, val) in zip(f_i, f_v)   # this is slower!

        zv=f_val1.*im_fd_0[f_mm1[1,1]:f_mm1[1,2],f_mm1[2,1]:f_mm1[2,2],f_mm1[3,1]:f_mm1[3,2]]
        S1[f1] = sum(abs2.(zv))/(Nx*Ny*Nz)

        zarr[f_mm1[1,1]:f_mm1[1,2],f_mm1[2,1]:f_mm1[2,2],f_mm1[3,1]:f_mm1[3,2]]=zv

        im_fdf_0_1[f_mm1[1,1]:f_mm1[1,2],f_mm1[2,1]:f_mm1[2,2],f_mm1[3,1]:f_mm1[3,2]] .= abs.(zv)


        im_rd_0_1=abs2.(P*CUDA.CUFFT.ifftshift(zarr))

        zarr[f_mm1[1,1]:f_mm1[1,2],f_mm1[2,1]:f_mm1[2,2],f_mm1[3,1]:f_mm1[3,2]] .= 0.

        im_rd_0_1 .= sqrt.(im_rd_0_1)
        im_fdf_1_1 = CUDA.CUFFT.fftshift(CUDA.CUFFT.fft(im_rd_0_1))
        for f2 = 1:Nf
            f_mm2 = f_mms[:,:,f2]
            f_val2 = f_vals[f2]

            S2[f1,f2] = sum(abs2.(f_val2 .* im_fdf_1_1[f_mm2[1,1]:f_mm2[1,2],f_mm2[2,1]:f_mm2[2,2],f_mm2[3,1]:f_mm2[3,2]] ))/(Nx*Ny*Nz)

            if !doS12
                continue
            end

            if f2>f1
                S12[f1,f2]=NaN
                continue
            end

            if f_mm1[1,1]>f_mm2[1,2] || f_mm2[1,1]>f_mm1[1,2] || f_mm1[2,1]>f_mm2[2,2] || f_mm2[2,1]>f_mm1[2,2] || f_mm1[3,1]>f_mm2[3,2] || f_mm2[3,1]>f_mm1[3,2]
                S12[f1,f2]=NaN
                continue
            end

            mx=max(f_mm1[1,1],f_mm2[1,1])
            my=max(f_mm1[2,1],f_mm2[2,1])
            mz=max(f_mm1[3,1],f_mm2[3,1])
            Mx=min(f_mm1[1,2],f_mm2[1,2])
            My=min(f_mm1[2,2],f_mm2[2,2])
            Mz=min(f_mm1[3,2],f_mm2[3,2])
            sfx=Mx-mx
            sfy=My-my
            sfz=Mz-mz
            st1x=mx-f_mm1[1,1]+1
            st1y=my-f_mm1[2,1]+1
            st1z=mz-f_mm1[3,1]+1
            st2x=mx-f_mm2[1,1]+1
            st2y=my-f_mm2[2,1]+1
            st2z=mz-f_mm2[3,1]+1

            S12[f1,f2]=sum(f_val1[st1x:st1x+sfx,st1y:st1y+sfy,st1z:st1z+sfz].*f_val2[st2x:st2x+sfx,st2y:st2y+sfy,st2z:st2z+sfz].*abs2.(im_fd_0[mx:Mx,my:My,mz:Mz]))
            CUDA.reclaim()
        end
        CUDA.reclaim()
    end
    append!(out_coeff,S0)
    append!(out_coeff, S1)
    if doS2 append!(out_coeff, S2) end
    if doS12 append!(out_coeff, S12) end
    return out_coeff
end

function eqws_compute_3d_gpu_fake(image::Array{Float64,3}, filter_hash;doS2::Bool=true, doS12::Bool=false, norm=true)
    # array sizes
    (Nx, Ny, Nz)  = size(image)
    (Nf, )    = size(filter_hash["filt_vals"])

    # allocate coeff arrays
    out_coeff = []
    S0  = zeros(Float64, 2)
    S1  = zeros(Float64, Nf)
    #S20 = zeros(Float64, Nf, Nf)
    if doS12 S12 = zeros(Float64, Nf, Nf) end
    if doS2 S2  = zeros(Float64, Nf, Nf)  end # traditional 2nd order

    # allocate image arrays for internal use
    im_fdf_0_1 = Array{Float64, 3}(undef, Nx, Ny, Nz)
    im_rd_0_1  = Array{Float64, 3}(undef, Nx, Ny, Nz)


    ## 0th Order
    S0[1]   = mean(image)
    norm_im = image.-S0[1]
    S0[2]   = sum(norm_im .* norm_im)/(Nx*Ny*Nz)
    if norm
        norm_im ./= sqrt(Nx*Ny*Nz*S0[2])
    else
        norm_im=image
    end

    ## 1st Order
    im_fd_0 = FFTW.fftshift(FFTW.fft(norm_im))  # total power=1.0

    # unpack filter_hash
    f_mms   = filter_hash["filt_mms"]  # (J, L) array of filters represented as index value pairs
    f_vals   = filter_hash["filt_vals"]

    #f_ind2   = filter_hash2["filt_index"]  # (J, L) array of filters represented as index value pairs
    #f_val2   = filter_hash2["filt_value"]

    zarr = zeros(ComplexF64, Nx, Ny, Nz)  # temporary array to fill with zvals

    # make a FFTW "plan" for an array of the given size and type
    P   = FFTW.plan_ifft(im_fd_0)   # P is an operator, P*im is ifft(im)

    #depth first search
    for f1 = 1:Nf
        im_fdf_0_1.=0.
        f_mm1 = f_mms[:,:,f1]  # CartesianIndex list for filter
        f_val1 = f_vals[f1]  # Values for f_i
        # for (ind, val) in zip(f_i, f_v)   # this is slower!

        zv=f_val1.*im_fd_0[f_mm1[1,1]:f_mm1[1,2],f_mm1[2,1]:f_mm1[2,2],f_mm1[3,1]:f_mm1[3,2]]
        S1[f1] = sum(abs2.(zv))/(Nx*Ny*Nz)

        zarr[f_mm1[1,1]:f_mm1[1,2],f_mm1[2,1]:f_mm1[2,2],f_mm1[3,1]:f_mm1[3,2]]=zv

        im_fdf_0_1[f_mm1[1,1]:f_mm1[1,2],f_mm1[2,1]:f_mm1[2,2],f_mm1[3,1]:f_mm1[3,2]] .= abs.(zv)


        im_rd_0_1=abs2.(P*FFTW.ifftshift(zarr))

        zarr[f_mm1[1,1]:f_mm1[1,2],f_mm1[2,1]:f_mm1[2,2],f_mm1[3,1]:f_mm1[3,2]] .= 0.

        im_rd_0_1 .= sqrt.(im_rd_0_1)
        im_fdf_1_1 = FFTW.fftshift(FFTW.fft(im_rd_0_1))
        for f2 = 1:Nf
            f_mm2 = f_mms[:,:,f2]
            f_val2 = f_vals[f2]

            S2[f1,f2] = sum(abs2.(f_val2 .* im_fdf_1_1[f_mm2[1,1]:f_mm2[1,2],f_mm2[2,1]:f_mm2[2,2],f_mm2[3,1]:f_mm2[3,2]] ))/(Nx*Ny*Nz)

            if !doS12
                continue
            end

            if f2>f1
                S12[f1,f2]=NaN
                continue
            end

            if f_mm1[1,1]>f_mm2[1,2] || f_mm2[1,1]>f_mm1[1,2] || f_mm1[2,1]>f_mm2[2,2] || f_mm2[2,1]>f_mm1[2,2] || f_mm1[3,1]>f_mm2[3,2] || f_mm2[3,1]>f_mm1[3,2]
                S12[f1,f2]=NaN
                continue
            end

            mx=max(f_mm1[1,1],f_mm2[1,1])
            my=max(f_mm1[2,1],f_mm2[2,1])
            mz=max(f_mm1[3,1],f_mm2[3,1])
            Mx=min(f_mm1[1,2],f_mm2[1,2])
            My=min(f_mm1[2,2],f_mm2[2,2])
            Mz=min(f_mm1[3,2],f_mm2[3,2])
            sfx=Mx-mx
            sfy=My-my
            sfz=Mz-mz
            st1x=mx-f_mm1[1,1]+1
            st1y=my-f_mm1[2,1]+1
            st1z=mz-f_mm1[3,1]+1
            st2x=mx-f_mm2[1,1]+1
            st2y=my-f_mm2[2,1]+1
            st2z=mz-f_mm2[3,1]+1

            S12[f1,f2]=sum(f_val1[st1x:st1x+sfx,st1y:st1y+sfy,st1z:st1z+sfz].*f_val2[st2x:st2x+sfx,st2y:st2y+sfy,st2z:st2z+sfz].*abs2.(im_fd_0[mx:Mx,my:My,mz:Mz]))
        end
    end
    append!(out_coeff,S0)
    append!(out_coeff, S1)
    if doS2 append!(out_coeff, S2) end
    if doS12 append!(out_coeff, S12) end
    return out_coeff
end
