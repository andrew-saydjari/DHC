## Preloads
module DHC_2DUtils

    using FFTW
    using LinearAlgebra
    using Statistics

    export finklet
    export fink_filter_bank
    export fink_filter_list
    export DHC_compute

    function fink_filter_bank(c, L; nx=256,wd=1,pc=1)
        #plane coverage (default 1, full 2Pi 2)
        #width of the wavelets (default 1, wide 2)
        #c sets the scale sampling rate (1 is dyadic, 2 is half dyadic)

        # -------- set parameters
        dθ   = pc*π/L
        wdθ  = wd*dθ
        dx   = nx/2-1

        im_scale = convert(Int8,log2(nx))
        J = (im_scale-2)*c

        # -------- allocate output array of zeros
        filt = zeros(nx, nx, J*L+1)

        # -------- allocate theta and logr arrays
        logr = zeros(nx, nx)
        θ    = zeros(nx, nx)

        # -------- allocate temp phi building zeros
        phi_b = zeros(nx, nx)

        for l = 0:L-1
            θ_l = dθ*l

        # -------- allocate anggood BitArray
            anggood = falses(nx, nx)

        # -------- loop over pixels
            for x = 1:nx
                sx = mod(x+dx,nx)-dx -1    # define sx,sy so that no fftshift() needed
                for y = 1:nx
                    sy = mod(y+dx,nx)-dx -1
                    θ_pix  = mod(atan(sy, sx)+π -θ_l, 2*π)
                    θ_good = abs(θ_pix-π) <= wdθ

                # If this is a pixel we might use, calculate log2(r)
                    if θ_good
                        anggood[y, x] = θ_good
                        θ[y, x]       = θ_pix
                        r = sqrt(sx^2 + sy^2)
                        logr[y, x] = log2(max(1,r))
                    end
                end
            end
            angmask = findall(anggood)
        # -------- compute the wavelet in the Fourier domain
        # -------- the angular factor is the same for all j
            F_angular = cos.((θ[angmask].-π).*L./(2 .*wd .*pc))

        # -------- loop over j for the radial part
            for (j_ind, j) in enumerate(1/c:1/c:im_scale-2)
                jrad  = 7-j
                Δj    = abs.(logr[angmask].-jrad)
                rmask = (Δj .<= 1/c)

        # -------- radial part
                F_radial = cos.(Δj[rmask] .* (c*π/2))
                ind      = angmask[rmask]
                filt[ind,(j_ind-1)*L+l+1] = F_radial .* F_angular[rmask]
            end
        # -------- handle the phi case (jrad=0, j=7)
            for (j_ind, j) in enumerate(im_scale-2+1/c:1/c:im_scale-1)
                jrad  = 7-j
                Δj    = abs.(logr[angmask].-jrad)
                rmask = (Δj .<= 1/c)

                # -------- radial part
                F_radial = cos.(Δj[rmask] .* (c*π/2))
                ind      = angmask[rmask]
                phi_b[ind] .+= (F_radial .* F_angular[rmask]).^2
            end
        end

        # -------- normalize the phi case correctly
        if pc == 1
            phi_b .+= circshift(phi_b[end:-1:1,end:-1:1],(1,1))
            phi_b[1,1] /= 2
        end

        phi_b .= sqrt.(phi_b)

        filt[:,:,J*L+1] .= phi_b
        return filt
    end

    ## Make a list of non-zero pixels for the (Fourier plane) filters
    function fink_filter_list(filt)
        (ny,nx,Nf) = size(filt)

        # Allocate output arrays
        filtind = fill(CartesianIndex{2}[], Nf)
        filtval = fill(Float64[], Nf)

        # Loop over filters and record non-zero values
        for l=1:Nf
            f = @view filt[:,:,l]
            ind = findall(f .> 1E-13)
            val = f[ind]
            filtind[l] = ind
            filtval[l] = val
        end
        return [filtind, filtval]
    end

    function DHC_compute(image::Array{Float64,2}, filter_list)
        # Use 2 threads for FFT
        FFTW.set_num_threads(2)

        # array sizes
        (Nx, Ny)  = size(image)
        (Nf, )    = size(filter_list[1])

        # allocate coeff arrays
        out_coeff = []
        S0  = zeros(Float64, 2)
        S1  = zeros(Float64, Nf)
        S20 = zeros(Float64, Nf, Nf)
        S12 = zeros(Float64, Nf, Nf)

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
        im_fd_0 = fft(norm_im)

        # unpack filter_list
        f_ind   = filter_list[1]  # (J, L) array of filters represented as index value pairs
        f_val   = filter_list[2]

        zarr = zeros(ComplexF64, Nx, Ny)  # temporary array to fill with zvals

        # make a FFTW "plan" for an array of the given size and type
        P   = plan_ifft(im_fd_0)   # P is an operator, P*im is ifft(im)

        ## Main 1st Order and Precompute 2nd Order
        for l = 1:Nf
            S1tot = 0.0
            f_i = f_ind[l]  # CartesianIndex list for filter
            f_v = f_val[l]  # Values for f_i
            # for (ind, val) in zip(f_i, f_v)   # this is slower!
            if length(f_i) > 0
                for i = 1:length(f_i)
                    ind       = f_i[i]
                    zval      = f_v[i] * im_fd_0[ind]
                    S1tot    += abs2(zval)
                    zarr[ind] = zval
                    im_fdf_0_1[ind,l] = abs(zval)
                end
                S1[l] = S1tot
                im_rd_0_1[:,:,l] .= abs2.(P*zarr)
                zarr[f_i] .= 0
            end
        end
        append!(out_coeff, S1[:])

        # we stored the abs()^2, so take sqrt (this is faster to do all at once)
        im_rd_0_1 .= sqrt.(im_rd_0_1)

        ## 2nd Order
        Amat = reshape(im_fdf_0_1, Nx*Ny, Nf)
        S12  = Amat' * Amat
        Amat = reshape(im_rd_0_1, Nx*Ny, Nf)
        S20  = Amat' * Amat

        append!(out_coeff, S20)
        append!(out_coeff, S12)

        return out_coeff
    end

end # of module
