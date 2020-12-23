## Preloads
module DHC_Utils

    using Statistics
    using FFTW
    using Plots
    using LinearAlgebra
    using StaticArrays
    using HybridArrays

    export finklet
    export fink_filter_bank_slow
    export fink_filter_bank
    export fink_filter_list
    export speedy_DHC


    ## Define wavelet in Fourier space
    function finklet(j, l)
        # -------- set filters
        jrad = 7-j
        dθ = π/8        # 8 angular bins hardwired
        θ_l = dθ*l
        # -------- define coordinates
        nx = 256
        xbox = LinRange(-nx/2, nx/2-1 , nx)
        # make a 256x256 grid of X
        sx = xbox' .* ones(nx)
        sy = ones(nx)' .* xbox
        r  = sqrt.((sx).^2 + (sy).^2)
        θ  = mod.(atan.(sy, sx).+π .-θ_l,2*π)
        nozeros = r .> 0
        logr = log2.(r[nozeros])
        r[nozeros] = logr
        # -------- in Fourier plane, envelope of psi_j,l
        mask = (abs.(θ.-π).<= dθ) .& (abs.(r.-jrad) .<= 1)
        # -------- angular part
        ang = cos.((θ.-π).*4)
        # -------- radial part
        rad = cos.((r.-jrad).*π./2)
        psi = mask.*ang.*rad             #mask times angular part times radial part
        return psi
    end


    ## Compute the whole filter bank.  Legacy code for comparison
    function fink_filter_bank_slow(J,L)
        fink_filter = Array{Float64, 4}(undef, 256, 256, J, L)
        for l = 1:L
            for j = 1:J
                @inbounds fink_filter[:,:,j,l]=fftshift(finklet(j-1,l-1))
            end
        end
        return fink_filter
    end


    ## Faster filter bank generation.  Only square filters allowed.
    function fink_filter_bank(J::Integer, L::Integer; nx::Integer=256, wid::Integer=1)

        # -------- set parameters
        dθ   = π/8        # 8 angular bins hardwired
        dx   = nx÷2-1

        # -------- allocate output array of zeros
        filt = zeros(Float64, nx, nx, J, L)

        # -------- allocate theta and logr arrays
        logr = zeros(Float64, nx, nx)
        θ    = zeros(Float64, nx, nx)

        for l = 0:L-1
            θ_l = dθ*l

        # -------- allocate anggood BitArray
            anggood = falses(nx, nx)

        # -------- loop over pixels
            for x = 1:nx
                sx = mod(x+dx, nx)-dx -1    # define sx,sy so that no fftshift() needed
                for y = 1:nx
                    sy = mod(y+dx, nx)-dx -1
                    θ_pix  = mod(atan(sy, sx)+π -θ_l, 2*π)
                    θ_good = abs(θ_pix-π) <= (dθ*wid)

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
        #          the angular factor is the same for all j
            F_angular = cos.((θ[angmask].-π).*(4/wid))

        # -------- loop over j for the radial part
            for j = 0:J-1
                jrad  = 7-j
                Δj    = abs.(logr[angmask].-jrad)
                rmask = (Δj .<= 1)

        # -------- radial part
                F_radial = cos.(Δj[rmask] .* (π/2))
                ind      = angmask[rmask]
                filt[ind,j+1,l+1] = F_radial .* F_angular[rmask]
            end
        end
        return filt
    end


    ## Make a list of non-zero pixels for the (Fourier plane) filters
    function fink_filter_list(filt)
        (ny,nx,J,L) = size(filt)

        # Allocate output arrays
        filtind = fill(CartesianIndex{2}[], J, L)
        filtval = fill(Float64[], J, L)

        # Loop over J,L and record non-zero values
        for l=1:L
            for j=1:J
                f = @view filt[:,:,j,l]
                ind = findall(f .> 1E-13)
                val = f[ind]
                filtind[j,l] = ind
                filtval[j,l] = val
            end
        end
        return [filtind, filtval]
    end


    ## Todo list
    # Check if 2 threads really help FFT when computer is busy
    function speedy_DHC(image::Array{Float64,2}, filter_list)
        # Use 2 threads for FFT
        FFTW.set_num_threads(2)

        # array sizes
        (Nx, Ny)  = size(image)
        (J,L)     = size(filter_list[1])

        # allocate coeff arrays
        out_coeff = []
        S0  = zeros(Float64, 2)
        S1  = zeros(Float64, J, L)
        S20 = zeros(Float64, J, L, J, L)
        S12 = zeros(Float64, J, L, J, L)

        # allocate image arrays for internal use
        im_fdf_0_1 = zeros(Float64,           Nx, Ny, J, L)   # this must be zeroed!
        im_rd_0_1  = Array{Float64, 4}(undef, Nx, Ny, J, L)

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
        for l = 1:L
            for j = 1:J
                S1tot = 0.0
                f_i = f_ind[j,l]  # CartesianIndex list for filter
                f_v = f_val[j,l]  # Values for f_i
                # for (ind, val) in zip(f_i, f_v)   # this is slower!
                if length(f_i) > 0
                    for i = 1:length(f_i)
                        ind       = f_i[i]
                        zval      = f_v[i] * im_fd_0[ind]
                        S1tot    += abs2(zval)
                        zarr[ind] = zval
                        im_fdf_0_1[ind,j,l] = abs(zval)
                    end
                    S1[j,l] = S1tot
                    im_rd_0_1[:,:,j,l] .= abs2.(P*zarr)
                    zarr[f_i] .= 0
                end
            end
        end
        append!(out_coeff, S1[:])

        # we stored the abs()^2, so take sqrt (this is faster to do all at once)
        im_rd_0_1 .= sqrt.(im_rd_0_1)

        ## 2nd Order
        Amat = reshape(im_fdf_0_1, Nx*Ny, J*L)
        S12  = reshape(Amat' * Amat, J, L, J, L)
        Amat = reshape(im_rd_0_1, Nx*Ny, J*L)
        S20  = reshape(Amat' * Amat, J, L, J, L)

        append!(out_coeff, S20)
        append!(out_coeff, S12)

        return out_coeff
    end

end # of module
