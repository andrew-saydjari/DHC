## Preloads
module DHC_2DUtils

    using FFTW
    using LinearAlgebra
    using Statistics

    export finklet
    export fink_filter_bank
    export fink_filter_list
    export fink_filter_hash
    export DHC_compute


    function fink_filter_bank(c, L; nx=256, wd=1, pc=1, shift=false)
        #c     - sets the scale sampling rate (1 is dyadic, 2 is half dyadic)
        #L     - number of angular bins (usually 8*pc or 16*pc)
        #wd    - width of the wavelets (default 1, wd=2 for a double covering)
        #pc    - plane coverage (default 1, full 2pi 2)
        #shift - shift in θ by 1/2 of the θ spacing

        # -------- set parameters
        dθ   = pc*π/L
        wdθ  = wd*dθ
        θ_sh = shift ? dθ/2 : 0.0
        dx   = nx/2-1
        norm = 1.0/sqrt(wd)

        im_scale = convert(Int8,log2(nx))
        # -------- number of bins in radial direction (size scales)
        J = (im_scale-2)*c

        # -------- allocate output array of zeros
        filt      = zeros(nx, nx, J*L+1)
        psi_index = zeros(Int32, J, L)
        theta     = zeros(Float64, L)
        j_value   = zeros(Float64, J)

        # -------- allocate theta and logr arrays
        θ    = zeros(nx, nx)
        logr = zeros(nx, nx)

        # -------- loop over l
        for l = 0:L-1
            θ_l        = dθ*l+θ_sh
            theta[l+1] = θ_l

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
                        r2            = sx^2 + sy^2
                        logr[y, x]    = 0.5*log2(max(1,r2))
                    end
                end
            end
            angmask = findall(anggood)
        # -------- compute the wavelet in the Fourier domain
        # -------- the angular factor is the same for all j
            F_angular = norm .* cos.((θ[angmask].-π).*(L/(2*wd*pc)))

        # -------- loop over j for the radial part
        #    for (j_ind, j) in enumerate(1/c:1/c:im_scale-2)
            for j_ind = 1:c*(im_scale-2)
                j = j_ind/c
                j_value[j_ind] = j  # store for later
                jrad  = im_scale-j-1
                Δj    = abs.(logr[angmask].-jrad)
                rmask = (Δj .<= 1/c)

        # -------- radial part
                F_radial = cos.(Δj[rmask] .* (c*π/2))
                ind      = angmask[rmask]
                f_ind    = (j_ind-1)*L+l+1
                filt[ind, f_ind] = F_radial .* F_angular[rmask]
                psi_index[j_ind,l+1] = f_ind
            end
        end

        # -------- phi contains power near k=0 not yet accounted for
        filter_power = (sum(filt.*filt, dims=3))[:,:,1]

        # -------- for plane half-covered (pc=1), add other half-plane
        if pc == 1
            filter_power .+= circshift(filter_power[end:-1:1,end:-1:1],(1,1))
        end

        # -------- compute power required to sum to 1.0
        i0 = round(Int16,nx/2-2)
        i1 = round(Int16,nx/2+4)
        center_power = 1.0 .- fftshift(filter_power)[i0:i1,i0:i1]
        zind = findall(center_power .< 1E-15)
        center_power[zind] .= 0.0  # set small numbers to zero
        phi_cen = zeros(nx, nx)
        phi_cen[i0:i1,i0:i1] = sqrt.(center_power)

        # -------- add result to filter array
        phi_index  = J*L+1
        filt[:,:,phi_index] .= fftshift(phi_cen)

        # -------- metadata dictionary
        info=Dict()
        info["npix"]         = nx
        info["j_value"]      = j_value
        info["theta_value"]  = theta
        info["psi_index"]    = psi_index
        info["phi_index"]    = phi_index
        info["pc"]           = pc
        info["wd"]           = wd

        return filt, info
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


    function fink_filter_hash(c, L; nx=256, wd=1, pc=1, shift=false)
        # -------- compute the filter bank
        filt, hash = fink_filter_bank(c, L; nx=nx, wd=wd, pc=pc, shift=shift)

        # -------- list of non-zero pixels
        flist = fink_filter_list(filt)

        # -------- pack everything you need into the info structure
        hash["filt_index"] = flist[1]
        hash["filt_value"] = flist[2]
        return hash
    end


    function DHC_compute(image::Array{Float64,2}, filter_hash)
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

        # unpack filter_hash
        f_ind   = filter_hash["filt_index"]  # (J, L) array of filters represented as index value pairs
        f_val   = filter_hash["filt_value"]

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
