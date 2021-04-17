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

function fink_filter_bank_p(c, L; nx=256, wd=2, t=1, shift=false, Omega=false, safety_on=true, wd_cutoff=1, p=2)
    #c     - sets the scale sampling rate (1 is dyadic, 2 is half dyadic)
    #L     - number of angular bins (usually 8*t or 16*t)
    #wd    - width of the wavelets (default 1, wd=2 for a double covering)
    #t    - plane coverage (default 1, full 2pi 2)
    #shift - shift in θ by 1/2 of the θ spacing
    #Omega - true= append Omega filter (all power beyond Nyquist) so the sum of filters is 1.0

    # -------- assertion errors to make sure arguments are reasonable
    #@test wd <= L/2

    # -------- set parameters
    dθ   = t*π/L
    θ_sh = shift ? dθ/2 : 0.0
    dx   = nx/2-1

    im_scale = convert(Int8,log2(nx))
    # -------- number of bins in radial direction (size scales)
    J = (im_scale-3)*c + 1
    normj = 1/sqrt(c)

    # -------- allocate output array of zeros
    filt      = zeros(nx, nx, J*L+(Omega ? 2 : 1))
    psi_index = zeros(Int32, J, L)
    psi_ind_in= zeros(Int32, J*L+(Omega ? 2 : 1), 2)
    psi_ind_L = zeros(Int32, J*L+(Omega ? 2 : 1))
    theta     = zeros(Float64, L)
    j_value   = zeros(Float64, J)
    info=Dict{String,Any}()

    # -------- compute the required wd
    j_rad_exp = zeros(J)
    for j_ind = 1:J
        j = (j_ind-1)/c
        jrad  = im_scale-j-2
        j_rad_exp[j_ind] = 2^(jrad)
    end

    wd_j = max.(ceil.(wd_cutoff.*L./(t.*π.*j_rad_exp)),wd)

    if !safety_on
        wd_j.=wd
    end

    # loop over wd from small to large
    ## there is some uneeded redundancy added by doing this esp in l loop
    for wd in sort(unique(wd_j))
        # -------- allocate theta and logr arrays
        θ    = zeros(nx, nx)
        logr = zeros(nx, nx)

        wdθ  = wd*dθ
        norm = 1.0/(sqrt(wd))
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
            F_angular = norm .* cos.((θ[angmask].-π).*(L/(2*wd*t))).^(2/p)

        # -------- loop over j for the radial part
        #    for (j_ind, j) in enumerate(1/c:1/c:im_scale-2)
            j_ind_w_wd = findall(wd_j.==wd)
            for j_ind in j_ind_w_wd
                j = (j_ind-1)/c
                j_value[j_ind] = 1+j  # store for later
                jrad  = im_scale-j-2
                Δj    = abs.(logr[angmask].-jrad)
                rmask = (Δj .<= 1) #deprecating the 1/c to 1, constant width

        # -------- radial part
                F_radial = normj .* cos.(Δj[rmask] .* (π/2)).^(2/p) #deprecating c*π/2 to π/2
                ind      = angmask[rmask]
        #      Let's have these be (J,L) if you reshape...
        #        f_ind    = (j_ind-1)*L+l+1
                f_ind    = j_ind + l*J
                filt[ind, f_ind] = F_radial .* F_angular[rmask]
                psi_index[j_ind,l+1] = f_ind
                psi_ind_in[f_ind,:] = [j_ind-1,l]
                psi_ind_L[f_ind] = 1
            end
        end
    end

    # -------- phi contains power near k=0 not yet accounted for
    filter_power = (sum(filt.*filt, dims=3))[:,:,1]

    # -------- for plane half-covered (t=1), add other half-plane
    if t == 1
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

    # -------- before adding ϕ to filter bank, renormalize ψ if t=1
    if t==1 filt .*= sqrt(2.0) end  # double power for half coverage

    # -------- add result to filter array
    phi_index  = J*L+1
    filt[:,:,phi_index] .= fftshift(phi_cen)
    psi_ind_in[phi_index,:] = [J,0]
    psi_ind_L[phi_index] = 0

    if Omega     # append a filter containing the rest (outside Nyquist)
        filter_power += filt[:,:,phi_index].^2
        edge_power    = 1.0 .- filter_power
        zind          = findall(edge_power .< 1E-15)
        edge_power[zind]     .= 0.0  # set small numbers to zero
        Omega_index           = J*L+2
        info["Omega_index"]   = Omega_index
        filt[:,:,Omega_index] = sqrt.(edge_power)
        psi_ind_in[Omega_index,:] = [J,1]
        psi_ind_L[Omega_index] = 0
    end

    # -------- metadata dictionary
    info["npix"]         = nx
    info["j_value"]      = j_value
    info["theta_value"]  = theta
    info["psi_index"]    = psi_index
    info["phi_index"]    = phi_index
    info["J_L"]          = psi_ind_in
    info["t"]            = t
    info["wd"]           = wd_j
    info["wd_cutoff"]    = wd_cutoff
    info["fs_center_r"]  = j_rad_exp
    info["psi_ind_L"]    = psi_ind_L

    return filt, info
end

test_p = fink_filter_bank_p(1,8)
test_bank = fink_filter_bank(1,8)
test_bank .== test_p

function DHC_compute_p(image::Array{Float64,2}, filter_hash::Dict, filter_hash2::Dict=filter_hash;
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
