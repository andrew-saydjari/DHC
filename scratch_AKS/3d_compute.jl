push!(LOAD_PATH, "/Users/saydjari/Dropbox/GradSchool_AKS/Doug/Projects/DHC/main")
using DHC_2DUtils
using Profile
using BenchmarkTools
using FFTW
using HDF5
using Test
using SparseArrays

function fink_filter_bank_3dizer(hash, cz; nz=256)
    # -------- set parameters
    nx = convert(Int64,hash["npix"])
    n2d = size(hash["filt_value"])[1]
    dz = nz/2-1
    J = size(hash["j_value"])[1]
    L = size(hash["theta_value"])[1]

    im_scale = convert(Int8,log2(nz))
    # -------- number of bins in radial direction (size scales)
    K = (im_scale-2)*cz

    # -------- allocate output array of zeros
    J_L_K     = zeros(Int32, n2d*K,3)
    psi_index = zeros(Int32, J+1, L, K)
    k_value   = zeros(Float64, K)
    logr      = zeros(Float64,nz)
    filt_temp = zeros(Float64,nx,nx,nz)
    F_z       = zeros(Float64,nx,nx,nz)
    #F_p       = zeros(Float64,nx,nx,nz)

    filtind = fill(CartesianIndex{3}[], K*n2d)
    filtval = fill(Float64[], K*n2d)

    for z = 1:convert(Int64,dz+1)
        sz = mod(z+dz,nz)-dz-1
        z2 = sz^2
        logr[z] = 0.5*log2(max(1,z2))
    end

    @inbounds for k_ind = 1:K
        k = k_ind/cz
        k_value[k_ind] = k  # store for later
        krad  = im_scale-k-1
        Δk    = abs.(logr.-krad)
        kmask = (Δk .<= 1/cz)
        k_count = count(kmask)
        k_vals = findall(kmask)

        # -------- radial part
        #I have somehow dropped a factor of sqrt(2) which has been reinserted as a 0.5... fix my brain
        @views F_z[:,:,kmask].= reshape(1/sqrt(2).*cos.(Δk[kmask] .* (cz*π/2)),1,1,k_count)
        @inbounds for index = 1:n2d
            p_ind = hash["filt_index"][index]
            p_filt = hash["filt_value"][index]
            #F_p[p_ind,:] .= p_filt
            f_ind    = k_ind + (index-1)*K
            @views filt_tmp = p_filt.*F_z[p_ind,kmask]
            #temp_ind = findall((F_p .> 1E-13) .& (F_z .> 1E-13))
            #@views filt_tmp = F_p[temp_ind].*F_z[temp_ind]
            ind = findall(filt_tmp .> 1E-13)
            @views filtind[f_ind] = map(x->CartesianIndex(p_ind[x[1]][1],p_ind[x[1]][2],k_vals[x[2]]),ind)
            #filtind[f_ind] = temp_ind[ind]
            @views filtval[f_ind] = filt_tmp[ind]
            @views J_L_K[f_ind,:] = [hash["J_L"][index,1],hash["J_L"][index,2],k_ind-1]
            psi_index[hash["J_L"][index,1]+1,hash["J_L"][index,2]+1,k_ind] = f_ind
        end
    end

    # -------- metadata dictionary
    info=Dict()
    info["2d_npix"]         = hash["npix"]
    info["2d_j_value"]      = hash["j_value"]
    info["2d_theta_value"]  = hash["theta_value"]
    info["2d_psi_index"]    = hash["psi_index"]
    info["2d_phi_index"]    = hash["phi_index"]
    info["2d_J_L"]          = hash["J_L"]
    info["2d_pc"]           = hash["pc"]
    info["2d_wd"]           = hash["wd"]
    info["2d_fs_center_r"]  = hash["fs_center_r"]
    info["2d_filt_index"]   = hash["filt_index"]
    info["2d_filt_value"]   = hash["filt_value"]

    info["nz"]              = nz
    info["cz"]              = cz
    info["J_L_K"]           = psi_index
    info["psi_index"]       = psi_index
    info["k_value"]         = k_value
    info["filt_index"]      = filtind
    info["filt_value"]      = filtval

    return info
end

filter_hash = fink_filter_hash(1, 8, nx=256, pc=1, wd=2)
filt_3d = fink_filter_bank_3dizer(filter_hash, 1, nz=256)

function DHC_compute_3d(image::Array{Float64,3}, filter_hash)
    # Use 2 threads for FFT
    FFTW.set_num_threads(2)

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
    im_rd_0_1  = Array{Float64, 3}(undef, Nx, Ny, Nz, Nf)

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
    append!(out_coeff, S2)

    return out_coeff
end
