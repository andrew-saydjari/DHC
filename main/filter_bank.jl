## Filter hash construct core

function fink_filter_hash(c, L; nx=256, wd=2, t=1, shift=false, Omega=false, safety_on=true, wd_cutoff=1)
    # -------- compute the filter bank
    filt, hash = fink_filter_bank(c, L; nx=nx, wd=wd, t=t, shift=shift, Omega=Omega, safety_on=safety_on, wd_cutoff=wd_cutoff)

    # -------- list of non-zero pixels
    flist = fink_filter_list(filt)

    # -------- pack everything you need into the info structure
    hash["filt_index"] = flist[1]
    hash["filt_value"] = flist[2]

    # -------- compute matrix that projects iso coeffs, add to hash
    S1_iso_mat = S1_iso_matrix(hash)
    hash["S1_iso_mat"] = S1_iso_mat
    S2_iso_mat = S2_iso_matrix(hash)
    hash["S2_iso_mat"] = S2_iso_mat
    hash["num_iso_coeff"] = size(S1_iso_mat)[1] + size(S2_iso_mat)[1] + 2
    hash["num_coeff"] = size(S1_iso_mat)[2] + size(S2_iso_mat)[2] + 2

    return hash
end

function fink_filter_hash_p(c, L; nx=256, wd=2, t=1, shift=false, Omega=false, safety_on=true, wd_cutoff=1, p=2)
    # -------- compute the filter bank
    filt, hash = fink_filter_bank_p(c, L; nx=nx, wd=wd, t=t, shift=shift, Omega=Omega, safety_on=safety_on, wd_cutoff=wd_cutoff, p=p)

    # -------- list of non-zero pixels
    flist = fink_filter_list(filt)

    # -------- pack everything you need into the info structure
    hash["filt_index"] = flist[1]
    hash["filt_value"] = flist[2]

    # -------- compute matrix that projects iso coeffs, add to hash
    S1_iso_mat = S1_iso_matrix(hash)
    hash["S1_iso_mat"] = S1_iso_mat
    S2_iso_mat = S2_iso_matrix(hash)
    hash["S2_iso_mat"] = S2_iso_mat
    hash["num_iso_coeff"] = size(S1_iso_mat)[1] + size(S2_iso_mat)[1] + 2
    hash["num_coeff"] = size(S1_iso_mat)[2] + size(S2_iso_mat)[2] + 2

    return hash
end

function fink_filter_hash_gpu(c, L; nx=256, wd=2, t=1, shift=false, Omega=false, safety_on=true,threeD=false,cz=1,nz=256)

    # -------- compute the filter bank
    filt, hash = fink_filter_bank(c, L; nx=nx, wd=wd, t=t, shift=shift, Omega=Omega, safety_on=safety_on)

    flist = fink_filter_list(filt)
    # -------- pack everything you need into the info structure
    hash["filt_index"] = flist[1]
    hash["filt_value"] = flist[2]

    if threeD
        hash=fink_filter_bank_3dizer(hash,cz,nz=nz)
    end

    hash=list_to_box(hash,dim=threeD ? 3 : 2)

    # send to gpu
    CUDA.reclaim()
    hash["filt_vals"] = [CUDA.CuArray(val) for val in hash["filt_vals"]]
    CUDA.reclaim()

    #Not implemented
    # -------- compute matrix that projects iso coeffs, add to hash
    #S1_iso_mat = S1_iso_matrix(hash)
    hash["S1_iso_mat"] = nothing
    #S2_iso_mat = S2_iso_matrix(hash)
    hash["S2_iso_mat"] = nothing

    return hash
end

function fink_filter_hash_gpu_sparse(c, L; nx=256, wd=2, t=1, shift=false, Omega=false, safety_on=true, wd_cutoff=1, prec=Float32)
    filt, hash = eqws.fink_filter_bank(c, L; nx=nx, wd=wd, t=t, shift=shift, Omega=Omega, safety_on=safety_on, wd_cutoff=wd_cutoff)

    Nf = size(filt)[3]
    hash["Nf"] = Nf

    gpu_filt = CUDA.zeros(Float64,size(filt))
    #send to GPU by convert to sparse CPU, transfer to sparse GPU, convert to dense GPU
    for i=1:Nf
        gpu_filt[:,:,i] .= CUDA.CuArray(CuSparseMatrixCSC(sparse(filt[:,:,i])))
    end
    hash["gpu_filt"] = gpu_filt

    S1_iso_mat = CuSparseMatrixCSC(S1_iso_matrix_gpu(hash, prec))
    hash["S1_iso_mat"] = S1_iso_mat
    S2_iso_mat = CuSparseMatrixCSC(S2_iso_matrix_gpu(hash, prec))
    hash["S2_iso_mat"] = S2_iso_mat
    hash["num_iso_coeff"] = size(S1_iso_mat)[1] + size(S2_iso_mat)[1] + 2
    hash["num_coeff"] = size(S1_iso_mat)[2] + size(S2_iso_mat)[2] + 2

    return hash
end

function fink_filter_hash_gpu_fake(c, L; nx=256, wd=2, t=1, shift=false, Omega=false, safety_on=true,threeD=false,cz=1,nz=256)

    # -------- compute the filter bank
    filt, hash = fink_filter_bank(c, L; nx=nx, wd=wd, t=t, shift=shift, Omega=Omega, safety_on=safety_on)

    flist = fink_filter_list(filt)
    # -------- pack everything you need into the info structure
    hash["filt_index"] = flist[1]
    hash["filt_value"] = flist[2]

    if threeD
        hash=fink_filter_bank_3dizer(hash,cz,nz=nz)
    end

    hash=list_to_box(hash,dim=threeD ? 3 : 2)


    #Not implemented
    # -------- compute matrix that projects iso coeffs, add to hash
    #S1_iso_mat = S1_iso_matrix(hash)
    hash["S1_iso_mat"] = nothing
    #S2_iso_mat = S2_iso_matrix(hash)
    hash["S2_iso_mat"] = nothing

    return hash
end

function fink_filter_bank_3dizer(hash, cz; nz=256, tz=1, Omega3d=false)
    @assert hash["t"]==2 #can remove later when we add support for tz=2
    @assert tz==1

    # -------- set parameters
    nx = convert(Int64,hash["npix"])
    n2d = size(hash["filt_value"])[1]
    dz = nz/2-1
    J = size(hash["j_value"])[1]
    L = size(hash["theta_value"])[1]
    psi_ind_L2d = hash["psi_ind_L"]

    im_scale = convert(Int8,log2(nz))
    # -------- number of bins in radial direction (size scales)
    K = (im_scale-3)*cz + 1
    normj = 1/sqrt(cz)

    Omega2d = haskey(hash, "Omega_index")

    if Omega2d
        tot_num_filt = n2d*K + 3
    else
        tot_num_filt = n2d*K + 1
    end

    # -------- allocate output array of zeros
    J_L_K     = zeros(Int32, tot_num_filt,3)
    psi_ind_L = zeros(Int32, tot_num_filt)
    psi_index = zeros(Int32, J+1, L, K+1)
    k_value   = zeros(Float64, K)
    logr      = zeros(Float64,nz)
    filt_temp = zeros(Float64,nx,nx,nz)
    F_z       = zeros(Float64,nx,nx,nz)
    #F_p       = zeros(Float64,nx,nx,nz)

    filtind = fill(CartesianIndex{3}[], tot_num_filt)
    filtval = fill(Float64[], tot_num_filt)

    for z = 1:convert(Int64,dz+1)
        sz = mod(z+dz,nz)-dz-1
        z2 = sz^2
        logr[z] = 0.5*log2(max(1,z2))
    end

    @inbounds for k_ind = 1:K
        k = (k_ind-1)/cz
        k_value[k_ind] = k  # store for later
        krad  = im_scale-k-2
        Δk    = abs.(logr.-krad)
        kmask = (Δk .<= 1)
        k_count = count(kmask)
        k_vals = findall(kmask)

        # -------- radial part
        #I have somehow dropped a factor of sqrt(2) which has been reinserted as a 0.5... fix my brain
        @views F_z[:,:,kmask].= reshape(cos.(Δk[kmask] .* (π/2)),1,1,k_count)
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
            psi_ind_L[f_ind] = psi_ind_L2d[index]
        end
    end

    filter_power = zeros(nx,nx,nz)
    center_power = zeros(nx,nx,nz)
    for j = 1:J
        for l = 1:L
            for k = 1:K
                index = psi_index[j,l,k]
                filter_power[filtind[index]] .+= filtval[index].^2
            end
        end
    end

    for k = 1:K
        index = psi_index[J+1,1,k]
        filter_power[filtind[index]] .+= filtval[index].^2
    end

    #### ------- This is phi0 containing the plane
    # phi contains power near k=0 not yet accounted for
    # for plane half-covered (tz=1), add other half-plane

    if tz == 1
        filter_power .+= circshift(filter_power[:,:,end:-1:1],(0,0,1))
    end

    center_power = fftshift(filter_power)
    phi_cen = zeros(nx, nx, nz)
    phi_cen[:,:,nz÷2:nz÷2+2] = center_power[:,:,nz÷2+3].*ones(1,1,3)
    phi_cen_shift = fftshift(phi_cen)
    ind = findall(phi_cen_shift .> 1E-13)
    val = phi_cen_shift[ind]

    # -------- before adding ϕ to filter bank, renormalize ψ if t=1
    if tz==1 filtval[1:n2d*K] .*= sqrt(2.0) end  # double power for half coverage

    # -------- add result to filter array
    filtind[n2d*K + 1] = ind
    filtval[n2d*K + 1] = sqrt.(val)
    J_L_K[n2d*K + 1,:] = [J+1,0,K+1]
    psi_index[J+1,1,K+1] = n2d*K + 1
    psi_ind_L[n2d*K + 1] = 0

    if Omega3d
        #### ------- This is omega0 containing the plane

        omega0 = zeros(nx, nx, nz)
        omega0[:,:,nz÷2:nz÷2+2] .= ones(nx,nx,3)
        omega0 .-= phi_cen

        omega0_shift = fftshift(omega0)
        ind = findall(omega0_shift .> 1E-13)
        val = omega0_shift[ind]

        # -------- add result to filter array
        filtind[n2d*K + 2] = ind
        filtval[n2d*K + 2] = sqrt.(val)
        J_L_K[n2d*K + 2,:] = [J+1,1,K+1]
        psi_index[J+1,2,K+1] = n2d*K + 2
        psi_ind_L[n2d*K + 2] = 0

        #### ------- This is omega3 around the edges

        filter_power = zeros(nx,nx,nz)
        center_power = zeros(nx,nx,nz)
        for index = 1:n2d*K + 2
            filter_power[filtind[index]] .+= filtval[index].^2
        end

        if tz == 1
            filter_power .+= circshift(filter_power[:,:,end:-1:1],(0,0,1))
        end

        center_power = ones(nx,nx,nz) .- filter_power
        ind = findall(center_power .> 1E-13)
        val = center_power[ind]

        # -------- add result to filter array
        filtind[n2d*K + 3] = ind
        filtval[n2d*K + 3] = sqrt.(val)
        J_L_K[n2d*K + 3,:] = [J+1,2,K+1]
        psi_index[J+1,3,K+1] = n2d*K + 3
        psi_ind_L[n2d*K + 3] = 0
    end

    # -------- metadata dictionary
    info=Dict()
    info["npix"]         = hash["npix"]
    info["j_value"]      = hash["j_value"]
    info["theta_value"]  = hash["theta_value"]
    info["2d_psi_index"]    = hash["psi_index"]
    info["2d_phi_index"]    = hash["phi_index"]
    info["2d_J_L"]          = hash["J_L"]
    info["2d_t"]           = hash["t"]
    info["2d_wd"]           = hash["wd"]
    info["2d_fs_center_r"]  = hash["fs_center_r"]
    info["2d_filt_index"]   = hash["filt_index"]
    info["2d_filt_value"]   = hash["filt_value"]
    if Omega2d
        info["Omega_index"]   = hash["Omega_index"]
    end

    info["nz"]              = nz
    info["cz"]              = cz
    info["J_L_K"]           = psi_index
    info["psi_index"]       = psi_index
    info["k_value"]         = k_value
    info["filt_index"]      = filtind
    info["filt_value"]      = filtval
    info["tz"]             = tz
    info["Omega3d"]        = Omega3d
    info["psi_ind_L"]       = psi_ind_L

    # -------- compute matrix that projects iso coeffs, add to hash
    S1_iso_mat = S1_iso_matrix3d(info)
    info["S1_iso_mat"] = S1_iso_mat
    S2_iso_mat = S2_iso_matrix3d(info)
    info["S2_iso_mat"] = S2_iso_mat
    #

    return info
end

## Filter bank utilities

function fink_filter_bank(c, L; nx=256, wd=2, t=1, shift=false, Omega=false, safety_on=true, wd_cutoff=1)
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
            F_angular = norm .* cos.((θ[angmask].-π).*(L/(2*wd*t)))

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
                F_radial = normj .* cos.(Δj[rmask] .* (π/2)) #deprecating c*π/2 to π/2
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
    info["p"]            = 2

    return filt, info
end

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
        norm = 1.0/((wd).^(1/p))
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
    filter_power = (sum(filt.^p, dims=3))[:,:,1]

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
    phi_cen[i0:i1,i0:i1] = (center_power).^(1/p)

    # -------- before adding ϕ to filter bank, renormalize ψ if t=1
    if t==1 filt .*= 2.0^(1/p) end  # double power for half coverage

    # -------- add result to filter array
    phi_index  = J*L+1
    filt[:,:,phi_index] .= fftshift(phi_cen)
    psi_ind_in[phi_index,:] = [J,0]
    psi_ind_L[phi_index] = 0

    if Omega     # append a filter containing the rest (outside Nyquist)
        filter_power += filt[:,:,phi_index].^p
        edge_power    = 1.0 .- filter_power
        zind          = findall(edge_power .< 1E-15)
        edge_power[zind]     .= 0.0  # set small numbers to zero
        Omega_index           = J*L+2
        info["Omega_index"]   = Omega_index
        filt[:,:,Omega_index] = (edge_power).^(1/p)
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
    info["p"]            = p

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

function list_to_box(hash;dim=2)#this modifies the hash
    (Nf,) = size(hash["filt_index"])
    nx=hash["npix"]
    if dim==3 nz=hash["nz"] end

    # Allocate output arrays
    filtmms = Array{Int64}(undef,dim,2,Nf)
    filtvals = Array{Any}(undef,Nf)#is this a terrible way to allocate memory?
    if dim==2
        filt=zeros(Float64,nx,nx)
    else
        filt=zeros(Float64,nx,nx,nz)
    end
    # Loop over filters and record non-zero values
    for l=1:Nf
        filt.=0.
        f_i = hash["filt_index"][l]
        v_i=hash["filt_value"][l]
        for i = 1:length(f_i)
            filt[f_i[i]]= v_i[i]
        end
        filt = FFTW.fftshift(filt)

        ind = findall(filt .> 1E-13)
        if dim==2
            ind=getindex.(ind,[1 2])
        else
            ind=getindex.(ind,[1 2 3])
        end
        mins=minimum(ind,dims=1)
        maxs=maximum(ind,dims=1)


        filtmms[:,1,l] = mins
        filtmms[:,2,l] = maxs
        if dim==2
            filtvals[l]=filt[mins[1]:maxs[1],mins[2]:maxs[2]]
        else
            filtvals[l]=filt[mins[1]:maxs[1],mins[2]:maxs[2],mins[3]:maxs[3]]
        end
    end

    hash["filt_index"] = nothing
    hash["filt_value"] = nothing

    hash["filt_mms"] = filtmms
    hash["filt_vals"]=filtvals
    return hash
end
