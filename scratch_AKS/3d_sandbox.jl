push!(LOAD_PATH, "/Users/saydjari/Dropbox/GradSchool_AKS/Doug/Projects/DHC/main")
using DHC_2DUtils
using Profile
using BenchmarkTools

function fink_filter_bank(c, L; nx=256, wd=1, pc=1, shift=false, Omega=false, safety_on=true)
    #c     - sets the scale sampling rate (1 is dyadic, 2 is half dyadic)
    #L     - number of angular bins (usually 8*pc or 16*pc)
    #wd    - width of the wavelets (default 1, wd=2 for a double covering)
    #pc    - plane coverage (default 1, full 2pi 2)
    #shift - shift in θ by 1/2 of the θ spacing
    #Omega - true= append Omega filter (all power beyond Nyquist) so the sum of filters is 1.0

    # -------- assertion errors to make sure arguments are reasonable
    #@test wd <= L/2

    # -------- set parameters
    dθ   = pc*π/L
    θ_sh = shift ? dθ/2 : 0.0
    dx   = nx/2-1

    im_scale = convert(Int8,log2(nx))
    # -------- number of bins in radial direction (size scales)
    J = (im_scale-2)*c

    # -------- allocate output array of zeros
    filt      = zeros(nx, nx, J*L+(Omega ? 2 : 1))
    psi_index = zeros(Int32, J, L)
    theta     = zeros(Float64, L)
    j_value   = zeros(Float64, J)

    # -------- compute the required wd
    j_rad_exp = zeros(J)
    for j_ind = 1:J
        j = j_ind/c
        jrad  = im_scale-j-1
        j_rad_exp[j_ind] = 2^(jrad)
    end

    wd_j = max.(ceil.(L./(pc.*π.*j_rad_exp)),wd)

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
        norm = 1.0/sqrt(wd)
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
            j_ind_w_wd = findall(wd_j.==wd)
            for j_ind = j_ind_w_wd
                j = j_ind/c
                j_value[j_ind] = j  # store for later
                jrad  = im_scale-j-1
                Δj    = abs.(logr[angmask].-jrad)
                rmask = (Δj .<= 1/c)

        # -------- radial part
                F_radial = cos.(Δj[rmask] .* (c*π/2))
                ind      = angmask[rmask]
        #      Let's have these be (J,L) if you reshape...
        #        f_ind    = (j_ind-1)*L+l+1
                f_ind    = j_ind + l*J
                filt[ind, f_ind] = F_radial .* F_angular[rmask]
                psi_index[j_ind,l+1] = f_ind
            end
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

    # -------- before adding ϕ to filter bank, renormalize ψ if pc=1
    if pc==1 filt .*= sqrt(2.0) end  # double power for half coverage

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
    info["wd"]           = wd_j
    info["fs_center_r"]  = j_rad_exp

    if Omega     # append a filter containing the rest (outside Nyquist)
        filter_power += filt[:,:,phi_index].^2
        edge_power    = 1.0 .- filter_power
        zind          = findall(edge_power .< 1E-15)
        edge_power[zind]     .= 0.0  # set small numbers to zero
        Omega_index           = J*L+2
        info["Omega_index"]   = Omega_index
        filt[:,:,Omega_index] = sqrt.(edge_power)
    end


    return filt, info
end

function fink_filter_bank_3dizer(hash, cz; nz=256)
    # -------- set parameters
    nx = convert(Int64,hash["npix"])
    n2d = size(hash["filt_value"])[1]
    dz = nz/2-1

    im_scale = convert(Int8,log2(nz))
    # -------- number of bins in radial direction (size scales)
    K = (im_scale-2)*cz

    # -------- allocate output array of zeros
    #filt      = zeros(Float64, nx, nx, nz, K*n2d)
    #psi_index = zeros(Int32, J, L, K)
    #theta     = zeros(Float64, L)
    k_value   = zeros(Float64, K)
    logr      = zeros(Float64,nz)
    filt_temp = zeros(Float64,nx,nx,nz)
    F_z       = zeros(Float64,nx,nx,nz)

    filtind = fill(CartesianIndex{2}[], K*n2d)
    filtval = fill(Float64[], K*n2d)

    for z = 1:convert(Int64,dz+1)
        sz = mod(z+dz,nz)-dz-1
        z2 = sz^2
        logr[z] = 0.5*log2(max(1,z2))
    end

    for k_ind = 1:K
        k = k_ind/cz
        k_value[k_ind] = k  # store for later
        krad  = im_scale-k-1
        Δk    = abs.(logr.-krad)
        kmask = (Δk .<= 1/cz)
        k_count = count(kmask)

        # -------- radial part
        F_z[:,:,kmask].= reshape(cos.(Δk[kmask] .* (cz*π/2)),1,1,k_count)

        for index = 1:n2d
            p_ind = hash["filt_index"][index]
            p_filt = hash["filt_value"][index]
            f_ind    = k_ind + (index-1)*K
            filt_tmp = p_filt.*F_z[p_ind,kmask] #
            ind = findall(filt_tmp .> 1E-13)
            val = filt_tmp[ind]
            filtind[f_ind] = ind
            filtval[f_ind] = val
            #psi_index[j_ind,l+1] = f_ind
        end
    end

    # -------- metadata dictionary
    info=Dict()
    info["2d_npix"]         = hash["npix"]
    info["2d_j_value"]      = hash["npix"]
    info["2d_theta_value"]  = hash["theta_value"]
    info["2d_psi_index"]    = hash["psi_index"]
    info["2d_phi_index"]    = hash["phi_index"]
    info["2d_pc"]           = hash["pc"]
    info["2d_wd"]           = hash["wd"]
    info["2d_fs_center_r"]  = hash["fs_center_r"]

    info["nz"]              = nz
    info["cz"]              = cz
    info["k_value"]         = k_value
    info["filt_index"]      = filtind
    info["filt_value"]      = filtval

    return info
end

hash = fink_filter_hash(1, 4, nx=128, pc=1, wd=1)

@time filt_3d = fink_filter_bank_3dizer(hash, 1, nz=128)

using HDF5

h5write("../DHC/scratch_AKS/data/filt_3d.h5", "main/data", filt_3d[:,:,:,49])

Profile.clear()
@profile filt_3d = fink_filter_bank_3dizer(hash, 1, nz=128)

Juno.profiler()

@benchmark filt_3d = fink_filter_bank_3dizer(hash, 1, nz=128)

hash = fink_filter_hash(1, 8, nx=128, pc=1, wd=1)
@benchmark filt_3d = fink_filter_bank_3dizer(hash, 1, nz=128)


hash = fink_filter_hash(1, 8, nx=256, pc=1, wd=1)
@benchmark filt_3d = fink_filter_bank_3dizer(hash, 1, nz=256)


hash = fink_filter_hash(1, 8, nx=256, pc=1, wd=2)
@benchmark filt_3d = fink_filter_bank_3dizer(hash, 1, nz=256)
