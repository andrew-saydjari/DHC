push!(LOAD_PATH, "/Users/saydjari/Dropbox/GradSchool_AKS/Doug/Projects/DHC/main")
using DHC_2DUtils
using DHC_tests
using Profile
using BenchmarkTools
using FFTW
using HDF5
using Test
using SparseArrays
using Plots
theme(:dark)

filter_hash = fink_filter_hash(1, 8, nx=256, pc=1)
filt_3d = fink_filter_bank_3dizer_AKS(filter_hash, 1, nz=256)

plot_filter_bank_QAxy(filt_3d, fname="/Users/saydjari/Dropbox/GradSchool_AKS/Doug/Projects/DHC/scratch_AKS/images/filter_bank_QAxy.png")
plot_filter_bank_QAxz(filt_3d, fname="/Users/saydjari/Dropbox/GradSchool_AKS/Doug/Projects/DHC/scratch_AKS/images/filter_bank_QAxz.png")

function fink_filter_bank_3dizer_AKS(hash, cz; nz=256)
    # -------- set parameters
    nx = convert(Int64,hash["npix"])
    n2d = size(hash["filt_value"])[1]
    dz = nz/2-1
    J = size(hash["j_value"])[1]
    L = size(hash["theta_value"])[1]

    im_scale = convert(Int8,log2(nz))
    # -------- number of bins in radial direction (size scales)
    K = (im_scale-3)*cz + 1
    normj = 1/sqrt(cz)

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
        end
    end

    # -------- metadata dictionary
    info=Dict()
    info["npix"]         = hash["npix"]
    info["j_value"]      = hash["j_value"]
    info["theta_value"]  = hash["theta_value"]
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

    # -------- compute matrix that projects iso coeffs, add to hash
    S1_iso_mat = S1_iso_matrix3d(info)
    info["S1_iso_mat"] = S1_iso_mat
    S2_iso_mat = S2_iso_matrix3d(info)
    info["S2_iso_mat"] = S2_iso_mat
    #

    return info
end

filter_hash = fink_filter_hash(1, 8, nx=256, pc=2)
filt_3d = fink_filter_bank_3dizer_AKS(filter_hash, 1, nz=256)

nx = convert(Int64,filt_3d["npix"])
nz = 256
lup  = filt_3d["psi_index"]
index= filt_3d["filt_index"]
value= filt_3d["filt_value"]

filt = zeros(nx,nx,nz)
for x in 1:288
    filt[index[x]] .+= value[x].^2
end
for x in 289:294
    filt[index[x]] .+= value[x].^2
end
heatmap(fftshift(dropdims(sum(filt,dims=3),dims=3)))
heatmap(fftshift(dropdims(sum(filt[:,:,2:3],dims=3),dims=3)))
heatmap(fftshift(filt[:,:,3]))

h5write("../DHC/scratch_AKS/data/filt_3d_fs.h5", "data3", fftshift(filt))

p_ind = filter_hash["filt_index"][2]
p_filt = filter_hash["filt_value"][2]

filt = zeros(nx,nx)
filt[p_ind] = p_filt
heatmap(fftshift(filt))

test = p_filt.*ones(1,1,7)

test[:,:,1] == test[:,:,2]

fftshift(filt)+circshift(fftshift(filt)[:,:,end:-1:1],(0,0,1))

h5write("../DHC/scratch_AKS/data/filt_3d_fs.h5", "data5", fftshift(filt)+circshift(fftshift(filt)[:,:,end:-1:1],(0,0,1)))

filter_hash = fink_filter_hash(1, 8, nx=256, pc=2, Omega=true)
filt_3d = fink_filter_bank_3dizer_AKS(filter_hash, 1, nz=256)

nx = convert(Int64,filt_3d["npix"])
nz = 256
lup  = filt_3d["psi_index"]
index= filt_3d["filt_index"]
value= filt_3d["filt_value"]

filt = zeros(nx,nx,nz)
for x in 1:288
    filt[index[x]] .+= value[x].^2
end
for x in 289:294
    filt[index[x]] .+= value[x].^2
end
for x in 295:300
    filt[index[x]] .+= value[x].^2
end
heatmap(fftshift(dropdims(sum(filt,dims=3),dims=3)))
heatmap(fftshift(dropdims(sum(filt[:,:,2:3],dims=3),dims=3)))
heatmap(fftshift(filt[:,:,3]))

filt[:,:,3].-1.0

h5write("../DHC/scratch_AKS/data/filt_3d_fs.h5", "data6", fftshift(filt)+circshift(fftshift(filt)[:,:,end:-1:1],(0,0,1)))

function fink_filter_hash_AKS(c, L; nx=256, wd=2, pc=1, shift=false, Omega=false, safety_on=true, wd_cutoff=1)
    # -------- compute the filter bank
    filt, hash = fink_filter_bank_AKS(c, L; nx=nx, wd=wd, pc=pc, shift=shift, Omega=Omega, safety_on=safety_on, wd_cutoff=wd_cutoff)

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

function fink_filter_bank_AKS(c, L; nx=256, wd=2, pc=1, shift=false, Omega=false, safety_on=true, wd_cutoff=1)
    #c     - sets the scale sampling rate (1 is dyadic, 2 is half dyadic)
    #L     - number of angular bins (usually 8*pc or 16*pc)
    #wd    - width of the wavelets (default 1, wd=2 for a double covering)
    #pc    - plane coverage (default 1, full 2pi 2)
    #shift - shift in θ by 1/2 of the θ spacing
    #Omega - true= append Omega filter (all power beyond Nyquist) so the sum of filters is 1.0

    # -------- assertion errors to make sure arguments are reasonable
    @test wd <= L/2

    # -------- set parameters
    dθ   = pc*π/L
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
    theta     = zeros(Float64, L)
    j_value   = zeros(Float64, J)

    # -------- compute the required wd
    j_rad_exp = zeros(J)
    for j_ind = 1:J
        j = (j_ind-1)/c
        jrad  = im_scale-j-2
        j_rad_exp[j_ind] = 2^(jrad)
    end

    wd_j = max.(ceil.(wd_cutoff.*L./(pc.*π.*j_rad_exp)),wd)

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
            F_angular = norm .* cos.((θ[angmask].-π).*(L/(2*wd*pc)))

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
    psi_ind_in[phi_index,:] = [J,0]

    # -------- metadata dictionary
    info=Dict{String,Any}()
    info["npix"]         = nx
    info["j_value"]      = j_value
    info["theta_value"]  = theta
    info["psi_index"]    = psi_index
    info["phi_index"]    = phi_index
    info["J_L"]          = psi_ind_in
    info["pc"]           = pc
    info["wd"]           = wd_j
    info["wd_cutoff"]    = wd_cutoff
    info["fs_center_r"]  = j_rad_exp

    if Omega     # append a filter containing the rest (outside Nyquist)
        filter_power += filt[:,:,phi_index].^2
        edge_power    = 1.0 .- filter_power
        zind          = findall(edge_power .< 1E-15)
        edge_power[zind]     .= 0.0  # set small numbers to zero
        Omega_index           = J*L+2
        info["Omega_index"]   = Omega_index
        filt[:,:,Omega_index] = sqrt.(edge_power)
        psi_ind_in[Omega_index,:] = [J,1]
    end


    return filt, info
end

filter_hash = fink_filter_hash_AKS(1, 8, nx=256, pc=2, Omega=true)
filt_3d = fink_filter_bank_3dizer_AKS(filter_hash, 1, nz=256)

nx = convert(Int64,filt_3d["npix"])
nz = 256
lup  = filt_3d["psi_index"]
index= filt_3d["filt_index"]
value= filt_3d["filt_value"]

filt = zeros(nx,nx,nz)
for x in 1:288
    filt[index[x]] .+= value[x].^2
end
for x in 289:294
    filt[index[x]] .+= value[x].^2
end
for x in 295:300
    filt[index[x]] .+= value[x].^2
end
heatmap(fftshift(dropdims(sum(filt,dims=3),dims=3)))
heatmap(fftshift(dropdims(sum(filt[:,:,2:3],dims=3),dims=3)))
heatmap(fftshift(filt[:,:,3]))

function fink_filter_bank_3dizer_AKS(hash, cz; nz=256, pcz=1)
    @assert hash["pc"]==2 #can remove later when we add support for pcz=2
    @assert pcz==1

    # -------- set parameters
    nx = convert(Int64,hash["npix"])
    n2d = size(hash["filt_value"])[1]
    dz = nz/2-1
    J = size(hash["j_value"])[1]
    L = size(hash["theta_value"])[1]

    if hash["Omega"]
        tot_num_filt = n2d*K + 3
    else
        tot_num_filt = n2d*K + 1
    end

    im_scale = convert(Int8,log2(nz))
    # -------- number of bins in radial direction (size scales)
    K = (im_scale-3)*cz + 1
    normj = 1/sqrt(cz)

    # -------- allocate output array of zeros
    J_L_K     = zeros(Int32, tot_num_filt,3)
    psi_index = zeros(Int32, J+1, L, K)
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

    ## ------- This is phi0 containing the plane
    # phi contains power near k=0 not yet accounted for
    # for plane half-covered (pcz=1), add other half-plane

    if pcz == 1
        filter_power .+= circshift(filter_power[:,:,end:-1:1],(0,0,1))
    end

    center_power = fftshift(filter_power)
    phi_cen = zeros(nx, nx, nz)
    phi_cen[:,:,nz÷2-1:nz÷2+1] = center_power[:,:,nz÷2+3].*ones(1,1,3)
    phi_cen_shift = fftshift(phi_cen)
    ind = findall(phi_cen_shift .> 1E-13)
    val = phi_cen_shift[ind]

    # -------- before adding ϕ to filter bank, renormalize ψ if pc=1
    # if pcz==1 filt .*= sqrt(2.0) end  # double power for half coverage

    # -------- add result to filter array
    filtind[n2d*K + 1] = ind
    filtval[n2d*K + 1] = val
    J_L_K[f_ind,:] = [J,2,k_ind-1]
    psi_index[J+1,2+1,k_ind] = n2d*K + 1

    # -------- metadata dictionary
    info=Dict()
    info["npix"]         = hash["npix"]
    info["j_value"]      = hash["j_value"]
    info["theta_value"]  = hash["theta_value"]
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

    # -------- compute matrix that projects iso coeffs, add to hash
    S1_iso_mat = S1_iso_matrix3d(info)
    info["S1_iso_mat"] = S1_iso_mat
    S2_iso_mat = S2_iso_matrix3d(info)
    info["S2_iso_mat"] = S2_iso_mat
    #

    return info
end

filter_hash = fink_filter_hash(1, 8, nx=256, pc=2, Omega=true)
filt_3d = fink_filter_bank_3dizer_AKS(filter_hash, 1, nz=256)

nx = convert(Int64,filt_3d["npix"])
nz = 256
lup  = filt_3d["psi_index"]
index= filt_3d["filt_index"]
value= filt_3d["filt_value"]

filt = zeros(nx,nx,nz)
for x in 1:288
    filt[index[x]] .+= value[x].^2
end
for x in 289:294
    filt[index[x]] .+= value[x].^2
end
for x in 295:300
    filt[index[x]] .+= value[x].^2
end
heatmap(fftshift(dropdims(sum(filt,dims=3),dims=3)))
heatmap(fftshift(dropdims(sum(filt[:,:,2:3],dims=3),dims=3)))
heatmap(fftshift(filt[:,:,3]))

filt[:,:,3].-1.0

h5write("../DHC/scratch_AKS/data/filt_3d_fs.h5", "data6", fftshift(filt)+circshift(fftshift(filt)[:,:,end:-1:1],(0,0,1)))

filter_hash["Omega_index"]
key_list = filter_hash.keys
issubset(x, key_list)
x = "Omega_index"
convert(Array{String,1},key_list)


function fink_filter_bank_AKS(c, L; nx=256, wd=2, pc=1, shift=false, Omega=false, safety_on=true, wd_cutoff=1)
    #c     - sets the scale sampling rate (1 is dyadic, 2 is half dyadic)
    #L     - number of angular bins (usually 8*pc or 16*pc)
    #wd    - width of the wavelets (default 1, wd=2 for a double covering)
    #pc    - plane coverage (default 1, full 2pi 2)
    #shift - shift in θ by 1/2 of the θ spacing
    #Omega - true= append Omega filter (all power beyond Nyquist) so the sum of filters is 1.0

    # -------- assertion errors to make sure arguments are reasonable
    @test wd <= L/2

    # -------- set parameters
    dθ   = pc*π/L
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
    theta     = zeros(Float64, L)
    j_value   = zeros(Float64, J)

    # -------- compute the required wd
    j_rad_exp = zeros(J)
    for j_ind = 1:J
        j = (j_ind-1)/c
        jrad  = im_scale-j-2
        j_rad_exp[j_ind] = 2^(jrad)
    end

    wd_j = max.(ceil.(wd_cutoff.*L./(pc.*π.*j_rad_exp)),wd)

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
            F_angular = norm .* cos.((θ[angmask].-π).*(L/(2*wd*pc)))

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
    psi_ind_in[phi_index,:] = [J,0]

    # -------- metadata dictionary
    info=Dict{String,Any}()
    info["npix"]         = nx
    info["j_value"]      = j_value
    info["theta_value"]  = theta
    info["psi_index"]    = psi_index
    info["phi_index"]    = phi_index
    info["J_L"]          = psi_ind_in
    info["pc"]           = pc
    info["wd"]           = wd_j
    info["wd_cutoff"]    = wd_cutoff
    info["fs_center_r"]  = j_rad_exp
    info["Omega"]        = Omega

    if Omega     # append a filter containing the rest (outside Nyquist)
        filter_power += filt[:,:,phi_index].^2
        edge_power    = 1.0 .- filter_power
        zind          = findall(edge_power .< 1E-15)
        edge_power[zind]     .= 0.0  # set small numbers to zero
        Omega_index           = J*L+2
        info["Omega_index"]   = Omega_index
        filt[:,:,Omega_index] = sqrt.(edge_power)
        psi_ind_in[Omega_index,:] = [J,1]
    end


    return filt, info
end

function fink_filter_bank_3dizer_AKS(hash, cz; nz=256, pcz=1)
    @assert hash["pc"]==2 #can remove later when we add support for pcz=2
    @assert pcz==1

    # -------- set parameters
    nx = convert(Int64,hash["npix"])
    n2d = size(hash["filt_value"])[1]
    dz = nz/2-1
    J = size(hash["j_value"])[1]
    L = size(hash["theta_value"])[1]

    im_scale = convert(Int8,log2(nz))
    # -------- number of bins in radial direction (size scales)
    K = (im_scale-3)*cz + 1
    normj = 1/sqrt(cz)

    if hash["Omega"]
        tot_num_filt = n2d*K + 3
    else
        tot_num_filt = n2d*K + 1
    end

    # -------- allocate output array of zeros
    J_L_K     = zeros(Int32, tot_num_filt,3)
    psi_index = zeros(Int32, J+1, L, K)
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

    ## ------- This is phi0 containing the plane
    # phi contains power near k=0 not yet accounted for
    # for plane half-covered (pcz=1), add other half-plane

    if pcz == 1
        filter_power .+= circshift(filter_power[:,:,end:-1:1],(0,0,1))
    end

    center_power = fftshift(filter_power)
    phi_cen = zeros(nx, nx, nz)
    phi_cen[:,:,nz÷2-1:nz÷2+1] = center_power[:,:,nz÷2+3].*ones(1,1,3)
    phi_cen_shift = fftshift(phi_cen)
    ind = findall(phi_cen_shift .> 1E-13)
    val = phi_cen_shift[ind]

    # -------- before adding ϕ to filter bank, renormalize ψ if pc=1
    # if pcz==1 filt .*= sqrt(2.0) end  # double power for half coverage

    # -------- add result to filter array
    filtind[n2d*K + 1] = ind
    filtval[n2d*K + 1] = val
    J_L_K[n2d*K + 1,:] = [J+1,0,K]
    psi_index[J+1,2+1,K] = n2d*K + 1

    # -------- metadata dictionary
    info=Dict()
    info["npix"]         = hash["npix"]
    info["j_value"]      = hash["j_value"]
    info["theta_value"]  = hash["theta_value"]
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

    # -------- compute matrix that projects iso coeffs, add to hash
    S1_iso_mat = S1_iso_matrix3d(info)
    info["S1_iso_mat"] = S1_iso_mat
    S2_iso_mat = S2_iso_matrix3d(info)
    info["S2_iso_mat"] = S2_iso_mat
    #

    return info
end

filter_hash = fink_filter_hash_AKS(1, 8, nx=256, pc=2, Omega=true)
filt_3d = fink_filter_bank_3dizer_AKS(filter_hash, 1, nz=256)

nx = convert(Int64,filt_3d["npix"])
nz = 256
lup  = filt_3d["psi_index"]
index= filt_3d["filt_index"]
value= filt_3d["filt_value"]

filt = zeros(nx,nx,nz)
for x in 1:288
    filt[index[x]] .+= value[x].^2
end
for x in 289:294
    filt[index[x]] .+= value[x].^2
end
for x in 295:303
    filt[index[x]] .+= value[x].^2
end
heatmap(fftshift(dropdims(sum(filt,dims=3),dims=3)))
heatmap(fftshift(dropdims(sum(filt[:,:,2:3],dims=3),dims=3)))
heatmap(fftshift(filt[:,:,3]))

h5write("../DHC/scratch_AKS/data/filt_3d_fs.h5", "data8", fftshift(filt))

function fink_filter_bank_AKS(c, L; nx=256, wd=2, pc=1, shift=false, Omega=false, safety_on=true, wd_cutoff=1)
    #c     - sets the scale sampling rate (1 is dyadic, 2 is half dyadic)
    #L     - number of angular bins (usually 8*pc or 16*pc)
    #wd    - width of the wavelets (default 1, wd=2 for a double covering)
    #pc    - plane coverage (default 1, full 2pi 2)
    #shift - shift in θ by 1/2 of the θ spacing
    #Omega - true= append Omega filter (all power beyond Nyquist) so the sum of filters is 1.0

    # -------- assertion errors to make sure arguments are reasonable
    @test wd <= L/2

    # -------- set parameters
    dθ   = pc*π/L
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
    theta     = zeros(Float64, L)
    j_value   = zeros(Float64, J)

    # -------- compute the required wd
    j_rad_exp = zeros(J)
    for j_ind = 1:J
        j = (j_ind-1)/c
        jrad  = im_scale-j-2
        j_rad_exp[j_ind] = 2^(jrad)
    end

    wd_j = max.(ceil.(wd_cutoff.*L./(pc.*π.*j_rad_exp)),wd)

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
            F_angular = norm .* cos.((θ[angmask].-π).*(L/(2*wd*pc)))

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
    psi_ind_in[phi_index,:] = [J,0]

    # -------- metadata dictionary
    info=Dict{String,Any}()
    info["npix"]         = nx
    info["j_value"]      = j_value
    info["theta_value"]  = theta
    info["psi_index"]    = psi_index
    info["phi_index"]    = phi_index
    info["J_L"]          = psi_ind_in
    info["pc"]           = pc
    info["wd"]           = wd_j
    info["wd_cutoff"]    = wd_cutoff
    info["fs_center_r"]  = j_rad_exp
    info["Omega"]        = Omega

    if Omega     # append a filter containing the rest (outside Nyquist)
        filter_power += filt[:,:,phi_index].^2
        edge_power    = 1.0 .- filter_power
        zind          = findall(edge_power .< 1E-15)
        edge_power[zind]     .= 0.0  # set small numbers to zero
        Omega_index           = J*L+2
        info["Omega_index"]   = Omega_index
        filt[:,:,Omega_index] = sqrt.(edge_power)
        psi_ind_in[Omega_index,:] = [J,1]
    end


    return filt, info
end

function fink_filter_bank_3dizer_AKS(hash, cz; nz=256, pcz=1)
    @assert hash["pc"]==2 #can remove later when we add support for pcz=2
    @assert pcz==1

    # -------- set parameters
    nx = convert(Int64,hash["npix"])
    n2d = size(hash["filt_value"])[1]
    dz = nz/2-1
    J = size(hash["j_value"])[1]
    L = size(hash["theta_value"])[1]

    im_scale = convert(Int8,log2(nz))
    # -------- number of bins in radial direction (size scales)
    K = (im_scale-3)*cz + 1
    normj = 1/sqrt(cz)

    if hash["Omega"]
        tot_num_filt = n2d*K + 3
    else
        tot_num_filt = n2d*K + 1
    end

    # -------- allocate output array of zeros
    J_L_K     = zeros(Int32, tot_num_filt,3)
    psi_index = zeros(Int32, J+1, L, K)
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

    ## ------- This is phi0 containing the plane
    # phi contains power near k=0 not yet accounted for
    # for plane half-covered (pcz=1), add other half-plane

    if pcz == 1
        filter_power .+= circshift(filter_power[:,:,end:-1:1],(0,0,1))
    end

    center_power = fftshift(filter_power)
    phi_cen = zeros(nx, nx, nz)
    phi_cen[:,:,nz÷2:nz÷2+2] = center_power[:,:,nz÷2+3].*ones(1,1,3)
    phi_cen_shift = fftshift(phi_cen)
    ind = findall(phi_cen_shift .> 1E-13)
    val = phi_cen_shift[ind]

    # -------- before adding ϕ to filter bank, renormalize ψ if pc=1
    # if pcz==1 filt .*= sqrt(2.0) end  # double power for half coverage

    # -------- add result to filter array
    filtind[n2d*K + 1] = ind
    filtval[n2d*K + 1] = val
    J_L_K[n2d*K + 1,:] = [J+1,0,K]
    psi_index[J+1,2+1,K] = n2d*K + 1

    # -------- metadata dictionary
    info=Dict()
    info["npix"]         = hash["npix"]
    info["j_value"]      = hash["j_value"]
    info["theta_value"]  = hash["theta_value"]
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

    # -------- compute matrix that projects iso coeffs, add to hash
    S1_iso_mat = S1_iso_matrix3d(info)
    info["S1_iso_mat"] = S1_iso_mat
    S2_iso_mat = S2_iso_matrix3d(info)
    info["S2_iso_mat"] = S2_iso_mat
    #

    return info
end

filter_hash = fink_filter_hash_AKS(1, 8, nx=256, pc=2, Omega=true)
filt_3d = fink_filter_bank_3dizer_AKS(filter_hash, 1, nz=256)

nx = convert(Int64,filt_3d["npix"])
nz = 256
lup  = filt_3d["psi_index"]
index= filt_3d["filt_index"]
value= filt_3d["filt_value"]

filt = zeros(nx,nx,nz)
for x in 1:303
    filt[index[x]] .+= value[x].^2
end
heatmap(fftshift(dropdims(sum(filt,dims=3),dims=3)))
heatmap(fftshift(dropdims(sum(filt[:,:,2:3],dims=3),dims=3)))
heatmap(fftshift(filt[:,:,1]))

h5write("../DHC/scratch_AKS/data/filt_3d_fs.h5", "data9", fftshift(filt))

function fink_filter_bank_AKS(c, L; nx=256, wd=2, pc=1, shift=false, Omega=false, safety_on=true, wd_cutoff=1)
    #c     - sets the scale sampling rate (1 is dyadic, 2 is half dyadic)
    #L     - number of angular bins (usually 8*pc or 16*pc)
    #wd    - width of the wavelets (default 1, wd=2 for a double covering)
    #pc    - plane coverage (default 1, full 2pi 2)
    #shift - shift in θ by 1/2 of the θ spacing
    #Omega - true= append Omega filter (all power beyond Nyquist) so the sum of filters is 1.0

    # -------- assertion errors to make sure arguments are reasonable
    @test wd <= L/2

    # -------- set parameters
    dθ   = pc*π/L
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
    theta     = zeros(Float64, L)
    j_value   = zeros(Float64, J)

    # -------- compute the required wd
    j_rad_exp = zeros(J)
    for j_ind = 1:J
        j = (j_ind-1)/c
        jrad  = im_scale-j-2
        j_rad_exp[j_ind] = 2^(jrad)
    end

    wd_j = max.(ceil.(wd_cutoff.*L./(pc.*π.*j_rad_exp)),wd)

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
            F_angular = norm .* cos.((θ[angmask].-π).*(L/(2*wd*pc)))

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
    psi_ind_in[phi_index,:] = [J,0]

    # -------- metadata dictionary
    info=Dict{String,Any}()
    info["npix"]         = nx
    info["j_value"]      = j_value
    info["theta_value"]  = theta
    info["psi_index"]    = psi_index
    info["phi_index"]    = phi_index
    info["J_L"]          = psi_ind_in
    info["pc"]           = pc
    info["wd"]           = wd_j
    info["wd_cutoff"]    = wd_cutoff
    info["fs_center_r"]  = j_rad_exp
    info["Omega"]        = Omega

    if Omega     # append a filter containing the rest (outside Nyquist)
        filter_power += filt[:,:,phi_index].^2
        edge_power    = 1.0 .- filter_power
        zind          = findall(edge_power .< 1E-15)
        edge_power[zind]     .= 0.0  # set small numbers to zero
        Omega_index           = J*L+2
        info["Omega_index"]   = Omega_index
        filt[:,:,Omega_index] = sqrt.(edge_power)
        psi_ind_in[Omega_index,:] = [J,1]
    end


    return filt, info
end

function fink_filter_bank_3dizer_AKS(hash, cz; nz=256, pcz=1, Omega3d=false)
    @assert hash["pc"]==2 #can remove later when we add support for pcz=2
    @assert pcz==1

    # -------- set parameters
    nx = convert(Int64,hash["npix"])
    n2d = size(hash["filt_value"])[1]
    dz = nz/2-1
    J = size(hash["j_value"])[1]
    L = size(hash["theta_value"])[1]

    im_scale = convert(Int8,log2(nz))
    # -------- number of bins in radial direction (size scales)
    K = (im_scale-3)*cz + 1
    normj = 1/sqrt(cz)

    if hash["Omega"]
        tot_num_filt = n2d*K + 3
    else
        tot_num_filt = n2d*K + 1
    end

    # -------- allocate output array of zeros
    J_L_K     = zeros(Int32, tot_num_filt,3)
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
    # for plane half-covered (pcz=1), add other half-plane

    if pcz == 1
        filter_power .+= circshift(filter_power[:,:,end:-1:1],(0,0,1))
    end

    center_power = fftshift(filter_power)
    phi_cen = zeros(nx, nx, nz)
    phi_cen[:,:,nz÷2:nz÷2+2] = center_power[:,:,nz÷2+3].*ones(1,1,3)
    phi_cen_shift = fftshift(phi_cen)
    ind = findall(phi_cen_shift .> 1E-13)
    val = phi_cen_shift[ind]

    # -------- before adding ϕ to filter bank, renormalize ψ if pc=1
    # if pcz==1 filt .*= sqrt(2.0) end  # double power for half coverage

    # -------- add result to filter array
    filtind[n2d*K + 1] = ind
    filtval[n2d*K + 1] = val
    J_L_K[n2d*K + 1,:] = [J+1,0,K+1]
    psi_index[J+1,1,K+1] = n2d*K + 1

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
        filtval[n2d*K + 2] = val
        J_L_K[n2d*K + 2,:] = [J+1,1,K+1]
        psi_index[J+1,2,K+1] = n2d*K + 2

        # #### ------- This is omega3 around the edges
        #
        # filter_power = zeros(nx,nx,nz)
        # center_power = zeros(nx,nx,nz)
        # for index = 1:n2d*K + 2
        #     filter_power[filtind[index]] .+= filtval[index].^2
        # end
        #
        # if pcz == 1
        #     filter_power .+= circshift(filter_power[:,:,end:-1:1],(0,0,1))
        # end
        #
        # center_power = ones(nx,nx,nz) .- filter_power
        # ind = findall(center_power .> 1E-13)
        # val = center_power[ind]
        #
        # # -------- before adding ϕ to filter bank, renormalize ψ if pc=1
        # # if pcz==1 filt .*= sqrt(2.0) end  # double power for half coverage
        #
        # # -------- add result to filter array
        # filtind[n2d*K + 1] = ind
        # filtval[n2d*K + 1] = val
        # J_L_K[n2d*K + 1,:] = [J+1,2,K+1]
        # psi_index[J+1,3,K+1] = n2d*K + 3
    end

    # -------- metadata dictionary
    info=Dict()
    info["npix"]         = hash["npix"]
    info["j_value"]      = hash["j_value"]
    info["theta_value"]  = hash["theta_value"]
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

    # -------- compute matrix that projects iso coeffs, add to hash
    S1_iso_mat = S1_iso_matrix3d(info)
    info["S1_iso_mat"] = S1_iso_mat
    S2_iso_mat = S2_iso_matrix3d(info)
    info["S2_iso_mat"] = S2_iso_mat
    #

    return info
end

filter_hash = fink_filter_hash_AKS(1, 8, nx=256, pc=2, Omega=true)
filt_3d = fink_filter_bank_3dizer_AKS(filter_hash, 1, nz=256, Omega3d=false)

nx = convert(Int64,filt_3d["npix"])
nz = 256
lup  = filt_3d["psi_index"]
index= filt_3d["filt_index"]
value= filt_3d["filt_value"]

filt = zeros(nx,nx,nz)
for x in 1:303
    filt[index[x]] .+= value[x].^2
end
heatmap(fftshift(dropdims(sum(filt,dims=3),dims=3)))
heatmap(fftshift(dropdims(sum(filt[:,:,2:3],dims=3),dims=3)))
heatmap(fftshift(filt[:,:,5]))

h5write("../DHC/scratch_AKS/data/filt_3d_fs.h5", "data11", fftshift(filt))

function fink_filter_bank_3dizer_AKS(hash, cz; nz=256, pcz=1, Omega3d=false)
    @assert hash["pc"]==2 #can remove later when we add support for pcz=2
    @assert pcz==1

    # -------- set parameters
    nx = convert(Int64,hash["npix"])
    n2d = size(hash["filt_value"])[1]
    dz = nz/2-1
    J = size(hash["j_value"])[1]
    L = size(hash["theta_value"])[1]

    im_scale = convert(Int8,log2(nz))
    # -------- number of bins in radial direction (size scales)
    K = (im_scale-3)*cz + 1
    normj = 1/sqrt(cz)

    if haskey(hash, "Omega_index")
        tot_num_filt = n2d*K + 3
    else
        tot_num_filt = n2d*K + 1
    end

    # -------- allocate output array of zeros
    J_L_K     = zeros(Int32, tot_num_filt,3)
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
    # for plane half-covered (pcz=1), add other half-plane

    if pcz == 1
        filter_power .+= circshift(filter_power[:,:,end:-1:1],(0,0,1))
    end

    center_power = fftshift(filter_power)
    phi_cen = zeros(nx, nx, nz)
    phi_cen[:,:,nz÷2:nz÷2+2] = center_power[:,:,nz÷2+3].*ones(1,1,3)
    phi_cen_shift = fftshift(phi_cen)
    ind = findall(phi_cen_shift .> 1E-13)
    val = phi_cen_shift[ind]

    # -------- before adding ϕ to filter bank, renormalize ψ if pc=1
    if pcz==1 filtval[1:n2d*K] .*= sqrt(2.0) end  # double power for half coverage

    # -------- add result to filter array
    filtind[n2d*K + 1] = ind
    filtval[n2d*K + 1] = sqrt.(val)
    J_L_K[n2d*K + 1,:] = [J+1,0,K+1]
    psi_index[J+1,1,K+1] = n2d*K + 1

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

        #### ------- This is omega3 around the edges

        filter_power = zeros(nx,nx,nz)
        center_power = zeros(nx,nx,nz)
        for index = 1:n2d*K + 2
            filter_power[filtind[index]] .+= filtval[index].^2
        end

        if pcz == 1
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
    end

    # -------- metadata dictionary
    info=Dict()
    info["npix"]         = hash["npix"]
    info["j_value"]      = hash["j_value"]
    info["theta_value"]  = hash["theta_value"]
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
    info["pcz"]             = pcz

    # -------- compute matrix that projects iso coeffs, add to hash
    S1_iso_mat = S1_iso_matrix3d(info)
    info["S1_iso_mat"] = S1_iso_mat
    S2_iso_mat = S2_iso_matrix3d(info)
    info["S2_iso_mat"] = S2_iso_mat
    #

    return info
end

filter_hash = fink_filter_hash_AKS(1, 8, nx=256, pc=2, Omega=true)
filt_3d = fink_filter_bank_3dizer_AKS(filter_hash, 1, nz=256, Omega3d=true)

nx = convert(Int64,filt_3d["npix"])
nz = 256
lup  = filt_3d["psi_index"]
index= filt_3d["filt_index"]
value= filt_3d["filt_value"]

filt = zeros(nx,nx,nz)
for x in 1:303
    filt[index[x]] .+= value[x].^2
end
heatmap(fftshift(dropdims(sum(filt,dims=3),dims=3)))
heatmap(fftshift(dropdims(sum(filt[:,:,2:3],dims=3),dims=3)))
heatmap(fftshift(filt[:,:,1]))

h5write("../DHC/scratch_AKS/data/filt_3d_fs.h5", "data13", fftshift(filt))

function fink_filter_bank_AKS(c, L; nx=256, wd=2, pc=1, shift=false, Omega=false, safety_on=true, wd_cutoff=1)
    #c     - sets the scale sampling rate (1 is dyadic, 2 is half dyadic)
    #L     - number of angular bins (usually 8*pc or 16*pc)
    #wd    - width of the wavelets (default 1, wd=2 for a double covering)
    #pc    - plane coverage (default 1, full 2pi 2)
    #shift - shift in θ by 1/2 of the θ spacing
    #Omega - true= append Omega filter (all power beyond Nyquist) so the sum of filters is 1.0

    # -------- assertion errors to make sure arguments are reasonable
    @test wd <= L/2

    # -------- set parameters
    dθ   = pc*π/L
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
    theta     = zeros(Float64, L)
    j_value   = zeros(Float64, J)

    # -------- compute the required wd
    j_rad_exp = zeros(J)
    for j_ind = 1:J
        j = (j_ind-1)/c
        jrad  = im_scale-j-2
        j_rad_exp[j_ind] = 2^(jrad)
    end

    wd_j = max.(ceil.(wd_cutoff.*L./(pc.*π.*j_rad_exp)),wd)

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
            F_angular = norm .* cos.((θ[angmask].-π).*(L/(2*wd*pc)))

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
    psi_ind_in[phi_index,:] = [J,0]

    # -------- metadata dictionary
    info=Dict{String,Any}()
    info["npix"]         = nx
    info["j_value"]      = j_value
    info["theta_value"]  = theta
    info["psi_index"]    = psi_index
    info["phi_index"]    = phi_index
    info["J_L"]          = psi_ind_in
    info["pc"]           = pc
    info["wd"]           = wd_j
    info["wd_cutoff"]    = wd_cutoff
    info["fs_center_r"]  = j_rad_exp

    if Omega     # append a filter containing the rest (outside Nyquist)
        filter_power += filt[:,:,phi_index].^2
        edge_power    = 1.0 .- filter_power
        zind          = findall(edge_power .< 1E-15)
        edge_power[zind]     .= 0.0  # set small numbers to zero
        Omega_index           = J*L+2
        info["Omega_index"]   = Omega_index
        filt[:,:,Omega_index] = sqrt.(edge_power)
        psi_ind_in[Omega_index,:] = [J,1]
    end

    return filt, info
end

filter_hash = fink_filter_hash_AKS(1, 8, nx=256, pc=1, Omega=true)

nx = convert(Int64,filter_hash["npix"])
index= filter_hash["filt_index"]
value= filter_hash["filt_value"]

filt = zeros(nx,nx)
for x in 1:50
    filt[index[x]] .+= value[x].^2
end
heatmap(fftshift(filt))
heatmap(fftshift(dropdims(sum(filt[:,:,2:3],dims=3),dims=3)))
heatmap(fftshift(filt[:,:,5]))

##

using Plots
using Measures

function S1_iso_matrix3d_AKS(fhash)
    # fhash is the filter hash output by fink_filter_hash
    # The output matrix converts an S1 coeff vector to S1iso by
    #   summing over l
    # Matrix is stored in sparse CSC format using SparseArrays.
    # DPF 2021-Feb-18
    # AKS 2021-Feb-22

    # Does hash contain Omega filter?
    Omega   = haskey(fhash, "Omega_index")

    # unpack fhash
    Nl      = length(fhash["theta_value"])
    Nj      = length(fhash["j_value"])
    Nk      = length(fhash["k_value"])
    Nf      = length(fhash["filt_value"])
    ψ_ind   = fhash["psi_index"]

    # number of iso coefficients
    Niso    = Omega ? (Nj+2)*Nk+3 : (Nj+1)*Nk+1
    Nstep   = Omega ? Nj+1 : Nj
    Mat     = zeros(Int32, Niso, Nf)

    # first J elements of iso
    for j = 1:Nj
        for l = 1:Nl
            for k=1:Nk
                λ = ψ_ind[j,l,k]
                Mat[j+(k-1)*Nstep, λ] = 1
            end
        end
    end

    #this is currently how I am storing the phi_k, omega_k
    j = Nj+1
    l = 1
    for k=1:Nk
        λ = ψ_ind[j,l,k]
        Mat[j+(k-1)*Nstep, λ] = 1
    end

    if (haskey(fhash, "Omega_index") & fhash["Omega_3d"])
    j = Nj+1
    l = 2
    for k=1:Nk
        λ = ψ_ind[j,l,k]
        Mat[j+1+(k-1)*Nstep, λ] = 1
    end

    ### these are the 3d globals

    #phi0
    λ = ψ_ind[J+1,1,K+1]
    Mat[Niso-2, λ] = 1

    #omega_0
    λ = ψ_ind[J+1,2,K+1]
    Mat[Niso-1, λ] = 1

    # omega_3d
    λ = ψ_ind[J+1,3,K+1]
    Mat[Niso, λ] = 1

    return sparse(Mat)
end

function S2_iso_matrix3d(fhash)
    # fhash is the filter hash output by fink_filter_hash
    # The output matrix converts an S2 coeff vector to S2iso by
    #   summing over l1,l2 and fixed Δl.
    # Matrix is stored in sparse CSC format using SparseArrays.
    # DPF 2021-Feb-18
    # AKS 2021-Feb-22

    # Does hash contain Omega filter?
    #Omega   = haskey(fhash, "Omega_index")
    #if Omega Ω_ind = fhash["Omega_index"] end

    # unpack fhash
    Nl      = length(fhash["theta_value"])
    Nj      = length(fhash["j_value"])
    Nk      = length(fhash["k_value"])
    Nf      = length(fhash["filt_value"])
    ψ_ind   = fhash["psi_index"]

    # number of iso coefficients
    #Niso    = Omega ? Nj*Nj*Nl+4*Nj+4 : Nj*Nj*Nl+2*Nj+1
    Niso    = Nj*Nj*Nk*Nk*Nl+2*Nj*Nk*Nk+Nk*Nk
    Mat     = zeros(Int32, Niso, Nf*Nf)

    #should probably think about reformatting to (Nj+1) so phi
    #phi cross terms are in the right place

    # first J*J*K*K*L elements of iso
    for j1 = 1:Nj
        for j2 = 1:Nj
            for l1 = 1:Nl
                for l2 = 1:Nl
                    for k1 = 1:Nk
                        for k2 = 1:Nk
                            DeltaL = mod(l1-l2, Nl)
                            λ1     = ψ_ind[j1,l1,k1]
                            λ2     = ψ_ind[j2,l2,k2]

                            Iiso   = k1+Nk*((k2-1)+Nk*((j1-1)+Nj*((j2-1)+Nj*DeltaL)))
                            Icoeff = λ1+Nf*(λ2-1)
                            Mat[Iiso, Icoeff] = 1
                        end
                    end
                end
            end
        end
    end

    # Next J elements are λϕ, then J elements ϕλ
    
    for j = 1:Nj
        for l = 1:Nl
            for k1 = 1:Nk
                for k2 =1:Nk
                    λ      = ψ_ind[j,l,k1]
                    Iiso   = Nj*Nj*Nk*Nk*Nl+k1+Nk*((k2-1)+Nk*(j-1))
                    Icoeff = λ+Nf*(ψ_ind[Nj+1,1,k2]-1)  # λϕ
                    Mat[Iiso, Icoeff] = 1

                    Iiso   = Nj*Nj*Nk*Nk*Nl+Nj*Nk*Nk+k1+Nk*((k2-1)+Nk*(j-1))
                    Icoeff = (ψ_ind[Nj+1,1,k2]-1)+Nf*(λ-1)  # ϕλ
                    Mat[Iiso, Icoeff] = 1
                end
            end
        end
    end

    # Next Nk*Nk elements are ϕ(k)ϕ(k)
    j = Nj+1
    l = 1
    for k1=1:Nk
        for k2=1:Nk
            λ1 = ψ_ind[j,l,k1]
            λ2 = ψ_ind[j,l,k2]

            Iiso = Nj*Nj*Nk*Nk*Nl+2*Nj*Nk*Nk+k1+Nk*(k2-1)
            Icoeff = λ1+Nf*(λ2-1)
            Mat[Iiso, Icoeff] = 1
        end
    end

    # # If the Omega filter exists, add more terms
    # if Omega
    #     # Next J elements are λΩ, then J elements Ωλ
    #     for j = 1:Nj
    #         for l = 1:Nl
    #             λ      = ψ_ind[j,l]
    #             Iiso   = I0+j
    #             Icoeff = λ+Nf*(Ω_ind-1)  # λΩ
    #             Mat[Iiso, Icoeff] = 1
    #
    #             Iiso   = I0+Nj+j
    #             Icoeff = Ω_ind+Nf*(λ-1)  # Ωλ
    #             Mat[Iiso, Icoeff] = 1
    #         end
    #     end
    #     # Next 3 elements are ϕΩ, Ωϕ, ΩΩ
    #     Iiso   = I0+Nj+Nj
    #     Mat[Iiso+1, ϕ_ind+Nf*(Ω_ind-1)] = 1
    #     Mat[Iiso+2, Ω_ind+Nf*(ϕ_ind-1)] = 1
    #     Mat[Iiso+3, Ω_ind+Nf*(Ω_ind-1)] = 1
    # end
    #
    return sparse(Mat)
end

filter_hash = fink_filter_hash(1, 8, nx=256, pc=2)
filt_3d = fink_filter_bank_3dizer_AKS(filter_hash, 1, nz=256)

function plot_filter_bank_QAxy_AKS(hash; fname="filter_bank_QAxy.png")

    # -------- define plot1 to append plot to a list
    function plot1(ps, image; clim=nothing, bin=1.0, fsz=12, label=nothing,color=:white)
        # ps    - list of plots to append to
        # image - image to heatmap
        # clim  - tuple of color limits (min, max)
        # bin   - bin down by this factor, centered on nx/2+1
        # fsz   - font size
        marg   = -1mm
        nx, ny = size(image)
        nxb    = nx/round(Integer, 2*bin)

        # -------- center on nx/2+1
        i0 = max(1,round(Integer, (nx/2+2)-nxb-1))
        i1 = min(nx,round(Integer, (nx/2)+nxb+1))
        lims = [i0,i1]
        subim = image[i0:i1,i0:i1]
        push!(ps, heatmap(image,aspect_ratio=:equal,clim=clim,
            xlims=lims, ylims=lims, size=(400,400),
            legend=false,xtickfontsize=fsz,ytickfontsize=fsz,#tick_direction=:out,
            rightmargin=marg,leftmargin=marg,topmargin=marg,bottommargin=marg))
        if label != nothing
            annotate!(lims'*[.96,.04],lims'*[.09,.91],text(label,:left,color,16))
        end
        return
    end

    # -------- initialize array of plots
    ps   = []
    jval = hash["j_value"]
    kval = hash["k_value"]
    J    = length(hash["j_value"])
    L    = length(hash["theta_value"])
    K    = length(hash["k_value"])
    nx   = hash["npix"]
    nz   = hash["nz"]
    lup  = hash["psi_index"]
    index= hash["filt_index"]
    value= hash["filt_value"]
    pcz  = hash["pcz"]
    n2d = size(hash["2d_filt_value"])[1]
    # plot the first 3 and last 3 j values
    jplt = [1:3;J-2:J]
    lplt = 1:3
    kplt = [1,2,K]

    # -------- renormalize ψ if pcz=1
    if pcz==1 value[1:n2d*K] ./= sqrt(2.0) end  # double power for half coverage

    # -------- loop over k
    for k_ind in kplt
        # -------- loop over j
        k = kval[k_ind]
        krs = convert(Int64,(max(1,2.0^(7-k))+nz/2))
        kfs = convert(Int64,(max(1,2.0^(7-k))+1))
        for j_ind in jplt
            # first 3 filters in Fourier space
            j = jval[j_ind]
            bfac = max(1,2.0^(j-1))
            # -------- loop over l
            for l_ind in lplt
                label=string("j=",j_ind," ℓ=",l_ind-1," k=",k_ind)
                filt = zeros(nx,nx,nz)
                ind = lup[j_ind,l_ind,k_ind]
                filt[index[ind]] = value[ind]
                #temp = dropdims(sum(filt,dims=3),dims=3)
                temp = filt[:,:,kfs]
                plot1(ps, fftshift(temp[:,:]),
                      bin=bfac, label=label)
            end
            fac = max(1,2.0^(5-j))
            # real part of config space of 3rd filter
            l_ind = 3
            filt = zeros(nx,nx,nz)
            ind = lup[j_ind,l_ind,k_ind]
            filt[index[ind]] = value[ind]
            rs = fftshift(real(ifft(filt)))
            rs_slice = rs[:,:,krs]
            #rs_sum = dropdims(sum(rs,dims=3),dims=3)
            plot1(ps, rs_slice,
                  bin=fac, label=string("j=",j_ind," ℓ=",l_ind-1," k=",k_ind))

            # sum all filters to form (partial) ring
            ring = zeros(nx,nx)
            filt = zeros(nx,nx,nz)
            for indx in lup[j_ind,:,k_ind]
                filt[index[indx]] .+= value[indx].^2
            end
            ring = filt[:,:,kfs]
            plot1(ps, fftshift(ring[:,:]),
                  bin=bfac, label=string("j=",j_ind," ℓ=",0,":",L-1," k=",k_ind))
        end
    end

    filt = zeros(nx,nx,nz)
    for indx in lup[1:J,:,2]
        filt[index[indx]] .+= value[indx].^2
    end

    wavepow = fftshift(dropdims(sum(filt, dims=3), dims=3))
    plot1(ps, wavepow, label=string("j=1:",J," ℓ=0:",L-1," K=1:",K),color=:white)

    if hash["2d_pc"]==1
        wavepow ./= 2
        wavepow += circshift(wavepow[end:-1:1,end:-1:1],(1,1))
    end
    plot1(ps, wavepow, #clim=(0,K),
        bin=32, label="missing power",color=:black)

    phi_z = zeros(nx,nx,nz)
    for indx in lup[J+1,1,2]
        phi_z[index[indx]] .+= value[indx].^2
    end

    phi_rs = zeros(nx,nx,nz)
    for indx in lup[J+1,1,2]
        phi_rs[index[indx]] .+= value[indx]
    end
    phi_rst = fftshift(real(ifft(phi_rs)))[:,:,krs]

    k = kval[K]
    krs = convert(Int64,(max(1,2.0^(7-k))+nz/2))
    kfs = convert(Int64,(max(1,2.0^(7-k))+1))

    phi_shift = fftshift(dropdims(sum(phi_z, dims=3), dims=3))
    plot1(ps, phi_shift, #clim=(0,K),
        bin=32, label="ϕ")
    disc = wavepow.+phi_shift
    plot1(ps, phi_rst, label=string("ϕ k=",convert(Int64,k)))
    plot1(ps, disc, #clim=(0,K),
        bin=32, label="all", color=:black)

    myplot = plot(ps..., layout=(19,5), size=(3000,8000))
    savefig(myplot, fname)
    return phi_z
end

function plot_filter_bank_QAxz_AKS(hash; fname="filter_bank_QAxz.png")

    # -------- define plot1 to append plot to a list
    function plot1(ps, image; clim=nothing, binx=1.0, biny=1.0, fsz=12, label=nothing,color=:white)
        # ps    - list of plots to append to
        # image - image to heatmap
        # clim  - tuple of color limits (min, max)
        # bin   - bin down by this factor, centered on nx/2+1
        # fsz   - font size
        marg   = -1mm
        nx, ny = size(image)
        nxb    = nx/round(Integer, 2*binx)
        nyb    = ny/round(Integer, 2*biny)

        # -------- center on nx/2+1
        xi0 = max(1,round(Integer, (nx/2+2)-nxb-1))
        xi1 = min(nx,round(Integer, (nx/2)+nxb+1))
        xlims = [xi0,xi1]

        yi0 = max(1,round(Integer, (ny/2+2)-nyb-1))
        yi1 = min(ny,round(Integer, (ny/2)+nyb+1))
        ylims = [yi0,yi1]

        subim = image[xi0:xi1,yi0:yi1]
        push!(ps, heatmap(image,aspect_ratio=biny/binx,clim=clim,
            xlims=xlims, ylims=ylims, size=(400,400),
            legend=false,xtickfontsize=fsz,ytickfontsize=fsz,#tick_direction=:out,
            rightmargin=marg,leftmargin=marg,topmargin=marg,bottommargin=marg))
        if label != nothing
            annotate!(xlims'*[.96,.04],ylims'*[.09,.91],text(label,:left,color,16))
        end
        return
    end

    # -------- initialize array of plots
    ps   = []
    jval = hash["j_value"]
    kval = hash["k_value"]
    J    = length(hash["j_value"])
    L    = length(hash["theta_value"])
    K    = length(hash["k_value"])
    nx   = hash["npix"]
    nz   = hash["nz"]
    lup  = hash["psi_index"]
    index= hash["filt_index"]
    value= hash["filt_value"]
    pcz  = hash["pcz"]
    n2d = size(hash["2d_filt_value"])[1]
    # plot the first 3 and last 3 j values
    jplt = [1:3;J-2:J]
    lplt = [1,2,L]
    kplt = [1,2,K]

    # -------- renormalize ψ if pcz=1
    if pcz==1 value[1:n2d*K] ./= sqrt(2.0) end  # double power for half coverage

    # -------- loop over k
    for k_ind in kplt
        # -------- loop over j
        k = kval[k_ind]
        krs = convert(Int64,(max(1,2.0^(7-k))+nz/2))
        kfs = convert(Int64,(max(1,2.0^(7-k))+1))
        kfac = max(1,2.0^(k-1))
        for j_ind in jplt
            # first 3 filters in Fourier space
            j = jval[j_ind]
            jrs = convert(Int64,(max(1,2.0^(7-j))+nx/2))
            jfs = convert(Int64,(max(1,2.0^(7-j))+1))

            bfac = max(1,2.0^(j-1))
            # -------- loop over l
            for l_ind in lplt
                label=string("j=",j_ind," ℓ=",l_ind-1," k=",k_ind)
                filt = zeros(nx,nx,nz)
                ind = lup[j_ind,l_ind,k_ind]
                filt[index[ind]] = value[ind]
                #temp = dropdims(sum(filt,dims=2),dims=2)
                #temp = filt[:,nx÷2,:]
                plot1(ps, fftshift(filt)[nx÷2,:,:], #clim=(0,1),
                      binx=kfac, biny=bfac, label=label)
            end
            j_fac = max(1,2.0^(5-j))
            k_fac = max(1,2.0^(5-k))
            # real part of config space of 1st filter
            l_ind = 1

            filt = zeros(nx,nx,nz)
            ind = lup[j_ind,l_ind,k_ind]
            filt[index[ind]] = value[ind]
            rs = fftshift(real(ifft(filt)))
            rs_slice = rs[nx÷2,:,:]
            plot1(ps, rs_slice,
                  binx=k_fac, biny=j_fac, label=string("j=",j_ind," ℓ=",l_ind-1," k=",k_ind))

            filt = zeros(nx,nx,nz)
            ind = lup[j_ind,l_ind,k_ind]
            filt[index[ind]] = value[ind]
            rs = fftshift(real(ifft(filt)))
            rs_slice = rs[:,nx÷2,:]
            plot1(ps, rs_slice,
                binx=k_fac, biny=j_fac, label=string("j=",j_ind," ℓ=",l_ind-1," k=",k_ind))

            # sum all filters to form (partial) ring
            ring = zeros(nx,nx)
            filt = zeros(nx,nx,nz)
            for indx in lup[j_ind,:,k_ind]
                filt[index[indx]] .+= value[indx].^2
            end
            ring = filt
            plot1(ps, fftshift(ring)[nx÷2,:,:], #clim=(0,1),
                  binx=kfac, biny=bfac, label=string("j=",j_ind," ℓ=",0,":",L-1," k=",k_ind))
        end
    end

    filt = zeros(nx,nx,nz)
    for indx in lup[1:J,:,1:K]
        filt[index[indx]] .+= value[indx].^2
    end

    wavepow = fftshift(filt)
    plot1(ps, wavepow[:,nx÷2,:], #clim=(-0.1,1),
    label=string("j=1:",J," ℓ=0:",L-1," K=1:",K))

    wavepow2 = wavepow[:,nx÷2,:]
    if hash["2d_pc"]==1
        wavepow2 += reverse(wavepow[:,nx÷2,:], dims = 2)
        wavepow2 += circshift(wavepow2[end:-1:1,end:-1:1],(1,1))
    end
    plot1(ps, wavepow2, #clim=(0,K),
        binx=16, label="missing power",color=:black)

    phi_z = zeros(nx,nx,nz)
    for indx in lup[J+1,1,:]
        phi_z[index[indx]] .+= value[indx].^2
    end

    phi_rs = zeros(nx,nx,nz)
    for indx in lup[J+1,1,:]
        phi_rs[index[indx]] .+= value[indx]
    end
    phi_rst = fftshift(real(ifft(phi_rs)))[:,:,krs]

    k = kval[K]
    krs = convert(Int64,(max(1,2.0^(7-k))+nz/2))
    kfs = convert(Int64,(max(1,2.0^(7-k))+1))

    phi_shift = fftshift(dropdims(sum(phi_z, dims=1), dims=1))
    plot1(ps, phi_shift, #clim=(0,K),
        binx=32, label="ϕ")
    disc = wavepow[:,nx÷2,:].+phi_shift
    plot1(ps, phi_rst, label=string("ϕ k=",convert(Int64,k)))
    plot1(ps, disc, #clim=(0,K),
        binx=32, label="all", color=:black)

    plot1(ps, disc, #clim=(0,K),
        binx=32, label="all", color=:black)

    myplot = plot(ps..., layout=(19,6), size=(3000,8000))
    savefig(myplot, fname)

    filt = zeros(nx,nx,nz)
    ind = lup[1,1,1]
    filt[index[ind]] = value[ind]

    return wavepow
end

plot_filter_bank_QAxy_AKS(filt_3d, fname="/Users/saydjari/Dropbox/GradSchool_AKS/Doug/Projects/DHC/scratch_AKS/images/filter_bank_QAxy.png")
plot_filter_bank_QAxz_AKS(filt_3d, fname="/Users/saydjari/Dropbox/GradSchool_AKS/Doug/Projects/DHC/scratch_AKS/images/filter_bank_QAxz.png")
