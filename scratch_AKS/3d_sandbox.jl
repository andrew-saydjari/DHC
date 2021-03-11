push!(LOAD_PATH, "/Users/saydjari/Dropbox/GradSchool_AKS/Doug/Projects/DHC/main")
using DHC_2DUtils
using Profile
using BenchmarkTools
using FFTW
using HDF5
using Test
using SparseArrays

filter_hash = fink_filter_hash(1, 8, nx=256, pc=1, wd=1)
out = DHC_compute(rand(256,256),filter_hash,doS2=true, doS12=false, doS20=false)
out = DHC_compute(rand(256,256),filter_hash,doS2=true, doS12=true, doS20=false)
out = DHC_compute(rand(256,256),filter_hash,doS2=true, doS12=true, doS20=true)
out = DHC_compute(rand(256,256),filter_hash,doS2=true, doS12=true, doS20=true)
out = DHC_compute(rand(256,256),filter_hash,doS2=true, doS12=true, doS20=true)
out = DHC_compute(rand(256,256),filter_hash,doS2=true, doS12=true, doS20=true)

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
    info["2d_filt_index"]   = hash["filt_index"]
    info["2d_filt_value"]   = hash["filt_value"]


    info["nz"]              = nz
    info["cz"]              = cz
    info["k_value"]         = k_value
    info["filt_index"]      = filtind
    info["filt_value"]      = filtval

    return info
end

hash = fink_filter_hash(1, 4, nx=128, pc=1, wd=1)

@time filt_3d = fink_filter_bank_3dizer(hash, 1, nz=128)

filter_temp = zeros(256,256,256)
filter_temp[filt_3d["filt_index"][12]] = filt_3d["filt_index"][12]

h5write("../DHC/scratch_AKS/data/filt_3d_rs.h5", "main/data", fft2())

Profile.clear()
@profile filt_3d = fink_filter_bank_3dizer(hash, 1, nz=128)

Juno.profiler()

@benchmark fink_filter_bank_3dizer(hash, 1, nz=128)

hash = fink_filter_hash(1, 8, nx=128, pc=1, wd=1)
@benchmark filt_3d = fink_filter_bank_3dizer(hash, 1, nz=128)


hash = fink_filter_hash(1, 8, nx=256, pc=1, wd=1)
@benchmark filt_3d = fink_filter_bank_3dizer(hash, 1, nz=256)


hash = fink_filter_hash(1, 8, nx=256, pc=1, wd=2)
@benchmark filt_3d = fink_filter_bank_3dizer(hash, 1, nz=256)

## Here we go with the plotting

function plot_filter_bank_QA(hash; fname="filter_bank_QA.png")

    # -------- define plot1 to append plot to a list
    function plot1(ps, image; clim=nothing, bin=1.0, fsz=12, label=nothing)
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
            annotate!(lims'*[.96,.04],lims'*[.09,.91],text(label,:left,:white,16))
        end
        return
    end

    # -------- initialize array of plots
    ps   = []
    jval = hash["2d_j_value"]
    J    = length(jval)
    L    = length(hash["2d_theta_value"])
    nx   = hash["2d_npix"]
    nz   = hash["nz"]
    # plot the first 3 and last 3 j values
    jplt = [1:3;J-2:J]

    ind = zeros(Int64,J,L)
    for j_ind=1:J
        for l=1:L
            ind[j_ind,l] = Int(j_ind-1 + (l-1)*J)
        end
    end
    ind .+= 1

    # -------- loop over j
    for j_ind in jplt
        # first 3 filters in Fourier space
        j = jval[j_ind]
        bfac = max(1,2.0^(j-1))
        for l = 1:3
            label=string("j=",j_ind," ℓ=",l-1)
            #label=latexstring("\\tilde{\\psi}_{$j_ind,$(l-1)}")
            filt = zeros(nx,nx,nz)
            println(nx)
            println(nz)
            filt[hash["2d_filt_index"][2],hash["filt_index"][2]] = hash["filt_value"][2]
            temp = dropdims(sum(filt,dims=3))
            plot1(ps, fftshift(filt[:,:]), clim=(0,1),
                  bin=bfac, label=label)
        end
        fac = max(1,2.0^(5-j))
        # real part of config space of 3rd filter
        plot1(ps, fftshift(real(ifft(filt[:,:,ind[j_ind,3]]))),
              bin=fac, label=string("j=",j_ind," ℓ=",3-1))

        # sum all filters to form (partial) ring
        ring = dropdims(sum(filt[:,:,ind[j_ind,:]].^2, dims=3), dims=3)
        plot1(ps, fftshift(ring[:,:,1]), clim=(0,1),
              bin=bfac, label=string("j=",j_ind," ℓ=",0,":",L-1))
    end

    # wavepow = fftshift(dropdims(sum(filt[:,:,ind].^2, dims=(3,4)), dims=(3,4)))
    # plot1(ps, wavepow, clim=(-0.1,1), label=string("j=1:",J," ℓ=0:",L-1))
    #
    # if info["pc"]==1
    #     wavepow += circshift(wavepow[end:-1:1,end:-1:1],(1,1))
    # end
    # plot1(ps, wavepow, clim=(-0.1,1), bin=32, label="missing power")
    # phi = filt[:,:,info["phi_index"]]
    # phi_shift = fftshift(phi)
    # plot1(ps, phi_shift.^2, clim=(-0.1,1), bin=32, label="ϕ")
    # disc = wavepow+(phi_shift.^2)
    # plot1(ps, fftshift(real(ifft(phi))), label="ϕ")
    # plot1(ps, disc, clim=(-0.1,1), bin=32, label="all")

    myplot = plot(ps..., layout=(7,5), size=(1400,2000))
    savefig(myplot, fname)
end

plot_filter_bank_QA(filt_3d, fname="/Users/saydjari/Dropbox/GradSchool_AKS/Doug/Projects/DHC/scratch_AKS/filt-3d.png")

## Plotting is not working because keeping track of indices is a mess. Lets
## try sparse matrices

function fink_filter_bank(c, L; nx=256, wd=1, pc=1, shift=false, Omega=false, safety_on=true)
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

@benchmark hash = fink_filter_bank(1, 8, nx=256, pc=1, wd=2)

function fink_filter_bank_AKS(c, L; nx=256, wd=1, pc=1, shift=false, Omega=false, safety_on=true)
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
    J = (im_scale-2)*c

    # -------- allocate output array of zeros
    filt      = spzeros(nx,nx,J*L+(Omega ? 2 : 1))
    filt_pow  = spzeros(nx,nx)
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
        θ    = spzeros(nx,nx)
        logr = spzeros(nx,nx)
        F_angular = spzeros(nx,nx)

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
        # -------- compute the wavelet in the Fourier domain
        # -------- the angular factor is the same for all j
            f = x -> norm*cos((x-π)*(L/(2*wd*pc)))
            map!(f,θ.nzval,θ.nzval) #inplace so theta becomes F_ang

        # -------- loop over j for the radial part
        #    for (j_ind, j) in enumerate(1/c:1/c:im_scale-2)
            j_ind_w_wd = findall(wd_j.==wd)
            for j_ind = j_ind_w_wd
                Δj    = deepcopy(logr)
                j = j_ind/c
                j_value[j_ind] = j  # store for later
                jrad  = im_scale-j-1
                g = x -> abs(x-jrad)
                map!(g,Δj.nzval,logr.nzval)
        # -------- radial part
                h = x -> norm*cos(x.*(c*π/2))
                map!(h,Δj.nzval,Δj.nzval) #inplace so Δj becomes F_rad
        #      Let's have these be (J,L) if you reshape...
                f_ind    = j_ind + l*J
                filt[:,:,f_ind] = Δj .* θ
            end
        end
    end
    return filt
    # -------- phi contains power near k=0 not yet accounted for
    filt_pow = (sum(filt.^2, dims=3))
    println(filt_pow)
    println(size(filt_pow))

    # -------- for plane half-covered (pc=1), add other half-plane
    if pc == 1
        filt_pow .+= circshift(filt_pow[end:-1:1,end:-1:1],(1,1))
    end

    # -------- compute power required to sum to 1.0
    i0 = round(Int16,nx/2-2)
    i1 = round(Int16,nx/2+4)
    center_power = 1.0 .- fftshift(filt_pow)[i0:i1,i0:i1]
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

@time hash = fink_filter_bank_AKS(1, 8, nx=256, pc=1, wd=2)

hash

A = SparseArrays.sprand(256,256,0.1)

B = sprand(256,256,0.1)

@time A*B

A1 = Matrix(A)
B1 = Matrix(B)

@time A1*B1

A.nzval.-=1

A

f = x -> cos(x)

map!(f,A1)

f = x -> 2*x
foo = [1   2   3];
foo1 = [7   8   9];
map!(f,foo,foo1)

foo
foo1

map!(f,A.nzval,A.nzval)
C = copy(A)

C.nzval .= 0

C

A

@benchmark A.^2

@benchmark A.*A

sum(A)

##

hash = fink_filter_hash(1, 8, nx=256, pc=1, wd=2)

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

        # -------- radial part
        @views F_z[:,:,kmask].= reshape(cos.(Δk[kmask] .* (cz*π/2)),1,1,k_count)

        @inbounds for index = 1:n2d
            p_ind = hash["filt_index"][index]
            p_filt = hash["filt_value"][index]
            f_ind    = k_ind + (index-1)*K
            @views filt_tmp = p_filt.*F_z[p_ind,kmask]
            ind = findall(filt_tmp .> 1E-13)
            val = filt_tmp[ind]
            map!(x->CartesianIndex(p_ind[x[1]][1],p_ind[x[1]][2],x[2]),ind,filtind[f_ind])
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
    info["2d_filt_index"]   = hash["filt_index"]
    info["2d_filt_value"]   = hash["filt_value"]


    info["nz"]              = nz
    info["cz"]              = cz
    info["k_value"]         = k_value
    info["filt_index"]      = filtind
    info["filt_value"]      = filtval

    return info
end

filt_3d = fink_filter_bank_3dizer(hash, 1, nz=256)

@time hash = fink_filter_hash(1, 8, nx=256, pc=1, wd=2)
@benchmark filt_3d = fink_filter_bank_3dizer(hash, 1, nz=256)

filt_3d[1].*filt_3d[2]

filt_3d

temp = map(x->[x[1],x[2]],filt_3d[1])

temp2 = collect(Base.product(temp, filt_3d[2]))
temp3 = [CartesianIndex(x[1][1],x[1][2],x[2]) for x in temp2]

temp4 = vec(temp3)

filt_3d["2d_filt_index"][2]

filt_3d["filt_index"]

function plot_filter_bank_QA(hash; fname="filter_bank_QA.png")

    # -------- define plot1 to append plot to a list
    function plot1(ps, image; clim=nothing, bin=1.0, fsz=12, label=nothing)
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
            annotate!(lims'*[.96,.04],lims'*[.09,.91],text(label,:left,:white,16))
        end
        return
    end

    # -------- initialize array of plots
    ps   = []
    jval = hash["2d_j_value"]
    J    = length(jval)
    L    = length(hash["2d_theta_value"])
    nx   = hash["2d_npix"]
    nz   = hash["nz"]
    # plot the first 3 and last 3 j values
    jplt = [1:3;J-2:J]

    ind = zeros(Int64,J,L)
    for j_ind=1:J
        for l=1:L
            ind[j_ind,l] = Int(j_ind-1 + (l-1)*J)
        end
    end
    ind .+= 1

    # -------- loop over j
    for j_ind in jplt
        # first 3 filters in Fourier space
        j = jval[j_ind]
        bfac = max(1,2.0^(j-1))
        for l = 1:3
            label=string("j=",j_ind," ℓ=",l-1)
            #label=latexstring("\\tilde{\\psi}_{$j_ind,$(l-1)}")
            filt = zeros(nx,nx,nz)
            println(nx)
            println(nz)
            filt[hash["2d_filt_index"][2],hash["filt_index"][2]] = hash["filt_value"][2]
            temp = dropdims(sum(filt,dims=3))
            plot1(ps, fftshift(filt[:,:]), clim=(0,1),
                  bin=bfac, label=label)
        end
        fac = max(1,2.0^(5-j))
        # real part of config space of 3rd filter
        plot1(ps, fftshift(real(ifft(filt[:,:,ind[j_ind,3]]))),
              bin=fac, label=string("j=",j_ind," ℓ=",3-1))

        # sum all filters to form (partial) ring
        ring = dropdims(sum(filt[:,:,ind[j_ind,:]].^2, dims=3), dims=3)
        plot1(ps, fftshift(ring[:,:,1]), clim=(0,1),
              bin=bfac, label=string("j=",j_ind," ℓ=",0,":",L-1))
    end

    # wavepow = fftshift(dropdims(sum(filt[:,:,ind].^2, dims=(3,4)), dims=(3,4)))
    # plot1(ps, wavepow, clim=(-0.1,1), label=string("j=1:",J," ℓ=0:",L-1))
    #
    # if info["pc"]==1
    #     wavepow += circshift(wavepow[end:-1:1,end:-1:1],(1,1))
    # end
    # plot1(ps, wavepow, clim=(-0.1,1), bin=32, label="missing power")
    # phi = filt[:,:,info["phi_index"]]
    # phi_shift = fftshift(phi)
    # plot1(ps, phi_shift.^2, clim=(-0.1,1), bin=32, label="ϕ")
    # disc = wavepow+(phi_shift.^2)
    # plot1(ps, fftshift(real(ifft(phi))), label="ϕ")
    # plot1(ps, disc, clim=(-0.1,1), bin=32, label="all")

    myplot = plot(ps..., layout=(7,5), size=(1400,2000))
    savefig(myplot, fname)
end

plot_filter_bank_QA(filt_3d, fname="/Users/saydjari/Dropbox/GradSchool_AKS/Doug/Projects/DHC/scratch_AKS/filt-3d.png")

Profile.clear()
@profile filt_3d = fink_filter_bank_3dizer(filter_hash, 1, nz=256)

Juno.profiler()

@benchmark zeros(256,256,256)
@benchmark zeros(128,128,128)

## Ok, so lets try to have a good workflow

filter_hash = fink_filter_hash(1, 8, nx=256, pc=1, wd=2)

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

filt_3d = fink_filter_bank_3dizer(filter_hash, 1, nz=256)

@time filt_3d = fink_filter_bank_3dizer(filter_hash, 1, nz=256)

@benchmark filt_3d = fink_filter_bank_3dizer(filter_hash, 1, nz=256)

using Plots; gr()
using Measures
theme(:dark)

function plot_filter_bank_QAxy(hash; fname="filter_bank_QA.png")

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
    jval = hash["2d_j_value"]
    kval = hash["k_value"]
    J    = length(hash["2d_j_value"])
    L    = length(hash["2d_theta_value"])
    K    = length(hash["k_value"])
    nx   = hash["2d_npix"]
    nz   = hash["nz"]
    lup  = hash["psi_index"]
    index= hash["filt_index"]
    value= hash["filt_value"]
    # plot the first 3 and last 3 j values
    jplt = [1:3;J-2:J]
    lplt = 1:3
    kplt = [1,2,K]

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
                plot1(ps, fftshift(temp[:,:]), clim=(0,1),
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
            plot1(ps, fftshift(ring[:,:]), clim=(0,1),
                  bin=bfac, label=string("j=",j_ind," ℓ=",0,":",L-1," k=",k_ind))
        end
    end

    filt = zeros(nx,nx,nz)
    for indx in lup[1:J,:,2]
        filt[index[indx]] .+= value[indx].^2
    end

    wavepow = fftshift(dropdims(sum(filt, dims=3), dims=3))
    plot1(ps, wavepow, clim=(-0.1,1), label=string("j=1:",J," ℓ=0:",L-1," K=1:",K),color=:black)

    if hash["2d_pc"]==1
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

temp1 = plot_filter_bank_QAxy(filt_3d, fname="/Users/saydjari/Dropbox/GradSchool_AKS/Doug/Projects/DHC/scratch_AKS/images/filt-3d.png")

function plot_filter_bank_QAxz(hash; fname="filter_bank_QA.png")

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
    jval = hash["2d_j_value"]
    kval = hash["k_value"]
    J    = length(hash["2d_j_value"])
    L    = length(hash["2d_theta_value"])
    K    = length(hash["k_value"])
    nx   = hash["2d_npix"]
    nz   = hash["nz"]
    lup  = hash["psi_index"]
    index= hash["filt_index"]
    value= hash["filt_value"]
    # plot the first 3 and last 3 j values
    jplt = [1:3;J-2:J]
    lplt = [1,2,L]
    kplt = [1,2,K]

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
    for indx in lup[1:J,:,:]
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

temp1 = plot_filter_bank_QAxz(filt_3d, fname="/Users/saydjari/Dropbox/GradSchool_AKS/Doug/Projects/DHC/scratch_AKS/images/filt-3dxz.png")

nx = convert(Int64,filt_3d["2d_npix"])
nz = 256
lup  = filt_3d["psi_index"]
index= filt_3d["filt_index"]
value= filt_3d["filt_value"]

filt = zeros(nx,nx,nz)
ind = lup[1,1,1]
filt[index[ind]] = value[ind]
plot(dropdims(sum(filt[1,].^2,dims=(1,2)),dims=(1,2)),xlims=(0,128))

filt = zeros(nx,nx,nz)
filt[filt_3d[1]] = filt_3d[2]
plot(dropdims(sum(filt.^2,dims=(1,2)),dims=(1,2)),xlims=(0,128))
