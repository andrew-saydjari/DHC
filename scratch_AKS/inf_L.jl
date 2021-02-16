using Test
using BenchmarkTools

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

using Plots; gr()
using Measures
using FFTW
push!(LOAD_PATH, "/Users/saydjari/Dropbox/GradSchool_AKS/Doug/Projects/DHC/main")
using DHC_2DUtils

function plot_filter_bank_QA(filt, info; fname="filter_bank_QA.png")

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
    jval = info["j_value"]
    J    = length(jval)
    L    = length(info["theta_value"])
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
            plot1(ps, fftshift(filt[:,:,ind[j_ind,l]]), clim=(0,1),
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

    wavepow = fftshift(dropdims(sum(filt[:,:,ind].^2, dims=(3,4)), dims=(3,4)))
    plot1(ps, wavepow, clim=(-0.1,1), label=string("j=1:",J," ℓ=0:",L-1))

    if info["pc"]==1
        wavepow += circshift(wavepow[end:-1:1,end:-1:1],(1,1))
    end
    plot1(ps, wavepow, clim=(-0.1,1), bin=32, label="missing power")
    phi = filt[:,:,info["phi_index"]]
    phi_shift = fftshift(phi)
    plot1(ps, phi_shift.^2, clim=(-0.1,1), bin=32, label="ϕ")
    disc = wavepow+(phi_shift.^2)
    plot1(ps, fftshift(real(ifft(phi))), label="ϕ")
    plot1(ps, disc, clim=(-0.1,1), bin=32, label="all")

    myplot = plot(ps..., layout=(7,5), size=(1400,2000))
    savefig(myplot, fname)
end

filt, info  = fink_filter_bank_AKS(1, 8, pc=1, wd=1, safety_on=false)

@benchmark filt, info  = fink_filter_bank(1, 8, pc=1, wd=1)

@benchmark filt, info  = fink_filter_bank_AKS(1, 8, pc=1, wd=1)

@benchmark filt, info  = fink_filter_bank_AKS(1, 8, pc=1, wd=1, safety_on=false)

function fink_filter_hash_AKS(c, L; nx=256, wd=1, pc=1, shift=false, Omega=false)
    # -------- compute the filter bank
    filt, hash = fink_filter_bank_AKS(c, L; nx=nx, wd=wd, pc=pc, shift=shift, Omega=Omega)

    # -------- list of non-zero pixels
    flist = fink_filter_list(filt)

    # -------- pack everything you need into the info structure
    hash["filt_index"] = flist[1]
    hash["filt_value"] = flist[2]
    return hash
end

filt, info  = fink_filter_bank_AKS(1, 8, pc=2, wd=1)

hash = fink_filter_hash_AKS(1, 8, pc=1, wd=1)

filt, info  = fink_filter_bank(1, 8, pc=2, wd=1)

theme(:dark)

plot_filter_bank_QA(filt, info, fname="/Users/saydjari/Dropbox/GradSchool_AKS/Doug/Projects/DHC/scratch_AKS/filt-8-pc1-wd1-aks.png")

filt, info  = fink_filter_bank_AKS(1, 8, pc=1, wd=1)

plot_filter_bank_QA(filt, info, fname="/Users/saydjari/Dropbox/GradSchool_AKS/Doug/Projects/DHC/scratch_AKS/filt-8-pc1-wd1-aks.png")

filt, info  = fink_filter_bank(1, 8, pc=1, wd=1)

plot_filter_bank_QA(filt, info, fname="/Users/saydjari/Dropbox/GradSchool_AKS/Doug/Projects/DHC/scratch_AKS/filt-8-pc1-wd1.png")

filt, info  = fink_filter_bank_AKS(1, 8, pc=1, wd=2)

plot_filter_bank_QA(filt, info, fname="/Users/saydjari/Dropbox/GradSchool_AKS/Doug/Projects/DHC/scratch_AKS/filt-8-pc1-wd2-aks.png")

filt, info  = fink_filter_bank(1, 8, pc=1, wd=2)

plot_filter_bank_QA(filt, info, fname="/Users/saydjari/Dropbox/GradSchool_AKS/Doug/Projects/DHC/scratch_AKS/filt-8-pc1-wd2.png")

hash = fink_filter_hash_AKS(1, 8, pc=1, wd=1)

DHC_compute(rand(256,256),hash,hash)
