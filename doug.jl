using Plots; gr()
using Measures
using FFTW
push!(LOAD_PATH, pwd())
using DHC_2DUtils


function plot_filter_bank_QA(filt, info; fname="filter_bank_QA.png")

    # -------- define plot1 to append plot to a list
    function plot1(ps, image; clim=nothing, bin=1.0, fsz=5)
        # ps    - list of plots to append to
        # image - image to heatmap
        # clim  - tuple of color limits (min, max)
        # bin   - bin down by this factor, centered on nx/2+1
        # fsz   - font size
        marg   = 0mm
        nx, ny = size(image)
        nxb    = nx/round(Integer, 2*bin)

        # -------- center on nx/2+1
        i0 = round(Integer, (nx/2+2)-nxb)
        i1 = round(Integer, (nx/2)+nxb)
        subim = image[i0:i1,i0:i1]
        push!(ps, heatmap(image,aspect_ratio=:equal,clim=clim,
            xlims=(i0,i1), ylims=(i0,i1),
            legend=false,xtickfontsize=fsz,ytickfontsize=fsz,
            rightmargin=marg,leftmargin=marg,topmargin=marg,bottommargin=marg))
        return
    end

    # -------- initialize array of plots
    ps   = []
    ind  = info["filter_index"]
    jval = info["j_value"]
    # -------- loop over j
    for j_ind in 1:6
        # first 3 filters in Fourier space
        j = jval[j_ind]
        bfac = max(1,2.0^(j-1))
        plot1(ps, fftshift(filt[:,:,ind[j_ind,1]]), clim=(0,1),bin=bfac)
        plot1(ps, fftshift(filt[:,:,ind[j_ind,2]]), clim=(0,1),bin=bfac)
        plot1(ps, fftshift(filt[:,:,ind[j_ind,3]]), clim=(0,1),bin=bfac)
        fac = max(1,2.0^(5-j))
        # real part of config space of 3rd filter
        plot1(ps, fftshift(real(ifft(filt[:,:,ind[j_ind,3]]))),bin=fac)

        # sum all filters to form (partial) ring
        ring = dropdims(sum(filt[:,:,ind[j_ind,:]].^2, dims=3), dims=3)
        plot1(ps, fftshift(ring[:,:,1]), clim=(0,1),bin=bfac)
    end

    wavepow = fftshift(dropdims(sum(filt[:,:,ind].^2, dims=(3,4)), dims=(3,4)))
    plot1(ps, wavepow, clim=(-0.1,1))

    if info["pc"]==1
        wavepow += circshift(wavepow[end:-1:1,end:-1:1],(1,1))
    end
    plot1(ps, wavepow, clim=(-0.1,1), bin=32)
    phi = fftshift(filt[:,:,info["phi_index"]])
    plot1(ps, phi.^2, clim=(-0.1,1), bin=32)
    disc = wavepow+(phi.^2)
    plot1(ps, disc, clim=(-0.1,1), bin=32)
    plot1(ps, disc, clim=(-0.1,1))

    myplot = plot(ps..., layout=(7,5), size = (700,1000))
    savefig(myplot, fname)
end


function fink_filter_bank2(c, L; nx=256, wd=1, pc=1)
    #c  - sets the scale sampling rate (1 is dyadic, 2 is half dyadic)
    #L  - number of angular bins (usually 8*pc or 16*pc)
    #wd - width of the wavelets (default 1, wide 2)
    #pc - plane coverage (default 1, full 2Pi 2)


    # -------- set parameters
    dθ   = pc*π/L
    wdθ  = wd*dθ
    dx   = nx/2-1
    norm = 1.0/sqrt(wd)

    im_scale = convert(Int8,log2(nx))
    # -------- number of bins in radial direction (size scales)
    J = (im_scale-2)*c

    # -------- allocate output array of zeros
    filt         = zeros(nx, nx, J*L+1)
    filter_index = zeros(Int32, J, L)
    theta        = zeros(Float64, L)
    j_value      = zeros(Float64, J)

    # -------- allocate theta and logr arrays
    θ    = zeros(nx, nx)
    logr = zeros(nx, nx)

    # -------- loop over l
    for l = 0:L-1
        θ_l        = dθ*l
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
            filter_index[j_ind,l+1] = f_ind
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
    info["filter_index"] = filter_index
    info["phi_index"]    = phi_index
    info["pc"]           = pc
    info["wd"]           = wd

    return filt, info
end

# -------- define a filter bank

filt, info  = fink_filter_bank2(1, 8, pc=1, wd=1)
plot_filter_bank_QA(filt, info, fname="filt-8-pc1-wd1.png")

filt, info  = fink_filter_bank2(1, 8, pc=1, wd=2)
plot_filter_bank_QA(filt, info, fname="filt-8-pc1-wd2.png")

filt, info  = fink_filter_bank2(1, 8, pc=1, wd=3)
plot_filter_bank_QA(filt, info, fname="filt-8-pc1-wd3.png")

filt, info  = fink_filter_bank2(1, 8, pc=2, wd=1)
plot_filter_bank_QA(filt, info, fname="filt-8-pc2-wd1.png")

filt, info  = fink_filter_bank2(1, 16, pc=2, wd=1)
plot_filter_bank_QA(filt, info, fname="filt-16-pc2-wd1.png")

filt, info  = fink_filter_bank2(1, 16, pc=2, wd=2)
plot_filter_bank_QA(filt, info, fname="filt-16-pc2-wd2.png")

filt, info  = fink_filter_bank2(1, 16, pc=1, wd=1)
plot_filter_bank_QA(filt, info, fname="filt-16-pc1-wd1.png")

filt, info  = fink_filter_bank2(2, 8, pc=1, wd=1)
plot_filter_bank_QA(filt, info, fname="filt2-8-pc1-wd1.png")

filt, info  = fink_filter_bank2(2, 8, pc=2, wd=1)
plot_filter_bank_QA(filt, info, fname="filt2-8-pc2-wd1.png")

filt, info  = fink_filter_bank2(2, 16, pc=2, wd=1)
plot_filter_bank_QA(filt, info, fname="filt2-16-pc2-wd1.png")

filt, info  = fink_filter_bank2(2, 16, pc=2, wd=2)
plot_filter_bank_QA(filt, info, fname="filt2-16-pc2-wd2.png")



""
