using Plots; gr()
using Measures
using FFTW
push!(LOAD_PATH, pwd())
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
    ind  = info["filter_index"]
    jval = info["j_value"]
    J    = length(jval)
    L    = length(info["theta_value"])
    # plot the first 3 and last 3 j values
    jplt = [1:3;J-2:J]

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

# -------- define a filter bank

filt, info  = fink_filter_bank(1, 8, pc=1, wd=1)
hash = fink_filter_hash(1, 8, pc=1, wd=1)
plot_filter_bank_QA(filt, info, fname="filt-8-pc1-wd1.png")

filt, info  = fink_filter_bank(1, 8, pc=1, wd=2)
plot_filter_bank_QA(filt, info, fname="filt-8-pc1-wd2.png")

filt, info  = fink_filter_bank(1, 8, pc=1, wd=3)
plot_filter_bank_QA(filt, info, fname="filt-8-pc1-wd3.png")

filt, info  = fink_filter_bank(1, 8, pc=2, wd=1)
plot_filter_bank_QA(filt, info, fname="filt-8-pc2-wd1.png")

filt, info  = fink_filter_bank(1, 16, pc=2, wd=1)
plot_filter_bank_QA(filt, info, fname="filt-16-pc2-wd1.png")

filt, info  = fink_filter_bank(1, 16, pc=2, wd=2)
plot_filter_bank_QA(filt, info, fname="filt-16-pc2-wd2.png")

filt, info  = fink_filter_bank(1, 16, pc=1, wd=1)
plot_filter_bank_QA(filt, info, fname="filt-16-pc1-wd1.png")

filt, info  = fink_filter_bank(2, 8, pc=1, wd=1)
plot_filter_bank_QA(filt, info, fname="filt2-8-pc1-wd1.png")

filt, info  = fink_filter_bank(2, 8, pc=2, wd=1)
plot_filter_bank_QA(filt, info, fname="filt2-8-pc2-wd1.png")

filt, info  = fink_filter_bank(2, 16, pc=2, wd=1)
plot_filter_bank_QA(filt, info, fname="filt2-16-pc2-wd1.png")

filt, info  = fink_filter_bank(2, 16, pc=2, wd=2)
plot_filter_bank_QA(filt, info, fname="filt2-16-pc2-wd2.png")

filt, info  = fink_filter_bank(1, 8, pc=1, wd=1, shift=true)
plot_filter_bank_QA(filt, info, fname="filt-8-pc1-wd1-shift.png")

filt, info  = fink_filter_bank(1, 16, pc=2, wd=1, shift=true)
plot_filter_bank_QA(filt, info, fname="filt-16-pc2-wd1-shift.png")

filt, info  = fink_filter_bank(1, 16, pc=2, wd=2, shift=true)
plot_filter_bank_QA(filt, info, fname="filt-16-pc2-wd2-shift.png")
DH
