
using Plots; gr()
using Measures
using FFTW
push!(LOAD_PATH, pwd())
using DHC_2DUtils


function plot_filter_bank_QA(filt; fname="filter_bank_QA.png")

    # -------- define plot1 to append plot to a list
    function plot1(ps, image; clim=nothing, bin=1, fsz=5)
        # ps    - list of plots to append to
        # image - image to heatmap
        # clim  - tuple of color limits (min, max)
        # bin   - bin down by this factor, centered on nx/2+1
        # fsz   - font size
        marg   = 0mm
        nx, ny = size(image)
        nxb    = nx/(2*bin)

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
    ps = []

    # -------- loop over j
    for j in 1:6
        # first 3 filters in Fourier space
        plot1(ps, fftshift(filt[:,:,(j-1)*8+1]), clim=(0,1),bin=2^(j-1))
        plot1(ps, fftshift(filt[:,:,(j-1)*8+2]), clim=(0,1),bin=2^(j-1))
        plot1(ps, fftshift(filt[:,:,(j-1)*8+3]), clim=(0,1),bin=2^(j-1))
        fac = max(1,2.0^(5-j))
        # real part of config space of 3rd filter
        plot1(ps, fftshift(real(ifft(filt[:,:,(j-1)*8+3]))),bin=fac)

        # sum all filters to form (partial) ring
        ring = sum(filt[:,:,(j-1)*8+1:j*8].^2,dims=3)
        plot1(ps, fftshift(ring[:,:,1]), clim=(0,2),bin=2^(j-1))
    end

    filtsum = (sum(filt[:,:,1:48].^2,dims=3))[:,:,1]
    plot1(ps, fftshift(filtsum), clim=(-0.1,1))
    dd = fftshift(filtsum)
    dsym = dd+circshift(dd[end:-1:1,end:-1:1],(1,1))
    plot1(ps, dsym, clim=(-0.1,2), bin=32)
    phi = fftshift(filt[:,:,49])
    plot1(ps, phi.^2, clim=(-0.1,2), bin=32)
    disc = dsym+(phi.^2)
    plot1(ps, disc, clim=(-0.1,2), bin=32)
    plot1(ps, disc, clim=(-0.1,2))

    myplot = plot(ps..., layout=(7,5), size = (700,1000))
    savefig(myplot, fname)
end

# -------- define a filter bank
filt  = fink_filter_bank(1, 8)
filt2 = fink_filter_bank(1, 8, wd=2)
plot_filter_bank_QA(filt, fname="filt.png")
plot_filter_bank_QA(filt2, fname="filt2.png")
