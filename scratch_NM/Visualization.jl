module Visualization
    using Plots
    using Measures

    function plot_synth_QA(ImTrue, ImInit, ImSynth; fname=nothing)

        # -------- define plot1 to append plot to a list
        function plot1(ps, image; clim=nothing, bin=1.0, fsz=16, label=nothing)
            # ps    - list of plots to append to
            # image - image to heatmap
            # clim  - tuple of color limits (min, max)
            # bin   - bin down by this factor, centered on nx/2+1
            # fsz   - font size
            marg   = 1mm
            nx, ny = size(image)
            nxb    = nx/round(Integer, 2*bin)

            # -------- center on nx/2+1
            i0 = max(1,round(Integer, (nx/2+2)-nxb-1))
            i1 = min(nx,round(Integer, (nx/2)+nxb+1))
            lims = [i0,i1]
            subim = image[i0:i1,i0:i1]
            push!(ps, heatmap(image, aspect_ratio=:equal, clim=clim,
                xlims=lims, ylims=lims, size=(400,400),
                legend=false, xtickfontsize=fsz, ytickfontsize=fsz,#tick_direction=:out,
                rightmargin=marg, leftmargin=marg, topmargin=marg, bottommargin=marg))
            if label != nothing
                annotate!(lims'*[.96,.04],lims'*[.09,.91],text(label,:left,:white,32))
            end
            return
        end

        # -------- initialize array of plots
        ps   = []
        clim  = (minimum(ImTrue), maximum(ImTrue))
        resit = ImInit-ImTrue
        resir = ImInit-ImSynth
        resrt = ImSynth-ImTrue
        clim2 = (minimum([minimum(resit), minimum(resir), minimum(resrt)]), maximum([maximum(resit), maximum(resir), maximum(resrt)]))
        println(clim, clim2)

        # -------- 6 panel QA plot

        plot1(ps, ImTrue, clim=clim, label="True")
        plot1(ps, ImSynth, clim=clim, label="Synth")
        plot1(ps, ImInit, clim=clim, label="Init")
        plot1(ps, resit, clim=clim2, label="Init-True")
        plot1(ps, resir, clim=clim2, label="Init-Synth")
        plot1(ps, resrt, clim=clim2, label="Synth-True")
        #=
        heatmap(ImTrue, clim=clim, label="True")
        heatmap(ImSynth, clim=clim, label="Synth")
        heatmap(ImInit, clim=clim, label="Init")
        heatmap(resit, clim=clim2, label="Init-True")
        heatmap(resir, clim=clim2, label="Init-Synth")
        heatmap(resrt, clim=clim2, label="Synth-True")
        =#
        myplot = plot(ps..., layout=(3,2), size=(1400,2000))
        if fname!=nothing
            savefig(myplot, fname)
        end
    end

    function plot_diffscales(images, titles; fname=nothing)
        pl12 = plot(
            heatmap(images[1], title=titles[1]),
            heatmap(images[2], title=titles[2]),
            heatmap(images[3],title= titles[3]),
            heatmap(images[4], title=titles[4]);
            layout=4,
        )
        plot!(size=(1000, 800))
        if fname!=nothing
            savefig(pl12, fname)
        end
    end
end
