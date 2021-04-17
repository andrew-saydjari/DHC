
module DHC_tests

    using FFTW
    using Plots
    using Measures
    theme(:dark)

    export plot_filter_bank_QA
    export plot_filter_bank_QAxy
    export plot_filter_bank_QAxz

    function plot_filter_bank_QA(filt, info; fname="filter_bank_QA.png",p=2)

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
        ind  = info["psi_index"]
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
                  bin=fac, label=string("j=",jval[j_ind]," ℓ=",3-1))

            # sum all filters to form (partial) ring
            ring = dropdims(sum(filt[:,:,ind[j_ind,:]].^2, dims=3), dims=3)
            plot1(ps, fftshift(ring[:,:,1]), clim=(0,1),
                  bin=bfac, label=string("j=",jval[j_ind]," ℓ=",0,":",L-1))
        end

        wavepow = fftshift(dropdims(sum(filt[:,:,ind].^p, dims=(3,4)), dims=(3,4)))
        plot1(ps, wavepow, label=string("j=1:",jval[end]," ℓ=0:",L-1)) #clim=(-0.1,1)

        if info["t"]==1
            wavepow ./= 2
            wavepow += circshift(wavepow[end:-1:1,end:-1:1],(1,1))
        end
        plot1(ps, wavepow, bin=32, label="missing power") #clim=(-0.1,1)
        phi = filt[:,:,info["phi_index"]]
        phi_shift = fftshift(phi)
        plot1(ps, phi_shift.^p, bin=32, label="ϕ") #clim=(-0.1,1)
        disc = wavepow+(phi_shift.^p)
        plot1(ps, fftshift(real(ifft(phi))), label="ϕ")
        plot1(ps, disc, bin=32, label="all") #clim=(-0.1,1)

        myplot = plot(ps..., layout=(7,5), size=(1400,2000))
        savefig(myplot, fname)
        return disc
    end

    #AKS added 3d plotting functions 2021_02_22; They are not fully complete
    #but do give an ok picture of what is going on. Enough to proceed with some applications
    function plot_filter_bank_QAxy(hash; fname="filter_bank_QAxy.png")

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

        if hash["2d_t"]==1
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

    function plot_filter_bank_QAxz(hash; fname="filter_bank_QAxz.png")

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
        if hash["2d_t"]==1
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
end # of module
