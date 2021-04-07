## Preloads
using Statistics
using FFTW
using Plots
using BenchmarkTools
using Profile
using LinearAlgebra
using Measures
#using StaticArrays
#using HybridArrays

# put the cwd on the path so Julia can find the module
push!(LOAD_PATH, pwd()*"/main")
using DHC_2DUtils

##
function rod_image(xcen, ycen, length, pa, fwhm; nx=256)
    # returns image of a rod with some (x,y) position, length,
    #   position angle, and FWHM in an (nx,nx) image.
    rodimage = zeros(nx,nx)

    x=0
    y=0
    sig = fwhm/2.355
    dtor = π/180
    # -------- define a unit vector in direction of rod at position angle pa
    ux = sin(pa*dtor)   #  0 deg is up
    uy = cos(pa*dtor)   # 90 deg to the right

    for i=1:nx
        for j=1:nx
            x=i-nx/2+xcen
            y=j-nx/2+ycen

            # -------- distance parallel and perpendicular to
            dpara =  ux*x + uy*y
            dperp = -uy*x + ux*y

            if abs(dpara)-length <0
                dpara= 0
            end
            dpara = abs(dpara)
            dpara = min(abs(dpara-length),dpara)

            rodimage[i,j] = exp(-(dperp^2+dpara^2)/(2*sig^2))
        end
    end
    rodimage ./= sqrt(sum(rodimage.^2))
    return rodimage
end


function rod_test_old(filter_hash, Nsam=1; doS20=false)

    Nf      = length(filter_hash["filt_value"])
    Nj      = length(filter_hash["j_value"])
    Nl      = length(filter_hash["theta_value"])
    Npix    = filter_hash["npix"]
    jlind   = filter_hash["psi_index"]
    angmax  = 180.0
    angstep = 2.5
    nang    = Int64(angmax ÷ angstep+1)
    energy     = zeros(nang)
    Siso       = zeros(Nj, Nj, Nl, nang)
    Sall       = zeros(Nl, Nj, Nl, Nj, nang)
    S2all      = zeros(Nf, Nf, nang)
    Ef         = zeros(Nf, nang)  # energy in each filter
    dang       = 180/Nl
    for i = 1:nang
        ang = (i-1)*angstep


        t = zeros(2+Nf+Nf*Nf)
        for k=0:Nsam-1
            rod1 = rod_image(10,10,30,ang+dang*k/4,8, nx=fhash["npix"])
            t   += DHC_compute(rod1, filter_hash, doS2=!doS20,doS20=doS20)
        end
        t   ./= Nsam

        #S20_big = reshape(t[2+Nf+1:2+Nf+Nf*Nf], Nf, Nf)
        #S20 = S20_big[1:Nj*Nl, 1:Nj*Nl]
        off2 = 2+Nf
        S2 = reshape(t[off2+1:off2+Nf*Nf], Nf, Nf)
        S2all[:,:,i] = S2
        # energy[i] = sum(diag(S20_big))
        # Ef[:,i]   = diag(S20_big)
        # if filter_hash["pc"]==1 Ef[1:Nj*Nl,i] .*= 2 end
        # Verify sum of diag(S20) = sum(S1)   (= energy)
        # println(i,"   ",energy[i],"   ",sum(t[3:2+Nf]))


        # what about j1, j2, Delta L
        #S = reshape(S20,Nj,Nl,Nj,Nl)  #  (j1, l1, j2, l2), symmetric in j and l
        S = S2[jlind,jlind]  # amazing this works...
        println("Sum of S2:  ",sum(S2))
        for j1 = 1:Nj
            for j2 = 1:Nj
                for l1 = 1:Nl
                    for l2 = 1:Nl
                        DeltaL = mod(l1-l2, Nl)
                        Siso[j1,j2,DeltaL+1,i] += S[j1,l1,j2,l2]
                        # Sall[j1,j2,l1,l2,i] = S[j1,l1,j2,l2]
                        Sall[l1,j1,l2,j2,i] = S[j1,l1,j2,l2] # flip indices
                        #println(Siso[j1,j2,DeltaL+1,i])
                    end
                end
            end
        end

    end
    println(std(Siso))
    return (Siso, Sall)
end


function make_mp4(fhash; doS20=false)

    clims = (-12,-3)
    if doS20
        clims=(-6,-1)
    end

    prefix = "frame"
    suffix = ".png"
    Sisofoo, Sall = rod_test_old(fhash, doS20=doS20)
    Nj      = length(fhash["j_value"])
    Nl      = length(fhash["theta_value"])
    Nframe  = size(Sall, 5)

    for i=1:Nframe
        p=heatmap(log10.(reshape(Sall[:,:,:,:,i],Nj*Nl,Nj*Nl)),
        size=(512,512),aspect_ratio=1,clims=clims)
        #display(p)
        outname = prefix*lpad(i,4,'0')*suffix
        println(outname)
        savefig(p, outname)
    end
    return 0
end


cmd=`ffmpeg -r 15 -f image2 -i frame%04d.png -vcodec libx264 -crf 25 -pix_fmt yuv420p test.mp4`
run(cmd)


function rod_test(filter_hash, Nsam=1; nx=256, doS20=false, doS1=false)
    # Loop over many rod position angles, return Siso for all
    Nf      = length(filter_hash["filt_value"])
    Nj      = length(filter_hash["j_value"])
    Nl      = length(filter_hash["theta_value"])
    Npix    = filter_hash["npix"]
    isomat  = doS1 ? filter_hash["S1_iso_mat"] : filter_hash["S2_iso_mat"]
    Niso    = size(isomat, 1)

    angmax  = 180.0
    angstep = 2.5
    Nang    = Int64(angmax ÷ angstep+1)

    Siso       = zeros(Niso, Nang)
    dang       = 180/Nl
    for i = 1:Nang
        rod1 = zeros(nx,nx)
        ang = i*angstep

        t = 0.0
        for k=1:Nsam
            rod1 = rod_image(1,1,30*(nx/256),ang+dang*k/4,8,nx=nx)
            rod1 .-= mean(rod1)
            doS2 = (!doS20 & !doS1)
            t   = t.+DHC_compute(rod1, filter_hash, doS2=doS2, doS20=doS20)
        end
        t   ./= Nsam

        off2 = 2+Nf
        Sarr = doS1 ? t[3:end] : t[off2+1:off2+Nf*Nf]

        println(i," of ",Nang,"  Sum of S:  ",sum(Sarr), "  Sum of rod: ",sum(rod1))
        Siso[:,i] = isomat * Sarr

    end
    println(std(Siso))
    return Siso
end


function rodtest_rms(Siso; frac=false)
    # reshape(Sisoin,Nj*Nj*Nl,37)
    sig = std(Siso, dims=2)
    mn  = mean(Siso, dims=2)
    if frac return sig./(mn.+1e-6) end
    return sig
end


function rodtest_plot_rms(Siso1,Siso4,Siso16;outname="test.png",
    label="Nsam=".*["1","4","16"], doS20=false, title=title)

    ylim=10.0 .^(-6,0)
    marg   = 10mm
    coefftype = doS20 ? "S20" : "S2"
    xlabel = coefftype*"_ISO Index"
    p1=plot(rodtest_rms(Siso1,frac=true),ylim=ylim,xlabel=xlabel,
            ylabel="Fractional RMS", yscale=:log10, line=(:gray, :dot),
            label=label[1],size=(700,1000),leftmargin=marg)
    plot!(p1,rodtest_rms(Siso4,frac=true), line=(:gray, :dash),label=label[2])
    plot!(p1,rodtest_rms(Siso16,frac=true), line=(:blue),label=label[3])
    annotate!(p1,10,0.6*ylim[2],text(title,:left,:blue,16))

    p2=plot(rodtest_rms(Siso1),ylim=[1E-10,1E-2],xlabel=xlabel,
            ylabel="RMS [Energy]", yscale=:log10, line=(:gray, :dot),
            label=label[1],size=(700,1000))
    plot!(p2,rodtest_rms(Siso4), line=(:gray, :dash),label=label[2])
    plot!(p2,rodtest_rms(Siso16), line=(:green),label=label[3])

    ps = [p1,p2]
    myplot = plot(ps..., layout=(2,1))
    savefig(myplot, outname)

    return
end


function rodtest_plot_rms_S1(Siso1,Siso4,Siso16;outname="test.png",
    label="Nsam=".*["1","4","16"], title=title)

    ylim=10.0 .^(-8,0)
    marg   = 10mm
    coefftype = "S1"
    xlabel = coefftype*"_ISO Index"
    p1=plot(rodtest_rms(Siso1,frac=true),ylim=ylim,xlabel=xlabel,
            ylabel="Fractional RMS", yscale=:log10, line=(:gray, :dot),
            label=label[1],size=(700,1000),leftmargin=marg)
    plot!(p1,rodtest_rms(Siso4,frac=true), line=(:gray, :dash),label=label[2])
    plot!(p1,rodtest_rms(Siso16,frac=true), line=(:blue),label=label[3])
    annotate!(p1,10,0.6*ylim[2],text(title,:left,:blue,16))

    p2=plot(rodtest_rms(Siso1),ylim=[1E-10,1E-2],xlabel=xlabel,
            ylabel="RMS [Energy]", yscale=:log10, line=(:gray, :dot),
            label=label[1],size=(700,1000))
    plot!(p2,rodtest_rms(Siso4), line=(:gray, :dash),label=label[2])
    plot!(p2,rodtest_rms(Siso16), line=(:green),label=label[3])

    ps = [p1,p2]
    myplot = plot(ps..., layout=(2,1))
    savefig(myplot, outname)

    return
end


function rodtest_plot_coeffs(Siso1,Siso4,Siso16;outname="test.png",
    label="Nsam=".*["1","4","16"], doS20=false, title=title)

    marg   = 4mm
    coefftype = doS20 ? "S20" : "S2"
    xval   = collect(0:2.5:180)
    xlabel = "Angle [deg]"
    ticks  = collect(0:45:180)

    p1=plot(xval,Siso1',yscale=:log10,legend=false,xlabel=xlabel,
             ylabel=coefftype,size=(700,1000),leftmargin=marg,xticks=ticks)
    annotate!(p1,0,.9,text(title,:left,:blue,16))

    p2=plot(xval,Siso16',yscale=:log10,legend=false,xlabel=xlabel,
             ylabel=coefftype,size=(700,1000),leftmargin=marg,xticks=ticks)
    #annotate!(p2,0,.9,text(title,:left,:blue,16))

    p3=plot(xval,Siso1',yscale=:log10,legend=false,xlabel=xlabel,
             ylabel=coefftype,size=(700,1000),leftmargin=marg,
             xticks=ticks,ylims=(1e-3,1e-2))
    annotate!(p3,0,.9,text(title,:left,:blue,16))

    p4=plot(xval,Siso16',yscale=:log10,legend=false,xlabel=xlabel,
             ylabel=coefftype,size=(700,1000),leftmargin=marg,
             xticks=ticks,ylims=(1e-3,1e-2))
    annotate!(p4,0,.9,text(title,:left,:blue,16))

    ps = [p1,p2,p3,p4]
    myplot = plot(ps..., layout=(2,2))
    savefig(myplot, outname)

    return
end


function rod_test_plot(Nsam; L=8, wd=1, nx=128,
                       outname="test.png", doS20=false)
    # run rod_test for 3 values of Nsam and output PNG file
    fhash = fink_filter_hash(1, L, wd=wd, nx=nx, Omega=true)
    Siso_A = rod_test(fhash, Nsam[1], nx=nx, doS20=doS20)
    Siso_B = rod_test(fhash, Nsam[2], nx=nx, doS20=doS20)
    Siso_C = rod_test(fhash, Nsam[3], nx=nx, doS20=doS20)
    mylabels = "Nsam=".*string.(Nsam)
    coefftype = doS20 ? "S20" : "S2"
    title = string(coefftype, "iso, L=", L, ", wd=", wd)
    rodtest_plot_rms(Siso_A, Siso_B, Siso_C, outname=outname,
                     label=mylabels, title=title, doS20=doS20)

    cname = replace(outname, "RMS" => "coeff")
    rodtest_plot_coeffs(Siso_A, Siso_B, Siso_C, outname=cname,
                        label=mylabels, title=title, doS20=doS20)

    return
end


function rod_test_plot_S1(Nsam; L=8, wd=1, nx=128,
                       outname="test.png", doS20=false)
    # run rod_test for 3 values of Nsam and output PNG file
    fhash = fink_filter_hash(1, L, wd=wd, nx=nx, Omega=true)
    println(fhash["wd"])
    Siso_A = rod_test(fhash, Nsam[1], nx=nx, doS1=true)
    Siso_B = rod_test(fhash, Nsam[2], nx=nx, doS1=true)
    Siso_C = rod_test(fhash, Nsam[3], nx=nx, doS1=true)
    mylabels = "Nsam=".*string.(Nsam)
    coefftype = doS20 ? "S20" : "S2"
    title = string(coefftype, "iso, L=", L, ", wd=", wd)
    rodtest_plot_rms_S1(Siso_A, Siso_B, Siso_C, outname=outname,
                     label=mylabels, title=title)

    return
end



# Just run this code to make the plots I sent on Mar 14, 2021 (π-day!)
nx=128
snx = string(nx)
Nsam = [1,4,16]

rod_test_plot(Nsam; L=8, wd=1, nx=nx, outname="S2-RMS-NL8-wd1-"*snx*".png")
rod_test_plot(Nsam; L=8, wd=2, nx=nx, outname="S2-RMS-NL8-wd2-"*snx*".png")
rod_test_plot(Nsam; L=8, wd=4, nx=nx, outname="S2-RMS-NL8-wd4-"*snx*".png")
rod_test_plot([1,4,32]; L=16, wd=4, nx=nx, outname="S2-RMS-NL16-wd4-"*snx*".png")

rod_test_plot(Nsam; L=8, wd=1, nx=nx, outname="S20-RMS-NL8-wd1-"*snx*".png", doS20=true)
rod_test_plot(Nsam; L=8, wd=2, nx=nx, outname="S20-RMS-NL8-wd2-"*snx*".png", doS20=true)
rod_test_plot(Nsam; L=8, wd=4, nx=nx, outname="S20-RMS-NL8-wd4-"*snx*".png", doS20=true)
rod_test_plot([1,4,32]; L=16, wd=4, nx=nx, outname="S20-RMS-NL16-wd4-"*snx*".png", doS20=true)

rod_test_plot_S1(Nsam; L=8, wd=1, nx=nx, outname="S1-RMS-NL8-wd1-"*snx*".png")
rod_test_plot_S1(Nsam; L=8, wd=2, nx=nx, outname="S1-RMS-NL8-wd2-"*snx*".png")
rod_test_plot_S1(Nsam; L=8, wd=4, nx=nx, outname="S1-RMS-NL8-wd4-"*snx*".png")
rod_test_plot_S1([1,4,32]; L=16, wd=4, nx=nx, outname="S1-RMS-NL16-wd4-"*snx*".png")

## AKS export
xval   = collect(0:2.5:180)

fhash = fink_filter_hash(1, 8, wd=2, nx=128, Omega=false)
Siso_A = rod_test(fhash, 1, nx=128, doS20=false)
h5write("../DHC/scratch_AKS/paper_data/rod_stable_L8_wd2_N1.h5", "data", Siso_A, deflate=3)
h5write("../DHC/scratch_AKS/paper_data/rod_stable_L8_wd2_N1.h5", "angles", xval, deflate=3)

fhash = fink_filter_hash(1, 16, wd=2, nx=128, Omega=false)
Siso_A = rod_test(fhash, 1, nx=128, doS20=false)
h5write("../DHC/scratch_AKS/paper_data/rod_stable_L16_wd2_N1.h5", "data", Siso_A, deflate=3)
h5write("../DHC/scratch_AKS/paper_data/rod_stable_L16_wd2_N1.h5", "angles", xval, deflate=3)

fhash = fink_filter_hash(1, 16, wd=4, nx=128, Omega=false)
Siso_A = rod_test(fhash, 1, nx=128, doS20=false)
h5write("../DHC/scratch_AKS/paper_data/rod_stable_L16_wd4_N1.h5", "data", Siso_A, deflate=3)
h5write("../DHC/scratch_AKS/paper_data/rod_stable_L16_wd4_N1.h5", "angles", xval, deflate=3)

fhash = fink_filter_hash(1, 8, wd=4, nx=128, Omega=false)
Siso_A = rod_test(fhash, 1, nx=128, doS20=false)
h5write("../DHC/scratch_AKS/paper_data/rod_stable_L8_wd4_N1.h5", "data", Siso_A, deflate=3)
h5write("../DHC/scratch_AKS/paper_data/rod_stable_L8_wd4_N1.h5", "angles", xval, deflate=3)

fhash = fink_filter_hash(1, 8, wd=1, nx=128, Omega=false)
Siso_A = rod_test(fhash, 1, nx=128, doS20=false)
h5write("../DHC/scratch_AKS/paper_data/rod_stable_L8_wd1_N1.h5", "data", Siso_A, deflate=3)
h5write("../DHC/scratch_AKS/paper_data/rod_stable_L8_wd1_N1.h5", "angles", xval, deflate=3)

fhash = fink_filter_hash(1, 8, wd=2, nx=128, Omega=false)
Siso_A = rod_test(fhash, 4, nx=128, doS20=false)
h5write("../DHC/scratch_AKS/paper_data/rod_stable_L8_wd2_N4.h5", "data", Siso_A, deflate=3)
h5write("../DHC/scratch_AKS/paper_data/rod_stable_L8_wd2_N4.h5", "angles", xval, deflate=3)

fhash = fink_filter_hash(1, 8, wd=2, nx=128, Omega=false)
Siso_A = rod_test(fhash, 16, nx=128, doS20=false)
h5write("../DHC/scratch_AKS/paper_data/rod_stable_L8_wd2_N16.h5", "data", Siso_A, deflate=3)
h5write("../DHC/scratch_AKS/paper_data/rod_stable_L8_wd2_N16.h5", "angles", xval, deflate=3)

test_image = rod_image(1,1,30*(128/256),0,8,nx=128)

heatmap(test_image,xlims = [46,81],ylims = [46,81])
