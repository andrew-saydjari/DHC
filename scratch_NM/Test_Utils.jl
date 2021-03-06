using Statistics
using Plots
using BenchmarkTools
using Profile
using FFTW
using Statistics
using Optim
using Images, FileIO, ImageIO
using Printf
using Revise
using Profile
using LinearAlgebra

push!(LOAD_PATH, pwd()*"/main")
using DHC_2DUtils
#include("Deriv_Utils_New.jl")
#import Deriv_Utils_New
push!(LOAD_PATH, pwd()*"/scratch_NM")
using Deriv_Utils_New

function wst_S1S2_derivfast_local(image::Array{Float64,2}, filter_hash::Dict)
    #Works now.
    #Possible Bug: You assumed zero freq in wrong place and N-k is the wrong transformation.
    # Use 2 threads for FFT
    FFTW.set_num_threads(2)

    # array sizes
    (Nx, Ny)  = size(image)
    (Nf, )    = size(filter_hash["filt_index"])

    # allocate output array
    dS1dp  = zeros(Float64, Nf, Nx, Nx) #Mod zeros(Float64, Nx, Nx, Nf)
    dS2dp  = zeros(Float64, Nf, Nf, Nx, Nx) #Mod

    # allocate image arrays for internal use
    im_rdc_0_1 = Array{ComplexF64, 3}(undef, Nx, Ny, Nf)  # real domain, complex
    im_rd_0_1  = Array{Float64, 3}(undef, Nx, Ny, Nf)  # real domain, complex

    # Not sure what to do here -- for gradients I don't think we want these
    ## 0th Order
    #S0[1]   = mean(image)
    #norm_im = image.-S0[1]
    #S0[2]   = sum(norm_im .* norm_im)/(Nx*Ny)
    #norm_im ./= sqrt(Nx*Ny*S0[2])

    ## 1st Order
    im_fd_0 = fft(image)  # total power=1.0

    # unpack filter_hash
    f_ind   = filter_hash["filt_index"]  # (J, L) array of filters represented as index value pairs
    f_val   = filter_hash["filt_value"]

    f_ind_rev = [[CartesianIndex(mod1(Nx+2 - ci[1],Nx), mod1(Nx+2 - ci[2],Nx)) for ci in f_Nf] for f_Nf in f_ind]

    #CartesianIndex(17,17) .- f_ind
    # tmp arrays for all tmp[f_i] = arr[f_i] .* f_v opns
    zarr1 = zeros(ComplexF64, Nx, Nx)
    fz_fψ1 = zeros(ComplexF64, Nx, Nx)
    fterm_a = zeros(ComplexF64, Nx, Nx)
    fterm_ct1 = zeros(ComplexF64, Nx, Nx)
    fterm_ct2 = zeros(ComplexF64, Nx, Nx)
    #rterm_bt2 = zeros(ComplexF64, Nx, Nx)
    # make a FFTW "plan" for an array of the given size and type
    P = plan_ifft(im_fd_0)  # P is an operator, P*im is ifft(im)

    # Loop over filters
    for f1 = 1:Nf
        f_i1 = f_ind[f1]  # CartesianIndex list for filter1
        f_v1 = f_val[f1]  # Values for f_i1
        f_i1rev = f_ind_rev[f1]

        zarr1[f_i1] = f_v1 .* im_fd_0[f_i1] #F[z] #BUG: here????
        zarr1_rd = P*zarr1 #for frzi
        fz_fψ1[f_i1] = f_v1 .* zarr1[f_i1]
        fz_fψ1_rd = P*fz_fψ1
        dS1dp[f1, :,:] = 2 .* real.(fz_fψ1_rd) #real.(conv(I_λ, ψ_λ)) #Mod
        #CHECK; that this equals derivS1fast code

        #dS2 loop prep
        #ψ_λ1  = realspace_filter(Nx, f_i1, f_v1) #Slow
        #fcψ_λ1 = fft(conj.(ψ_λ1)) #Slow
        I_λ1 = sqrt.(abs2.(zarr1_rd))
        fI_λ1 = fft(I_λ1)
        rterm_bt1 = zarr1_rd./I_λ1 #Z_λ1/I_λ1_bar.
        rterm_bt2 = conj.(zarr1_rd)./I_λ1
        for f2 = 1:Nf
            f_i2 = f_ind[f2]  # CartesianIndex list for filter2
            f_v2 = f_val[f2]  # Values for f_i2

            fterm_a[f_i2] = fI_λ1[f_i2] .* (f_v2).^2 #fterm_a = F[I_λ1].F[ψ_λ]^2
            rterm_a = P*fterm_a                #Finv[fterm_a]
            fterm_bt1 = fft(rterm_a .* rterm_bt1) #F[Finv[fterm_a].*Z_λ1/I_λ1_bar] for T1
            fterm_bt2 = fft(rterm_a .* rterm_bt2) #F[Finv[fterm_a].*Z_λ1_bar/I_λ1_bar] for T2
            fterm_ct1[f_i1] = fterm_bt1[f_i1] .* f_v1    #fterm_b*F[ψ_λ]
            #fterm_ct2 = fterm_bt2 .* fcψ_λ1             #Slow
            #println(size(fterm_ct2[f_i1rev]),size(fterm_bt2[f_i1rev]),size(f_v1),size(conj.(f_v1)))
            fterm_ct2[f_i1rev] = fterm_bt2[f_i1rev] .* f_v1 #f_v1 is real
            #fterm_ct2slow = fterm_bt2 .* fcψ_λ1
            dS2dp[f1, f2, :, :] = real.(P*(fterm_ct1 + fterm_ct2)) #Mod THIS SHOULD be Nx*f2-1 + f1 or do the reshaping in loss func.

            #Reset f2 specific tmp arrays to 0
            fterm_a[f_i2] .= 0
            fterm_ct1[f_i1] .=0
            fterm_ct2[f_i1rev] .=0

        end
        # reset all reused variables to 0
        zarr1[f_i1] .= 0
        fz_fψ1[f_i1] .= 0


    end
    return vcat(dS1dp, reshape(dS2dp, (Nf^2, Nx, Nx))) #Mod
end

function wst_S1S2_deriv_sbori(image::Array{Float64,2}, filter_hash::Dict)
    #Works as is: some superfluous convolution and FFT opns. Working on fixing this.
    # Use 2 threads for FFT
    FFTW.set_num_threads(2)

    # array sizes
    (Nx, Ny)  = size(image)
    (Nf, )    = size(filter_hash["filt_index"])

    # allocate output array
    dS1dp  = zeros(Float64, Nx, Nx, Nf)
    dS2dp  = zeros(Float64, Nx, Nx, Nf, Nf)
    #debug

    # allocate image arrays for internal use
    im_rdc_0_1 = Array{ComplexF64, 3}(undef, Nx, Ny, Nf)  # real domain, complex
    im_rd_0_1  = Array{Float64, 3}(undef, Nx, Ny, Nf)  # real domain, complex

    # Not sure what to do here -- for gradients I don't think we want these
    ## 0th Order
    #S0[1]   = mean(image)
    #norm_im = image.-S0[1]
    #S0[2]   = sum(norm_im .* norm_im)/(Nx*Ny)
    #norm_im ./= sqrt(Nx*Ny*S0[2])

    ## 1st Order
    im_fd_0 = fft(image)  # total power=1.0

    # unpack filter_hash
    f_ind   = filter_hash["filt_index"]  # (J, L) array of filters represented as index value pairs
    f_val   = filter_hash["filt_value"]

    # tmp arrays for all tmp[f_i] = arr[f_i] .* f_v opns
    zarr1 = zeros(ComplexF64, Nx, Nx)
    fz_fψ1 = zeros(ComplexF64, Nx, Nx)
    fterm_a = zeros(ComplexF64, Nx, Nx)
    fterm_ct1 = zeros(ComplexF64, Nx, Nx)
    #rterm_bt2 = zeros(ComplexF64, Nx, Nx)
    # make a FFTW "plan" for an array of the given size and type
    P = plan_ifft(im_fd_0)  # P is an operator, P*im is ifft(im)

    # Loop over filters
    for f1 = 1:Nf
        f_i1 = f_ind[f1]  # CartesianIndex list for filter1
        f_v1 = f_val[f1]  # Values for f_i1

        zarr1[f_i1] = f_v1 .* im_fd_0[f_i1] #F[z]
        zarr1_rd = P*zarr1
        fz_fψ1[f_i1] = f_v1 .* zarr1[f_i1]
        fz_fψ1_rd = P*fz_fψ1
        dS1dp[:,:,f1] = 2 .* real.(fz_fψ1_rd) #real.(conv(I_λ, ψ_λ))
        #CHECK; that this equals derivS1fast code

        #dS2 loop prep
        ψ_λ1  = realspace_filter(Nx, f_i1, f_v1)
        fcψ_λ1 = fft(conj.(ψ_λ1))
        I_λ1 = sqrt.(abs2.(zarr1_rd)) #NM: Alloc+find faster way
        fI_λ1 = fft(I_λ1)
        rterm_bt1 = zarr1_rd./I_λ1 #Z_λ1/I_λ1_bar. Check that I_lambda_bar equals I_lambda. MUST be true.
        rterm_bt2 = conj.(zarr1_rd)./I_λ1
        for f2 = 1:Nf
            f_i2 = f_ind[f2]  # CartesianIndex list for filter2
            f_v2 = f_val[f2]  # Values for f_i2

            fterm_a[f_i2] = fI_λ1[f_i2] .* (f_v2).^2 #fterm_a = F[I_λ1].F[ψ_λ]^2
            rterm_a = P*fterm_a                #Finv[fterm_a]
            fterm_bt1 = fft(rterm_a .* rterm_bt1) #F[Finv[fterm_a].*Z_λ1/I_λ1_bar] for T1
            fterm_bt2 = fft(rterm_a .* rterm_bt2) #F[Finv[fterm_a].*Z_λ1_bar/I_λ1_bar] for T2
            fterm_ct1[f_i1] = fterm_bt1[f_i1] .* f_v1    #fterm_b*F[ψ_λ]
            fterm_ct2 = fterm_bt2 .* fcψ_λ1    #Slow version
            dS2dp[:, :, f1, f2] = real.((P*fterm_ct1) + (P*fterm_ct2)) #CHECK: Should this 2 not be there?? Why does it give an answer thats larger by 2x

            #Reset f2 specific tmp arrays to 0
            fterm_a[f_i2] .= 0
            fterm_ct1[f_i1] .=0

        end
        # reset all reused variables to 0
        zarr1[f_i1] .= 0
        fz_fψ1[f_i1] .= 0


    end
    return dS1dp, dS2dp
end


function derivtestS1(Nx)
   eps = 1e-4
   fhash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1)
   im = rand(Float64, Nx, Nx).*0.1
   im[6,6]=1.0
   im1 = copy(im)
   im0 = copy(im)
   im1[2,3] += eps/2
   im0[2,3] -= eps/2
   blarg = Deriv_Utils_New.wst_S1_deriv(im, fhash)
   der0=DHC_compute(im0,fhash,doS2=false)
   der1=DHC_compute(im1,fhash,doS2=false)
   dS = (der1-der0) ./ eps

   diff_old = dS[3:end]-blarg[2,3,:]

   println(dS[3:end])
   println("and")
   println(blarg[2,3,:])
   println("Mean abs | Mean abs frac", mean(abs.(diff_old)), mean(abs.(diff_old./dS)))
   println("stdev: ",std(diff_old))
   return
end

function derivtestS1S2(Nx) #MUST RUN DHC_COMPUTE WITH NORM=FALSE.
    eps = 1e-4
    fhash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1)
    im = rand(Float64, Nx, Nx).*0.1
    im[6,6]=1.0
    im1 = copy(im)
    im0 = copy(im)
    im1[2,3] += eps/2
    im0[2,3] -= eps/2
    dS1dp_existing = Deriv_Utils_New.wst_S1_deriv(im, fhash)
    dS1S2dp = Deriv_Utils_New.wst_S1S2_deriv(im, fhash) #Deriv_Utils_New.wst_S1S2_deriv(im, fhash)

    der0=DHC_compute(im0,fhash,doS2=true,doS20=false, norm=false)
    der1=DHC_compute(im1,fhash,doS2=true,doS20=false, norm=false)
    dSlim = (der1-der0) ./ eps
    Nf = length(fhash["filt_index"])
    lasts1ind = 2+Nf

    dS1lim23 = dSlim[3:lasts1ind]
    dS2lim23 = dSlim[lasts1ind+1:end]
    dS1dp_23 = dS1S2dp[2, 3, 1:Nf]
    dS2dp_23 = dS1S2dp[2, 3, Nf+1:end]
    #=
    println("Checking dS1dp using existing deriv")
    Nf = length(fhash["filt_index"])
    lasts1ind = 2+Nf
    diff = dS1dp_existing[2, 3, :]-dSlim[3:lasts1ind]
    println(dSlim[3:lasts1ind])
    println("and")
    println(dS1dp_existing[2, 3, :])
    println("stdev: ",std(diff))
    println("--------------------------")
    =#
    println("Checking dS1dp using dS1S2")
    derdiff = dS1dp_23 - dS1lim23
    #println(dS1lim23)
    #println("and")
    #println(dS1dp_23)
    txt= @sprintf("Range of S1 der0, der1: Max= %.3E, %.3E Min=%.3E, %.3E ",maximum(der0[3:lasts1ind]),maximum(der1[3:lasts1ind]),minimum(der0[3:lasts1ind]),minimum(der1[3:lasts1ind]))
    print(txt)
    txt = @sprintf("Checking dS1dp using S1S2 deriv, mean(abs(diff/dS1lim)): %.3E", mean(abs.(derdiff./dS1lim23)))
    print(txt)
    println("stdev: ",std(derdiff))
    txt= @sprintf("Range of dS1lim: Max= %.3E Min= %.3E",maximum(abs.(dS1lim23)),minimum(abs.(dS1lim23)))
    print(txt)
    println("--------------------------")
    println("Checking dS2dp using S1S2 deriv")
    Nf = length(fhash["filt_index"])
    println("Shape check",size(dS2dp_23),size(dS2lim23))
    derdiff = dS2dp_23 - dS2lim23 #Column major issue here?
    println("Difference",derdiff)
    txt = @sprintf("Range of S2 der0, der1: Max= %.3E, %.3E Min=%.3E, %.3E \n",maximum(der0[lasts1ind+1:end]),maximum(der1[lasts1ind+1:end]),minimum(der0[lasts1ind+1:end]),minimum(der1[lasts1ind+1:end]))
    print(txt)
    txt = @sprintf("Range of dS2lim: Max=%.3E  Min=%.3E \n",maximum(abs.(dS2lim23)),minimum(abs.(dS2lim23)))
    print(txt)
    txt = @sprintf("Difference between dS2dp and dSlim Mean: %.3E stdev: %.3E mean(abs(derdiff/dS2lim)): %.3E \n",mean(derdiff),std(derdiff), mean(abs.(derdiff./dS2lim23)))
    print(txt)

    println("Examining only those S2 coeffs which are greater than eps=1e-3 for der0")
    eps_mask = findall(der0[lasts1ind+1:end] .> 1E-3)
    der0_s2good = der0[lasts1ind+1:end][eps_mask]
    der1_s2good = der1[lasts1ind+1:end][eps_mask]
    dSlim_s2good = dS2lim23[eps_mask]
    derdiff_good = derdiff[eps_mask]
    print(derdiff_good)
    txt = @sprintf("\nRange of S2 der0, der1: Max= %.3E, %.3E Min=%.3E, %.3E \n",maximum(der0_s2good),maximum(der1_s2good),minimum(der0_s2good),minimum(der1_s2good))
    print(txt)
    txt = @sprintf("Range of dS2lim: Max=%.3E  Min=%.3E \n",maximum(abs.(dSlim_s2good)),minimum(abs.(dSlim_s2good)))
    print(txt)
    txt = @sprintf("Mean: %.3E stdev: %.3E mean(abs(derdiff/dS2lim)): %.3E \n",mean(derdiff_good),std(derdiff_good), mean(abs.(derdiff_good ./dSlim_s2good)))
    print(txt)
    return dS1dp_23, dS1lim23, reshape(dS2dp_23, (Nf, Nf)), reshape(dS2lim23, (Nf, Nf))
end

function derivtestS12(Nx)
    eps = 1e-4
    fhash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1)
    im = rand(Float64, Nx, Nx).*0.1
    im[6,6]=1.0
    im1 = copy(im)
    im0 = copy(im)
    im1[2,3] += eps/2
    im0[2,3] -= eps/2
    dS12dp = Deriv_Utils_New.wst_S12_deriv(im, fhash)

    der0=DHC_compute(im0, fhash, doS2=false, doS20=false, doS12=true, norm=false)
    der1=DHC_compute(im1, fhash, doS2=false, doS20=false, doS12=true, norm=false)
    dS = (der1-der0) ./ eps

    Nf = length(fhash["filt_index"])
    i0 = 3+Nf
    blarg = dS12dp[2, 3, :, :]
    diff = dS[i0:end]-reshape(blarg,Nf*Nf)
    println(dS[i0:end])
    println("and")
    println(blarg)
    println("stdev: ",std(diff))
    println(diff)
    println(size(diff), size(dS), size(dS12dp))
    txt = @sprintf("Mean: %.3E stdev: %.3E mean(abs(diff/dS12dp)): %.3E, mean(abs(dS12dp/dS)) : %.3E \n",mean(diff),std(diff), mean(abs.(diff ./dS12dp[2, 3, :, :][:])), mean(abs.(dS12dp[2, 3, :, :][:]/dS[i0:end])))
    print(txt)

    #plot(dS[3:end])
    #plot!(blarg[2,3,:])
    #plot(diff)
    return reshape(dS12dp[2, 3, :, :], (Nf, Nf)), reshape(dS[i0:end], (Nf, Nf))
end

ds1code, ds1lim, ds2code, ds2lim = derivtestS1S2(16)

ds12code, ds12lim = derivtestS12(16)
