module Deriv_Utils
    using Statistics
    using Plots
    using BenchmarkTools
    using Profile
    using FFTW
    using Statistics
    using Optim
    using Images, FileIO, ImageIO
    using Printf

    #using Profile
    using LinearAlgebra

    # put the cwd on the path so Julia can find the module
    push!(LOAD_PATH, pwd()*"/main")
    using DHC_2DUtils


    function wst_S1_deriv_fast(image::Array{Float64,2}, filter_hash::Dict)

        # Use 2 threads for FFT
        FFTW.set_num_threads(2)

        # array sizes
        (Nx, Ny)  = size(image)
        (Nf, )    = size(filter_hash["filt_index"])

        # allocate output array
        dSdp  = zeros(Float64, Nx, Nx, Nf)

        # allocate image arrays for internal use
        #im_rdc_0_1 = Array{ComplexF64, 3}(undef, Nx, Ny, Nf)  # real domain, complex

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

        zarr = zeros(ComplexF64, Nx, Nx)  # temporary array to fill with zvals

        # make a FFTW "plan" for an array of the given size and type
        P = plan_ifft(im_fd_0)  # P is an operator, P*im is ifft(im)

        # Loop over filters
        for f = 1:Nf
            f_i = f_ind[f]  # CartesianIndex list for filter
            f_v = f_val[f]  # Values for f_i

            zarr[f_i] = f_v .* im_fd_0[f_i]
            #Old: I_λ = P*zarr  # complex valued ifft of zarr, Should have been z_lambda in the latex conv
            #im_rdc_0_1[:,:,f] = I_lambda


            #xfac = 2 .* real(I_λ)
            #yfac = 2 .* imag(I_λ)
            # it is clearly stupid to transform back and forth, but we will need these steps for the S20 part
            #Old: ψ_λ  = realspace_filter(Nx, f_i, f_v)
            # convolution requires flipping wavelet direction, equivalent to flipping sign of imaginary part.
            # dSdp[:,:,f] = real.(conv(xfac,real(ψ_λ))) - real.(conv(yfac,imag(ψ_λ)))
            zarr[f_i] = f_v .* zarr[f_i] #Reusing zarr for F[conv(z_λ,ψ_λ)] = F[z_λ].*F[ψ_λ] = fz_fψ
            dSdp[:,:,f] = 2 .* real.(P*(zarr)) #real.(conv(I_λ, ψ_λ))
            zarr[f_i] .= 0   # reset zarr for next loop

        end
        return dSdp

    end

    function wst_S1S2_derivfast(image::Array{Float64,2}, filter_hash::Dict)
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

    function wst_S20_deriv(image::Array{Float64,2}, filter_hash::Dict, nthread::Int=1)

        # Use nthread threads for FFT -- but for Nx<512 nthread=1 is fastest.  Overhead?
        FFTW.set_num_threads(nthread)

        # array sizes
        (Nx, Ny)  = size(image)
        (Nf, )    = size(filter_hash["filt_index"])

        # allocate output array
        dS20dα  = zeros(Float64, Nx, Nx, Nf, Nf)

        # allocate image arrays for internal use
        im_rdc = Array{ComplexF64, 3}(undef, Nx, Ny, Nf)  # real domain, complex
        im_rd  = Array{Float64, 3}(undef, Nx, Ny, Nf)  # real domain, complex
        im_fd  = fft(image)

        # unpack filter_hash
        f_ind   = filter_hash["filt_index"]  # (J, L) array of filters represented as index value pairs
        f_val   = filter_hash["filt_value"]
        zarr = zeros(ComplexF64, Nx, Nx)  # temporary array to fill with zvals

        # make a FFTW "plan" a complex array, both forward and inverse transform
        P_fft  = plan_fft(im_fd)   # P_fft is an operator,  P_fft*im is fft(im)
        P_ifft = plan_ifft(im_fd)  # P_ifft is an operator, P_ifft*im is ifft(im)

        # Loop over filters
        for f = 1:Nf
            f_i = f_ind[f]  # CartesianIndex list for filter
            f_v = f_val[f]  # Values for f_i

            zarr[f_i] = f_v .* im_fd[f_i]
            Z_λ = P_ifft*zarr  # complex valued ifft of zarr
            zarr[f_i] .= 0   # reset zarr for next loop
            im_rdc[:,:,f] = Z_λ
            im_rd[:,:,f]  = abs.(Z_λ)
        end

        zarr = zeros(ComplexF64, Nx, Nx)  # temporary array to fill with zvals
        for f2 = 1:Nf
            f_i = f_ind[f2]  # CartesianIndex list for filter
            f_v = f_val[f2]  # Values for f_i
            uvec = im_rdc[:,:,f2] ./ im_rd[:,:,f2]
            for f1 = 1:Nf
                temp = P_fft*(im_rd[:,:,f1].*uvec)
                zarr[f_i] = f_v .* temp[f_i]

                Z1dZ2 = real.(P_ifft*zarr)
                #  It is possible to do this with rifft, but it is not much faster...
                #   Z1dZ2 = myrealifft(zarr)
                dS20dα[:,:,f1,f2] += Z1dZ2
                dS20dα[:,:,f2,f1] += Z1dZ2
                zarr[f_i] .= 0   # reset zarr for next loop
            end
        end
        return dS20dα
    end


    function wst_S12_deriv(image::Array{Float64,2}, filter_hash::Dict)
        FFTW.set_num_threads(2)

        # array sizes
        (Nx, Ny)  = size(image)
        (Nf, )    = size(filter_hash["filt_index"])


        # unpack filter_hash
        f_ind   = filter_hash["filt_index"]  # (J, L) array of filters represented as index value pairs
        f_val   = filter_hash["filt_value"]

        zarr = zeros(ComplexF64, Nf, Nx, Nx)  # temporary array to fill with zvals
        ## 1st Order
        im_fd_0 = fft(image)  # total power=1.0
        # make a FFTW "plan" for an array of the given size and type
        P = plan_ifft(im_fd_0)  # P is an operator, P*im is ifft(im)
        for f1 = 1:Nf
            f_i1 = f_ind[f1]  # CartesianIndex list for filter1
            f_v1 = f_val[f1]  # Values for f_i1
            zarr[f1, f_i1] = im_fd_0[f_i1] .* f_v1 #check if this notation is fine
        end
        zarrc =conj.(zarr)
        absz = abs.(zarr)


        #Implemented in the more memory intensive / parallelized ops way
        #fhash_indsarr = [hcat((x->[x[1], x[2]]).(fh)...)' for fh in fhash["filt_index"]] #Each elem gives you fhash["filt_index"] in non-CartesianIndices
        #dabsz = #Mem Req: Nf*Nx^2*sp*Nx^2 and sp~2%

        #Less Mem Route
        dabsz_absz = zeros(Float64, Nf, Nf, Nx, Nx)  # Lookup arrays for dS12 terms
        xigrid = reshape(CartesianIndices((1:Nx, 1:Nx)), (1, Nx^2))
        xigrid = hcat((x->[x[1], x[2]]).(xigrid)...) #2*Nx^2

        for f1 = 1:Nf
            f_i1 = f_ind[f1]  # CartesianIndex list for filter1
            f_v1 = f_val[f1]  # Values for f_i1

            for f2 = 1:Nf
                f_i2 = f_ind[f2]  # CartesianIndex list for filter1
                f_v2 = f_val[f2]  # Values for f_i1
                pnz = intersect(f_i1, f_i2) #Fd indices where both filters are nonzero
                if length(pnz)!=0
                    pnz_arr = vcat((x->[x[1], x[2]]').(pnz)...) #pnz*2
                    Φmat =  exp.((-2π*im)/Nx .* ((pnz_arr .- 1) * (xigrid .- 1))) #pnz*Nx^2
                    f_v1pnz = f_v1[findall(in(pnz), f_i1)]
                    t2 = real.(zarrc[f1, pnz] .* Φmat) #Check
                    #println(size(f_v1pnz), size(t2), size(absz[f2, pnz]))
                    term = sum(((absz[f2, pnz] ./ absz[f1, pnz]) .* f_v1pnz) .* t2, dims=1) #p*Nx^2 -> 1*Nx^2
                    dabsz_absz[f1, f2, :, :] = reshape(term, (Nx, Nx))
                end
            end
        end
        dS12 = dabsz_absz + permutedims(dabsz_absz, [2, 1, 3, 4])
        return dS12
    end


    function wst_S2_deriv_sum(image::Array{Float64,2}, filter_hash::Dict, wt::Array{Float64})
        FFTW.set_num_threads(2)

        # array sizes
        (Nx, Ny)  = size(image)
        (Nf, )    = size(filter_hash["filt_index"])

        if size(wt)!=(Nf, Nf) error("Wt has wrong shape") end

        f_ind   = filter_hash["filt_index"]  # (J, L) array of filters represented as index value pairs
        f_val   = filter_hash["filt_value"]
        f_ind_rev = [[CartesianIndex(mod1(Nx+2 - ci[1],Nx), mod1(Nx+2 - ci[2],Nx)) for ci in f_Nf] for f_Nf in f_ind]


        im_fd_0 = fft(image)
        P = plan_fft(im_fd_0)
        Pi = plan_ifft(im_fd_0)
        zarr1 = zeros(ComplexF64, Nx, Nx)
        Ilam1 = zeros(ComplexF64, Nx, Nx)
        fterm_bt1 = zeros(ComplexF64, Nx, Nx)
        fterm_bt2 = zeros(ComplexF64, Nx, Nx)
        ifterm = zeros(ComplexF64, Nx, Nx)
        dLossdα = zeros(Float64, Nx, Nx)
        for f1=1:Nf
            vlam1 = zeros(ComplexF64, Nx, Nx) #defined for each lambda1, computed by summing over lambda2
            f_i1 = f_ind[f1]
            f_v1 = f_val[f1]
            f_i1rev = f_ind_rev[f1]

            zarr1[f_i1] = im_fd_0[f_i1] .* f_v1
            zarr1_rd = Pi*(zarr1)
            Ilam1 = abs.(zarr1_rd)
            fIlam1 = P*Ilam1

            for f2=1:Nf
                f_i2 = f_ind[f2]
                f_v2 = f_val[f2]
                vlam1[f_i2] += wt[f1, f2] .* f_v2.^2
            end
            #vlam1= sum_λ2 w.ψl1,l2^2
            vlam1 = (Pi*(fIlam1 .* vlam1))./Ilam1 #sum over f2 -> (Nx, Nx) -> real space Vλ1/Iλ1
            fterm_bt1 = P*(vlam1 .* zarr1_rd)
            fterm_bt2 = P*(vlam1 .* conj.(zarr1_rd))
            ifterm[f_i1] = fterm_bt1[f_i1] .* f_v1
            ifterm[f_i1rev] += fterm_bt2[f_i1rev] .* f_v1
            dLossdα += real.(Pi*ifterm)

            #zero out alloc
            zarr1[f_i1] .= 0
            ifterm[f_i1] .=0
            ifterm[f_i1rev] .=0

        end
        return dLossdα

    end

    function wst_S1S2_derivsum_comb(image::Array{Float64,2}, filter_hash::Dict, wt::Array{Float64})
        # array sizes
        (Nx, Ny)  = size(image)
        (Nf, )    = size(filter_hash["filt_index"])

        if size(wt)!=(Nf + (Nf*Nf),) error("Wt has wrong shape: the argument should contain both S1 and S2 weights") end
        wts1 = reshape(wt[1:Nf], (Nf, 1))
        wts2 = reshape(wt[Nf+1:end], (Nf, Nf))

        dwS1 = reshape(DHC_2DUtils.wst_S1_deriv(image, filter_hash), (Nx^2, Nf)) * wts1 #NOTE: Replace this with DHC_2DUtils.wst_S1_deriv
        dwS2 = reshape(wst_S2_deriv_sum(image, filter_hash, wts2), (Nx^2, 1))
        #println(size(dwS1), size(dwS2))
        return reshape(dwS1 + dwS2, Nx^2)

    end

    #######Testing Utilities Begin Here##################
    function derivtestfast(Nx)
       eps = 1e-4
       fhash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1)
       im = rand(Float64, Nx, Nx).*0.1
       im[6,6]=1.0
       im1 = copy(im)
       im0 = copy(im)
       im1[2,3] += eps/2
       im0[2,3] -= eps/2
       blarg = wst_S1_deriv(im, fhash)
       blargfast = wst_S1_deriv_fast(im, fhash)

       der0=DHC(im0,fhash,doS2=false)
       der1=DHC(im1,fhash,doS2=false)
       dS = (der1-der0) ./ eps


       diff = dS[3:end]-blarg[2,3,:]
       println(dS[3:end])
       println("and")
       println(blarg[2,3,:])
       println("stdev: ",std(diff))
       println("New fewer conv ds1")
       println(dS[3:end])
       println("and")
       println(blargfast[2,3,:])
       println("stdev: ",std(diff))
       #=#Delete: blarg shouldnt be 0 for all other pixels since the coefficient isnt constant wrt other pixels!!
       #Check: that the derivative is 0 for all pixels?? THink about this... need to check against autodiff?
       Nf = length(fhash["filt_index"])
       mask = trues((Nx,Nx, Nf))
       println("Nf",Nf,"Check",(size(mask)==size(blargfast)))
       mask[2, 3, :] .= false
       println("Mean deriv wrt to other pixels (should be 0 for all)",mean(blargfast[mask]), "Maximum",maximum(blargfast[mask]))
       #For Nx=128, Mean is actually 0.002, the max is 0.10. Seems concerning since eps is 1e-4!?
       #For Nx=8, Mean is 0.01, max is 0.90 -- some kind of normalization issue?
       #plot(dS[3:end])
       #plot!(blarg[2,3,:])
       #plot(diff)=#
       return
   end


   function derivtestS1S2(Nx; mode="slow")
       eps = 1e-4
       fhash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1)
       im = rand(Float64, Nx, Nx).*0.1
       im[6,6]=1.0
       im1 = copy(im)
       im0 = copy(im)
       im1[2,3] += eps/2
       im0[2,3] -= eps/2
       dS1dp_existing = wst_S1_deriv(im, fhash)
       if mode=="slow"
           dS1dp, dS2dp = wst_S1S2_deriv(im, fhash)
       else
           dS1dp, dS2dp = wst_S1S2_derivfast(im, fhash)
       end

       der0=DHC(im0,fhash,doS2=true,doS20=false)
       der1=DHC(im1,fhash,doS2=true,doS20=false)
       dSlim = (der1-der0) ./ eps
       Nf = length(fhash["filt_index"])
       lasts1ind = 2+Nf

       dS1lim23 = dSlim[3:lasts1ind]
       dS2lim23 = dSlim[lasts1ind+1:end]
       dS1dp_23 = dS1dp[2, 3, :]
       dS2dp_23 = dS2dp[2, 3, :, :]
       #=
       println("Checking dS1dp using existing deriv")
       Nf = length(fhash["filt_index"])
       lasts1ind = 2+Nf
       diff = dS1dp_existing[2, 3, :]-dSlim[3:lasts1ind]
       println(dSlim[3:lasts1ind])
       println("and")
       println(dS1dp_existing[2, 3, :])
       println("stdev: ",std(diff))
       println("--------------------------")=#
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
       derdiff = reshape(dS2dp_23, Nf*Nf) - dS2lim23 #Column major issue here?
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

       #plot(dS[3:end])
       #plot!(blarg[2,3,:])
       #plot(diff)
       return
   end
end
