module Deriv_Utils_New
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
    using eqws




    #Jacobians of coeffs wrt image #######################################################

    function wst_S1_deriv(image::Array{Float64,2}, filter_hash::Dict; FFTthreads::Int=1)
        #=
        output: Nx, Nx, Nf
        =#
        # Use nthread threads for FFT -- but for Nx<512 nthread=1 is fastest.  Overhead?
        FFTW.set_num_threads(FFTthreads)

        # array sizes
        (Nx, Ny)  = size(image)
        (Nf, )    = size(filter_hash["filt_index"])

        # allocate output array
        dS1dα  = zeros(Float64, Nx, Nx, Nf)

        ## 1st Order
        im_fd = fft(image)

        # unpack filter_hash
        f_ind   = filter_hash["filt_index"]  # (J, L) array of filters represented as index value pairs
        f_val   = filter_hash["filt_value"]
        zarr = zeros(ComplexF64, Nx, Nx)  # temporary array to fill with zvals

        # make a FFTW "plan" for an array of the given size and type
        P = plan_ifft(im_fd)  # P is an operator, P*im is ifft(im)

        # Loop over filters
        for f = 1:Nf
            f_i = f_ind[f]  # CartesianIndex list for filter
            f_v = f_val[f]  # Values for f_i

            zarr[f_i] = (f_v.*f_v) .* im_fd[f_i]
            dS1dα[:,:,f] = 2 .* real.(P*zarr)

            zarr[f_i] .= 0   # reset zarr for next loop
        end
        return dS1dα
    end

    function wst_S20_deriv(image::Array{Float64,2}, filter_hash::Dict; FFTthreads::Int=1)
        #=
        output: Nx, Nx, Nf, Nf
        =#
        # Use nthread threads for FFT -- but for Nx<512 nthread=1 is fastest.  Overhead?
        FFTW.set_num_threads(FFTthreads)

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



    function wst_S12_deriv(image::Array{Float64,2}, filter_hash::Dict; FFTthreads::Int=1)
        #=
        output: Nx, Nx, Nf, Nf
        =#
        FFTW.set_num_threads(FFTthreads)

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
        #Calculate pnz's here and pre-allocate memory

        for f1 = 1:Nf
            f_i1 = f_ind[f1]  # CartesianIndex list for filter1
            f_v1 = f_val[f1]  # Values for f_i1

            for f2 = 1:Nf
                f_i2 = f_ind[f2]  # CartesianIndex list for filter1
                f_v2 = f_val[f2]  # Values for f_i1
                pnz = intersect(f_i1, f_i2) #Fd indices where both filters are nonzero
                if length(pnz)!=0
                    pnz_arr = vcat((x->[x[1], x[2]]').(pnz)...) #pnz*2
                    Φmat =  exp.((-2π*im)/Nx .* ((pnz_arr .- 1) * (xigrid .- 1))) #pnz*Nx^2, BN1
                    f_v1pnz = f_v1[findall(in(pnz), f_i1)]
                    t2 = real.(zarrc[f1, pnz] .* Φmat) #Check, BN3
                    #println(size(f_v1pnz), size(t2), size(absz[f2, pnz]))
                    term = sum(((absz[f2, pnz] ./ absz[f1, pnz]) .* f_v1pnz) .* t2, dims=1) #p*Nx^2 -> 1*Nx^2, BN2
                    dabsz_absz[f1, f2, :, :] = reshape(term, (Nx, Nx))
                end
            end
        end
        dS12 = dabsz_absz + permutedims(dabsz_absz, [2, 1, 3, 4])
        return permutedims(dS12, [3, 4, 1, 2]) #MODIFIED
    end
    #=
    function wst_S12_derivfast(image::Array{Float64,2}, filter_hash::Dict; FFTthreads::Int=1)
        #=
        output: Nx, Nx, Nf, Nf
        =#
        FFTW.set_num_threads(FFTthreads)

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
        #Calculate pnz's here and pre-allocate memory
        for f1  = 1:Nf
            f_i1 = f_ind[f1]  # CartesianIndex list for filter1
            f_v1 = f_val[f1]  # Values for f_i1
            for f2 = 1:Nf
                f_i2 = f_ind[f2]  # CartesianIndex list for filter1
                f_v2 = f_val[f2]  # Values for f_i1

        for f1 = 1:Nf
            f_i1 = f_ind[f1]  # CartesianIndex list for filter1
            f_v1 = f_val[f1]  # Values for f_i1

            for f2 = 1:Nf
                f_i2 = f_ind[f2]  # CartesianIndex list for filter1
                f_v2 = f_val[f2]  # Values for f_i1
                pnz = intersect(f_i1, f_i2) #Fd indices where both filters are nonzero
                if length(pnz)!=0
                    pnz_arr = vcat((x->[x[1], x[2]]').(pnz)...) #pnz*2
                    Φmat =  exp.((-2π*im)/Nx .* ((pnz_arr .- 1) * (xigrid .- 1))) #pnz*Nx^2, BN1
                    f_v1pnz = f_v1[findall(in(pnz), f_i1)]
                    t2 = real.(zarrc[f1, pnz] .* Φmat) #Check, BN3
                    #println(size(f_v1pnz), size(t2), size(absz[f2, pnz]))
                    term = sum(((absz[f2, pnz] ./ absz[f1, pnz]) .* f_v1pnz) .* t2, dims=1) #p*Nx^2 -> 1*Nx^2, BN2
                    dabsz_absz[f1, f2, :, :] = reshape(term, (Nx, Nx))
                end
            end
        end
        dS12 = dabsz_absz + permutedims(dabsz_absz, [2, 1, 3, 4])
        return permutedims(dS12, [3, 4, 1, 2]) #MODIFIED
    end
    =#
    function wst_S1S2_deriv(image::Array{Float64,2}, filter_hash::Dict; FFTthreads::Int=1)
        #=
        output: Nx, Nx, Nf+Nf^2
        =#
        # Use 2 threads for FFT
        FFTW.set_num_threads(FFTthreads)

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
        return permutedims(vcat(dS1dp, reshape(dS2dp, (Nf^2, Nx, Nx))), [2, 3, 1]) #Mod #MODIFIED
    end

    #Chi-Square Derivatives###############################################################
    function wst_S2_deriv_sum(image::Array{Float64,2}, filter_hash::Dict, wt::Array{Float64}; FFTthreads::Int=1)
        #=
        output:
        =#
        FFTW.set_num_threads(FFTthreads)

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
        return reshape(dLossdα, (Nx^2, 1))

    end

    function wst_S1S2_derivsum_comb(image::Array{Float64,2}, filter_hash::Dict, wt::Array{Float64}; FFTthreads::Int=1)
        FFTW.set_num_threads(FFTthreads)
        # array sizes
        (Nx, Ny)  = size(image)
        (Nf, )    = size(filter_hash["filt_index"])

        if size(wt)!=(Nf + (Nf*Nf),) error("Wt has wrong shape: the argument should contain both S1 and S2 weights") end
        wts1 = reshape(wt[1:Nf], (Nf, 1))
        wts2 = reshape(wt[Nf+1:end], (Nf, Nf))

        dwS1 = reshape(wst_S1_deriv(image, filter_hash), (Nx^2, Nf)) * wts1 #NOTE: Replace this with eqws_2DUtils.wst_S1_deriv
        dwS2 = reshape(wst_S2_deriv_sum(image, filter_hash, wts2), (Nx^2, 1))
        #println(size(dwS1), size(dwS2))
        return dwS1 + dwS2 #Why was another reshape needed here?

    end
    #=
    function wst_S20_deriv_sum_old(image::Array{Float64,2}, filter_hash::Dict, wt::Array{Float64}; FFTthreads::Int=1)
        # Sum over (f1,f2) filter pairs for S20 derivative.  This is much faster
        #   than calling wst_S20_deriv() because the sum can be moved inside the FFT.
        # Use FFTthreads threads for FFT -- but for Nx<512 FFTthreads=1 is fastest.  Overhead?
        # On Cascade Lake box, 4 is good for 2D, 8 or 16 for 3D FFTs
        FFTW.set_num_threads(FFTthreads)

        # array sizes
        (Nx, Ny)  = size(image)
        (Nf, )    = size(filter_hash["filt_index"])

        # allocate image arrays for internal use
        Uvec   = Array{ComplexF64, 3}(undef, Nx, Ny, Nf)  # real domain, complex
        im_rd  = Array{Float64, 3}(undef, Nx, Ny, Nf)  # real domain, complex
        im_fd  = fft(image) #Should this be the apodized image?

        # unpack filter_hash
        f_ind   = filter_hash["filt_index"]  # (J, L) array of filters represented as index value pairs
        f_val   = filter_hash["filt_value"]
        zarr    = zeros(ComplexF64, Nx, Nx)  # temporary array to fill with zvals

        # make a FFTW "plan" a complex array, both forward and inverse transform
        P_fft  = plan_fft(im_fd)   # P_fft is an operator,  P_fft*im is fft(im)
        P_ifft = plan_ifft(im_fd)  # P_ifft is an operator, P_ifft*im is ifft(im)

        # Loop over filters
        for f = 1:Nf
            f_i = f_ind[f]  # CartesianIndex list for filter
            f_v = f_val[f]  # Values for f_i

            zarr[f_i] = f_v .* im_fd[f_i]
            Z_λ = P_ifft*zarr  # complex valued ifft of zarr
            zarr[f_i] .= 0     # reset zarr for next loop
            im_rd[:,:,f] = abs.(Z_λ)
            Uvec[:,:,f]  = Z_λ ./ im_rd[:,:,f]
        end

        zarr = zeros(ComplexF64, Nx, Nx)  # temporary array to fill with zvals
        for f2 = 1:Nf
            f_i = f_ind[f2]  # CartesianIndex list for filter
            f_v = f_val[f2]  # Values for f_i

            Wtot = reshape( reshape(im_rd,Nx*Nx,Nf)*wt[:,f2], Nx, Nx) #SPEED: Preallocate mem for Wtot?
            temp = P_fft*(Wtot.*Uvec[:,:,f2])
            zarr[f_i] .+= f_v .* temp[f_i]
        end
        ΣdS20dα = 2* real.(P_ifft*zarr) #added here

        return reshape(ΣdS20dα, (Nx^2, 1))
    end
    =#

    function wst_S20_deriv_sum(image::Array{Float64,2}, filter_hash::Dict, wt::Array{Float64}; FFTthreads::Int=1)
        # Sum over (f1,f2) filter pairs for S20 derivative.  This is much faster
        #   than calling wst_S20_deriv() because the sum can be moved inside the FFT.
        # Use FFTthreads threads for FFT -- but for Nx<512 FFTthreads=1 is fastest.  Overhead?
        # On Cascade Lake box, 4 is good for 2D, 8 or 16 for 3D FFTs
        FFTW.set_num_threads(FFTthreads)

        # array sizes
        (Nx, Ny)  = size(image)
        (Nf, )    = size(filter_hash["filt_index"])

        # allocate image arrays for internal use
        Uvec   = Array{ComplexF64, 3}(undef, Nx, Ny, Nf)  # real domain, complex
        im_rd  = Array{Float64, 3}(undef, Nx, Ny, Nf)  # real domain, complex
        im_fd  = fft(image) #Should this be the apodized image?

        # unpack filter_hash
        f_ind   = filter_hash["filt_index"]  # (J, L) array of filters represented as index value pairs
        f_val   = filter_hash["filt_value"]
        zarr    = zeros(ComplexF64, Nx, Nx)  # temporary array to fill with zvals

        # make a FFTW "plan" a complex array, both forward and inverse transform
        P_fft  = plan_fft(im_fd)   # P_fft is an operator,  P_fft*im is fft(im)
        P_ifft = plan_ifft(im_fd)  # P_ifft is an operator, P_ifft*im is ifft(im)

        # Loop over filters
        for f = 1:Nf
            f_i = f_ind[f]  # CartesianIndex list for filter
            f_v = f_val[f]  # Values for f_i

            zarr[f_i] = f_v .* im_fd[f_i]
            Z_λ = P_ifft*zarr  # complex valued ifft of zarr
            zarr[f_i] .= 0     # reset zarr for next loop
            im_rd[:,:,f] = abs.(Z_λ)
            Uvec[:,:,f]  = Z_λ ./ im_rd[:,:,f]
        end

        zarr = zeros(ComplexF64, Nx, Nx)  # temporary array to fill with zvals
        for f2 = 1:Nf
            f_i = f_ind[f2]  # CartesianIndex list for filter
            f_v = f_val[f2]  # Values for f_i

            Wtot = reshape( reshape(im_rd,Nx*Nx,Nf)* (wt[:,f2] + wt'[:, f2]), Nx, Nx) #SPEED: Preallocate mem for Wtot?
            temp = P_fft*(Wtot.*Uvec[:,:,f2])
            zarr[f_i] .+= f_v .* temp[f_i]
        end
        ΣdS20dα = real.(P_ifft*zarr) #added here

        return reshape(ΣdS20dα, (Nx^2, 1))
    end

    function wst_S20_deriv_sum_norm(image::Array{Float64,2}, filter_hash::Dict, wt::Array{Float64}, std::Float64; FFTthreads::Int=1)
        # Sum over (f1,f2) filter pairs for S20 derivative.  This is much faster
        #   than calling wst_S20_deriv() because the sum can be moved inside the FFT.
        # Use FFTthreads threads for FFT -- but for Nx<512 FFTthreads=1 is fastest.  Overhead?
        # On Cascade Lake box, 4 is good for 2D, 8 or 16 for 3D FFTs
        FFTW.set_num_threads(FFTthreads)

        # array sizes
        (Nx, Ny)  = size(image)
        (Nf, )    = size(filter_hash["filt_index"])

        # allocate image arrays for internal use
        Uvec   = Array{ComplexF64, 3}(undef, Nx, Ny, Nf)  # real domain, complex
        im_rd  = Array{Float64, 3}(undef, Nx, Ny, Nf)  # real domain, complex
        im_fd  = fft(image) #Should this be the apodized image?

        # unpack filter_hash
        f_ind   = filter_hash["filt_index"]  # (J, L) array of filters represented as index value pairs
        f_val   = filter_hash["filt_value"]
        zarr    = zeros(ComplexF64, Nx, Nx)  # temporary array to fill with zvals

        # make a FFTW "plan" a complex array, both forward and inverse transform
        P_fft  = plan_fft(im_fd)   # P_fft is an operator,  P_fft*im is fft(im)
        P_ifft = plan_ifft(im_fd)  # P_ifft is an operator, P_ifft*im is ifft(im)

        # Loop over filters
        for f = 1:Nf
            f_i = f_ind[f]  # CartesianIndex list for filter
            f_v = f_val[f]  # Values for f_i

            zarr[f_i] = f_v .* im_fd[f_i]
            Z_λ = P_ifft*zarr  # complex valued ifft of zarr
            zarr[f_i] .= 0     # reset zarr for next loop
            im_rd[:,:,f] = abs.(Z_λ)
            Uvec[:,:,f]  = Z_λ ./ im_rd[:,:,f]
        end

        zarr = zeros(ComplexF64, Nx, Nx)  # temporary array to fill with zvals
        for f2 = 1:Nf
            f_i = f_ind[f2]  # CartesianIndex list for filter
            f_v = f_val[f2]  # Values for f_i

            Wtot = reshape(reshape(im_rd,Nx*Nx,Nf)* (wt[:,f2] + wt'[:, f2]), Nx, Nx) #SPEED: Preallocate mem for Wtot?
            temp = P_fft*(Wtot.*Uvec[:,:,f2])
            zarr[f_i] .+= f_v .* temp[f_i]
        end
        ΣdS20dα = real.(P_ifft*zarr).*(1.0/(Nx*std)) #added here

        return reshape(ΣdS20dα, (Nx^2, 1))
    end

    #Image Reconstruction Funcs##########################################################
    #=
    function image_recon_derivsum(input::Array{Float64, 2}, filter_hash::Dict, s_targ_mean::Array{Float64, 1}, s_targ_invcov::Array{Float64, 2}, pixmask::BitArray{2}, coeff_choice;
        FFTthreads::Int=1, optim_settings=Dict([("iterations", 10)]), coeff_mask=nothing) #add iso here and a check that returns error if both coeffmask is not nothing and iso is present.
        #=
        Cases to be handled by this function:
        S2 | S20 | S12
        Select pixels masked or all floating
        Select coefficients optimized wrt

        Cases NOT handled by this function:
        Iso
        Only S1
        No Regularization
        Non-Gaussian loss function
        Using the S0 params (mean and variance) in the loss function
        eqws_compute with norm=true
        Log
        Apodizing
        =#
        (Nf,) = size(filter_hash["filt_index"])
        println("Coeff mask:", (coeff_mask!=nothing))

        (Nx, Ny)  = size(input)
        if Nx != Ny error("Input image must be square") end

        if Nf == 0 error("filter hash corrupted") end
        if coeff_choice=="S2"
            println("S2")
            pixmask = pixmask[:] #flattened: Nx^2s

            if coeff_mask!=nothing
                if length(coeff_mask)!= (2+Nf + Nf^2) error("Wrong dim mask") end
                if size(s_targ_mean)[1]!=count((i->(i==true)), coeff_mask) error("s_targ_mean should only contain coeffs to be optimized") end
                if size(s_targ_invcov)[1]!=count((i->(i==true)), coeff_mask) error("s_targ_invcov should only have coeffs to be optimized") end
            else #No mask: all coeffs (default: All S1 and S2 coeffs will be optim wrt. Mean, var not incl)
                #Currently assuming inputs have mean, var params in case that's something we wanna optimize at some point
                if size(s_targ_mean)[1]!=(Nf+Nf^2) error("s_targ_mean should only contain coeffs to be optimized") end
                if (size(s_targ_invcov)!=(Nf+Nf^2, Nf+Nf^2))  error("s_targ_invcov of wrong size") end #(s_targ_invcov!=I) &
                #At this point both have dims |S1+S2|
                #Create coeff_mask subsetting 3:end
                coeff_mask = fill(true, 2+Nf+Nf^2)
                coeff_mask[1] = false
                coeff_mask[2] = false
            end

            num_freecoeff = count((i->(i==true)), coeff_mask) #Nf+Nf^2 if argument coeff_mask was empty
            num_freecoeffS1 = count((i->(i==true)), coeff_mask[3:Nf+2])
            num_freecoeffS2 = count((i->(i==true)), coeff_mask[Nf+3:end])

            #After this all cases have a coeffmask, and s_targ_mean and s_targ_invcov have the shapes of the coefficients that we want to select.

            #Optimization Hyperparameters
            norm=false
            numitns_dict = get(optim_settings, "iterations", 100)


            function loss_func(img_curr)
                #size(img_curr) must be (Nx^2, 1)
                s_curr = eqws_compute(reshape(img_curr, (Nx, Nx)),  filter_hash, doS2=true, doS12=false, doS20=false, norm=tonorm)[coeff_mask]
                neglogloss = 0.5 .* (s_curr - s_targ_mean)' * s_targ_invcov * (s_curr - s_targ_mean)
                return neglogloss[1]
            end

            function dloss(storage_grad, img_curr)
                #storage_grad, img_curr must be (Nx^2, 1)
                s_curr = eqws_compute(reshape(img_curr, (Nx, Nx)), filter_hash, doS2=true, doS12=false, doS20=false, norm=tonorm)[coeff_mask]
                diff = s_curr - s_targ_mean
                wts1s2 = zeros(Nf+Nf^2)
                wtall = reshape(diff' * s_targ_invcov, num_freecoeff)
                wts1s2[coeff_mask[3:end]] .= wtall
                dterm = wst_S1S2_derivsum_comb(reshape(img_curr, (Nx, Nx)), filter_hash, wts1s2, FFTthreads=FFTthreads)
                storage_grad .= reshape(dterm, (Nx^2, 1))
                storage_grad[pixmask, 1] .= 0 # better way to do this by taking pixmask as an argument wst_s2_deriv_sum?
                return
            end

            #Debugging stuff
            println("Diff check")
            eps = zeros(size(input))
            eps[1, 2] = 1e-4
            chisq1 = loss_func(input+eps./2)
            chisq0 = loss_func(input-eps./2)
            brute  = (chisq1-chisq0)/1e-4
            #df_brute = eqws_compute(reshape(input, (Nx, Nx)), filter_hash, doS2=true, doS12=false, doS20=false, norm=false)[coeff_mask] - s_targ_mean
            clever = reshape(zeros(size(input)), (Nx*Nx, 1))
            _bar = dloss(clever, reshape(input, (Nx^2, 1)))
            #println("dS1S2comb check")
            #wts1s2 = zeros(Nf+Nf^2)
            #wtall = reshape(df' * s_targ_invcov, num_freecoeff)
            #wts1s2[coeff_mask[3:end]] .= wtall
            #dterm = Deriv_Utils.wst_S1S2_derivsum_comb(reshape(input, (Nx, Nx)), filter_hash, wts1s2)
            #dS1S2comb = Deriv_Utils.wst_S1S2_derivfast(reshape(input, (Nx, Nx)), filter_hash)
            #dS1dp = permutedims(dS1S2comb[1:Nf, :, :], [2, 3, 1])
            #dS2dp = permutedims(dS1S2comb[Nf+1:end, :, :], [2, 3, 1])
            #dS2dp = reshape(dS2dp, Nx, Nx, Nf^2)
            #sum2 = zeros(Float64, Nx, Nx)
            #for i=1:Nf*Nf sum2 += (dS2dp[:,:,i].*wts1s2[Nf+i]) end
            #term1 = reshape(dS1dp, (Nx^2, Nf)) * reshape(wts1s2[1:Nf], (Nf, 1))
            #dterm_brute = term1 + reshape(sum2, (Nx^2, 1))
            println("Chisq Derve Check")
            println("Brute:  ",brute)
            println("Clever: ",clever[Nx*(1)+1])


            res = optimize(loss_func, dloss, reshape(input, (Nx^2, 1)), ConjugateGradient(), Optim.Options(iterations = numitns_dict, store_trace = true, show_trace = true))
            result_img = Optim.minimizer(res)
        elseif coeff_choice=="S20"
            println("S20")
            pixmask = pixmask[:] #flattened: Nx^2s

            if coeff_mask!=nothing
                if count((i->i==true), coeff_mask[1:Nf+2])!=0 println("Warning: S1 coeffs being double counted / S0 aren't 0") end #wrap
                if length(coeff_mask)!= (2+Nf + Nf^2) error("Wrong dim mask") end #wrap
                if size(s_targ_mean)[1]!=count((i->(i==true)), coeff_mask) error("s_targ_mean should only contain coeffs to be optimized") end
                if size(s_targ_invcov)[1]!=count((i->(i==true)), coeff_mask) error("s_targ_invcov should only have coeffs to be optimized") end
            else #No mask: all coeffs (default: Only S20 coeffs will be optim wrt.)
                #Currently assuming inputs have mean, var params in case that's something we wanna optimize at some point
                if size(s_targ_mean)[1]!=(Nf^2) error("s_targ_mean should only contain coeffs to be optimized") end
                if (size(s_targ_invcov)!=(Nf^2, Nf^2))  error("s_targ_invcov of wrong size") end #(s_targ_invcov!=I) &
                #At this point both have dims |S1+S2|
                #Create coeff_mask subsetting 3:end
                coeff_mask = fill(true, 2+Nf+Nf^2)
                coeff_mask[1:Nf+2] .= false #Mask out S0 and S1
            end

            num_freecoeff = count((i->(i==true)), coeff_mask) #Nf^2 if argument coeff_mask was empty

            #After this all cases have a coeffmask, and s_targ_mean and s_targ_invcov have the shapes of the coefficients that we want to select.

            #Optimization Hyperparameters
            tonorm = get(optim_settings, "norm", false)
            numitns_dict = get(optim_settings, "iterations", 100)

            function loss_func20(img_curr)
                #size(img_curr) must be (Nx^2, 1)
                s_curr = eqws_compute(reshape(img_curr, (Nx, Nx)),  filter_hash, doS2=false, doS12=false, doS20=true, norm=tonorm)[coeff_mask]
                neglogloss = 0.5 .* (s_curr - s_targ_mean)' * s_targ_invcov * (s_curr - s_targ_mean)
                return neglogloss[1]
            end

            function dloss20(storage_grad, img_curr)
                #storage_grad, img_curr must be (Nx^2, 1)
                s_curr = eqws_compute(reshape(img_curr, (Nx, Nx)), filter_hash, doS2=false, doS12=false, doS20=true, norm=tonorm)[coeff_mask]
                diff = s_curr - s_targ_mean
                wt = reshape(convert(Array{Float64, 2}, transpose(diff) * s_targ_invcov), (Nf, Nf))
                dterm = wst_S20_deriv_sum(reshape(img_curr, (Nx, Nx)), filter_hash, wt, FFTthreads=FFTthreads)
                storage_grad .= reshape(dterm, (Nx^2, 1))
                storage_grad[pixmask, 1] .= 0 # better way to do this by taking pixmask as an argument wst_s2_deriv_sum?
                return
            end

            #Debugging stuff
            println("Diff check")
            eps = zeros(size(input))
            eps[1, 2] = 1e-4
            chisq1 = loss_func20(input+eps./2)
            chisq0 = loss_func20(input-eps./2)
            brute  = (chisq1-chisq0)/1e-4
            #df_brute = eqws_compute(reshape(input, (Nx, Nx)), filter_hash, doS2=true, doS12=false, doS20=false, norm=false)[coeff_mask] - s_targ_mean
            clever = reshape(zeros(size(input)), (Nx*Nx, 1))
            _bar = dloss20(clever, reshape(input, (Nx^2, 1)))
            println("Chisq Derve Check")
            println("Brute:  ",brute)
            println("Clever: ",clever[Nx*(1)+1])

            res = optimize(loss_func20, dloss20, reshape(input, (Nx^2, 1)), ConjugateGradient(), Optim.Options(iterations = numitns_dict, store_trace = true, show_trace = true))
            result_img = Optim.minimizer(res)
        else
             if coeff_choice!="S12" error("Invalid coeff_choice") end
             println("S12")
             pixmask = pixmask[:] #flattened: Nx^2s

             if coeff_mask!=nothing
                 num_freecoeff = count((i->(i==true)), coeff_mask)
                 println("Number of coefficients being optimized= ", num_freecoeff)
                 if count((i->i==true), coeff_mask[1:Nf+2])!=0 println("Warning: S1 coeffs being double counted / S0 aren't 0") end #wrap
                 if length(coeff_mask)!= (2+Nf + Nf^2) error("Wrong dim mask") end #wrap
                 if size(s_targ_mean)[1]!=count((i->(i==true)), coeff_mask) error("s_targ_mean should only contain coeffs to be optimized") end
                 if size(s_targ_invcov)[1]!=count((i->(i==true)), coeff_mask) error("s_targ_invcov should only have coeffs to be optimized") end
             else #No mask: all coeffs (default: Only S20 coeffs will be optim wrt.)
                 error("S12 must be run with the coeff_mask argument")
                 #=
                 if size(s_targ_mean)[1]!=(Nf^2) error("s_targ_mean should only contain coeffs to be optimized") end
                 if (size(s_targ_invcov)!=(Nf^2, Nf^2))  error("s_targ_invcov of wrong size") end #(s_targ_invcov!=I) &
                 #At this point both have dims |S1+S2|
                 #Create coeff_mask subsetting 3:end
                 coeff_mask = fill(true, 2+Nf+Nf^2)
                 coeff_mask[1:Nf+2] .= false #Mask out S0 and S1 =#
             end

             #After this all cases have a coeffmask, and s_targ_mean and s_targ_invcov have the shapes of the coefficients that we want to select.

             #Optimization Hyperparameters
             tonorm = get(optim_settings, "norm", false)
             numitns_dict = get(optim_settings, "iterations", 100)
             coeffmask_S12 = coeff_mask[Nf+3:end]

             function loss_func12(img_curr)
                 #size(img_curr) must be (Nx^2, 1)
                 s_curr = eqws_compute(reshape(img_curr, (Nx, Nx)),  filter_hash, doS2=false, doS12=true, doS20=false, norm=tonorm)[coeff_mask]
                 neglogloss = 0.5 .* (s_curr - s_targ_mean)' * s_targ_invcov * (s_curr - s_targ_mean)
                 return neglogloss[1]
             end

             function dloss12(storage_grad, img_curr)
                 #storage_grad, img_curr must be (Nx^2, 1)
                 s_curr = eqws_compute(reshape(img_curr, (Nx, Nx)), filter_hash, doS2=false, doS12=true, doS20=false, norm=tonorm)[coeff_mask]
                 diff = s_curr - s_targ_mean
                 wt = transpose(diff) * s_targ_invcov
                 storage_grad .= reshape(wst_S12_deriv(reshape(img_curr, (Nx, Nx)), filter_hash, FFTthreads=FFTthreads), (Nx^2, Nf^2))[:, coeffmask_S12] * transpose(wt) #->(nx^2, 1)
                 storage_grad[pixmask, 1] .= 0 # better way to do this by taking pixmask as an argument wst_s2_deriv_sum?
                 return
             end

             #Debugging stuff
             println("Diff check")
             eps = zeros(size(input))
             eps[1, 2] = 1e-4
             chisq1 = loss_func12(input+eps./2)
             chisq0 = loss_func12(input-eps./2)
             brute  = (chisq1-chisq0)/1e-4
             #df_brute = eqws_compute(reshape(input, (Nx, Nx)), filter_hash, doS2=true, doS12=false, doS20=false, norm=false)[coeff_mask] - s_targ_mean
             clever = reshape(zeros(size(input)), (Nx*Nx, 1))
             _bar = dloss12(clever, reshape(input, (Nx^2, 1)))
             println("Chisq Derve Check")
             println("Brute:  ",brute)
             println("Clever: ",clever[Nx*(1)+1])

             res = optimize(loss_func12, dloss12, reshape(input, (Nx^2, 1)), ConjugateGradient(), Optim.Options(iterations = numitns_dict, store_trace = true, show_trace = true))
             result_img = Optim.minimizer(res)

        end
        return reshape(result_img, (Nx, Nx))
    end
    =#
    function image_recon_derivsum_regularized(input::Array{Float64, 2}, filter_hash::Dict, s_targ_mean::Array{Float64, 1}, s_targ_invcov, pixmask::BitArray{2}, dhc_args;
        FFTthreads::Int=1, optim_settings=Dict([("iterations", 10)]), coeff_mask::BitArray{1}=nothing, lambda=0.001,show_trace0::Bool=true) #add iso here and a check that returns error if both coeffmask is not nothing and iso is present.
        #=
        Cases to be handled by this function:
        S20
        Apodized or not
        Iso
        Log or Regular: If log then input, targ and cov should have been constructed using log
        Select pixels masked or all floating--not using for masked currently
        Select coefficients optimized wrt
        Lambda: regularization

        Cases NOT handled by this function:
        Only S1
        Non-Gaussian loss function
        Using the S0 params (mean and variance) in the loss function
        eqws_compute with norm=true
        =#
        (N1iso, Nf) = size(filter_hash["S1_iso_mat"])
        println("Coeff mask:", (coeff_mask!=nothing))

        (Nx, Ny)  = size(input)
        if Nx != Ny error("Input image must be square") end
        if !dhc_args[:doS20] error("Not implemented") end
        if Nf == 0 error("filter hash corrupted") end

        println("S20")
        #pixmask = pixmask[:] #flattened: Nx^2s #DEB
        ori_input = reshape(copy(input), (Nx^2, 1))

        if coeff_mask!=nothing
            if (!dhc_args[:iso]) & (count((i->i==true), coeff_mask[1:Nf+2])!=0) error("Code assumes S1 coeffs and S0 are masked out") end #wrap
            if (dhc_args[:iso]) & (count((i->i==true), coeff_mask[1:N1iso+2])!=0) error("Code assumes S1iso coeffs and S0 are masked out") end
            if size(s_targ_mean)[1]!=count((i->(i==true)), coeff_mask) error("s_targ_mean should only contain coeffs to be optimized") end
            if size(s_targ_invcov)[1]!=count((i->(i==true)), coeff_mask) error("s_targ_invcov should only have coeffs to be optimized") end
        else #No mask: all coeffs (default: Only S20 coeffs will be optim wrt.)
            error("You must supply a coeff_mask")
        end

        #Apodization adjustments
        function adaptive_apodizer(input_image::Array{Float64, 2})
            if dhc_args[:apodize]
                apd_image = apodizer(input_image)
            else
                apd_image = input_image
            end
            return apd_image
        end
        #For the Jacobian dA*P/dP
        if dhc_args[:apodize]
            Ap = wind_2d(Nx)
            cA = sum(Ap)
            Apflat = reshape(Ap, Nx^2)
            od_nz_idx = findall(!iszero, Apflat) #findall((x->((x!=0) & (x!=1))), Apflat)
            #od_zidx = findall(iszero, Apflat)
            avals = Apflat[od_nz_idx]
            dA = zeros(Nx^2, Nx^2)
            dA[:, od_nz_idx] .= ((1.0 .- Apflat) * avals')./cA
            dA[diagind(dA)] += Apflat
        else
            dA = I
        end


        num_freecoeff = count((i->(i==true)), coeff_mask)

        #After this all cases have a coeffmask, and s_targ_mean and s_targ_invcov have the shapes of the coefficients that we want to select.
        #Optimization Hyperparameters
        tonorm = false
        numitns_dict = get(optim_settings, "iterations", 100)
        minmethod = get(optim_settings, "minmethod", ConjugateGradient())

        if (dhc_args[:doS20]) & (dhc_args[:iso])
            iso2nf2mask = zeros(Int64, Nf^2)
            M20 = filter_hash["S2_iso_mat"]
            for id=1:Nf^2 iso2nf2mask[M20.colptr[id]] = M20.rowval[id] end
            coeff_masks20 = coeff_mask[3+N1iso:end]
        else# (dhc_args[:doS20]) & (!dhc_args[:iso])
            coeff_masks20 = coeff_mask[3+Nf:end]
        end

        function augment_weights(inpwt::Array{Float64})
            if (dhc_args[:doS20]) & (dhc_args[:iso]) #wt (Nc) -> |S2_iso| -> |Nf^2|
                w_s2iso = zeros(size(filter_hash["S2_iso_mat"])[1])
                w_s2iso[coeff_masks20] .= inpwt
                w_nf2 = zeros(Nf^2)
                w_nf2 .= w_s2iso[iso2nf2mask]
                return reshape(w_nf2, (Nf, Nf))
            else#if (dhc_args[:doS20]) & (!dhc_args[:iso])
                w_nf2 = zeros(Nf^2)
                w_nf2[coeff_masks20] .= inpwt
                return reshape(w_nf2, (Nf, Nf))
            end
        end

        function loss_func20(img_curr::Array{Float64, 2})
            s_curr = eqws_compute_wrapper(img_curr, filter_hash, norm=tonorm; dhc_args...)[coeff_mask]
            regterm =  0.5*lambda*sum((adaptive_apodizer(img_curr) - adaptive_apodizer(input)).^2)
            lnlik = ( 0.5 .* (s_curr - s_targ_mean)' * s_targ_invcov * (s_curr - s_targ_mean))
            neglogloss = lnlik[1] + regterm
            return neglogloss
        end
        #TODO: Need to replace the deriv_sums20 with a deriv_sum wrapper that handles all cases (S12, S20, S2)

        function dloss20(storage_grad::Array{Float64, 2}, img_curr::Array{Float64, 2})
            s_curr = eqws_compute_wrapper(img_curr, filter_hash, norm=tonorm; dhc_args...)[coeff_mask]
            diff = s_curr - s_targ_mean
            #Add branches here
            wt = augment_weights(convert(Array{Float64,1}, reshape(transpose(diff) * s_targ_invcov, (length(diff),))))
            apdimg_curr = adaptive_apodizer(img_curr)
            storage_grad .= reshape((wst_S20_deriv_sum(apdimg_curr, filter_hash, wt, FFTthreads=FFTthreads)' + reshape(lambda.*(apdimg_curr - adaptive_apodizer(input)), (1, Nx^2))) * dA, (Nx, Nx))
            storage_grad[pixmask] .= 0 # better way to do this by taking pixmask as an argument wst_s2_deriv_sum?
            return
        end


        println("Diff check")
        eps = zeros(size(input))
        row, col = 24, 18 #convert(Int8, Nx/2), convert(Int8, Nx/2)+3
        epsmag = 1e-4
        eps[row, col] = epsmag
        chisq1 = loss_func20(input+eps./2) #DEB
        chisq0 = loss_func20(input-eps./2) #DEB
        brute  = (chisq1-chisq0)/epsmag
        #df_brute = eqws_compute(reshape(input, (Nx, Nx)), filter_hash, doS2=true, doS12=false, doS20=false, norm=false)[coeff_mask] - s_targ_mean
        clever = zeros(size(input)) #DEB
        meanval = mean(adaptive_apodizer(input)) ./ mean(wind_2d(Nx))
        _bar = dloss20(clever, input) #DEB
        println("Chisq Derve Check")
        println("Brute:  ",brute)
        println("Clever: ",clever[row, col], " Difference: ", brute - clever[row, col], " Mean ", meanval) #DEB
        println("Initial Loss ", loss_func20(input))
        res = optimize(loss_func20, dloss20, input, minmethod, Optim.Options(iterations = numitns_dict, store_trace = show_trace0, show_trace = show_trace0))
        result_img = Optim.minimizer(res)
        println("Final Loss ", loss_func20(result_img))
        return res, reshape(result_img, (Nx, Nx))
    end

    function image_recon_derivsum_regularized_S2(input::Array{Float64, 2}, filter_hash::Dict, s_targ_mean::Array{Float64, 1}, s_targ_invcov, pixmask::BitArray{2}, dhc_args;
        FFTthreads::Int=1, optim_settings=Dict([("iterations", 10)]), coeff_mask=nothing, lambda=0.001,show_trace0::Bool=true) #add iso here and a check that returns error if both coeffmask is not nothing and iso is present.
        #=
        Cases to be handled by this function:
        S20
        Apodized or not
        Iso
        Log or Regular: If log then input, targ and cov should have been constructed using log
        Select pixels masked or all floating--not using for masked currently
        Select coefficients optimized wrt
        Lambda: regularization

        Cases NOT handled by this function:
        Only S1
        Non-Gaussian loss function
        Using the S0 params (mean and variance) in the loss function
        eqws_compute with norm=true
        =#
        (N1iso, Nf) = size(filter_hash["S1_iso_mat"])
        println("Coeff mask:", (coeff_mask!=nothing))

        (Nx, Ny)  = size(input)
        if Nx != Ny error("Input image must be square") end
        if !dhc_args[:doS2] error("Not implemented") end
        if Nf == 0 error("filter hash corrupted") end

        println("S2")
        #pixmask = pixmask[:] #flattened: Nx^2s #DEB
        ori_input = reshape(copy(input), (Nx^2, 1))

        if coeff_mask!=nothing
            if (!dhc_args[:iso]) & (count((i->i==true), coeff_mask[1:Nf+2])!=0) error("Code assumes S1 coeffs and S0 are masked out") end #wrap
            if (dhc_args[:iso]) & (count((i->i==true), coeff_mask[1:N1iso+2])!=0) error("Code assumes S1iso coeffs and S0 are masked out") end
            if size(s_targ_mean)[1]!=count((i->(i==true)), coeff_mask) error("s_targ_mean should only contain coeffs to be optimized") end
            if size(s_targ_invcov)[1]!=count((i->(i==true)), coeff_mask) error("s_targ_invcov should only have coeffs to be optimized") end
        else #No mask: all coeffs (default: Only S20 coeffs will be optim wrt.)
            error("You must supply a coeff_mask")
        end

        #Apodization adjustments
        function adaptive_apodizer(input_image::Array{Float64, 2})
            if dhc_args[:apodize]
                apd_image = apodizer(input_image)
            else
                apd_image = input_image
            end
            return apd_image
        end
        #For the Jacobian dA*P/dP
        if dhc_args[:apodize]
            Ap = wind_2d(Nx)
            cA = sum(Ap)
            Apflat = reshape(Ap, Nx^2)
            od_nz_idx = findall(!iszero, Apflat) #findall((x->((x!=0) & (x!=1))), Apflat)
            #od_zidx = findall(iszero, Apflat)
            avals = Apflat[od_nz_idx]
            dA = zeros(Nx^2, Nx^2)
            dA[:, od_nz_idx] .= ((1.0 .- Apflat) * avals')./cA
            dA[diagind(dA)] += Apflat
        else
            dA = I
        end


        num_freecoeff = count((i->(i==true)), coeff_mask)

        #After this all cases have a coeffmask, and s_targ_mean and s_targ_invcov have the shapes of the coefficients that we want to select.
        #Optimization Hyperparameters
        tonorm = false
        numitns_dict = get(optim_settings, "iterations", 100)
        minmethod = get(optim_settings, "minmethod", ConjugateGradient())

        if (dhc_args[:doS2]) & (dhc_args[:iso])
            iso2nf2mask = zeros(Int64, Nf^2)
            M20 = filter_hash["S2_iso_mat"]
            for id=1:Nf^2 iso2nf2mask[M20.colptr[id]] = M20.rowval[id] end
            coeff_masks20 = coeff_mask[3+N1iso:end]
        else# (dhc_args[:doS20]) & (!dhc_args[:iso])
            coeff_masks20 = coeff_mask[3+Nf:end]
        end

        function augment_weights(inpwt::Array{Float64})
            if (dhc_args[:doS2]) & (dhc_args[:iso]) #wt (Nc) -> |S2_iso| -> |Nf^2|
                w_s2iso = zeros(size(filter_hash["S2_iso_mat"])[1])
                w_s2iso[coeff_masks20] .= inpwt
                w_nf2 = zeros(Nf^2)
                w_nf2 .= w_s2iso[iso2nf2mask]
                return reshape(w_nf2, (Nf, Nf))
            else#if (dhc_args[:doS20]) & (!dhc_args[:iso])
                w_nf2 = zeros(Nf^2)
                w_nf2[coeff_masks20] .= inpwt
                return reshape(w_nf2, (Nf, Nf))
            end
        end

        function loss_func2(img_curr::Array{Float64, 2})
            s_curr = eqws_compute_wrapper(img_curr, filter_hash, norm=tonorm; dhc_args...)[coeff_mask]
            regterm =  0.5*lambda*sum((adaptive_apodizer(img_curr) - adaptive_apodizer(input)).^2)
            lnlik = ( 0.5 .* (s_curr - s_targ_mean)' * s_targ_invcov * (s_curr - s_targ_mean))
            neglogloss = lnlik[1] + regterm
            return neglogloss
        end
        #TODO: Need to replace the deriv_sums20 with a deriv_sum wrapper that handles all cases (S12, S20, S2)

        function dloss2(storage_grad::Array{Float64, 2}, img_curr::Array{Float64, 2})
            s_curr = eqws_compute_wrapper(img_curr, filter_hash, norm=tonorm; dhc_args...)[coeff_mask]
            diff = s_curr - s_targ_mean
            #Add branches here
            wt = augment_weights(convert(Array{Float64,1}, reshape(transpose(diff) * s_targ_invcov, (length(diff),))))
            apdimg_curr = adaptive_apodizer(img_curr)
            storage_grad .= reshape((wst_S2_deriv_sum(apdimg_curr, filter_hash, wt, FFTthreads=FFTthreads)' + reshape(lambda.*(apdimg_curr - adaptive_apodizer(input)), (1, Nx^2))) * dA, (Nx, Nx))
            storage_grad[pixmask] .= 0 # better way to do this by taking pixmask as an argument wst_s2_deriv_sum?
            return
        end


        println("Diff check")
        eps = zeros(size(input))
        row, col = 24, 18 #convert(Int8, Nx/2), convert(Int8, Nx/2)+3
        epsmag = 1e-4
        eps[row, col] = epsmag
        chisq1 = loss_func2(input+eps./2) #DEB
        chisq0 = loss_func2(input-eps./2) #DEB
        brute  = (chisq1-chisq0)/epsmag
        #df_brute = eqws_compute(reshape(input, (Nx, Nx)), filter_hash, doS2=true, doS12=false, doS20=false, norm=false)[coeff_mask] - s_targ_mean
        clever = zeros(size(input)) #DEB
        meanval = mean(adaptive_apodizer(input)) ./ mean(wind_2d(Nx))
        _bar = dloss2(clever, input) #DEB
        println("Chisq Derve Check")
        println("Brute:  ",brute)
        println("Clever: ",clever[row, col], " Difference: ", brute - clever[row, col], " Mean ", meanval) #DEB
        println("Initial Loss ", loss_func2(input))
        res = optimize(loss_func2, dloss2, input, minmethod, Optim.Options(iterations = numitns_dict, store_trace = show_trace0, show_trace = show_trace0))
        result_img = Optim.minimizer(res)
        println("Final Loss ", loss_func2(result_img))
        return res, reshape(result_img, (Nx, Nx))
    end

    function image_recon_derivsum_regularized_transformed_gaussian(input::Array{Float64, 2}, filter_hash::Dict, fs_targ_mean::Array{Float64, 1}, fs_targ_invcov::Array{Float64, 2}, pixmask::BitArray{2}, dhc_args, func, dfunc;
        FFTthreads::Int=1, optim_settings=Dict([("iterations", 10)]), coeff_mask=nothing, lambda=0.001) #add iso here and a check that returns error if both coeffmask is not nothing and iso is present.
        #=
        f(S) ~ G(fs_targ_mean, fs_targ_invcov)
        =#
        (N1iso, Nf) = size(filter_hash["S1_iso_mat"])
        println("Coeff mask:", (coeff_mask!=nothing))

        (Nx, Ny)  = size(input)
        if Nx != Ny error("Input image must be square") end
        if !dhc_args[:doS20] error("Not implemented") end
        if Nf == 0 error("filter hash corrupted") end

        println("S20")
        #pixmask = pixmask[:] #flattened: Nx^2s #DEB
        ori_input = reshape(copy(input), (Nx^2, 1))

        if coeff_mask!=nothing
            if (!dhc_args[:iso]) & (count((i->i==true), coeff_mask[1:Nf+2])!=0) error("Code assumes S1 coeffs and S0 are masked out") end #wrap
            if (dhc_args[:iso]) & (count((i->i==true), coeff_mask[1:N1iso+2])!=0) error("Code assumes S1iso coeffs and S0 are masked out") end
            #if length(coeff_mask)!= (2+Nf + Nf^2) error("Wrong dim mask") end #wrap
            if size(fs_targ_mean)[1]!=count((i->(i==true)), coeff_mask) error("s_targ_mean should only contain coeffs to be optimized") end
            if size(fs_targ_invcov)[1]!=count((i->(i==true)), coeff_mask) error("s_targ_invcov should only have coeffs to be optimized") end
        else #No mask: all coeffs (default: Only S20 coeffs will be optim wrt.)
            error("You must supply a coeff_mask")
        end

        #Apodization adjustments
        function adaptive_apodizer(input_image::Array{Float64, 2})
            if dhc_args[:apodize]
                apd_image = apodizer(input_image)
            else
                apd_image = input_image
            end
            return apd_image
        end
        #For the Jacobian dA*P/dP
        if dhc_args[:apodize]
            Ap = wind_2d(Nx)
            cA = sum(Ap)
            Apflat = reshape(Ap, Nx^2)
            od_nz_idx = findall(!iszero, Apflat) #findall((x->((x!=0) & (x!=1))), Apflat)
            #od_zidx = findall(iszero, Apflat)
            avals = Apflat[od_nz_idx]
            dA = zeros(Nx^2, Nx^2)
            dA[:, od_nz_idx] .= ((1.0 .- Apflat) * avals')./cA
            dA[diagind(dA)] += Apflat
        else
            dA = I
        end


        num_freecoeff = count((i->(i==true)), coeff_mask)

        #After this all cases have a coeffmask, and s_targ_mean and s_targ_invcov have the shapes of the coefficients that we want to select.
        #Optimization Hyperparameters
        tonorm = false
        numitns_dict = get(optim_settings, "iterations", 100)
        minmethod = get(optim_settings, "minmethod", ConjugateGradient())

        if (dhc_args[:doS20]) & (dhc_args[:iso])
            iso2nf2mask = zeros(Int64, Nf^2)
            M20 = filter_hash["S2_iso_mat"]
            for id=1:Nf^2 iso2nf2mask[M20.colptr[id]] = M20.rowval[id] end
            coeff_masks20 = coeff_mask[3+N1iso:end]
        else# (dhc_args[:doS20]) & (!dhc_args[:iso])
            coeff_masks20 = coeff_mask[3+Nf:end]
        end

        function augment_weights(inpwt::Array{Float64})
            if (dhc_args[:doS20]) & (dhc_args[:iso]) #wt (Nc) -> |S2_iso| -> |Nf^2|
                w_s2iso = zeros(size(filter_hash["S2_iso_mat"])[1])
                w_s2iso[coeff_masks20] .= inpwt
                w_nf2 = zeros(Nf^2)
                w_nf2 .= w_s2iso[iso2nf2mask]
                return reshape(w_nf2, (Nf, Nf))
            else#if (dhc_args[:doS20]) & (!dhc_args[:iso])
                w_nf2 = zeros(Nf^2)
                w_nf2[coeff_masks20] .= inpwt
                return reshape(w_nf2, (Nf, Nf))
            end
        end

        function loss_func20(img_curr::Array{Float64, 2})
            #println(typeof(eqws_compute_wrapper(img_curr, filter_hash, norm=tonorm; dhc_args...)[coeff_mask]))
            s_curr = func(eqws_compute_wrapper(img_curr, filter_hash, norm=tonorm; dhc_args...)[coeff_mask]) #Edit
            regterm =  0.5*lambda*sum((adaptive_apodizer(img_curr) - adaptive_apodizer(input)).^2)
            lnlik = ( 0.5 .* (s_curr - fs_targ_mean)' * fs_targ_invcov * (s_curr - fs_targ_mean))
            neglogloss = lnlik[1] + regterm
            return neglogloss
        end
        #TODO: Need to replace the deriv_sums20 with a deriv_sum wrapper that handles all cases (S12, S20, S2)

        function dloss20(storage_grad::Array{Float64, 2}, img_curr::Array{Float64, 2})
            s_curr = eqws_compute_wrapper(img_curr, filter_hash, norm=tonorm; dhc_args...)[coeff_mask]
            diff = func(s_curr) - fs_targ_mean
            #Add branches here
            wt = convert(Array{Float64,1}, reshape(transpose(diff) * fs_targ_invcov, (length(diff),))) .* dfunc(s_curr)
            wtaug = augment_weights(wt)
            #println(size(wtaug))
            apdimg_curr = adaptive_apodizer(img_curr)
            storage_grad .= reshape((wst_S20_deriv_sum(apdimg_curr, filter_hash, wtaug, FFTthreads=FFTthreads)' + reshape(lambda.*(apdimg_curr - adaptive_apodizer(input)), (1, Nx^2))) * dA, (Nx, Nx))
            storage_grad[pixmask] .= 0 # better way to do this by taking pixmask as an argument wst_s2_deriv_sum?
            return
        end


        println("Diff check")
        eps = zeros(size(input))
        row, col = 24, 18 #convert(Int8, Nx/2), convert(Int8, Nx/2)+3
        epsmag = 1e-6
        eps[row, col] = epsmag
        chisq1 = loss_func20(input+eps./2) #DEB
        chisq0 = loss_func20(input-eps./2) #DEB
        brute  = (chisq1-chisq0)/epsmag
        #df_brute = eqws_compute(reshape(input, (Nx, Nx)), filter_hash, doS2=true, doS12=false, doS20=false, norm=false)[coeff_mask] - s_targ_mean
        clever = zeros(size(input)) #DEB
        meanval = mean(adaptive_apodizer(input)) ./ mean(wind_2d(Nx))
        _bar = dloss20(clever, input) #DEB
        println("Chisq Derve Check")
        println("Brute:  ",brute)
        println("Clever: ",clever[row, col], " Difference: ", brute - clever[row, col], " Mean ", meanval) #DEB
        println("Initial Loss ", loss_func20(input))
        res = optimize(loss_func20, dloss20, input, minmethod, Optim.Options(iterations = numitns_dict, store_trace = true, show_trace = true))
        result_img = Optim.minimizer(res)
        println("Final Loss ", loss_func20(result_img))
        return res, reshape(result_img, (Nx, Nx))
    end

    function image_recon_derivsum_separate(input::Array{Float64, 2}, iteration_input::Array{Float64, 2}, filter_hash::Dict, s_targ_mean::Array{Float64, 1}, s_targ_invcov, pixmask::BitArray{2}, dhc_args;
        FFTthreads::Int=1, optim_settings=Dict([("iterations", 10)]), coeff_mask=nothing, lambda=0.001) #add iso here and a check that returns error if both coeffmask is not nothing and iso is present.
        #=
        Cases to be handled by this function:
        S20
        Apodized or not
        Iso
        Log or Regular: If log then input, targ and cov should have been constructed using log
        Select pixels masked or all floating--not using for masked currently
        Select coefficients optimized wrt
        Lambda: regularization

        Cases NOT handled by this function:
        Only S1
        Non-Gaussian loss function
        Using the S0 params (mean and variance) in the loss function
        eqws_compute with norm=true
        =#
        (N1iso, Nf) = size(filter_hash["S1_iso_mat"])
        println("Coeff mask:", (coeff_mask!=nothing))

        (Nx, Ny)  = size(input)
        if Nx != Ny error("Input image must be square") end
        if !dhc_args[:doS20] error("Not implemented") end
        if Nf == 0 error("filter hash corrupted") end

        println("S20")
        #pixmask = pixmask[:] #flattened: Nx^2s #DEB
        ori_input = reshape(copy(input), (Nx^2, 1))

        if coeff_mask!=nothing
            if (!dhc_args[:iso]) & (count((i->i==true), coeff_mask[1:Nf+2])!=0) error("Code assumes S1 coeffs and S0 are masked out") end #wrap
            if (dhc_args[:iso]) & (count((i->i==true), coeff_mask[1:N1iso+2])!=0) error("Code assumes S1iso coeffs and S0 are masked out") end
            if size(s_targ_mean)[1]!=count((i->(i==true)), coeff_mask) error("s_targ_mean should only contain coeffs to be optimized") end
            if size(s_targ_invcov)[1]!=count((i->(i==true)), coeff_mask) error("s_targ_invcov should only have coeffs to be optimized") end
        else #No mask: all coeffs (default: Only S20 coeffs will be optim wrt.)
            error("You must supply a coeff_mask")
        end

        #Apodization adjustments
        function adaptive_apodizer(input_image::Array{Float64, 2})
            if dhc_args[:apodize]
                apd_image = apodizer(input_image)
            else
                apd_image = input_image
            end
            return apd_image
        end
        #For the Jacobian dA*P/dP
        if dhc_args[:apodize]
            Ap = wind_2d(Nx)
            cA = sum(Ap)
            Apflat = reshape(Ap, Nx^2)
            od_nz_idx = findall(!iszero, Apflat) #findall((x->((x!=0) & (x!=1))), Apflat)
            #od_zidx = findall(iszero, Apflat)
            avals = Apflat[od_nz_idx]
            dA = zeros(Nx^2, Nx^2)
            dA[:, od_nz_idx] .= ((1.0 .- Apflat) * avals')./cA
            dA[diagind(dA)] += Apflat
        else
            dA = I
        end


        num_freecoeff = count((i->(i==true)), coeff_mask)

        #After this all cases have a coeffmask, and s_targ_mean and s_targ_invcov have the shapes of the coefficients that we want to select.
        #Optimization Hyperparameters
        tonorm = false
        numitns_dict = get(optim_settings, "iterations", 100)
        minmethod = get(optim_settings, "minmethod", ConjugateGradient())

        if (dhc_args[:doS20]) & (dhc_args[:iso])
            iso2nf2mask = zeros(Int64, Nf^2)
            M20 = filter_hash["S2_iso_mat"]
            for id=1:Nf^2 iso2nf2mask[M20.colptr[id]] = M20.rowval[id] end
            coeff_masks20 = coeff_mask[3+N1iso:end]
        else# (dhc_args[:doS20]) & (!dhc_args[:iso])
            coeff_masks20 = coeff_mask[3+Nf:end]
        end

        function augment_weights(inpwt::Array{Float64})
            if (dhc_args[:doS20]) & (dhc_args[:iso]) #wt (Nc) -> |S2_iso| -> |Nf^2|
                w_s2iso = zeros(size(filter_hash["S2_iso_mat"])[1])
                w_s2iso[coeff_masks20] .= inpwt
                w_nf2 = zeros(Nf^2)
                w_nf2 .= w_s2iso[iso2nf2mask]
                return reshape(w_nf2, (Nf, Nf))
            else#if (dhc_args[:doS20]) & (!dhc_args[:iso])
                w_nf2 = zeros(Nf^2)
                w_nf2[coeff_masks20] .= inpwt
                return reshape(w_nf2, (Nf, Nf))
            end
        end

        function loss_func20(img_curr::Array{Float64, 2})
            s_curr = eqws_compute_wrapper(img_curr, filter_hash, norm=tonorm; dhc_args...)[coeff_mask]
            regterm =  0.5*lambda*sum((adaptive_apodizer(img_curr) - adaptive_apodizer(input)).^2)
            lnlik = ( 0.5 .* (s_curr - s_targ_mean)' * s_targ_invcov * (s_curr - s_targ_mean))
            neglogloss = lnlik[1] + regterm
            return neglogloss
        end
        #TODO: Need to replace the deriv_sums20 with a deriv_sum wrapper that handles all cases (S12, S20, S2)

        function dloss20(storage_grad::Array{Float64, 2}, img_curr::Array{Float64, 2})
            s_curr = eqws_compute_wrapper(img_curr, filter_hash, norm=tonorm; dhc_args...)[coeff_mask]
            diff = s_curr - s_targ_mean
            #Add branches here
            wt = augment_weights(convert(Array{Float64,1}, reshape(transpose(diff) * s_targ_invcov, (length(diff),))))
            apdimg_curr = adaptive_apodizer(img_curr)
            storage_grad .= reshape((wst_S20_deriv_sum(apdimg_curr, filter_hash, wt, FFTthreads=FFTthreads)' + reshape(lambda.*(apdimg_curr - adaptive_apodizer(input)), (1, Nx^2))) * dA, (Nx, Nx))
            storage_grad[pixmask] .= 0 # better way to do this by taking pixmask as an argument wst_s2_deriv_sum?
            return
        end


        println("Diff check")
        eps = zeros(size(input))
        row, col = 24, 18 #convert(Int8, Nx/2), convert(Int8, Nx/2)+3
        epsmag = 1e-4
        eps[row, col] = epsmag
        chisq1 = loss_func20(input+eps./2) #DEB
        chisq0 = loss_func20(input-eps./2) #DEB
        brute  = (chisq1-chisq0)/epsmag
        #df_brute = eqws_compute(reshape(input, (Nx, Nx)), filter_hash, doS2=true, doS12=false, doS20=false, norm=false)[coeff_mask] - s_targ_mean
        clever = zeros(size(input)) #DEB
        meanval = mean(adaptive_apodizer(input)) ./ mean(wind_2d(Nx))
        _bar = dloss20(clever, input) #DEB
        println("Chisq Derve Check")
        println("Brute:  ",brute)
        println("Clever: ",clever[row, col], " Difference: ", brute - clever[row, col], " Mean ", meanval) #DEB
        println("Initial Loss ", loss_func20(input))
        res = optimize(loss_func20, dloss20, iteration_input, minmethod, Optim.Options(iterations = numitns_dict, store_trace = true, show_trace = true))
        result_img = Optim.minimizer(res)
        println("Final Loss ", loss_func20(result_img))
        return res, reshape(result_img, (Nx, Nx))
    end
end
