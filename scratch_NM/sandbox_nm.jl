using Statistics
using Plots
using BenchmarkTools
using Profile
using FFTW
using Statistics
using Optim
using Images, FileIO, ImageIO
using Printf
using SparseArrays

#using Profile
using LinearAlgebra

# put the cwd on the path so Julia can find the module
push!(LOAD_PATH, pwd()*"/main")
using DHC_2DUtils


@time fhash = fink_filter_hash(1, 8, nx=64, pc=1, wd=1)

function realspace_filter(Nx, f_i, f_v)

    zarr = zeros(ComplexF64, Nx, Nx)
    for i = 1:length(f_i)
        zarr[f_i[i]] = f_v[i] # filter*image in Fourier domain
    end
    filt = ifft(zarr)  # real space, complex
    return filt
end

function DHC(image::Array{Float64,2}, filter_hash::Dict, filter_hash2::Dict=filter_hash;
    doS2::Bool=true, doS12::Bool=false, doS20::Bool=false)
    # image        - input for WST
    # filter_hash  - filter hash from fink_filter_hash
    # filter_hash2 - filters for second order.  Default to same as first order.
    # doS2         - compute S2 coeffs
    # doS12        - compute S2 coeffs
    # doS20        - compute S2 coeffs

    # Use 2 threads for FFT
    FFTW.set_num_threads(2)

    # array sizes
    (Nx, Ny)  = size(image)
    if Nx != Ny error("Input image must be square") end
    (Nf, )    = size(filter_hash["filt_index"])
    if Nf == 0 error("filter hash corrupted") end

    # allocate coeff arrays
    out_coeff = []
    S0  = zeros(Float64, 2)
    S1  = zeros(Float64, Nf)
    if doS2  S2  = zeros(Float64, Nf, Nf) end  # traditional 2nd order
    if doS12 S12 = zeros(Float64, Nf, Nf) end  # Fourier correlation
    if doS20 S20 = zeros(Float64, Nf, Nf) end  # real space correlation
    anyM2 = doS2 | doS12 | doS20
    anyrd = doS2 | doS20             # compute real domain with iFFT

    # allocate image arrays for internal use
    if doS12 im_fdf_0_1 = zeros(Float64,           Nx, Ny, Nf) end   # this must be zeroed!
    if anyrd im_rd_0_1  = Array{Float64, 3}(undef, Nx, Ny, Nf) end

    ## 0th Order
    S0[1]   = mean(image)
    norm_im = image.-S0[1]
    S0[2]   = sum(norm_im .* norm_im)/(Nx*Ny)
    # norm_im ./= sqrt(Nx*Ny*S0[2])
    # println("norm off")
    norm_im = copy(image)

    append!(out_coeff,S0[:])

    ## 1st Order
    im_fd_0 = fft(norm_im)  # total power=1.0

    # unpack filter_hash
    f_ind   = filter_hash["filt_index"]  # (J, L) array of filters represented as index value pairs
    f_val   = filter_hash["filt_value"]

    zarr = zeros(ComplexF64, Nx, Ny)  # temporary array to fill with zvals

    # make a FFTW "plan" for an array of the given size and type
    if anyrd
        P = plan_ifft(im_fd_0) end  # P is an operator, P*im is ifft(im)

    ## Main 1st Order and Precompute 2nd Order
    for f = 1:Nf
        S1tot = 0.0
        f_i = f_ind[f]  # CartesianIndex list for filter
        f_v = f_val[f]  # Values for f_i
        # for (ind, val) in zip(f_i, f_v)   # this is slower!
        if length(f_i) > 0
            for i = 1:length(f_i)
                ind       = f_i[i]
                zval      = f_v[i] * im_fd_0[ind]
                S1tot    += abs2(zval)
                zarr[ind] = zval        # filter*image in Fourier domain
                if doS12 im_fdf_0_1[ind,f] = abs(zval) end
            end
            S1[f] = S1tot/(Nx*Ny)  # image power
            if anyrd
                im_rd_0_1[:,:,f] .= abs2.(P*zarr) end
            zarr[f_i] .= 0
        end
    end
    append!(out_coeff, S1[:]) #Why need the :?

    # we stored the abs()^2, so take sqrt (this is faster to do all at once)
    if anyrd im_rd_0_1 .= sqrt.(im_rd_0_1) end


    if doS2
        f_ind2   = filter_hash2["filt_index"]  # (J, L) array of filters represented as index value pairs
        f_val2   = filter_hash2["filt_value"]

        ## Traditional second order
        for f1 = 1:Nf
            thisim = fft(im_rd_0_1[:,:,f1])  # Could try rfft here
            # println("  f1",f1,"  sum(fft):",sum(abs2.(thisim))/Nx^2, "  sum(im): ",sum(abs2.(im_rd_0_1[:,:,f1])))
            # Loop over f2 and do second-order convolution
            for f2 = 1:Nf
                f_i = f_ind2[f2]  # CartesianIndex list for filter
                f_v = f_val2[f2]  # Values for f_i
                # sum im^2 = sum(|fft|^2/npix)
                S2[f1,f2] = sum(abs2.(f_v .* thisim[f_i]))/(Nx*Ny)
            end
        end
        append!(out_coeff, S2)
    end


    # Fourier domain 2nd order
    if doS12
        Amat = reshape(im_fdf_0_1, Nx*Nx, Nf)
        S12  = Amat' * Amat
        append!(out_coeff, S12)
    end


    # Real domain 2nd order
    if doS20
        Amat = reshape(im_rd_0_1, Nx*Nx, Nf)
        S20  = Amat' * Amat
        append!(out_coeff, S20)
    end


    return out_coeff
end

##Derivative Functions

function wst_S1_deriv(image::Array{Float64,2}, filter_hash::Dict)
    function conv(a,b)
        ifft(fft(a) .* fft(b))
    end

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

        zarr[f_i] = f_v .* im_fd_0[f_i] #Why do we need .* here??
        #Commented: I_λ = P*zarr  # complex valued ifft of zarr
        #im_rdc_0_1[:,:,f] = I_lambda


        #xfac = 2 .* real(I_λ)
        #yfac = 2 .* imag(I_λ)
        # it is clearly stupid to transform back and forth, but we will need these steps for the S20 part
        ψ_λ  = realspace_filter(Nx, f_i, f_v)
        # convolution requires flipping wavelet direction, equivalent to flipping sign of imaginary part.
        # dSdp[:,:,f] = real.(conv(xfac,real(ψ_λ))) - real.(conv(yfac,imag(ψ_λ)))
        zarr[f_i] = f_v .* zarr[f_i] #Reusing zarr for F[conv(z_λ,ψ_λ)] = F[z_λ].*F[ψ_λ]
        dSdp[:,:,f] = 2 .* real.(P*(zarr)) #real.(conv(I_λ, ψ_λ))
        zarr[f_i] .= 0   # reset zarr for next loop

    end
    return dSdp

end

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

        zarr[f_i] = f_v .* im_fd_0[f_i] #Why do we need .* here??
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



function wst_S20_deriv(image::Array{Float64,2}, filter_hash::Dict) #make this faster
    function conv(a,b)
        ifft(fft(a) .* fft(b))
    end

    # Use 2 threads for FFT
    FFTW.set_num_threads(2)

    # array sizes
    (Nx, Ny)  = size(image)
    (Nf, )    = size(filter_hash["filt_index"])

    # allocate output array
    dS20dp  = zeros(Float64, Nx, Nx, Nf, Nf)

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

    zarr = zeros(ComplexF64, Nx, Nx)  # temporary array to fill with zvals

    # make a FFTW "plan" for an array of the given size and type
    P = plan_ifft(im_fd_0)  # P is an operator, P*im is ifft(im)

    # Loop over filters
    for f = 1:Nf
        f_i = f_ind[f]  # CartesianIndex list for filter
        f_v = f_val[f]  # Values for f_i

        zarr[f_i] = f_v .* im_fd_0[f_i]
        I_λ = P*zarr  # complex valued ifft of zarr
        zarr[f_i] .= 0   # reset zarr for next loop
        im_rdc_0_1[:,:,f] = I_λ
        im_rd_0_1[:,:,f]  = abs.(I_λ)
    end

    for f2 = 1:Nf
        ψ_λ  = realspace_filter(Nx, f_ind[f2], f_val[f2])
        uvec = im_rdc_0_1[:,:,f2] ./ im_rd_0_1[:,:,f2]
        for f1 = 1:Nf
            cfac = im_rd_0_1[:,:,f1].*uvec
            I1dI2 = real.(conv(cfac,ψ_λ))  #NM: Make the conv to fft change here too
            dS20dp[:,:,f1,f2] += I1dI2
            dS20dp[:,:,f2,f1] += I1dI2
        end
    end

    return dS20dp

end


function wst_S1S2_deriv(image::Array{Float64,2}, filter_hash::Dict)
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


function wst_S1S2_derivfast(image::Array{Float64,2}, filter_hash::Dict)
    #Works now.
    #Possible Bug: You assumed zero freq in wrong place and N-k is the wrong transformation.
    # Use 2 threads for FFT
    FFTW.set_num_threads(2)

    # array sizes
    (Nx, Ny)  = size(image)
    (Nf, )    = size(filter_hash["filt_index"])

    # allocate output array
    dS1dp  = zeros(Float64, Nx, Nx, Nf)
    dS2dp  = zeros(Float64, Nx, Nx, Nf, Nf)

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

        zarr1[f_i1] = f_v1 .* im_fd_0[f_i1] #F[z]
        zarr1_rd = P*zarr1 #for frzi
        fz_fψ1[f_i1] = f_v1 .* zarr1[f_i1]
        fz_fψ1_rd = P*fz_fψ1
        dS1dp[:,:,f1] = 2 .* real.(fz_fψ1_rd) #real.(conv(I_λ, ψ_λ))
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
            dS2dp[:, :, f1, f2] = real.(P*(fterm_ct1 + fterm_ct2))

            #Reset f2 specific tmp arrays to 0
            fterm_a[f_i2] .= 0
            fterm_ct1[f_i1] .=0
            fterm_ct2[f_i1rev] .=0

        end
        # reset all reused variables to 0
        zarr1[f_i1] .= 0
        fz_fψ1[f_i1] .= 0


    end
    return dS1dp, dS2dp
end

#=function wst_S1S2_deriv_fd(image::Array{Float64,2}, filter_hash::Dict)
    #Works now.
    #Possible Bug: You assumed zero freq in wrong place and N-k is the wrong transformation.
    # Use 2 threads for FFT
    FFTW.set_num_threads(2)

    # array sizes
    (Nx, Ny)  = size(image)
    (Nf, )    = size(filter_hash["filt_index"])

    # allocate output array
    dS1dp  = zeros(Float64, Nx, Nx, Nf)
    dS2dp  = zeros(Float64, Nx, Nx, Nf, Nf)

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
    #phase_mat = reshape(collect(1:1:Nx), (Nx, 1)) .* reshape(collect(1:1:Nx), (1, Nx)) #Need to inst an Nx^2Nx2 mat. Instead just use it on the fly
    #phase_mat = 2π*
    #phase_mat = exp()
    # Loop over filters
    xImat = CartesianIndices((Nx, Nx))
    for f1 = 1:Nf
        f_i1 = f_ind[f1]  # CartesianIndex list for filter1
        f_v1 = f_val[f1]  # Values for f_i1
        f_i1rev = f_ind_rev[f1]

        zarr1[f_i1] = f_v1 .* im_fd_0[f_i1] #F[z]
        #zarr1_rd = P*zarr1 #for frzi
        fz_fψ1[f_i1] = f_v1 .* zarr1[f_i1]
        fz_fψ1_rd = P*fz_fψ1
        dS1dp[:,:,f1] = 2 .* real.(fz_fψ1_rd) #real.(conv(I_λ, ψ_λ))
        #Step 2: Do this using the fd space trick

        #dS2 loop prep
        #=I_λ1 = sqrt.(abs2.(zarr1_rd))
        fI_λ1 = fft(I_λ1)
        rterm_bt1 = zarr1_rd./I_λ1 #Z_λ1/I_λ1_bar.
        rterm_bt2 = conj.(zarr1_rd)./I_λ1=#

        for f2 = 1:Nf
            f_i2 = f_ind[f2]  # CartesianIndex list for filter2
            f_v2 = f_val[f2]  # Values for f_i2

            a1 = fz_fψ1[f_i2] .* (f_v2).^2
            b1 = exp.((-2π*im*) .* (f_i2 .* xImat))  #Shape: #f_i2*Nx*Nx
            term1 = a1 .* b1
            term2 =
            #fterm_ct2 = fterm_bt2 .* fcψ_λ1             #Slow
            #println(size(fterm_ct2[f_i1rev]),size(fterm_bt2[f_i1rev]),size(f_v1),size(conj.(f_v1)))
            fterm_ct2[f_i1rev] = fterm_bt2[f_i1rev] .* f_v1 #f_v1 is real
            #fterm_ct2slow = fterm_bt2 .* fcψ_λ1
            dS2dp[:, :, f1, f2] = real.(P*(fterm_ct1 + fterm_ct2))

            #Reset f2 specific tmp arrays to 0
            fterm_a[f_i2] .= 0
            fterm_ct1[f_i1] .=0
            fterm_ct2[f_i1rev] .=0

        end
        # reset all reused variables to 0
        zarr1[f_i1] .= 0
        fz_fψ1[f_i1] .= 0


    end
    return dS1dp, dS2dp
end
=#

function wst_S1S2_deriv_fdsparse(image::Array{Float64,2}, filter_hash::Dict)
    #Works now.
    #Possible Bug: You assumed zero freq in wrong place and N-k is the wrong transformation.
    # Use 2 threads for FFT
    FFTW.set_num_threads(2)

    # array sizes
    (Nx, Ny)  = size(image)
    (Nf, )    = size(filter_hash["filt_index"])

    # allocate output array
    dS1dp  = zeros(Float64, Nx, Nx, Nf)
    dS2dp  = zeros(Float64, Nx, Nx, Nf, Nf)


    # unpack filter_hash
    f_ind   = filter_hash["filt_index"]  # (J, L) array of filters represented as index value pairs
    f_indflat = [(tup-> (Nx*(tup[2]-1) + tup[1])).(Tuple.(fh)) for fh in filter_hash["filt_index"]]
    nf_nz = vcat(fill.(collect(1:1:Nf), length.(f_indflat))) #Filter id repeated as many times as nz elemes
    f_indsparse = vcat(hcat.(nf_nz, f_indflat)...) #Array of [filter_id, flattened_nz_idx]
    f_val   = filter_hash["filt_value"]


    im_fd_0 = reshape(fft(image), (Nx^2))
    f_ind_comb = (x->CartesianIndex(x)).(f_indsparse[:, 2]) #k_nz for all filters concat
    fzfψ = sparse(f_indsparse[:, 1], f_indsparse[:, 2], im_fd_0[f_ind_comb].*vcat(f_val...).^2) #Nf * Nx^2 sparse
    #Phi values
    knz_vals = hcat((p->[p[1], p[2]]).(vcat(filter_hash["filt_index"]...))...)' #|k_nz| x 2
    xigrid = reshape(CartesianIndices((1:Nx, 1:Nx)), (1, Nx^2))
    xigrid = hcat((x->[x[1], x[2]]).(xigrid)...)   #2*Nx^2
    spvals = exp.((-2π*im)/Nx .* reshape((knz_vals*xigrid), (size(knz_vals)[1]*Nx^2))) #BUG: Fix shape issue here
    spvals_idz = findnz(spvals)
    Φmat = sparse(spvals_idz[1], spvals_idz[2], spvals) #sparse(repeat(f_indsparse[:, 2], Nx.^2), vcat(fill.(collect(1:1:Nx^2), size(f_indsparse)[1])...), spvals) #Nx^2 * Nx^2 sparse k_nz should vary faster
    dS1_term = Array(fzfψ*Φmat)
    dS1dp = reshape((2/Nx^2) .* real.(dS1_term), (Nf, Nx, Nx)) #Divide by Nx^2!!!
    #Check the math and dS1 first
    #fzfψ_ψ2sq = sparse(f_indsparse[:, 1], f_indsparse[:, 2], im_fd_0[f_ind_comb].*vcat(f_val...).^2) #basically want something Nf^2 * Nx^2 analogous to fzfψ

    return dS1dp
end

#=
function realconv_test(Nx, ar1, ar2)
    soln = zeros(Float64, 8)
    for x=1:8:
        soln[x] = ar1 .* ar2[x-collect(1:1:8)]
=#

##Derivative test functions
# wst_S1_deriv agrees with brute force at 1e-10 level.
 function derivtest(Nx)
    eps = 1e-4
    fhash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1)
    im = rand(Float64, Nx, Nx).*0.1
    im[6,6]=1.0
    im1 = copy(im)
    im0 = copy(im)
    im1[2,3] += eps/2
    im0[2,3] -= eps/2
    blarg = wst_S1_deriv(im, fhash)
    println("Bl size",size(blarg))

    der0=DHC(im0,fhash,doS2=false)
    der1=DHC(im1,fhash,doS2=false)
    dS = (der1-der0) ./ eps #since eps in only (2, 3) this should equal blarg[2, 3, :]


    diff = dS[3:end]-blarg[2,3,:]
    println(dS[3:end])
    println("and")
    println(blarg[2,3,:])
    println("stdev: ",std(diff))

    #=#Delete: blarg shouldnt be 0 for all other pixels since the coefficient isnt constant wrt other pixels!!
    Nf = length(fhash["filt_index"])
    mask = trues((Nx,Nx, Nf))
    println("Nf",Nf,"Check",(size(mask)==size(blarg)))
    mask[2, 3, :] .= false
    println("Mean deriv wrt to other pixels (should be 0 for all)",mean(blarg[mask]), "Maximum",maximum(blarg[mask]))
    #For Nx=128, Mean is actually 0.002 Seems concerning since eps is 1e-4!? the max is 0.10
    #For Nx=8, Mean is 0.01, max is 1.0 -- some kind of normalization issue?=#
    return
end

# wst_S1_deriv agrees with brute force at 1e-10 level.
 function derivtestfast(Nx, mode="fourier")
    eps = 1e-4
    fhash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1)
    im = rand(Float64, Nx, Nx).*0.1
    im[6,6]=1.0
    im1 = copy(im)
    im0 = copy(im)
    im1[2,3] += eps/2
    im0[2,3] -= eps/2
    blarg = wst_S1_deriv(im, fhash)
    if mode=="fourier"
        blargfast = wst_S1S2_deriv_fdsparse(im, fhash)
    else
        blargfast = wst_S1_deriv_fast(im, fhash)
    end
    der0=DHC(im0,fhash,doS2=false)
    der1=DHC(im1,fhash,doS2=false)
    dS = (der1-der0) ./ eps

    diff_old = dS[3:end]-blarg[2,3,:]
    diff = dS[3:end]-blargfast[2,3,:]
    println(dS[3:end])
    println("and")
    println(blarg[2,3,:])
    println("stdev: ",std(diff_old))
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

derivtestfast(16, "fourier")


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

#=OLD
function derivtestS1S2(Nx)
    eps = 1e-4
    fhash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1)
    im = rand(Float64, Nx, Nx).*0.1
    im[6,6]=1.0
    im1 = copy(im)
    im0 = copy(im)
    im1[2,3] += eps/2
    im0[2,3] -= eps/2
    dS1dp_existing = wst_S1_deriv(im, fhash)
    dS1dp, dS2dp = wst_S1S2_deriv(im, fhash)

    der0=DHC(im0,fhash,doS2=true,doS20=false)
    der1=DHC(im1,fhash,doS2=true,doS20=false)
    dSlim = (der1-der0) ./ eps
    #=
    println("Checking dS1dp using existing deriv")
    Nf = length(fhash["filt_index"])
    lasts1ind = 2+Nf
    diff = dS1dp_existing[2, 3, :]-dSlim[3:lasts1ind]
    println(dSlim[3:lasts1ind])
    println("and")
    println(dS1dp_existing[2, 3, :])
    println("stdev: ",std(diff)) =#
    println("--------------------------")
    Nf = length(fhash["filt_index"])
    lasts1ind = 2+Nf
    diff = dS1dp[2, 3, :]-dSlim[3:lasts1ind]
    println(dSlim[3:lasts1ind])
    println("and")
    println(dS1dp[2,3,:])
    txt= @sprintf("Range of S1 der0, der1: Max= %.3E, %.3E Min=%.3E, %.3E ",maximum(der0[3:lasts1ind]),maximum(der1[3:lasts1ind]),minimum(der0[3:lasts1ind]),minimum(der1[3:lasts1ind]))
    print(txt)
    txt = @sprintf("Checking dS1dp using S1S2 deriv"," mean(abs(diff/dS1lim)): %.3E", mean(abs.(diff./dSlim[3:lasts1ind])))
    print(txt)
    println("stdev: ",std(diff)," mean(abs(diff/dS1lim)): ", mean(abs.(diff./dSlim[3:lasts1ind])))
    txt= @sprintf("Range of dS1lim: Max= ",maximum(abs.(dSlim[3:lasts1ind]))," Min= ",minimum(abs.(dSlim[3:lasts1ind])))
    print(txt)
    println("--------------------------")
    println("Checking dS2dp using S1S2 deriv")
    Nf = length(fhash["filt_index"])

    println("Shape check",size(dS2dp[2,3,:,:]),size(dSlim[lasts1ind+1:end]))
    diff = dSlim[lasts1ind+1:end] - reshape(dS2dp[2, 3, :, :], Nf*Nf) #Column major issue here?
    println("Difference",diff)
    println("Range of S2 der0, der1: Max",)
    print("Range of dS2lim: Max= ",maximum(abs.(dSlim[lasts1ind+1:end]))," Min= ",minimum(abs.(dSlim[lasts1ind+1:end])))
    println("Mean: ",mean(diff)," stdev: ",std(diff)," mean(abs(diff/dS2lim)): ", mean(abs.(diff./dSlim[lasts1ind+1:end])))


    #plot(dS[3:end])
    #plot!(blarg[2,3,:])
    #plot(diff)
    return
end
=#
function derivtestS20(Nx)
    eps = 1e-4
    fhash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1)
    im = rand(Float64, Nx, Nx).*0.1
    im[6,6]=1.0
    im1 = copy(im)
    im0 = copy(im)
    im1[2,3] += eps/2
    im0[2,3] -= eps/2
    dS20dp = wst_S20_deriv(im, fhash)

    der0=DHC(im0,fhash,doS2=false,doS20=true)
    der1=DHC(im1,fhash,doS2=false,doS20=true)
    dS = (der1-der0) ./ eps

    Nf = length(fhash["filt_index"])
    i0 = 3+Nf
    blarg = dS20dp[2,3,:,:]
    diff = dS[i0:end]-reshape(blarg,Nf*Nf)
    println(dS[i0:end])
    println("and")
    println(blarg)
    println("stdev: ",std(diff))
    println()
    println(diff)

    #plot(dS[3:end])
    #plot!(blarg[2,3,:])
    #plot(diff)
    return
end



derivtestS1S2(16, mode="fast")

Nx=16
fhash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1)
img = zeros(Float64, Nx, Nx)
img[6,6]=1.0
@benchmark ds1ds2slow = wst_S1S2_deriv(img, fhash)
@benchmark ds1s2 = wst_S1S2_derivfast(img, fhash)
@benchmark blarg = wst_S1_deriv(img, fhash)
@benchmark blarg = wst_S1_deriv_fast(img, fhash)
@benchmark
# S1 deriv mean time, NM Feb10??
# 32
# 64    3|6|23 ms
# 128   15|21|22 ms
# 256
# 512   1|2|2 s

# S1 derivFAST mean time, NM Feb10??
# 32
# 64    1.5|2|12 ms
# 128   6|10|29 ms
# 256
# 512   582|721|856




# S1 deriv time, Jan 30: Doug / Andrew??
# 32    17 ms
# 64    34 ms
# 128   115 ms
# 256   520 ms
# 512   2500 ms

Nx = 64
fhash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1)
im = zeros(Float64, Nx, Nx)
im[6,6]=1.0
@benchmark blarg = wst_S20_deriv(im, fhash)

Profile.clear()
@profile blarg = wst_S20_deriv(im, fhash)
Juno.profiler()


# S2 deriv time, Jan 30
# 8     28 ms
# 16    112
# 32    320
# 64    1000
# 128   5 sec
# 256   ---
# 512   ---



print(1)


derivtest(128)



Nx=32
im = rand(Nx,Nx)
@time fhash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1)
@benchmark Sarr = DHC(im, fhash, doS2=false)  # 0.36 ms
@benchmark Sarr = DHC(im, fhash, doS2=true)   # 6.4 ms
@benchmark Sarr = DHC(im, fhash, doS2=false, doS12=true)  # 0.7 ms
@benchmark Sarr = DHC(im, fhash, doS2=false, doS20=true)  # 2.1 ms


Nx=64
im = rand(Nx,Nx)
@time fhash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1)
@benchmark Sarr = DHC(im, fhash, doS2=false)  # 1.3 ms
@benchmark Sarr = DHC(im, fhash, doS2=true)   # 19 ms
@benchmark Sarr = DHC(im, fhash, doS2=false, doS12=true)  # 2.8 ms
@benchmark Sarr = DHC(im, fhash, doS2=false, doS20=true)  # 6.6 ms


Nx=256
im = rand(Nx,Nx)
@time fhash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1)
@benchmark Sarr = DHC(im, fhash, doS2=false)  # 25 ms
@benchmark Sarr = DHC(im, fhash, doS2=true)   # 375 ms
@benchmark Sarr = DHC(im, fhash, doS2=false, doS12=true)  # 55 ms
@benchmark Sarr = DHC(im, fhash, doS2=false, doS20=true)  # 130 ms


Nx = 64
fhash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1)
im = rand(Nx,Nx)
S_targ = DHC(im, fhash, doS2=false)

im = rand(Nx,Nx)





derivtestS1S2(128)

function wst_synth(im_init, fixmask)
    # fixmask -  0=float in fit   1=fixed

    function wst_synth_chisq(vec_in)

        thisim = copy(im_init)
        thisim[indfloat] = vec_in

        Sarr  = DHC(thisim, fhash, doS2=false)
        diff  = (Sarr .- S_targ)[3:end]

        # should have some kind of weight here
        chisq = diff'*diff
        return chisq

    end

    function wst_synth_dchisq(storage, vec_in)

        thisim = copy(im_init)
        thisim[indfloat] = vec_in
        dSdp   = wst_S1_deriv(thisim, fhash)
        S1arr  = DHC(thisim, fhash, doS2=false)
        diff  = (S1arr - S_targ)[3:end]

        # dSdp matrix * S1-S_targ is dchisq
        dchisq_im = (reshape(dSdp, Nx*Nx, Nf) * diff).*2
        dchisq = reshape(dchisq_im, Nx, Nx)[indfloat]

        storage .= dchisq
    end

    (Nx, Ny)  = size(im_init)
    if Nx != Ny error("Input image must be square") end
    (Nf, )    = size(fhash["filt_index"])


    # index list of pixels to float in fit
    indfloat = findall(fixmask .== 0)

    # initial values for floating pixels
    vec_init = im_init[indfloat]
    println(length(vec_init), " pixels floating")

    eps = zeros(size(vec_init))
    eps[1] = 1e-4
    chisq1 = wst_synth_chisq(vec_init+eps./2)
    chisq0 = wst_synth_chisq(vec_init-eps./2)
    brute  = (chisq1-chisq0)/1e-4

    clever = zeros(size(vec_init))
    _bar = wst_synth_dchisq(clever, vec_init)
    println("Brute:  ",brute)
    println("Clever: ",clever[1])

    # call optimizer
    res = optimize(wst_synth_chisq, wst_synth_dchisq, vec_init, BFGS())

    # copy results into pixels of output image
    im_synth = copy(im_init)
    im_synth[indfloat] = Optim.minimizer(res)
    println(res)

    return im_synth
end

Nx    = 128
fhash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1)
im    = rand(Nx,Nx)
fixmask = im .> 0.5
S_targ = DHC(im, fhash, doS2=false)

init = copy(im)
init[findall(fixmask .==0)] .= 0

foo = wst_synth(init, fixmask)






#Optim.minimizer(optimize(FOM, im, LBFGS()))

# using dchisq function
# size   t(BFGS) t(LBFGS) [sec]
# 16x16    1
# 32x32    4        9
# 64x64    20      52
# 128x128         153 (fitting 50% of pix)


# without dchisq function
# size   t(BFGS) t(LBFGS) [sec]
# 16x16   27
# 32x32  355      110
# 64x64          1645
# 128x128        est 28,000  (8 hrs)


function readdust()

    RGBA_img = load(pwd()*"/scratch_DF/t115_clean_small.png")
    img = reinterpret(UInt8,rawview(channelview(RGBA_img)))

    return img[1,:,:]
end





function wst_synthS20(im_init, fixmask, S_targ, S20sig)
    # fixmask -  0=float in fit   1=fixed

    function wst_synth_chisq(vec_in)

        thisim = copy(im_init)
        thisim[indfloat] = vec_in

        S20arr  = DHC(thisim, fhash, doS2=false, doS20=true)
        i0 = 3+Nf
        diff  = ((S20arr - S_targ)./S20sig)[i0:end]

        # should have some kind of weight here
        chisq = diff'*diff
        println(chisq)
        return chisq

    end

    function wst_synth_dchisq(storage, vec_in)

        thisim = copy(im_init)
        thisim[indfloat] = vec_in
        dS20dp = wst_S20_deriv(thisim, fhash)
        S20arr = DHC(thisim, fhash, doS2=false, doS20=true)
        i0 = 3+Nf
        # put both factors of S20sig in this array to weight
        diff   = ((S20arr - S_targ)./(S20sig.^2))[i0:end]

        # dSdp matrix * S1-S_targ is dchisq
        dchisq_im = (reshape(dS20dp, Nx*Nx, Nf*Nf) * diff).*2
        dchisq = reshape(dchisq_im, Nx, Nx)[indfloat]

        storage .= dchisq
    end

    (Nx, Ny)  = size(im_init)
    if Nx != Ny error("Input image must be square") end
    (Nf, )    = size(fhash["filt_index"])


    # index list of pixels to float in fit
    indfloat = findall(fixmask .== 0)

    # initial values for floating pixels
    vec_init = im_init[indfloat]
    println(length(vec_init), " pixels floating")

    eps = zeros(size(vec_init))
    eps[1] = 1e-4
    chisq1 = wst_synth_chisq(vec_init+eps./2)
    chisq0 = wst_synth_chisq(vec_init-eps./2)
    brute  = (chisq1-chisq0)/1e-4

    clever = zeros(size(vec_init))
    _bar = wst_synth_dchisq(clever, vec_init)
    println("Brute:  ",brute)
    println("Clever: ",clever[1])

    # call optimizer
    res = optimize(wst_synth_chisq, wst_synth_dchisq, vec_init, BFGS())

    # copy results into pixels of output image
    im_synth = copy(im_init)
    im_synth[indfloat] = Optim.minimizer(res)
    println(res)

    return im_synth
end


function S20_weights(im, fhash, Nsam=10)

    (Nx, Ny)  = size(im)
    if Nx != Ny error("Input image must be square") end
    (Nf, )    = size(fhash["filt_index"])

    # fhash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1)

    S20   = DHC(im, fhash, doS2=false, doS20=true)
    Ns    = length(S20)
    S20arr = zeros(Float64, Ns, Nsam)
    for j=1:Nsam
        noise = rand(Nx,Nx)
        S20arr[:,j] = DHC(im+noise, fhash, doS2=false, doS20=true)
    end

    wt = zeros(Float64, Ns)
    for i=1:Ns
        wt[i] = std(S20arr[i,:])
    end

    return wt
end



# read dust map
dust = Float64.(readdust())
dust = dust[1:16,1:16]
println("Real=",sum(abs2.(dust)))
println("Real=",sum(abs2.(fft(dust))))


Nx    = 64
fhash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1, Omega=true)
im    = imresize(dust,(Nx,Nx))
fixmask = rand(Nx,Nx) .< 0.1

S_targ = DHC(im, fhash, doS2=false, doS20=true)
S_targ[end] = 0 #regularizing start
init = copy(im)
floatind = findall(fixmask .==0)
init[floatind] .+= rand(length(floatind)).*5 .-25

S20sig = S20_weights(im, fhash, 100)
foo = wst_synthS20(init, fixmask, S_targ, S20sig)
S_foo = DHC(foo, fhash, doS2=false, doS20=true)


# using dchisq function with S20
# size   t(BFGS) t(LBFGS) [sec]
# 8x8      33
# 16x16    -
# 32x32



#DOING dS1S2 and derivtest code in MAIN TO EXAMINE VARS in workspace
#=
dust = Float64.(readdust())
Nx=16
dust = dust[1:Nx,1:Nx]
eps = 1e-4
fhash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1)
im = rand(Float64, Nx, Nx).*0.1 #dust[1:Nx, 1:Nx]
im[6,6]=1.0
im1 = copy(im)
im0 = copy(im)
im1[2,3] += eps/2
im0[2,3] -= eps/2
dS1dp_existing = wst_S1_deriv(im, fhash)
dS1dp, dS2dp = wst_S1S2_deriv(im, fhash)

der0=DHC(im0,fhash,doS2=true,doS20=false)
der1=DHC(im1,fhash,doS2=true,doS20=false)
dSlim = (der1-der0) ./ eps



println("Checking dS1dp using existing deriv")
Nf = length(fhash["filt_index"])
lasts1ind = 2+Nf
derdiff = dS1dp_existing[2, 3, :]-dSlim[3:lasts1ind]
println(dSlim[3:lasts1ind])
println("and")
println(dS1dp_existing[2, 3, :])
println("stdev: ",std(derdiff))
println("--------------------------")
Nf = length(fhash["filt_index"])
lasts1ind = 2+Nf
derdiff = dS1dp[2, 3, :]-dSlim[3:lasts1ind]
println(dSlim[3:lasts1ind])
println("and")
println(dS1dp[2,3,:])
txt= @sprintf("Range of S1 der0, der1: Max= %.3E, %.3E Min=%.3E, %.3E \n",maximum(der0[3:lasts1ind]),maximum(der1[3:lasts1ind]),minimum(der0[3:lasts1ind]),minimum(der1[3:lasts1ind]))
print(txt)
txt = @sprintf("Checking dS1dp using S1S2 deriv, mean(abs(derdiff/dS1lim)): %.3E \n", mean(abs.(derdiff./dSlim[3:lasts1ind])))
print(txt)
println("stdev: ",std(derdiff))
txt= @sprintf("Range of dS1lim: Max= %.3E, Min= %.3E \n",maximum(abs.(dSlim[3:lasts1ind])),minimum(abs.(dSlim[3:lasts1ind])))
print(txt)
txt = @sprintf("Mean: %.3E stdev: %.3E mean(abs(derdiff/dS1lim)): %.3E ",mean(derdiff),std(derdiff), mean(abs.(derdiff./dSlim[3:lasts1ind])))
print(txt)
println("--------------------------")
println("Checking dS2dp using S1S2 deriv")
Nf = length(fhash["filt_index"])

println("Shape check",size(dS2dp[2,3,:,:]),size(dSlim[lasts1ind+1:end]))
derdiff = dSlim[lasts1ind+1:end] - reshape(dS2dp[2, 3, :, :], Nf*Nf) #Column major issue here?
println("derdifference",derdiff)
txt = @sprintf("Range of S2 der0, der1: Max= %.3E, %.3E Min=%.3E, %.3E \n",maximum(der0[lasts1ind+1:end]),maximum(der1[lasts1ind+1:end]),minimum(der0[lasts1ind+1:end]),minimum(der1[lasts1ind+1:end]))
print(txt)
txt = @sprintf("Range of dS2lim: Max=%.3E  Min=%.3E \n",maximum(abs.(dSlim[lasts1ind+1:end])),minimum(abs.(dSlim[lasts1ind+1:end])))
print(txt)
txt = @sprintf("Mean: %.3E stdev: %.3E mean(abs(derdiff/dS2lim)): %.3E \n",mean(derdiff),std(derdiff), mean(abs.(derdiff./dSlim[lasts1ind+1:end])))
print(txt)
println("Examining only those S2 coeffs which are greater than eps=1e-5 for der0")
eps_mask = findall(der0[lasts1ind+1:end] .> 1E-3)
der0_s2good = der0[lasts1ind+1:end][eps_mask]
der1_s2good = der1[lasts1ind+1:end][eps_mask]
dSlim_s2good = dSlim[lasts1ind+1:end][eps_mask]
derdiff_good = derdiff[eps_mask]
print(derdiff_good)
txt = @sprintf("\nRange of S2 der0, der1: Max= %.3E, %.3E Min=%.3E, %.3E \n",maximum(der0_s2good),maximum(der1_s2good),minimum(der0_s2good),minimum(der1_s2good))
print(txt)
txt = @sprintf("Range of dS2lim: Max=%.3E  Min=%.3E \n",maximum(abs.(dSlim_s2good)),minimum(abs.(dSlim_s2good)))
print(txt)
txt = @sprintf("Mean: %.3E stdev: %.3E mean(abs(derdiff/dS2lim)): %.3E \n",mean(derdiff_good),std(derdiff_good), mean(abs.(derdiff_good ./dSlim_s2good)))
print(txt)
=#
