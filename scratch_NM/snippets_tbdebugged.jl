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

#=
function wst_S1S2_deriv_fdsparse(image::Array{Float64,2}, filter_hash::Dict)
    #Works now.
    #Possible Bug: You assumed zero freq in wrong place and N-k is the wrong transformation.
    # Use 2 threads for FFT
    FFTW.set_num_threads(2)

    # array sizes
    (Nx, Ny)  = size(image)
    (Nf, )    = size(filter_hash["filt_index"])
    # unpack filter_hash
    f_ind   = filter_hash["filt_index"]  # (J, L) array of filters represented as index value pairs
    f_val   = filter_hash["filt_value"]

    # allocate output array
    dS1dp  = zeros(Float64, Nx, Nx, Nf)
    dS2dp  = zeros(Float64, Nx, Nx, Nf, Nf)

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
    im_fdc = conj.(im_fd_0)
    fim_fac = im_fd_0 ./ im_fdc


    #Implemented in the more memory intensive / parallelized ops way
    #fhash_indsarr = [hcat((x->[x[1], x[2]]).(fh)...)' for fh in fhash["filt_index"]] #Each elem gives you fhash["filt_index"] in non-CartesianIndices
    #dabsz = #Mem Req: Nf*Nx^2*sp*Nx^2 and sp~2%

    #Less Mem Route
    dS2dp = zeros(Float64, Nf, Nf, Nx, Nx)  # Lookup arrays for dS12 terms
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
                pnz_arr = vcat((x->[x[1], x[2]]').(pnz)...) #pnz*2: Assumption that the correct p to use in the phase are the indices!!! THIS IS WRONG instead create a kgrid and use pnz to index from that
                Φmat =  exp.((-2π*im)/Nx .* ((pnz_arr .- 1) * (xigrid .- 1))) #pnz*Nx^2
                f_v1pnz = f_v1[findall(in(pnz), f_i1)]
                f_v2pnz = f_v2[findall(in(pnz), f_i2)]
                t2 = real.(zarr[f1, pnz] .* Φmat)
                #println(size(f_v1pnz), size(t2), size(absz[f2, pnz]))
                term = 2*real.(sum((f_v2pnz.^2 .* f_v1pnz .*fim_fac[pnz]) .* t2, dims=1)) #p*Nx^2 -> 1*Nx^2
                dS2dp[f1, f2, :, :] = reshape(term/(Nx^2), (Nx, Nx))
            end
        end
    end
    return dS2dp
end
=#
#=
function realconv_test(Nx, ar1, ar2)
    soln = zeros(Float64, 8)
    for x=1:8:
        soln[x] = ar1 .* ar2[x-collect(1:1:8)]
=#


#=#Inner function code
#Code for img_reconfunc using wst_S2_deriv_sum: doesn't pass chi-square check even though inner machinery (derivsumcombtest, S1 and S2 separate, adding) all verified??




=#
