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
input=init
filter_hash=fhash
s_targ_mean=s2targ[3:end]
s_targ_invcov=invcovarS2[3:end, 3:end]
pixmask= fixmask
optim_settings=Dict([("iterations", 10)])
coeff_mask = nothing
println("Coeff mask:", (coeff_mask!=nothing))

(Nx, Ny)  = size(input)
if Nx != Ny error("Input image must be square") end

(Nf, )    = size(filter_hash["filt_index"])
if Nf == 0 error("filter hash corrupted") end

println("S2")
pixmask = pixmask[:] #flattened: Nx^2s
#cpvals = copy(input[:])[pixmask] #constrained pix values

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
#Does alloc mem here help?
#wtall = zeros(Float64, Nf+Nf^2)

function loss_func(img_curr)
    #size(img_curr) must be (Nx^2, 1)
    s_curr = DHC_compute(reshape(img_curr, (Nx, Nx)),  filter_hash, doS2=true, doS12=false, doS20=false)[coeff_mask]
    neglogloss = 0.5 .* (s_curr - s_targ_mean)' * s_targ_invcov * (s_curr - s_targ_mean)
    return neglogloss[1]
end

function dloss(storage_grad, img_curr)
    #storage_grad, img_curr must be (Nx^2, 1)
    s_curr = DHC_compute(reshape(img_curr, (Nx, Nx)), filter_hash, doS2=true, doS12=false, doS20=false)[coeff_mask]
    diff = s_curr - s_targ_mean
    wts1s2 = zeros(Nf+Nf^2)
    wtall = reshape(diff' * s_targ_invcov, num_freecoeff) #DEBUG: This isnt eval to zero when you test it outside
    #=wts1[coeff_mask[3:Nf+2]] .= wtall[1:num_freecoeffS1] #WARNING: Assumes that mean, var not being optimized
    wts2[coeff_mask[3+Nf:end]] .= wtall[num_freecoeffS1+1:end] #Len= Nz(coeff_mask). #Selection func for only S2 coeffs
    #Works because coeff_mask is a boolean mask
    #Since wst_S2_deriv_sum works with the full set of S2 weights, need to go back to Nf+Nf^2 length
    =#
    wts1s2[coeff_mask[3:end]] .= wtall
    dterm = Deriv_Utils.wst_S1S2_derivsum_comb(reshape(img_curr, (Nx, Nx)), filter_hash, wts1s2)
    storage_grad .= reshape(dterm, (Nx^2, 1))
    #=
    #diff = diff
    dS1S2 = Transpose(reshape(Deriv_Utils.wst_S1S2_derivfast(reshape(img_curr, (Nx, Nx)), filter_hash), (:, Nx^2))) #(Nf+Nf^2, Nx, Nx)->(Nx^2, Nf+Nf^2) Cant directly do this without handling S1 coeffs
    dS1S2 = dS1S2[:, coeff_mask[3:end]] #Nx^2 * |SelCoeff|
    dS1S2[pixmask, :] .= 0 #Zeroing out wrt fixed params
    term1 = s_targ_invcov * diff #(Nf+Nf^2) or |SelCoeff| x 1
    #WARNING: Uses the Deriv_Utils version of dS1S2. Need to rewrite if you want to use the DHC_2DUtils one.
    #println("In dLoss:",size(diff), size(term1), size(term2))
    #println(size(term1),size(term2),size(storage_grad))
    mul!(storage_grad, dS1S2, term1) #Nx^2x1=#
    #TODO: Move into one line
    #S1contrib = reshape(wst_S1_deriv(reshape(img_curr, (Nx, Nx)), filter_hash), (Nx^2, Nf)) * reshape(wts1, (Nf, 1))
    #S2contrib = reshape(Deriv_Utils.wst_S2_deriv_sum(reshape(img_curr, (Nx, Nx)), filter_hash, reshape(wts2, (Nf, Nf))), (Nx^2, 1))
    #storage_grad .=( S1contrib + S2contrib) #
    storage_grad[pixmask, 1] .= 0 # better way to do this by taking pixmask as an argument wst_s2_deriv_sum
    return diff, dterm# meansub, wtall, wts1, wts2, S1contrib, S2contrib, storage_grad
end



#Debugging stuff
println("Diff check")
eps = zeros(size(input))
eps[1, 2] = 1e-4
chisq1 = loss_func(input+eps./2)
chisq0 = loss_func(input-eps./2)
brute  = (chisq1-chisq0)/1e-4
df_brute = DHC_compute(reshape(input, (Nx, Nx)), filter_hash, doS2=true, doS12=false, doS20=false)[coeff_mask] - s_targ_mean
clever = reshape(zeros(size(input)), (Nx*Nx, 1))
df, _bar = dloss(clever, reshape(input, (Nx^2, 1)))
println("dS1S2comb check")
wts1s2 = zeros(Nf+Nf^2)
wtall = reshape(df' * s_targ_invcov, num_freecoeff)
wts1s2[coeff_mask[3:end]] .= wtall
dterm = Deriv_Utils.wst_S1S2_derivsum_comb(reshape(input, (Nx, Nx)), filter_hash, wts1s2)
dS1S2comb = Deriv_Utils.wst_S1S2_derivfast(reshape(input, (Nx, Nx)), filter_hash)
dS1dp = permutedims(dS1S2comb[1:Nf, :, :], [2, 3, 1])
dS2dp = permutedims(dS1S2comb[Nf+1:end, :, :], [2, 3, 1])
dS2dp = reshape(dS2dp, Nx, Nx, Nf^2)
sum2 = zeros(Float64, Nx, Nx)
for i=1:Nf*Nf sum2 += (dS2dp[:,:,i].*wts1s2[Nf+i]) end
term1 = reshape(dS1dp, (Nx^2, Nf)) * reshape(wts1s2[1:Nf], (Nf, 1))
dterm_brute = term1 + reshape(sum2, (Nx^2, 1))
println("Chisq Derve Check")
println("Brute:  ",brute)
println("Clever: ",clever[Nx*(1)+1])


res = optimize(loss_func, dloss, reshape(input, (Nx^2, 1)), ConjugateGradient(), Optim.Options(iterations = get(optim_settings, "iterations", 100), store_trace = true, show_trace = true))
result_img = zeros(Float64, Nx, Nx)
result_img = Optim.minimizer(res)
return reshape(result_img, (Nx, Nx))
=#
