using Plots

using Statistics
using BenchmarkTools
using Profile
using FFTW

using Optim
using Measures
using Images, FileIO

using LinearAlgebra
using SparseArrays
using FITSIO
using Distributions

# put the cwd on the path so Julia can find the module
push!(LOAD_PATH, pwd()*"/main")
using eqws
#using DHC_tests

@time fhash = fink_filter_hash(1, 8, nx=64, t=1, wd=1)



function realspace_filter(Nx, f_i, f_v)

    zarr = zeros(ComplexF64, Nx, Nx)
    for i = 1:length(f_i)
        zarr[f_i[i]] = f_v[i] # filter*image in Fourier domain
    end
    filt = ifft(zarr)  # real space, complex
    return filt
end


# spitballing here...

filt = realspace_filter(Nx, f_i, f_v)
filt2 = abs2.(filt)
predvar =ifft(fft(sigim.^2) .* fft(filt2))

function foobar(image, Ntrial, sigim)

    Nx    = size(image,1)
    fhash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1, Omega=true)
    psi_ind = fhash["psi_index"]
    filt_ind = psi_ind[3,3]

    f_ind = fhash["filt_index"]
    f_val = fhash["filt_value"]
    im_fd_0 = fft(image)
    zarr = zeros(ComplexF64,Nx,Nx)
    out  = zeros(ComplexF64,Nx,Nx,Ntrial)
    P = plan_ifft(im_fd_0)

    f = filt_ind
    S1tot = 0.0
    f_i = f_ind[f]  # CartesianIndex list for filter
    f_v = f_val[f]  # Values for f_i

    #for f = 1:Nf

    zval       = f_v .* im_fd_0[f_i]
    S1tot      = sum(abs2.(zval))
    zarr[f_i] .= zval        # filter*image in Fourier domain
    Zλ = P*zarr
    zarr[f_i] .= 0


    for itrial = 1:Ntrial

        noise      = sigim .* randn(Nx,Nx)
        dimage     = image+noise
        im_fd_0    = fft(dimage)
        zval       = f_v .* im_fd_0[f_i]
        S1tot      = sum(abs2.(zval))
        zarr[f_i] .= zval        # filter*image in Fourier domain

        #S1[f] = S1tot/(Nx*Ny)  # image power
        #im_rd_0_1[:,:,f] .= abs2.(P*zarr)
        out[:,:,itrial] = P*zarr
        zarr[f_i] .= 0

    end
    return out, Zλ
end
🙄=5
🌖angle = 10
sigim = ones(Nx,Nx).*10
#sigim[33,33:34].=1
out,Zλ = foobar(im,10000,sigim)
bar    = std(out,dims=3)[:,:,1]


Nx = 256
bar = rand(2500,100)
fhash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1, Omega=true)
#M = S2_iso_matrix(fhash, sparse=false)
#Ms = sparse(M)
Ms = fhash["S2_iso_mat"]
M = Array(Ms)
@benchmark M*bar
@benchmark Ms*bar

# M*bar test
# Nx         full   sparse
# 128       0.8 ms     4 μs
# 128x100   45  ms   500 μs
# 256       1.9 ms     7 μs
# 256x100   90  ms   800 μs



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

    ## 1st Order
    im_fd_0 = fft(image)  # total power=1.0

    # unpack filter_hash
    f_ind   = filter_hash["filt_index"]  # (J, L) array of filters represented as index value pairs
    f_val   = filter_hash["filt_value"]

    f_ind_rev = [[CartesianIndex(mod1(Nx+2 - ci[1],Nx), mod1(Nx+2 - ci[2],Nx)) for ci in f_Nf] for f_Nf in f_ind]

    # tmp arrays for all tmp[f_i] = arr[f_i] .* f_v opns
    zarr1     = zeros(ComplexF64, Nx, Nx)
    fz_fψ1    = zeros(ComplexF64, Nx, Nx)
    fterm_a   = zeros(ComplexF64, Nx, Nx)
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
            #println("Term2",fterm_ct2)

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




function derivtestS1S2(Nx)
    eps = 1e-4
    fhash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1)
    im = rand(Float64, Nx, Nx).*0.1
    im[6,6]=1.0
    im1 = copy(im)
    im0 = copy(im)
    im1[2,3] += eps/2
    im0[2,3] -= eps/2
    dS1dp, dS2dp = wst_S1S2_derivfast(im, fhash)
    println(size(dS2dp))
    der0=DHC_compute(im0,fhash,doS2=true)
    der1=DHC_compute(im1,fhash,doS2=true)
    dS = (der1-der0) ./ eps

    Nf = length(fhash["filt_index"])
    i0 = 3+Nf
    blarg = dS2dp[2,3,:,:]
    diff = dS[i0:end]-reshape(blarg,Nf*Nf)
    println(dS[i0:end])
    println("and")
    println(blarg)
    println("stdev: ",std(diff))
    println()
    println(diff)

    return
end

derivtestS1S2(8)


println(1)

function derivtest2(Nx)
    eps = 1e-5
    fhash = fink_filter_hash(1, 8, nx=Nx, t=1, wd=1)
    im = rand(Float64, Nx, Nx).*0.1
    im[6,6]=1.0
    im1 = copy(im)
    im0 = copy(im)
    im1[2,3] += eps/2
    im0[2,3] -= eps/2
    dS20dp = wst_S20_deriv(im, fhash)

    der0=eqws_compute(im0,fhash,doS2=false,doS20=true,norm=false)
    der1=eqws_compute(im1,fhash,doS2=false,doS20=true,norm=false)
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

    return
end

derivtest2(8)



Nx=1024
fhash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1)
im = zeros(Float64, Nx, Nx)
im[6,6]=1.0
@benchmark blarg = wst_S1_deriv(im, fhash)


# S1 deriv time, Jan 30 version (old) compared to Feb 14 version from NM
# Nx    old      Feb 14                            Holyfink01
#   32     17 ms   0.5 ms                        FFTW nth
#   64     34 ms   3.5 ms                        ~10% effect
#  128    115 ms    20 ms                            9 ms
#  256    520 ms    92 ms                           60 ms
#  512   2500 ms   720 ms   540 ms with 2 threads  370 ms
# 1024   9500 ms  3300 ms  2500 ms with 2         2100 ms

Nx = 32
fhash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1)
im = zeros(Float64, Nx, Nx);
im[6,6]=1.0
@benchmark blarg,blafb2 = wst_S1S2_derivfast(im, fhash)

# S20 vs S2 deriv time, Feb 15
# Nx     S20
#   8      1 ms    27 ms
#  16      7      104
#  32     50      300
#  64    400 ms   1.1 sec
# 128    3.3 sec  5.0
# 256   17.2 sec
# 512   ---



function S20test(fhash)
    (Nf, )    = size(fhash["filt_index"])
    Nx        = fhash["npix"]
    im = rand(Nx,Nx)
    mywts = rand(Nf*Nf)

    # this is symmetrical, but not because S20 is symmetrical!
    wtgrid = reshape(mywts, Nf, Nf) + reshape(mywts, Nf, Nf)'
    wtvec  = reshape(wtgrid, Nf*Nf)

    # Use new faster code
    sum1 = wst_S20_deriv_sum(im, fhash, wtgrid, 1)

    # Compare to established code
    dS20 = reshape(wst_S20_deriv(im, fhash, 1),Nx,Nx,Nf*Nf)
    sum2 = zeros(Float64, Nx, Nx)
    for i=1:Nf*Nf sum2 += (dS20[:,:,i].*mywts[i]) end

    println("Stdev: ",std(sum1-sum2))

    return
end

Nx = 16
fhash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1)
(Nf, )    = size(fhash["filt_index"])
im = rand(Nx,Nx);
wts = rand(Nf,Nf);
#@benchmark wst_S20_deriv(im, fhash, 1)

@benchmark wst_S20_deriv_sum(im, fhash, wts, 1)


Profile.clear()
@profile wst_S20_deriv_sum(im, fhash, wts, 1)
Juno.profiler()


# S20 deriv time, Mac laptop
# Nx     Jan 30  Feb 14  Feb 21    HF01
#   8     28 ms   1 ms
#  16    112      7      0.6 ms    0.4 ms
#  32    320     50        2       1.3
#  64   1000    400       17        10
# 128   5 sec     3.3 s   90        50
# 256   ---      17.2 s  0.65 s   0.48 s
# 512   ---               2.9 s   1.85



print(1)


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

    der0=DHC_compute(im0,fhash,doS2=false)
    der1=DHC_compute(im1,fhash,doS2=false)
    dS = (der1-der0) ./ eps


    diff = dS[3:end]-blarg[2,3,:]
    println(dS[3:end])
    println("and")
    println(blarg[2,3,:])
    println("stdev: ",std(diff))

    #plot(dS[3:end])
    #plot!(blarg[2,3,:])
    #plot(diff)
    return
end

derivtest(128)



Nx=32
im = rand(Nx,Nx)
@time fhash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1)
@benchmark Sarr = DHC_compute(im, fhash, doS2=false)  # 0.36 ms
@benchmark Sarr = DHC_compute(im, fhash, doS2=true)   # 6.4 ms
@benchmark Sarr = DHC_compute(im, fhash, doS2=false, doS12=true)  # 0.7 ms
@benchmark Sarr = DHC_compute(im, fhash, doS2=false, doS20=true)  # 2.1 ms


Nx=64
im = rand(Nx,Nx)
@time fhash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1)
@benchmark Sarr = DHC_compute(im, fhash, doS2=false)  # 1.3 ms
@benchmark Sarr = DHC_compute(im, fhash, doS2=true)   # 19 ms
@benchmark Sarr = DHC_compute(im, fhash, doS2=false, doS12=true)  # 2.8 ms
@benchmark Sarr = DHC_compute(im, fhash, doS2=false, doS20=true)  # 6.6 ms



Nx=256
im = rand(Nx,Nx)
@time fhash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1)
@benchmark Sarr = DHC_compute(im, fhash, doS2=false)  # 25 ms
@benchmark Sarr = DHC_compute(im, fhash, doS2=true)   # 375 ms
@benchmark Sarr = DHC_compute(im, fhash, doS2=false, doS12=true)  # 55 ms
@benchmark Sarr = DHC_compute(im, fhash, doS2=false, doS20=true)  # 130 ms


Nx = 64
fhash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1)
im = rand(Nx,Nx)
S_targ = DHC_compute(im, fhash, doS2=false)

im = rand(Nx,Nx)







function wst_synth(im_init, fixmask)
    # fixmask -  0=float in fit   1=fixed

    function wst_synth_chisq(vec_in)

        thisim = copy(im_init)
        thisim[indfloat] = vec_in

        Sarr  = DHC_compute(thisim, fhash, doS2=false)
        diff  = (Sarr .- S_targ)[3:end]

        # should have some kind of weight here
        chisq = diff'*diff
        return chisq

    end

    function wst_synth_dchisq(storage, vec_in)

        thisim = copy(im_init)
        thisim[indfloat] = vec_in
        dSdp   = wst_S1_deriv(thisim, fhash)
        S1arr  = DHC_compute(thisim, fhash, doS2=false)
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
S_targ = DHC_compute(im, fhash, doS2=false)

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
    #RGBA_img = load("/n/home08/dfink/DHC/scratch_DF/t115_clean_small.png")
    img = reinterpret(UInt8,rawview(channelview(RGBA_img)))

    return img[1,:,:]
end


function readphoto(N)

    flist = ["IMG_2320.png",  # flower 1
             "IMG_3995.png",  # flower 2
             "IMG_4058.png",  # butterfly
             "IMG_4971.png",  # crab
             "IMG_5939.png",  # canyon
             "IMG_6764.png"]  # eclipse

    RGBA_img = load(pwd()*"/../WSTphotos/"*flist[N])
    #RGBA_img = load("/n/home08/dfink/DHC/scratch_DF/t115_clean_small.png")
    img = reinterpret(UInt8,rawview(channelview(RGBA_img)))
    temp = (Float64.(sum(img,dims=1)[1,:,:]))[end:-1:1,:]
    mintemp = minimum(temp)
    maxtemp = maximum(temp)
    output = (temp.-mintemp).*(255.0 / (maxtemp-mintemp))
    return output
end


function wst_synthS20(im_init, fixmask, S_targ, S20sig; iso=false)
    # fixmask -  0=float in fit   1=fixed

    function wst_synth_chisq(vec_in)

        thisim = copy(im_init)
        thisim[indfloat] = vec_in

        M20 = fhash["S2_iso_mat"]

        S20 = eqws_compute(thisim, fhash, doS2=false, doS20=true, norm=false, iso=iso)
        i0 = 3+(iso ? N1iso : Nf)
        diff  = ((S20[i0:end] - S_targ)./S20sig)



        # should have some kind of weight here
        chisq = diff'*diff
        if mod(iter,10) == 0 println(iter, "   ", chisq) end
        iter += 1
        return chisq

    end

    function wst_synth_dchisq(storage, vec_in)

        thisim = copy(im_init)
        thisim[indfloat] = vec_in

        i0 = 3+(iso ? N1iso : Nf)
        S20arr = (eqws_compute(thisim, fhash, doS2=false, doS20=true, norm=false, iso=iso))[i0:end]
        diff   = (S20arr - S_targ)./(S20sig.^2)
        if iso
            diffwt = diff[indiso]
        else
            diffwt = diff
        end

        wtgrid = 2 .* (reshape(diffwt, Nf, Nf) + reshape(diffwt, Nf, Nf)')

        dchisq_im = wst_S20_deriv_sum(thisim, fhash, wtgrid, 1)


        # dSdp matrix * S1-S_targ is dchisq
        # dchisq_im = (reshape(dS20dp, Nx*Nx, Nf*Nf) * diff).*2
        # dchisq_im = (dS20dp * diff).*2
        dchisq = reshape(dchisq_im, Nx, Nx)[indfloat]

        storage .= dchisq
    end

    iter = 0
    (Nx, Ny)  = size(im_init)
    if Nx != Ny error("Input image must be square") end
    (N1iso, Nf)    = size(fhash["S1_iso_mat"])
    if iso
        M20 = fhash["S2_iso_mat"]
        (Nwt,Nind) = size(M20)
        indiso = zeros(Int64,Nind)
        for ind = 1:Nind  indiso[M20.colptr[ind]] = M20.rowval[ind]  end
    end


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
    #res = optimize(wst_synth_chisq, wst_synth_dchisq, copy(vec_init), BFGS())
    res = optimize(wst_synth_chisq, wst_synth_dchisq, copy(vec_init),
        ConjugateGradient(), Optim.Options(iterations=1000))

    # copy results into pixels of output image
    im_synth = copy(im_init)
    im_synth[indfloat] = Optim.minimizer(res)
    println(res)

    return im_synth
end


function S20_weights(im, fhash, Nsam=10; iso=iso)

    (Nx, Ny)  = size(im)
    if Nx != Ny error("Input image must be square") end
    (N1iso, Nf)    = size(fhash["S1_iso_mat"])

    # fhash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1)

    S20   = eqws_compute(im, fhash, doS2=false, doS20=true, norm=false, iso=iso)
    Ns    = length(S20)
    S20arr = zeros(Float64, Ns, Nsam)
    for j=1:Nsam
        noise = rand(Nx,Nx)
        S20arr[:,j] = eqws_compute(im+noise, fhash, doS2=false, doS20=true, norm=false, iso=iso)
    end

    wt = zeros(Float64, Ns)
    for i=1:Ns
        wt[i] = std(S20arr[i,:])
    end

    return wt
end



# read dust map
dust = Float64.(readdust())
dust = dust[1:256,1:256]

Nx     = 32
doiso  = true
initimg = imresize(readphoto(2),Nx,Nx)
initimg = 250 .* (initimg./maximum(initimg))

fhash = fink_filter_hash(1, 8, nx=Nx, t=1, wd=1, Omega=true)
(N1iso, Nf)    = size(fhash["S1_iso_mat"])
i0 = 3+(doiso ? N1iso : Nf)
im    = imresize(dust,(Nx,Nx))
fixmask = rand(Nx,Nx) .< 0.01


S_targ = eqws_compute(im, fhash, doS2=false, doS20=true, norm=false, iso=doiso)
S_targ = S_targ[i0:end]

init = copy(im)
floatind = findall(fixmask .==0)
fixind = findall(fixmask .==1)
#init[floatind] .+= rand(length(floatind)).*50 .-25
#init[floatind] = (fftshift(im))[floatind]

init = copy(initimg)
init .*= (std(dust)/std(init))
init .+= (mean(dust)-mean(init))
init[fixind] .= im[fixind]
S20sig = S20_weights(im, fhash, 100, iso=doiso)
S20sig = S20sig[i0:end]
foo = wst_synthS20(init, fixmask, S_targ, S20sig, iso=doiso)
S_foo = eqws_compute(foo, fhash, doS2=false, doS20=true, norm=false, iso=doiso)


Profile.clear()
@profile foo=wst_synthS20(init, fixmask, S_targ, S20sig, iso=doiso)
Juno.profiler()


plot_synth_QA(im, init, foo, fhash)


function writestuff(im, init, im_out, fhash)
    f = FITS("newfile.fits", "w")
    write(f, im)
    write(f, init)
    write(f, im_out)
    close(f)
end


function scramble(im)

    fim = fft(im)
    (Nx,Ny) = size(fim)
    ϕ = 2π .* rand(Nx,Ny) .* 0.5
    ϕ[1,1] = 0.0
    i = complex(0,1)
    return real.(ifft(fim .* exp.(i*ϕ)))

end



function plot_synth_QA(ImTrue, ImInit, ImSynth, fhash; fname="test256.png")

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
    clim  = (0,200)
    clim2 = (0,200).-100

    # -------- 6 panel QA plot
    plot1(ps, ImTrue, clim=clim, label="True")
    plot1(ps, ImSynth, clim=clim, label="Synth")
    plot1(ps, ImInit, clim=clim, label="Init")
    plot1(ps, ImInit-ImTrue, clim=clim2, label="Init-True")
    plot1(ps, ImInit-ImSynth, clim=clim2, label="Init-Synth")
    plot1(ps, ImSynth-ImTrue, clim=clim2, label="Synth-True")

    myplot = plot(ps..., layout=(3,2), size=(1400,2000))
    savefig(myplot, fname)
end

plot_synth_QA(im, init, foo, fhash)



function sfft(Nx,Nind)
    zarr = zeros(ComplexF64, Nx, Nx)
    out  = zeros(ComplexF64, Nx, Nx, Nind)
    for i = 1:Nind
        zarr[1,i] = 1.0
        out[:,:,i] = ifft(zarr)
    end
    return out
end

function stest(fhash, f2, uvec)
    f_ind   = fhash["filt_index"]  # (J, L) array of filters represented as index value pairs
    f_val   = fhash["filt_value"]
    Nx = ?
    Nf = ?
    #for f2 = 1:Nf
        f_i = f_ind[f2]  # CartesianIndex list for filter
        f_v = f_val[f2]  # Values for f_i
        #uvec = im_rdc[:,:,f2] ./ im_rd[:,:,f2]


        # pre-compute some exp(ikx)
        expikx = zeros(ComplexF64, Nx, Nx, Nf)
        for i=1:Nf expikx[:,:,i] = ifft()


        A = reshape(im_rd, Nx*Nx, Nf) .* uvec
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

    return
end


function Fourierbasis2D(Nx, f_i)
    Nf = length(f_i)

    [[2,3]'*[ci[1],ci[2]] for ci in f_i]
    return
end

function f5(xs, ys, kx, ky)
    lx, ly = length(xs), length(ys)
    res = Array{ComplexF64, 2}(undef,lx*ly, 2)
    ind = 1
    ikx = Complex(0,kx)
    iky = Complex(0,ky)
    for y in ys, x in xs
        res[ind, 1] = exp(ikx*x + iky*y)
        #res[ind, 2] = y
        ind += 1
    end
    return res
end

# using dchisq function with S20
# size   t(BFGS) t(LBFGS) [sec]
# 8x8      30
# 16x16    90                                           iso on HF01
# 32x32   189                           74                  32
# 64x64  1637                          531      642 iso
# 128x128                              2 hrs
# 256x256                                                 8800 sec

# using new wst_S20_deriv_sum on Feb 21, 2021
# size     t(GG)  t(CG)ISO  HF01  [sec]   mem
# 8x8
# 16x16
# 32x32
# 64x64   134
# 128x128           221     116
# 256x256                   712           3.5 GB
# 512x512
# laptop 4800 sec


function GP_constrained_2D(im, mask, covar; seed=false)
    # Compute a Gaussian Process mean or draw from covariance
    # im    - input image.  Condition on mask==0 pixels
    # mask  - 0=good, 1=pixels to predict
    # covar - covariance matrix
    # seed  - set seed to return random draw; seed=false returns mean
    # follows notation from Rasmussan & Williams Chap. 2
    # http://www.gaussianprocess.org

    k      = findall(mask .== 0)
    nk     = length(k)
    kstar  = findall(mask .!= 0)
    nkstar = length(kstar)

    cov_kk         = (covar[:, k])[k, :]
    cov_kkstar     = (covar[:, kstar])[k, :]  # dim [nk, nkstar]
    cov_kstark     = (covar[:, k])[kstar, :]
    cov_kstarkstar = (covar[:, kstar])[kstar, :]

    # could do a smarter inverse here for positive definite matrix!
    icov_kk = inv(cov_kk)  # slow step

    # mean prediction: condition on k pixels and predict kstar pixels
    kstarpred = cov_kkstar' * (icov_kk * im[k])

    if (seed != false)
        # -------- get the prediction covariance
        predcovar = cov_kstarkstar - (cov_kstark*icov_kk*cov_kkstar)
        println("Cholesky")
        chol = cholesky(Hermitian(predcovar))
        noise = chol.L*rand(Normal(), nkstar)
        draw = copy(im)
        draw[kstar] = kstarpred+noise    # a draw from the distribution
        return draw
    else
        pred = copy(im)
        pred[kstar] = kstarpred
        return pred         # mean image
    end
end


function dust_covar_matrix(im)
    # derive covariance matrix from [Nx,Nx,Nimage] array

    (nx,__,Nslice) = size(im)
    eps=1E-6

    # make it mean zero with eps noise, otherwise covar is singular
    println("Mean zero")
    for i=1:Nslice  im[:, :, i] .-= (mean(im[:, :, i])+(rand()-0.5)*eps) end

    #println("Unit variance")
    #for i=1:Nslice  im[:, :, i] ./= std(im[:, :, i]) end

    dat = reshape(im, nx*nx, Nslice)

    # covariance matrix
    println("covariance")
    covar = (dat * dat') ./ Nslice

    # Check condition number
    println("Condition number  ", cond(covar))
    return covar

end



function mldust()

    # read FITS file with images
    # file in /n/fink2/dfink/mldust/dust10000.fits
    #     OR  /n/fink2/dfink/mldust/dust100000.fits

    fname = "dust10000.fits"
    f = FITS(fname, "r")
    big = read(f[1])

    (_,__,Nslice) = size(big)
    println(Nslice, " slices")

    #nx = 96
    #im = Float64.(big[11:11+nx-1, 11:11+nx-1, :])
    #nx = 48

    # rebin it to something smaller and take the log
    nx = 64
    im = log.(imresize(Float64.(big), nx, nx, Nslice))

    covar = dust_covar_matrix(im)

    #writefits, 'covar48_new.fits', covar

    # Cholesky factorization; chol.U and chol.L will be upper,lower triangular matrix
    println("Cholesky")
    chol = cholesky(covar)

    # generate some mock maps
    Nmock = 800
    ran = rand(Normal(), nx*nx, Nmock)

    recon = reshape(chol.L*ran, nx, nx, Nmock)

    return im, covar
end

# run these lines to play with GP prediction
images, covar = mldust()
Nx   = size(images,1)
mask = rand(Nx*Nx) .> 0.1
im   = images[:,:,5]
mnpred = GP_constrained_2D(im, mask, covar)
draw   = GP_constrained_2D(im, mask, covar, seed=1)

heatmap(im,clim=(-0.6,0.6))
heatmap(draw,clim=(-0.6,0.6))
heatmap(mnpred,clim=(-0.6,0.6))


doiso  = true
fhash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1, Omega=true)
(N1iso, Nf)    = size(fhash["S1_iso_mat"])
i0 = 3+(doiso ? N1iso : Nf)
fixmask = reshape(mask .==0, Nx,Nx)

S_targ = DHC_compute(im, fhash, doS2=false, doS20=true, norm=false, iso=doiso)
S_targ = S_targ[i0:end]

floatind = findall(fixmask .==0)
fixind = findall(fixmask .==1)

# use GP draw as initial guess
init = copy(draw)
init[fixind] .= im[fixind]
S20sig = S20_weights(im, fhash, 100, iso=doiso)
S20sig = S20sig[i0:end]
foo = wst_synthS20(init, fixmask, S_targ, S20sig, iso=doiso)
S_foo = DHC_compute(foo, fhash, doS2=false, doS20=true, norm=false, iso=doiso)

heatmap(foo,clim=(-0.6,0.6))
heatmap(init,clim=(-0.6,0.6))

heatmap(exp.(im),clim=(0.5,1.8))
heatmap(exp.(foo),clim=(0.5,1.8))
heatmap(exp.(init),clim=(0.5,1.8))



function my_DHC_compute(image::Array{Float64,2}, filter_hash::Dict, filter_hash2::Dict=filter_hash, sigim=nothing;
    doS2::Bool=true, doS12::Bool=false, doS20::Bool=false, norm=true, iso=false, FFTthreads=2)
    # image        - input for WST
    # filter_hash  - filter hash from fink_filter_hash
    # filter_hash2 - filters for second order.  Default to same as first order.
    # doS2         - compute S2 coeffs
    # doS12        - compute S2 coeffs
    # doS20        - compute S2 coeffs
    # norm         - scale to mean zero, unit variance
    # iso          - sum over angles to obtain isotropic coeffs

    # Use 2 threads for FFT
    FFTW.set_num_threads(FFTthreads)

    # array sizes
    (Nx, Ny)  = size(image)
    if Nx != Ny error("Input image must be square") end
    (Nf, )    = size(filter_hash["filt_index"])
    if Nf == 0  error("filter hash corrupted") end
    @assert Nx==filter_hash["npix"] "Filter size should match npix"
    @assert Nx==filter_hash2["npix"] "Filter2 size should match npix"

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
    if norm
        norm_im ./= sqrt(Nx*Ny*S0[2])
    else
        norm_im = copy(image)
    end

    append!(out_coeff,S0[:])

    ## 1st Order
    im_fd_0 = fft(norm_im)  # total power=1.0

    # unpack filter_hash
    f_ind   = filter_hash["filt_index"]  # (J, L) array of filters represented as index value pairs
    f_val   = filter_hash["filt_value"]
    fvar = fft(sigim.^2)
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
                fpsi = fft(abs2.(realspace_filter(Nx, f_ind[f], f_val[f])))
                extrapower = ifft(fvar.*fpsi)
                im_rd_0_1[:,:,f] .= abs2.(P*zarr) .+ abs.(extrapower)
            #    im_rd_0_1[:,:,f] .= abs2.(P*zarr) .+ sigim.^2 .*(sum(f_v.^2)/(Nx*Nx))
            end
            zarr[f_i] .= 0
        end
    end

    append!(out_coeff, iso ? filter_hash["S1_iso_mat"]*S1 : S1)


    # we stored the abs()^2, so take sqrt (this is faster to do all at once)
    if anyrd im_rd_0_1 .= sqrt.(im_rd_0_1) end

    Mat2 = filter_hash["S2_iso_mat"]
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
        append!(out_coeff, iso ? Mat2*S2[:] : S2[:])
    end

    # Fourier domain 2nd order
    if doS12
        Amat = reshape(im_fdf_0_1, Nx*Ny, Nf)
        S12  = Amat' * Amat
        append!(out_coeff, iso ? Mat2*S12[:] : S12[:])
    end

    # Real domain 2nd order
    if doS20
        Amat = reshape(im_rd_0_1, Nx*Ny, Nf)
        S20  = Amat' * Amat
        append!(out_coeff, iso ? Mat2*S20[:] : S20[:])
    end

    return out_coeff
end


function psi_matrix(fhash)
    (N1iso, Nf)    = size(fhash["S1_iso_mat"])
    Nx = fhash["npix"]
    rfilt = zeros(ComplexF64, Nx, Nx, Nf)
    f_ind = fhash["filt_index"]
    f_val = fhash["filt_value"]

    for i = 1:Nf
        rfilt[:,:,i] = realspace_filter(Nx, f_ind[i], f_val[i])
    end
    rfilt = abs.(reshape(rfilt,Nx*Nx,Nf))
    prod = rfilt' * rfilt

    return prod
end

# read dust map
dust = Float64.(readdust())
dust = dust[1:256,1:256]

Nx     = 16
Nsam   = 40000
doiso  = false
doS2   = false
doS20  = true
img    = imresize(dust,Nx,Nx)
fhash = fink_filter_hash(1, 8, nx=Nx, t=1, wd=1, Omega=true)
(N1iso, Nf)    = size(fhash["S1_iso_mat"])
(N2iso, _)    = size(fhash["S2_iso_mat"])
i0 = doiso ? 3+N1iso : 3+Nf
sigim = zeros(Nx,Nx)  # sigma for each pixel
sigim[1:Nx÷2,:] .= 20
#sigim = zeros(Nx,Nx)  # sigma for each pixel
#sigim[Nx÷2,Nx÷2] = 10
#sigim[2,5] = 10

img0  = img .- mean(img)
Smean, Scov = S20_noisecovar(img0, fhash, sigim, Nsam, iso=doiso, doS2=doS2, doS20=doS20)

dS20dp = wst_S20_deriv(img0, fhash)
Scoeff = (eqws_compute(img0, fhash,  doS2=doS2, doS20=doS20, norm=false, iso=doiso))[i0:end]
Stry = (my_DHC_compute(img0, fhash, fhash, sigim,  doS2=doS2, doS20=doS20, norm=false, iso=doiso))[i0:end]

G = reshape(dS20dp, Nx*Nx, Nf*Nf) .* reshape(sigim, Nx*Nx, 1)
pred = G'*G
fracerr = (Scov - pred) ./ Scov

fval = fhash["filt_value"]
psi2 = [sum(vals.^2)/Nx^2 for vals in fval]
diff = Smean-Scoeff

prod = psi_matrix(fhash)

plot(diff)
plot!((Stry-Scoeff))  # why 0.66 ????


plot!(reshape(prod.*25600 .*2,Nf*Nf))




filt1 = realspace_filter(Nx, f_ind[1], f_val[1])
filt2 = realspace_filter(Nx, f_ind[3], f_val[3])

img = randn(Nx,Nx)
img0  = img .- mean(img)

Z1 = ifft(fft(img0).*fft(filt1))
cplot(Z1)
cplot!(Z1 + 2. *filt1)


Z2 = ifft(fft(img0).*fft(filt2))

heatmap(abs.(Z1))
S2 = sum(abs.(Z1).*abs.(Z2))
S2c = sum(abs.(Z1+filt1).*abs.(Z2+filt2))

S2all = [sum(abs.(Z1+α.*filt1).*abs.(Z2+α.*filt2)) for α in randn(100000) ]

foo=Z1.*Z2 + filt1.*Z2 + filt2.*Z1 + filt1.*filt2

function cplot(Zarr)
    ZZ = reshape(Zarr,length(Zarr))
    plot(real.(ZZ), imag.(ZZ), seriestype=:scatter,size=(300,300),aspect_ratio=1)
end

function cplot!(Zarr)
    ZZ = reshape(Zarr,length(Zarr))
    plot!(real.(ZZ), imag.(ZZ), seriestype=:scatter)
end



##########################
# Bruno's method
##########################


# Bruno's method: estimate mean and covariance from noise realizations (slow)
function S20_noisecovar(im, fhash, σmap, Nsam=10; iso=iso, doS2=false, doS20=false)

    (Nx, Ny)  = size(im)
    if Nx != Ny error("Input image must be square") end
    (N1iso, Nf) = size(fhash["S1_iso_mat"])
    (N2iso, _)  = size(fhash["S2_iso_mat"])
    # fhash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1)
    if iso
        Sarr = zeros(Float64, N2iso, Nsam)
        i0 = 3+N1iso
        Smask = nothing # not implemented yet
    else
        # triangle mask
        tri_mask = reshape(tril(trues(Nf,Nf)),Nf*Nf)
        #tri_mask[1,1] = false
        Smask = vcat(falses(2+Nf), tri_mask)
        # sum(S1[jmax,l=odd]) = sum(S1[jmax,l=even]) so we must remove one more to avoid redundancy
        #Smask[19] = false
        Sarr  = zeros(Float64, sum(Smask), Nsam)
    end

    for j=1:Nsam
        noise = randn(Nx,Nx) .* σmap
        Scoeff = eqws_compute(im+noise, fhash, doS2=doS2, doS20=doS20, norm=false, iso=iso)
        Sarr[:,j] = (Scoeff[Smask])
    end
    Smean = mean(Sarr,dims=2)
    Smean = reshape(Smean, length(Smean))
    ΔS    = Sarr .- Smean
    Scov  = ΔS*ΔS' ./ (Nsam-1)
    return Smean, Scov, Smask
end


function bruno_synthS2R(im_targ, im_ivar, S_targ, S_icov, S_mask, fhash; im_init=nothing, iso=false)
    # im_targ - target (noisy) image
    # im_ivar - pixelwise inverse variance of im_targ
    # S_targ  - target WST vector
    # S_icov  - covariance of S_targ due to noise expressed in im_ivar
    # S_mask  - Boolean mask of coefficients to include
    # im_init - (optional) starting image for gradient descent
    # iso     - use only ISO coeffs (not implemented!)

    function wst_synth_chisq(vec_in)
        # As global vars need
        # fhash, iso, i0, Nx, S_icov, verbose, im_targ, im_ivar
        # could use Nayantara's wrapper here
        image_in = reshape(vec_in, Nx, Nx)
        #if mod(iter,20) == 0
        #    pframe = heatmap(image_in)
        #    display(pframe)
        #end

        ΔI    = vec_in - reshape(im_targ,Nx*Nx)
        im_χ2 = (reshape(im_ivar,1,Nx*Nx) * (ΔI.^2))[1]

        # Right now this only does S2R, but we could generalize
        S_vec  = (eqws_compute(image_in, fhash, doS2=false, doS20=true, norm=false, iso=iso))[S_mask]

        S_diff = (S_vec- S_targ)
        chisq  = (S_diff' * S_icov * S_diff)[1]

        if verbose != nothing
            if mod(iter,20) == 0
                println(iter, "   ", "chi2 function   S_TS: ", chisq, "   im_TS: ",im_χ2)
            end
            iter += 1
        end

        return Sχfac*chisq+im_χ2
        #return im_χ2
    end

    function wst_synth_dchisq(storage, vec_in)
        # As global vars need
        # fhash, iso, i0, Nx, S_icov, verbose
        image_in = reshape(vec_in, Nx, Nx)
        ΔI    = vec_in - reshape(im_targ,Nx*Nx)

        dchisq_image = 2 .* (reshape(im_ivar,Nx*Nx) .* ΔI)

        S_vec  = (eqws_compute(image_in, fhash, doS2=false, doS20=true, norm=false, iso=iso))[S_mask]
        S_diff = (S_vec - S_targ)
        if iso
            diffwt = S_icov * S_diff[indiso]  # not checked
        else
            diffwt = S_icov * S_diff
        end
        # this is a hack -- paint values on grid so we can transpose
        gwt = zeros(Nf,Nf)
        gwt[tril(trues(Nf,Nf))] = diffwt

        # derivative of chisq is 2*Sdiff*Cinv*grad(S) (see paper)
        # wtgrid = 2 .* (reshape(diffwt, Nf, Nf) + reshape(diffwt, Nf, Nf)')
        wtgrid = 2 .* (gwt + gwt')
        dchisq_im = wst_S20_deriv_sum(image_in, fhash, wtgrid, 1)

        # dSdp matrix * S1-S_targ is dchisq
        # dchisq_im = (reshape(dS20dp, Nx*Nx, Nf*Nf) * diff).*2
        # dchisq_im = (dS20dp * diff).*2
        # dchisq = reshape(dchisq_im, Nx, Nx)[indfloat]
        dchisq = reshape(dchisq_im,length(dchisq_im))
         storage .= Sχfac.*dchisq + dchisq_image
        #storage .= dchisq_image
    end

    # preliminaries
    verbose = true
    iter = 0
    Sχfac = 1.0
    (Nx, Ny)  = size(im_targ)
    if Nx != Ny error("Input image must be square") end

    # get number of filters (keep this iso-compatible for the moment)
    (N1iso, Nf) = size(fhash["S1_iso_mat"])
    if iso
        M20 = fhash["S2_iso_mat"]
        (Nwt,Nind) = size(M20)
        indiso = zeros(Int64,Nind)
        for ind = 1:Nind  indiso[M20.colptr[ind]] = M20.rowval[ind]  end
    end
    i0 = 3+(iso ? N1iso : Nf)

    # initialize vector for gradient descent, default to zeros
    vec_init = (im_init == nothing) ? zeros(Nx*Nx) : reshape(im_init,Nx*Nx)

    # spot check that derivative works
    if verbose
        eps = zeros(size(vec_init))
        eps[1] = 1e-4
        chisq1 = wst_synth_chisq(vec_init+eps./2)
        chisq0 = wst_synth_chisq(vec_init-eps./2)
        brute  = (chisq1-chisq0)[1]/eps[1]
        clever = zeros(size(vec_init))
        println("clever",size(clever))
        ___bar = wst_synth_dchisq(clever, vec_init)
        println("Brute:  ",brute, "   Clever: ",clever[1], "   Diff: ",abs(brute-clever[1]))

    end
    println("vec_init  ",size(vec_init))
    # call optimizer
    #res = optimize(wst_synth_chisq, wst_synth_dchisq, copy(vec_init), BFGS())
    res = optimize(wst_synth_chisq, wst_synth_dchisq, copy(vec_init),
        ConjugateGradient(), Optim.Options(iterations=2000,f_tol=1E-12))

    # copy results into pixels of output image
    im_synth = Optim.minimizer(res)
    println(res)

    return im_synth
end


function write_results(outname, results)
    #(img0, im_targ, im_ivar, im_init, im_denoise, Smean, Scov, Smask, S_true, S_noisy)
    f = FITS(outname, "w")
    for arr in results
        write(f, arr)
    end
    close(f)
end


function read_results(fname)
    f = FITS(fname, "r")
    arr = [read(hdu) for hdu in f]
    close(f)
    return arr
end


function bruno_plot_synth(dat; fname="test.png")

    # -------- define plot1 to append plot to a list
    function plot1(ps, image; clim=nothing, bin=1.0, fsz=14, label=nothing)
        # ps    - list of plots to append to
        # image - image to heatmap
        # clim  - tuple of color limits (min, max)
        # bin   - bin down by this factor, centered on nx/2+1
        # fsz   - font size
        marg   = 2mm
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
            annotate!(lims'*[.96,.04],lims'*[.09,.91],text(label,:left,:white,28))
        end
        return
    end

    # -------- initialize array of plots
    ps   = []
    clim  = (-75,125)
    clim2 = clim
    im_true = dat[1]
    im_targ = dat[2]
    im_sig = 1 ./sqrt.(dat[3])
    im_init = dat[4][:,:,1]
    im_denoise = dat[5][:,:,4] # number 4 starts near answer

    #(img0, im_targ, im_ivar, im_init, im_denoise, Smean, Scov, UInt8.(Smask), Float64.(S_true), Float64.(S_noisy))
   # g
    # -------- 6 panel QA plot
    plot1(ps, im_true, clim=clim, label="True")
    plot1(ps, im_denoise, clim=clim, label="Synth")
    #plot1(ps, im_targ-im_true, clim=clim2, label="Noise")
    p = plot(dat[6],label="S mean")
    plot!(p,dat[10],label="S noisy")
    plot!(p,dat[9],label="S true")

    push!(ps, p)
    plot1(ps, im_targ, clim=clim, label="Noisy")
    plot1(ps, im_targ-im_true, clim=clim2, label="Noisy-True")
    plot1(ps, im_sig, clim=clim2, label="Sigma")
    plot1(ps, im_targ-im_denoise, clim=clim2, label="Noisy-Synth")
    plot1(ps, im_denoise-im_true, clim=clim2, label="Synth-True")

    plot1(ps, dat[5][:,:,1]-dat[5][:,:,2], clim=clim2, label="Synth 1-2")
    plot1(ps, dat[5][:,:,1]-dat[5][:,:,3], clim=clim2, label="Synth 1-3")
    plot1(ps, dat[5][:,:,1]-dat[5][:,:,4], clim=clim2, label="Synth 1-4")
    plot1(ps, dat[5][:,:,2]-dat[5][:,:,3], clim=clim2, label="Synth 2-3")


    myplot = plot(ps..., layout=(4,3), size=(1400,2000))
    savefig(myplot, fname)
end
bruno_plot_synth(foo, fname="test.png")


println(1)



function S20_covariance(img, im_sigma, fhash)
    (N1iso, Nf) = size(fhash["S1_iso_mat"])
    Nx = size(img,1)
    dS20dα = wst_S20_deriv(img, fhash)
    G = reshape(dS20dα, Nx*Nx, Nf*Nf) .* reshape(im_sigma, Nx*Nx, 1)
    pred = G'*G
    return pred
end


function test_S20_covariance()
    # test covariance
    Nx = 16
    fhash  = fink_filter_hash(1, 8, nx=Nx, t=1, wd=1, Omega=true, safety_on=false)
    (N1iso, Nf) = size(fhash["S1_iso_mat"])
    img = randn(Nx,Nx)
    im_sigma = ones(Nx,Nx)
    pred = S20_covariance(img, im_sigma, fhash)

    Smean, Scov, Smask = S20_noisecovar(img, fhash, im_sigma, 4000, iso=false,doS2=false, doS20=true)
    Cmask = Smask[Nf+3:end]
    return pred, Scov, Cmask
end

pred, Scov, Cmask = test_S20_covariance()



function bruno_denoise(im_true, im_sigma, Nx=64, Nsam=10000)
    # Options we could have
    # Non-uniform noise
    # generate filter bank, safety_on=false forces wd=1 to avoid redundancy
    fhash  = fink_filter_hash(1, 8, nx=Nx, t=1, wd=1, Omega=true, safety_on=false)
    (N1iso, Nf) = size(fhash["S1_iso_mat"])

    # do S20 (aka S2R)
    doiso  = false
    doS2   = false
    doS20  = true

    # get Monte Carlo Scov
    if Nsam != 0
        Smean, Scov, Smask = S20_noisecovar(im_true, fhash, im_sigma, Nsam, iso=doiso, doS2=doS2, doS20=doS20)
    else
        Smean, Scov, Smask = S20_noisecovar(im_true, fhash, im_sigma, 10, iso=doiso, doS2=doS2, doS20=doS20)
        S20cov = S20_covariance(im_true, im_sigma, fhash)
        println("S20cov", size(S20cov))
        println("Smask", size(Smask))
        Cmask = Smask[Nf+3:end]
        println("Cmask", size(Cmask))
        Scov = S20cov[Cmask,Cmask] + 0.1I     # choose this regularization better???
        println("using true image for covar")
    end

    # Eigenvalue check
    evals = eigvals(Scov)
    println("MC Scov, minmax eigenvalues:  ", minimum(evals), "    ", maximum(evals))

    # invert with Cholesky
    Sicov = inv(cholesky(Scov))

    # make noisy image and its inverse variance
    im_targ = im_true .+ im_sigma.*randn(Nx,Nx)
    im_ivar = 1 ./ (im_sigma.*im_sigma)

    # EQWS coeffs for true image and noisy image
    S_true  = eqws_compute(im_true, fhash, doS2=false, doS20=true, norm=false)[Smask]
    S_noisy = eqws_compute(im_targ, fhash, doS2=false, doS20=true, norm=false)[Smask]

    # Attempt synthesis with estimated Smean, Scov
    im_init = zeros(Nx,Nx,4)
    im_denoise = zeros(Nx,Nx,4)
    for i=1:4
        im_init[:,:,i] = imresize(readphoto(i+1),Nx,Nx)
        if i==4
            im_init[:,:,i] = copy(im_targ)
        end
        im_denoise[:,:,i] = bruno_synthS2R(im_targ, im_ivar, S_true, Sicov, Smask, fhash; im_init=im_init[:,:,i], iso=false)
    end

    return (im_true, im_targ, im_ivar, im_init, im_denoise, Smean, Scov, UInt8.(Smask), Float64.(S_true), Float64.(S_noisy))
    # gradient descent converges to machine precision regardless of starting point.
end




function call_bruno(sigma_index, Nx=64, Nsam=10000)
    # read dust map
    dust = Float64.(readdust())
    dust = dust[1:256,1:256]

    # mean subtract resized image
    img     = imresize(dust,Nx,Nx)
    im_true = img .- mean(img)

    # noise scenarios
    Ntrial = 4
    sigim = zeros(Nx,Nx,Ntrial)
    sigim[:,:,1] .= 10
    sigim[:,:,2] .= 20

    sigim[1:Nx÷4,:,3] .= 1
    sigim[Nx÷4+1:Nx÷2,:,3] .= 5
    sigim[Nx÷2+1:3*Nx÷4,:,3] .= 10
    sigim[3*Nx÷4:Nx,:,3] .= 50

    sigim[:,:,4] .= rand(Nx,Nx).*10
    rx = ceil.(Int,rand(round(Int,Nx*Nx*0.1))*Nx)
    ry = ceil.(Int,rand(round(Int,Nx*Nx*0.1))*Nx)
    for i=1:length(rx) sigim[rx[i],ry[i],4] = 100.0 end

    im_sigma = sigim[:,:,sigma_index]
    arr = bruno_denoise(im_true, im_sigma, Nx, Nsam)
    return arr
end



function bruno_noise_loop(Nx=64,Nsam=10000)

    prefix = "bruno"*string(Nx)
    for noise_index=1:4
        @time result = call_bruno(noise_index, Nx, Nsam)
        fname = prefix*"-"*string(noise_index)
        write_results(fname*".fits", result)
        bruno_plot_synth(result, fname=fname*".png")
    end
end

bruno_noise_loop(128,40000)




bruno_plot_synth(read_results("bruno64-1.fits"), fname="bruno64-1.png")N
bruno_plot_synth(read_results("bruno64-2.fits"), fname="bruno64-2.png")
bruno_plot_synth(read_results("bruno64-3.fits"), fname="bruno64-3.png")
bruno_plot_synth(read_results("bruno64-4.fits"), fname="bruno64-4.png")



bar = randn(100,200)
cov = bar*bar'
U=cholesky(cov).U
plot(eigvals(cov),label="Eval(cov)")
plot!(eigvals(U),label="Eval(U)")

heatmap(U,clim=(-2,2))

C
Cinv = inv(C)
Cinv*C
L = cholesky(C).L
Λ = cholesky(Symmetric(Cinv)).U
Λ'*Λ*L*L'


Stry = (my_DHC_compute(img, fhash, fhash, sigim,  doS2=doS2, doS20=doS20, norm=false, iso=doiso))[i0:end]

fval = fhash["filt_value"]
psi2 = [sum(vals.^2)/Nx^2 for vals in fval]
diff = Smean-Scoeff

prod = psi_matrix(fhash)

plot(diff)
plot!((Stry-Scoeff))  # why 0.66 ????


plot!(reshape(prod.*25600 .*2,Nf*Nf))





function gaussian_draw_test(Ns=100)
    # make a set of correlated (x,y) pairs
    mockx = randn(Ns)
    mocky = randn(Ns)+mockx.*2 .+1
    mydata = hcat(mockx,mocky)

    # subtract the mean
    data_mean = mean(mydata,dims=1)
    data_meansub = mydata .- data_mean
    cov = (data_meansub' * data_meansub)./(Ns-1)

    # U is upper triagular matrix, sqrt of cov, so U' * U = cov
    U = cholesky(cov).U

    # random draw is unit Gaussian draw times U plus mean
    ran = randn(Ns,2)
    draw = (ran * U) .+ data_mean

    # plot origina distribution and draws
    scatter(mydata[:,1],mydata[:,2])
    scatter!(draw[:,1],draw[:,2])
end


struct Foo2
    a::Float64
    b::Any
    c::Array{Float64}
end

fname = "/Users/dfink/Downloads/1581898633071O-result.fits"
ff = FITS(fname,"r")
