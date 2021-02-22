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
    eps = 1e-4
    fhash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1)
    im = rand(Float64, Nx, Nx).*0.1
    im[6,6]=1.0
    im1 = copy(im)
    im0 = copy(im)
    im1[2,3] += eps/2
    im0[2,3] -= eps/2
    dS20dp = wst_S20_deriv(im, fhash)

    der0=DHC_compute(im0,fhash,doS2=false,doS20=true)
    der1=DHC_compute(im1,fhash,doS2=false,doS20=true)
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
@benchmark wst_S20_deriv(im, fhash, 1)

@benchmark wst_S20_deriv_sum(im, fhash, wts, 1)


Profile.clear()
@profile wst_S20_deriv_sum(im, fhash, wts, 1)
Juno.profiler()


# S20 deriv time, Mac laptop
# Nx     Jan 30  Feb 14  Feb 21
#   8     28 ms   1 ms
#  16    112      7      0.6 ms
#  32    320     50        2
#  64   1000    400       17
# 128   5 sec     3.3 s   90
# 256   ---      17.2 s  0.65 s
# 512   ---               2.9 s



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


function wst_synthS20(im_init, fixmask, S_targ, S20sig; iso=false)
    # fixmask -  0=float in fit   1=fixed

    function wst_synth_chisq(vec_in)

        thisim = copy(im_init)
        thisim[indfloat] = vec_in

        M20 = fhash["S2_iso_mat"]

        S20 = DHC_compute(thisim, fhash, doS2=false, doS20=true, norm=false, iso=iso)
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
        S20arr = (DHC_compute(thisim, fhash, doS2=false, doS20=true, norm=false, iso=iso))[i0:end]
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

    S20   = DHC_compute(im, fhash, doS2=false, doS20=true, norm=false, iso=iso)
    Ns    = length(S20)
    S20arr = zeros(Float64, Ns, Nsam)
    for j=1:Nsam
        noise = rand(Nx,Nx)
        S20arr[:,j] = DHC_compute(im+noise, fhash, doS2=false, doS20=true, norm=false, iso=iso)
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

Nx     = 128
doiso  = true
fhash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1, Omega=true)
(N1iso, Nf)    = size(fhash["S1_iso_mat"])
i0 = 3+(doiso ? N1iso : Nf)
im    = imresize(dust,(Nx,Nx))
fixmask = rand(Nx,Nx) .< 0.01


S_targ = DHC_compute(im, fhash, doS2=false, doS20=true, norm=false, iso=doiso)
S_targ = S_targ[i0:end]

init = copy(im)
floatind = findall(fixmask .==0)
init[floatind] .+= rand(length(floatind)).*50 .-25

S20sig = S20_weights(im, fhash, 100, iso=doiso)
S20sig = S20sig[i0:end]
foo = wst_synthS20(init, fixmask, S_targ, S20sig, iso=doiso)
S_foo = DHC_compute(foo, fhash, doS2=false, doS20=true, norm=false, iso=doiso)


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

# using new wst_S20_deriv_sum on Feb 21, 2021
# size     t(GG)  t(CG)ISO  [sec]
# 8x8
# 16x16
# 32x32
# 64x64   134
# 128x128           221
