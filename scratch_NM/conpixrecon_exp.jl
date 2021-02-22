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
using Distributions
using Measures
#using Weave

#using Profile
using LinearAlgebra


push!(LOAD_PATH, pwd()*"/scratch_NM")
using Deriv_Utils
push!(LOAD_PATH, pwd()*"/main")
using DHC_2DUtils#: fink_filter_bank, fink_filter_hash, DHC_compute

#=function img_reconfunc(input, s_targ_mean, s_targ_invcov, mask, coeff_choice)
    #Only for denoising rn. Need to adapt to use mask and coeff_choice

    (Nx, Ny)  = size(input)
    if Nx != Ny error("Input image must be square") end
    filter_hash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1)

    (Nf, )    = size(filter_hash["filt_index"])
    if Nf == 0 error("filter hash corrupted") end

    storage_beta = Array{Float64, 2}(undef, 1, Nx^2)

    function loss_func(img_curr)
        s_curr = Deriv_Utils.DHC(img_curr,  filter_hash, doS2=true, doS12=false, doS20=false)
        neglogloss = 0.5 .* (s_curr - s_targ_mean)' * s_targ_invcov * (s_curr - s_targ_mean)
        return neglogloss[1]
    end

    function dloss(storage_grad, img_curr)
        s_curr = Deriv_Utils.DHC(img_curr, filter_hash, doS2=true, doS12=false, doS20=false)
        diff = s_curr - s_targ_mean
        diff = diff[3:end]
        dS1S2 = Deriv_Utils.wst_S1S2_derivfast(img_curr, filter_hash)
        term1 = Transpose(s_targ_invcov * diff)
        term2 = reshape(dS1S2, (size(dS1S2)[1], Nx^2))
        mul!(storage_beta, term1, term2)
        storage_grad = reshape(storage_beta, 1, Nx, Nx) #term1: 1x (Nf+Nf^2) | term2: #(Nf+Nf^2) x Nx x Nx
        #storage_grad .=  term1 * term2
    end

    res = optimize(loss_func, dloss, input, BFGS())
    result_img = zeros(Float64, Nx, Nx)
    result_img = Optim.minimizer(res)
    return result_img

end
=#


#function optim_subroutine()


function img_reconfunc(input, s_targ_mean, s_targ_invcov, mask, coeff_choice, optim_settings)
    #=
    input: Square.
    s_targ_mean, s_targ_invcov should have the same format as the raw output of the appropriate DHC call. i.e. you need to exclude S0/S1 from s_targ_mean below.

    =#
    #Only for denoising rn. Need to adapt to use mask and coeff_choice
    #Consider making this the wrapper for a subroutine that is constant irrespective of the choice of coeffs, mask etc. SO ths wrapper handles
    #all the choices of S2 vs S20 or only S1 etc--nope probably messier because would need to pass wrapper for appropriate type of DHC
    #
    (Nx, Ny)  = size(input)
    if Nx != Ny error("Input image must be square") end
    filter_hash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1, Omega=true)

    (Nf, )    = size(filter_hash["filt_index"])
    if Nf == 0 error("filter hash corrupted") end

    #Handle optimizer settings
    #storage_beta = Array{Float64, 2}(undef, 1, Nx^2)
    if coeff_choice=="S2"
        #Doesn't work currently.
        println("S2")
        if size(s_targ_mean)[1]!=(Nf+Nf^2)
            s_targ_mean = s_targ_mean[3:end]
        end
        if (s_targ_invcov!=I) & (size(s_targ_invcov)!=(Nf+Nf^2, Nf+Nf^2))
            s_targ_invcov = s_targ_invcov[3:end, 3:end]
        end

        function loss_func(img_curr)
            #println("Img",size(img_curr))
            s_curr = Deriv_Utils.DHC(reshape(img_curr, (Nx, Nx)),  filter_hash, doS2=true, doS12=false, doS20=false)[3:end]
            neglogloss = 0.5 .* (s_curr - s_targ_mean)' * s_targ_invcov * (s_curr - s_targ_mean)
            #println("In Loss:",size(s_curr))
            #println("NegLogLoss",neglogloss[1])
            return neglogloss[1]
        end

        function dloss(storage_grad, img_curr)
            s_curr = Deriv_Utils.DHC(reshape(img_curr, (Nx, Nx)), filter_hash, doS2=true, doS12=false, doS20=false)[3:end]
            diff = s_curr - s_targ_mean
            #diff = diff
            dS1S2 = Transpose(reshape(Deriv_Utils.wst_S1S2_derivfast(reshape(img_curr, (Nx, Nx)), filter_hash), (:, Nx^2))) #(Nf+Nf^2, Nx, Nx)->(Nx^2, Nf+Nf^2) Cant directly do this without handling S1 coeffs
            term1 = s_targ_invcov * diff #(Nf+Nf^2) x 1
            #WARNING: Uses the Deriv_Utils version of dS1S2. Need to rewrite if you want to use the DHC_2DUtils one.
            #println("In dLoss:",size(diff), size(term1), size(term2))
            #println(size(term1),size(term2),size(storage_grad))
            mul!(storage_grad, dS1S2, term1) #Nx^2x1
            #storage_grad .=  term1 * term2
        end
        #Debugging stuff
        println("Chisq derivative check")
        eps = zeros(size(input))
        eps[1, 2] = 1e-4
        chisq1 = loss_func(input+eps./2)
        chisq0 = loss_func(input-eps./2)
        brute  = (chisq1-chisq0)/1e-4

        clever = reshape(zeros(size(input)), (Nx*Nx, 1))
        _bar = dloss(clever, reshape(input, (Nx^2, 1)))
        println("Brute:  ",brute)
        println("Clever: ",clever[Nx*(1)+1])


        res = optimize(loss_func, dloss, reshape(input, (Nx^2, 1)), ConjugateGradient(), Optim.Options(iterations = get(optim_settings, "iterations", 100), store_trace = true, show_trace = true))
        result_img = zeros(Float64, Nx, Nx)
        result_img = Optim.minimizer(res)

    else #Currently S20
        #DO NOT RENAME THESE FUNCS TO loss_func and dloss_func: otherwise Julia calls the s20 version of the loss in the S2 case.
        println("S20")
        if size(s_targ_mean)[1]!=Nf^2
            s_targ_mean = s_targ_mean[3+Nf:end]
        end
        if (s_targ_invcov!=I) & (size(s_targ_invcov)!=(Nf^2, Nf^2))
            s_targ_invcov = s_targ_invcov[3+Nf:end, 3+Nf:end]
        end

        #println("Shapes:s_targ_mean, s_targ_invcov", size(s_targ_mean), size(s_targ_invcov))
        function loss_func20(img_curr)
            s_curr = Deriv_Utils.DHC(reshape(img_curr, (Nx, Nx)),  filter_hash, doS2=false, doS12=false, doS20=true)[3+Nf:end]
            neglogloss = 0.5 .* (s_curr - s_targ_mean)' * s_targ_invcov * (s_curr - s_targ_mean)
            #println("s_curr",size(s_curr))
            return neglogloss[1]
        end

        function dloss20(storage_grad, img_curr)
            #Diagonal covar: This works!
            #=
            s_curr = Deriv_Utils.DHC(reshape(img_curr, (Nx, Nx)), filter_hash, doS2=false, doS12=false, doS20=true)[3+Nf:end]
            diff = (s_curr - s_targ_mean).* Array(s_targ_invcov[diagind(s_targ_invcov)])
            dS20 = reshape(Deriv_Utils.wst_S20_deriv(reshape(img_curr, (Nx, Nx)), filter_hash), (Nx^2, Nf^2)) #Check (Nx, Nx, Nf, Nf) -> (Nx^2, Nf^2)
            mul!(storage_grad, dS20, diff) #term1: 1x (Nf+Nf^2) | term2: #(Nf+Nf^2) x Nx^2
            =#
            #More general:
            s_curr = Deriv_Utils.DHC(reshape(img_curr, (Nx, Nx)), filter_hash, doS2=false, doS12=false, doS20=true)[3+Nf:end]
            diff = s_curr - s_targ_mean
            dS20 = reshape(Deriv_Utils.wst_S20_deriv(reshape(img_curr, (Nx, Nx)), filter_hash), (Nx^2, Nf^2)) #Check (Nx, Nx, Nf, Nf) -> (Nx^2, Nf^2)
            term1 = s_targ_invcov * diff #Nf^2 x1
            mul!(storage_grad, dS20, term1) #term1: 1x (Nf+Nf^2) | term2: #(Nf+Nf^2) x Nx^2

        end
        #
        println("Chisq derivative check")
        eps = zeros(size(input))
        eps[1, 2] = 1e-4
        chisq1 = loss_func20(input+eps./2)
        chisq0 = loss_func20(input-eps./2)
        brute  = (chisq1-chisq0)/1e-4

        clever = reshape(zeros(size(input)), (Nx*Nx, 1))
        _bar = dloss20(clever, reshape(input, (Nx^2)))
        println("Brute:  ",brute)
        println("Clever: ",clever[Nx*(1)+1])


        res = optimize(loss_func20, dloss20, reshape(input, (Nx^2, 1)), ConjugateGradient(), Optim.Options(iterations = get(optim_settings, "iterations", 100), store_trace = true, show_trace = true))
        result_img = zeros(Float64, Nx, Nx)
        result_img = Optim.minimizer(res)

    end

    return reshape(result_img, (Nx, Ny))

end


function readdust()

    RGBA_img = load(pwd()*"/scratch_DF/t115_clean_small.png")
    img = reinterpret(UInt8,rawview(channelview(RGBA_img)))

    return img[1,:,:]
end

function S2_weights(im, fhash, Nsam=10; iso=iso)

    (Nx, Ny)  = size(im)
    if Nx != Ny error("Input image must be square") end
    (N1iso, Nf)    = size(fhash["S1_iso_mat"])

    # fhash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1)

    S2   = DHC_compute(im, fhash, doS2=true, doS20=false, norm=false, iso=iso)
    Ns    = length(S2)
    S2arr = zeros(Float64, Ns, Nsam)
    println("Ns", Ns)
    for j=1:Nsam
        noise = rand(Nx,Nx)
        S2arr[:,j] = DHC_compute(im+noise, fhash, doS2=true, doS20=false, norm=false, iso=iso)
    end

    wt = zeros(Float64, Ns)
    for i=1:Ns
        wt[i] = std(S2arr[i,:])
    end

    return wt
end
####Comparison Functions#########
#Basically sandbox_DF funcs
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
        println(chisq)
        return chisq

    end

    function wst_synth_dchisq(storage, vec_in)

        thisim = copy(im_init)
        thisim[indfloat] = vec_in

        dS20dp = reshape(wst_S20_deriv(thisim, fhash), Nx*Nx, Nf*Nf)
        if iso
            M20 = fhash["S2_iso_mat"]
            dS20dp = dS20dp * M20'
        end
        i0 = 3+(iso ? N1iso : Nf)

        S20arr = (DHC_compute(thisim, fhash, doS2=false, doS20=true, norm=false, iso=iso))[i0:end]

        # put both factors of S20sig in this array to weight
        diff   = (S20arr - S_targ)./(S20sig.^2)
        #println("size of diff", size(diff))
        #println("size of dS20dp", size(reshape(dS20dp, Nx*Nx, Nf*Nf)))
        # dSdp matrix * S1-S_targ is dchisq
        #dchisq_im = (reshape(dS20dp, Nx*Nx, Nf*Nf) * diff).*2
        dchisq_im = (dS20dp * diff).*2
        dchisq = reshape(dchisq_im, Nx, Nx)[indfloat]

        storage .= dchisq
    end

    (Nx, Ny)  = size(im_init)
    if Nx != Ny error("Input image must be square") end
    (N1iso, Nf)    = size(fhash["S1_iso_mat"])


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
        ConjugateGradient(), Optim.Options(iterations=100))

    # copy results into pixels of output image
    im_synth = copy(im_init)
    im_synth[indfloat] = Optim.minimizer(res)
    println(res)

    return im_synth
end

function plot_synth_QA(ImTrue, ImInit, ImSynth, fhash; fname="test2.png")

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
####Testing Examples#############
#Sec1
dust = readdust()
dust = Array{Float64}(dust[1:16, 1:16])

n = Normal()
noise = reshape(rand(n, 256), (16, 16))
noisyinp = dust+noise

fhash = fink_filter_hash(1, 8, nx=16, pc=1, wd=1)
s_targ = DHC_2DUtils.DHC_compute(dust, fhash, doS2=false, doS20=true)[3+length(fhash["filt_index"]):end]
Nx=16
#noisyinp = zeros(Float64, Nx, Nx)
#noisyinp .= dust
#noisyinp[3, 5] += dust[3, 5] + 8
denoised_s20 = img_reconfunc(noisyinp, s_targ, I, trues(16, 16), "S20", Dict([("iterations", 1000)]))

#Comparing with wst_synthS20
denoised_comp = wst_synthS20(noisyinp, falses((Nx, Nx)), s_targ, 0.2)
#Doug's test code
Nx=16
dust = Float64.(readdust())
dust = dust[1:256,1:256]
img   = imresize(dust,(Nx,Nx))
n = reshape(rand(Normal(0, mean(img)/20.0), Nx^2), (Nx, Nx))
noisyimg = img + n
s_targ = DHC_2DUtils.DHC_compute(img, fhash, doS2=false, doS20=true)[3+length(fhash["filt_index"]):end]
denoised_img = wst_synthS20(noisyimg, falses(Nx,Nx), s_targ, mean(img)/20.0)

#For some reason the above code doesnt work.
#Sec2: This now works!
#Using the exact same example as in sandbox_DFcopy.
dust = Float64.(readdust())
dust = dust[1:256,1:256]

Nx     = 64
doiso  = false
fhash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1, Omega=true)
(N1iso, Nf)    = size(fhash["S1_iso_mat"])
i0 = 3+(doiso ? N1iso : Nf)
img    = imresize(dust,(Nx,Nx))
#fixmask = rand(Nx,Nx) .< 0.1
fixmask = falses((Nx, Nx))


S_targ = DHC_2DUtils.DHC_compute(img, fhash, doS2=false, doS20=true, norm=false, iso=doiso)
S_targ = S_targ[i0:end]

init = copy(img)
floatind = findall(fixmask .==0)
init[floatind] .+= rand(length(floatind)).*50 .-25

S20sig = S20_weights(img, fhash, 100, iso=doiso)
S20sig = S20sig[i0:end]

#2.1. Using wst_synthS20
foo = wst_synthS20(init, fixmask, S_targ, S20sig, iso=doiso)

#2.2. Using your code.
invcovar = Diagonal(S20_weights(img, fhash, 100, iso=doiso).^(-2))
s20targ = DHC_2DUtils.DHC_compute(img, fhash, doS2=false, doS20=true, norm=false, iso=doiso)
myfoogens20 = img_reconfunc(init, s20targ,  invcovar, fixmask, "S20", Dict([("iterations", 100)]))


#Need a new invcovar (S2weights) for doing S2
invcovarS2 = Diagonal(S2_weights(img, fhash, 100, iso=doiso).^(-2)) #This doesn't work for S2: use invcovar computed for S20 for S2 as well for the timebeing.
s2targ = DHC_2DUtils.DHC_compute(img, fhash, doS2=true, doS20=false, norm=false, iso=doiso)
println(size(img), size(init), size(s2targ), size(invcovar), size(fixmask))
reconS2cg = img_reconfunc(init, s2targ,  invcovar, fixmask, "S2", Dict([("iterations", 100)]))
#Weird bug where the code works using invcovar but doesnt when you use invcovars2
#myfoogen used S20's invcovar
#myfoogenS2 uses the correct incovar
#myfoogens20 uses s20

plot_synth_QA(img, init, myfoogens20, fhash, fname="scratch_NM/TestPlots/s20_100itns.png")
plot_synth_QA(img, init, myfoogenS2, fhash, fname="scratch_NM/TestPlots/s2_100itns.png")
plot_synth_QA(img, init, reconS2cg, fhash, fname="scratch_NM/TestPlots/s2_100itns_cg.png")



###Old DEBUG SECTION: Isolated funcs for debugging
#filename = normpath("scratch_NM", "test.jmd")
#weave(filename, out_path = :pwd)
#=
Nf = length(fhash["filt_index"])
Nx=16
storage_grad = zeros(Float64, 2+Nf+(Nf^2), Nx, Nx)

function dloss(storage_grad, img_curr, filter_hash, s_targ_mean, s_targ_invcov)
    s_curr = Deriv_Utils.DHC(img_curr, filter_hash, doS2=true, doS12=false, doS20=false)
    diff = s_curr - s_targ_mean
    diff = diff[3:end]
    term1 = Transpose(s_targ_invcov * diff)  #1x (Nf+Nf^2)
    storage_beta = Array{Float64, 2}(undef, size(term1)[1], Nx^2)
    dS1S2 = Deriv_Utils.wst_S1S2_derivfast(img_curr, filter_hash)
    term2 = reshape(dS1S2, (size(dS1S2)[1], Nx^2)) #(Nf+Nf^2) x Nx x Nx
    storage_grad = reshape(mul!(storage_beta, term1, term2), (size(term1)[1], Nx, Nx))
    #storage_grad .=  term1 * term2
end


function loss_func(img_curr, filter_hash, s_targ_mean, s_targ_invcov)
    s_curr = Deriv_Utils.DHC(img_curr,  filter_hash, doS2=true, doS12=false, doS20=false)
    neglogloss = 0.5 .* (s_curr - s_targ_mean)' * s_targ_invcov * (s_curr - s_targ_mean)
    return neglogloss
end


loss_func(dust, fhash, s_targ, I)  #Tested: with input=0, with white noise default=10^7
dloss(storage_grad, noisyinp, fhash, s_targ, I)
=#
#=
#Checking Loss Func only
s_targ = Deriv_Utils.DHC(dust, fhash, doS2=true)
s_targ_mean = s_targ[3:end]
s_targ_invcov = I
function loss_func(img_curr, filter_hash, s_targ_mean, s_targ_invcov)
    s_curr = Deriv_Utils.DHC(reshape(img_curr, (Nx, Nx)),  filter_hash, doS2=true, doS12=false, doS20=false)[3:end]
    neglogloss = 0.5 .* (s_curr - s_targ_mean)' * s_targ_invcov * (s_curr - s_targ_mean)
    println("In Loss:",size(s_curr))
    #println("NegLogLoss",neglogloss[1])
    return neglogloss[1]
end

loss_func(reshape(dust, (1, Nx^2)), fhash, s_targ_mean, s_targ_invcov)
=#
