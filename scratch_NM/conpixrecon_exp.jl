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

#using Profile
using LinearAlgebra


push!(LOAD_PATH, pwd()*"/scratch_NM")
using Deriv_Utils


function img_reconfunc(input, s_targ_mean, s_targ_invcov, mask, coeff_choice)
    #Only for denoising rn. Need to adapt to use mask and coeff_choice

    (Nx, Ny)  = size(input)
    if Nx != Ny error("Input image must be square") end
    filter_hash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1)

    (Nf, )    = size(filter_hash["filt_index"])
    if Nf == 0 error("filter hash corrupted") end

    function loss_func(img_curr)
        s_curr = DHC(img_curr,  filter_hash, doS2=false, doS12=false, doS20=false)
        neglogloss = 0.5 .* (s_curr - s_targ_mean)' * s_targ_invcov * (s_curr - s_targ_mean)
        return neglogloss
    end

    function dloss(storage_grad, img_curr)
        storage_grad .= (s_targ_invcov * (s_curr - s_targ_mean))' *  wst_S1S2_derivfast(img_curr, filter_hash)
    end

    res = optimize(loss_func, dloss, input, BFGS())
    result_img = zeros(Float64, Nx, Nx)
    result_img = Optim.minimizer(res)
    return result_img

end



function readdust()

    RGBA_img = load(pwd()*"/scratch_DF/t115_clean_small.png")
    img = reinterpret(UInt8,rawview(channelview(RGBA_img)))

    return img[1,:,:]
end

dust = readdust()
dust = Array{Float64}(dust[1:16, 1:16])

n = Normal()
noise = reshape(rand(n, 256), (16, 16))
noisyinp = dust+noise

fhash = fink_filter_hash(1, 8, nx=16, pc=1, wd=1)
s_targ = DHC(dust, fhash, doS2=true)

denoised_s2 = img_reconfunc(noisyinp, s_targ, I, trues(16, 16), "S2")
