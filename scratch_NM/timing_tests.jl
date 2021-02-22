using Statistics
using Plots
using BenchmarkTools
using Profile
using FFTW
using Statistics
using Optim
using Images, FileIO, ImageIO
using Printf
using ReverseDiff: GradientTape, GradientConfig, gradient, gradient!, compile, DiffResults
using LinearAlgebra
using Zygote

# put the cwd on the path so Julia can find the module
push!(LOAD_PATH, pwd()*"/main")
using DHC_2DUtils

push!(LOAD_PATH, pwd()*"/scratch_NM")
using Deriv_Utils: wst_S1S2_derivfast, DHC


#Image Cases
Nx=16
fhash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1)
img = zeros(Float64, Nx, Nx)
img[6,6]=1.0

#ds1s2 = wst_S1S2_derivfast(img, fhash)
#Create wrappers
function DHC_S2(input)
    return DHC(input, fhash, doS2=true)
end

function proxy(input)
    return [input[1]+input[2], input[1]*input[2], input[1]/input[2]]
end

fjac = Zygote.pullback(proxy, [3, 4])


#DIFFERENT REVERSEDIFF CONFIGS: Which is the fastest reversediff?
#Zyg.forward_jacobian doesnt work: No method matching DHC.FwdDiff
#Zyg.gradient only works if the output is scalar. Can try to benchmark with the chisq function.
#=
function jacobian(f, x)
    y = f(x)
    n = length(y)
    m = length(x)
    j = Array{Float64, 2}(undef, n, m)
    for i in 1:n
        j[i, :] .= gradient(x -> f(x)[i], x)
    end
    return j
end
=#
#This doesnt work either.
#I wont be surprised if there is no off-the-shelf autodiff for the DHC function--complex intermediaries, Jacobian, FFTs
#and if the way we've implemented this is the only way.
