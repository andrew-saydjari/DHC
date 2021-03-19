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
using Deriv_Utils_New
using Data_Utils
push!(LOAD_PATH, pwd()*"/scratch_NM")
using LossFuncs

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


arr = rand(256, 256)
@time foo1 = arr[:]
@time foo2 = reshape(arr, (256*256)) #100x faster

#0.000214 seconds (2 allocations: 512.078 KiB)
#0.000004 seconds (2 allocations: 80 bytes)

img = readdust(64)
fhash = fink_filter_hash(1, 8, nx=64, pc=1, wd=1, Omega=true)
@benchmark ds20 = Deriv_Utils_New.wst_S20_deriv(img, fhash) #157ms
@benchmark ds12 = Deriv_Utils_New.wst_S12_deriv(img, fhash) #3.28 s

Profile.clear()
@profile ds12 = Deriv_Utils_New.wst_S12_deriv(img, fhash)
Juno.profiler()

pnzlist = Array{Array, 2}

function intersect_p(fhash)
    f_ind   = fhash["filt_index"]  # (J, L) array of filters represented as index value pairs
    f_val   = fhash["filt_value"]

    (Nf,) = size(fhash["filt_index"])
    for f1=1:Nf
        f_i1 = f_ind[f1]
        f_v1 =f_val[f1]
        for f2=1:Nf
            f_i2 = f_ind[f2]
            f_v2 =f_val[f2]
            pnz = intersect(f_i1, f_i2)
        end
    end
end

function intersect_p_prealloc(fhash)
    f_ind   = fhash["filt_index"]  # (J, L) array of filters represented as index value pairs
    f_val   = fhash["filt_value"]
    pnz_store = []
    (Nf,) = size(fhash["filt_index"])
    for f1=1:Nf
        f_i1 = f_ind[f1]
        f_v1 =f_val[f1]
        for f2=1:Nf
            f_i2 = f_ind[f2]
            f_v2 =f_val[f2]
            push!(pnz_store, intersect(f_i1, f_i2))
        end
    end
end

@benchmark intersect_p_prealloc(fhash) #42ms, same time  as intersect_p



function foo_wrapperout(var)
    a=var+2
    function foo_wrapperin() #Generic Loss wrapper that can see internal variables but only takes vector as an arg
        LossFuncs.foo(a) #Specific loss function to be used here that has all 
    end
    outval = foo_wrapperin()
    return outval
end


varout = foo_wrapperout(10)
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
