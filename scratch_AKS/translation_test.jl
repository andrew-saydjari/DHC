## Preloads
using Statistics
using FFTW
using Plots
using BenchmarkTools
using Profile
using LinearAlgebra
using Measures
using ProgressMeter
push!(LOAD_PATH, pwd()*"/main")
using DHC_2DUtils

fhash = fink_filter_hash(1, 8, wd=2, nx=256, Omega=false)

randA = rand(128,128)

no_shift = DHC_compute(randA,fhash)

shift = DHC_compute(circshift(randA,(1,1)),fhash)

maximum(shift - no_shift)

coeff_vec = zeros(256*256,2452)
@showprogress for i=1:256
    @showprogress for j=1:256
        coeff_vec[i+(j-1)*256,:] = DHC_compute(circshift(randA,(i,j)),fhash)
    end
end

using Distributed
addprocs(7);

@everywhere begin
    using Statistics
    using FFTW
    using Plots
    using BenchmarkTools
    using Profile
    using LinearAlgebra
    using Measures
    using ProgressMeter
    push!(LOAD_PATH, pwd()*"/main")
    using DHC_2DUtils

    fhash = fink_filter_hash(1, 8, wd=2, nx=128, Omega=false)

    function calc_shift(x)
        i,j,img = x
        return DHC_compute(circshift(img,(i,j)),fhash)
    end
end

mnist_DHC_out = @showprogress pmap(calc_shift, Iterators.product(1:128,1:128,[randA]))

mnist_DHC_out = hcat(mnist_DHC_out...)

out_test = mnist_DHC_out

maximum(out_test .- out_test[:,1])
