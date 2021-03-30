## Preloads
using Revise
using ImageTransformations
using Statistics
using FFTW
using Plots
using BenchmarkTools
using Profile
using LinearAlgebra
using Measures
using HDF5
using Images
theme(:dark)
using DSP
using Interpolations
using Distributed
using ProgressMeter

test_img = rand(64,64)

Plots.heatmap(test_img)

ImageTransformations.imrotate(test_img,0.3,axes(test_img), Cubic(Throw(OnGrid())))

function box_extrapolation(parent::AbstractArray{T,N}, aitp::Interpolations.AbstractInterpolation, args...) where {T,N}
    itp = interpolate(parent, aitp)
    ImageTransformations.box_extrapolation(itp, args...)
end

function box_extrapolation(parent::AbstractArray{T,N}, aitp::Lanczos4OpenCV, args...) where {T,N}
    itp = interpolate(parent, aitp)
    ImageTransformations.box_extrapolation(itp, args...)
end


box_extrapolation(test_img,Lanczos4OpenCV())

out = ImageTransformations.imrotate(test_img,0.3,axes(test_img), Lanczos4OpenCV())

Plots.heatmap(out)

out[findall(out .!= out)] .= 0.0

Plots.heatmap(out)

test1 = ImageTransformations.imresize(test_img,Dims((128,128)))

test2 = ImageTransformations.imresize(test_img,Dims((128,128)),Lanczos4OpenCV())

Dims([128,128])

test1 .== test2

Plots.heatmap(test1)

Plots.heatmap(test2)

Plots.heatmap(test_img)


function imresize(original::AbstractArray, short_size::Union{Indices{M},Dims{M}}, aitp::Interpolations.Lanczos4OpenCV) where M
    len_short = length(short_size)
    len_short > ndims(original) && throw(DimensionMismatch("$short_size has too many dimensions for a $(ndims(original))-dimensional array"))
    new_size = ntuple(i -> (i > len_short ? odims(original, i, short_size) : short_size[i]), ndims(original))
    imresize(original, new_size, aitp)
end

Interpolations.Cubic(Throw(OnGrid()))

img = rand(Gray{N0f8}, 10, 10)
imgfloat = Float64.(img)

etp = @inferred ImageTransformations.box_extrapolation(imgfloat, Lanczos4OpenCV())
summary(etp)
@test typeof(etp) <: Interpolations.FilledExtrapolation
@test summary(etp) == "2×2 extrapolate(::Interpolations.LanczosInterpolation{Float64,2,Lanczos4OpenCV,OffsetArray{Float64,2,Array{Float64,2}},Tuple{Base.OneTo{Int64},Base.OneTo{Int64}}}, NaN) with element type Float64"

etp = @inferred ImageTransformations.imrotate(imgfloat,0.1,axes(imgfloat), Lanczos4OpenCV())
summary(etp)
imgfloat

etp = @inferred ImageTransformations.imresize(test_img,(128,128),Lanczos4OpenCV())
summary(etp)

img = rand(C{T},10,10)

etp = @inferred ImageTransformations.imresize(test_img,(128,128),Lanczos4OpenCV())
summary(etp)
@test summary(etp) == "128×128 Array{Float64,2}"

etp = @inferred ImageTransformations.imresize(test_img,(64,64),Lanczos4OpenCV())
summary(etp)
@test summary(etp) == "64×64 Array{Float64,2}"

etp = @inferred ImageTransformations.box_extrapolation(imgfloat, Cubic(Flat(OnGrid())), Flat())
@test typeof(etp) <: Interpolations.Extrapolation
summary(etp)
