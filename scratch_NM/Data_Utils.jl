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

#using Profile
using LinearAlgebra

# put the cwd on the path so Julia can find the module
push!(LOAD_PATH, pwd()*"/main")
using DHC_2DUtils

function readdust(Nx)

    RGBA_img = load(pwd()*"/scratch_DF/t115_clean_small.png")
    img = reinterpret(UInt8,rawview(channelview(RGBA_img)))

    return img[1,:,:][1:Nx, 1:Nx]
end
