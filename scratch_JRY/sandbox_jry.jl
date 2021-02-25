# File for miscellaneous testing

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
