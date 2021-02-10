using Plots
using BenchmarkTools
using Profile
using FFTW
using Statistics
using Optim
using Images, FileIO, ImageIO

push!(LOAD_PATH, pwd()*"/main")
using DHC_2DUtils


filt, info = fink_filter_bank(1, 8, nx=256, wd=2, pc=1, shift=false, Omega=true)
#= Sums of intensities: sum(fftshift(filt[:, :, :]))/(256^2)
fink_filter_bank(1, 8, nx=256, wd=1, pc=1, shift=false, Omega=false) <~0.5
fink_filter_bank(1, 8, nx=256, wd=1, pc=1, shift=false, Omega=true) =1.15. 2 in Q1-2, 1 elsewhere, 0 in Q3-4. Main config.
fink_filter_bank(1, 8, nx=256, wd=1, pc=2, shift=false, Omega=false) = 0.70
fink_filter_bank(1, 8, nx=256, wd=1, pc=2, shift=false, Omega=true) = 1.35
fink_filter_bank(2, 8, nx=256, wd=1, pc=1, shift=false, Omega=false) = 0.65
fink_filter_bank(2, 8, nx=256, wd=1, pc=1, shift=false, Omega=false) = 1.14
fink_filter_bank(1, 8, nx=256, wd=2, pc=1, shift=false, Omega=true) = 1.36, blurrier
=#

#Vis: heatmap(sum(fftshift(filt[:, :, :]), dims=3)[:, :, 1])
