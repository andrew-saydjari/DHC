using Statistics
using FFTW
using Plots
using BenchmarkTools
using Profile
using LinearAlgebra
using Measures
using ProgressMeter
using Plots
theme(:dark)
using Revise
push!(LOAD_PATH, pwd()*"/main")
using eqws

filter_hash = fink_filter_hash(1,8)
a = rand(256,256)
d = eqws_compute_convmap(a,filter_hash)

heatmap(d[57])

d2 = sum.(d)

d1 = eqws_compute(a,filter_hash)

count(d1 .≈ d2)

maximum(abs.(d1.-d2)) < 1e-15

rod_test = rod_image(1,1,30,35,8)

d1 = eqws_compute(rod_test,filter_hash)
d = eqws_compute_convmap(rod_test,filter_hash)
d2 = sum.(d)
count(d1 .≈ d2)
test2 = (d1 .≈ d2)
argmin(test2)

d1[57]
d2[57]

d1[57]
d2[57]

maximum(abs.(d1.-d2)) < 1e-16

scatter(d1 .≈ d2)

b = [im_rd_0_1[:,:,i]  for i in 1:size(im_rd_0_1)[3]]

b[1]

filter_hash["S1_iso_mat"]*b

filter_hash["S1_iso_mat"]

filter_hash["S1_iso_mat"]*b
