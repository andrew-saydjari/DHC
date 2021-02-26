push!(LOAD_PATH, "/Users/saydjari/Dropbox/GradSchool_AKS/Doug/Projects/DHC/main")
using DHC_2DUtils
using Profile
using BenchmarkTools
using FFTW
using HDF5
using Test
using SparseArrays

filter_hash = fink_filter_hash(1, 4, nx=256, pc=1, wd=1)

@time filt_3d = fink_filter_bank_3dizer(filter_hash, 1, nz=256)

filter_temp = zeros(256,256,256)
filter_temp[filt_3d["filt_index"][12]] = filt_3d["filt_index"][12]

h5write("../DHC/scratch_AKS/data/filt_3d_rs.h5", "main/data", fft2())
