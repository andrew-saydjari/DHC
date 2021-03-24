push!(LOAD_PATH, "/Users/saydjari/Dropbox/GradSchool_AKS/Doug/Projects/DHC/main")
using DHC_2DUtils
using Profile
using BenchmarkTools
using FFTW
using HDF5
using Test
using SparseArrays

bank = fink_filter_bank(1, 16, nx=256, pc=2, wd=2)

h5write("../DHC/scratch_AKS/data/filter_bank_paper_export.h5", "main/data", bank[1])
h5write("../DHC/scratch_AKS/data/filter_bank_paper_export.h5", "main/J_L", bank[2]["J_L"])
h5write("../DHC/scratch_AKS/data/filter_bank_paper_export.h5", "main/psi_index", bank[2]["psi_index"])
h5write("../DHC/scratch_AKS/data/filter_bank_paper_export.h5", "main/phi_index", [bank[2]["phi_index"]])

hash_filt = fink_filter_hash(1, 8, nx=256, pc=1, wd=1)
filt_3d = fink_filter_bank_3dizer(hash_filt, 1, nz=128)
