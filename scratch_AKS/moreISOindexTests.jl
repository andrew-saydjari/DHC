push!(LOAD_PATH, "/Users/saydjari/Dropbox/GradSchool_AKS/Doug/Projects/DHC/main")
using DHC_2DUtils
using DHC_tests
using Profile
using BenchmarkTools
using FFTW
using HDF5
using Test
using SparseArrays
using Plots
theme(:dark)

J = 6
L = 8
test = zeros(1,J*L+3+(J*L+1)*(J*L+1))
for j1=1:6
    for j2=1:6
        for l1=1:8
            for l2=1:8
                test[:,J*L+2+(j1-1+(l1-1)*6)+(j2-1+(l2-1)*6)*(J*L+1)] .= j1*8+j2*8
            end
        end
    end
end

fash = fink_filter_hash(1,8)

S1mat = S1_iso_matrix(fash)
S2mat = S2_iso_matrix(fash)


out = transformMaker(test,S1mat,S2mat)

h5write("../DHC/scratch_AKS/data/isoPlotTest.h5", "data", out)
