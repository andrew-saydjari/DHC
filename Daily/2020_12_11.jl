## Preloads
using Statistics
using FFTW
using Plots
using BenchmarkTools
using Profile
using LinearAlgebra
using Measures

#internal AREPO units in cgs
arepoLength = 3.0856e20
arepoMass = 1.991e33
arepoVel = 1.0e5

#AREPO conversion factors
arepoTime = arepoLength/arepoVel
arepoDensity = arepoMass/arepoLength/arepoLength/arepoLength
arepoEnergy= arepoMass*arepoVel*arepoVel
arepoColumnDensity = arepoMass/arepoLength/arepoLength

#Helium abundance
ABHE = 0.1

#Mass of proton
mp=1.67e-24

#Grid size
nx,ny,nz=500,500,500

#Trial Read in
y =  Array{Float32}(undef, 125000000)

test = read("/Users/aksaydjari/Dropbox/GradSchool_AKS/Doug/ExtData/CloudFactory/density_grid_x1480y1390z1395",y)
