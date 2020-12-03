## Preloads
using Statistics
using FFTW
using Plots
using BenchmarkTools
using Profile
using LinearAlgebra
using Measures

##
function fink_filter_bank_fast(J, L;wd=1,pc=1)
    #plane coverage (default 1, full 2Pi 2)
    #width of the wavelets (default 1, wide 2)

    # -------- set parameters
    dθ   = pc*π/L
    wdθ  = wd*dθ
    nx   = 256
    dx   = nx/2-1

    # -------- allocate output array of zeros
    filt = zeros(256, 256, J, L)

    # -------- allocate theta and logr arrays
    logr = zeros(nx, nx)
    θ    = zeros(nx, nx)

    for l = 0:L-1
        θ_l = dθ*l

    # -------- allocate anggood BitArray
        anggood = falses(nx, nx)

    # -------- loop over pixels
        for x = 1:nx
            sx = mod(x+dx,nx)-dx -1    # define sx,sy so that no fftshift() needed
            for y = 1:nx
                sy = mod(y+dx,nx)-dx -1
                θ_pix  = mod(atan(sy, sx)+π -θ_l, 2*π)
                θ_good = abs(θ_pix-π) <= wdθ

            # If this is a pixel we might use, calculate log2(r)
                if θ_good
                    anggood[y, x] = θ_good
                    θ[y, x]       = θ_pix
                    r = sqrt(sx^2 + sy^2)
                    logr[y, x] = log2(max(1,r))
                end
            end
        end
        angmask = findall(anggood)
    # -------- compute the wavelet in the Fourier domain
    # -------- the angular factor is the same for all j
        F_angular = cos.((θ[angmask].-π).*L./(2 .*wd .*pc))

    # -------- loop over j for the radial part
        for j = 0:J-1
            jrad  = 7-j
            Δj    = abs.(logr[angmask].-jrad)
            rmask = (Δj .<= 1)

    # -------- radial part
            F_radial = cos.(Δj[rmask] .* (π/2))
            ind      = angmask[rmask]
            filt[ind,j+1,l+1] = F_radial .* F_angular[rmask]
        end
    end
    return filt
end

@time fink_filter_set = fink_filter_bank_fast(8,16,wd=1)

plot(Plots.heatmap(fftshift(fink_filter_set[:,:,2,1]),
    xlims=(1,256),
    ylims=(1,256),
    aspectratio=1,
    axis=nothing,
    border=:none,
    c=:cubehelix,
    legend = :none,
    margin=0mm))

Profile.clear()
@profile fink_filter_bank_fast(8,8)
Juno.profiler()

theme(:juno)

plot_array = Any[]
for i in 1:8
    for j in 1:8
    stride = min(128,2^(9-j))-1
    push!(plot_array,
    plot(Plots.heatmap(fftshift(fink_filter_set[:,:,j,i])[128-stride:129+stride,128-stride:129+stride],
        xlims=(1,256),
        ylims=(1,256),
        aspectratio=1,
        axis=nothing,
        border=:none,
        c=:cubehelix,
        legend = :none,
        margin=0mm)))
    end
end

plot(plot_array...,layout=(8,8))
plot!(size=(1000,1000))

p = plot(
    axis = nothing,
    layout = @layout(grid(8,8)),
    size = (1000,1000)
)

J=8
L=8

for i=1:J
    for j=1:L
        stride = min(128,2^(9-j))-1
        Plots.heatmap!(p[i], fftshift(fink_filter_set[:,:,j,i])[128-stride:129+stride,128-stride:129+stride],
            ratio=1)
    end
end
p

#Grrr ignore all plots and just do my rotational invariance tests...

# generate an image of a rod with some position, length, position angle,
# and FWHM
function rod(xcen, ycen, length, pa, fwhm)
    rodimage = zeros(256,256)
    nx = 256
    x=0
    y=0
    sig = fwhm/2.355
    dtor = π/180
    # -------- define a unit vector in direction of rod at position angle pa
    ux = sin(pa*dtor)   #  0 deg is up
    uy = cos(pa*dtor)   # 90 deg to the right

    for i=1:nx
        for j=1:nx
            x=i-nx/2+xcen
            y=j-nx/2+ycen

            # -------- distance parallel and perpendicular to
            dpara =  ux*x + uy*y
            dperp = -uy*x + ux*y

            if abs(dpara)-length <0
                dpara= 0
            end
            dpara = abs(dpara)
            dpara = min(abs(dpara-length),dpara)

            rodimage[i,j] = exp(-(dperp^2+dpara^2)/(2*sig^2))
        end
    end
    return rodimage
end

@time test_rod = rod(10,10,30,45,4)

fink_filter_set1 = fink_filter_list(fink_filter_bank(8,16))

plot(Plots.heatmap(test_rod,
    xlims=(1,256),
    ylims=(1,256),
    aspectratio=1,
    axis=nothing,
    border=:none,
    c=:cubehelix,
    legend = :none,
    margin=0mm),
    size=(2000,2000))

speedy_DHC(test_rod,fink_filter_set2)

##Check rotational invariance of DHC on S20 for
# J=8, L=8, wd=1, pc=1
# J=8, L=8, wd=1, pc=2
# J=8, L=8, wd=2, pc=1
# J=8, L=8, wd=2, pc=2

# J=8, L=16, wd=1, pc=1
# J=8, L=16, wd=1, pc=2
# J=8, L=16, wd=2, pc=1
# J=8, L=16, wd=2, pc=2

test_rod = rod(10,10,30,45,4)
fink_filter_set2 = fink_filter_list(fink_filter_bank_fast(8,8))
@time speedy_DHC(test_rod,fink_filter_set2)

out0 = zeros(8258,360)
@views for angle=1:360
    test_rod=rod(10,10,30,angle,4)
    out0[:,angle].=speedy_DHC(test_rod,fink_filter_set2)
end

using HDF5

h5write("./Data/j8l8w1p1.h5", "main/data", out0)

#Wow, awesome plot idea for smoothness is to plot coefficients as function of rotation

fink_filter_set2 = fink_filter_list(fink_filter_bank_fast(8,8,wd=1,pc=1))
out0 = zeros(8258,360)
@views for angle=1:360
    test_rod=rod(10,10,30,angle,4)
    out0[:,angle].=speedy_DHC(test_rod,fink_filter_set2)
end

h5write("./Data/j8l8w1p1.h5", "main/data", out0)

fink_filter_set2 = fink_filter_list(fink_filter_bank_fast(8,8,wd=2,pc=1))
out0 = zeros(8258,360)
@views for angle=1:360
    test_rod=rod(10,10,30,angle,4)
    out0[:,angle].=speedy_DHC(test_rod,fink_filter_set2)
end

h5write("./Data/j8l8w2p1.h5", "main/data", out0)

fink_filter_set2 = fink_filter_list(fink_filter_bank_fast(8,8,wd=1,pc=2))
out0 = zeros(8258,360)
@views for angle=1:360
    test_rod=rod(10,10,30,angle,4)
    out0[:,angle].=speedy_DHC(test_rod,fink_filter_set2)
end

h5write("./Data/j8l8w1p2.h5", "main/data", out0)

fink_filter_set2 = fink_filter_list(fink_filter_bank_fast(8,8,wd=2,pc=2))
out0 = zeros(8258,360)
@views for angle=1:360
    test_rod=rod(10,10,30,angle,4)
    out0[:,angle].=speedy_DHC(test_rod,fink_filter_set2)
end

h5write("./Data/j8l8w2p2.h5", "main/data", out0)

test_rod = rod(10,10,30,45,4)
fink_filter_set2 = fink_filter_list(fink_filter_bank_fast(8,16))
@time speedy_DHC(test_rod,fink_filter_set2)

fink_filter_set2 = fink_filter_list(fink_filter_bank_fast(8,16,wd=1,pc=1))
out0 = zeros(32898,360)
@views for angle=1:360
    test_rod=rod(10,10,30,angle,4)
    out0[:,angle].=speedy_DHC(test_rod,fink_filter_set2)
end

h5write("./Data/j8l16w1p1.h5", "main/data", out0)

fink_filter_set2 = fink_filter_list(fink_filter_bank_fast(8,16,wd=2,pc=1))
out0 = zeros(32898,360)
@views for angle=1:360
    test_rod=rod(10,10,30,angle,4)
    out0[:,angle].=speedy_DHC(test_rod,fink_filter_set2)
end

h5write("./Data/j8l16w2p1.h5", "main/data", out0)

fink_filter_set2 = fink_filter_list(fink_filter_bank_fast(8,16,wd=1,pc=2))
out0 = zeros(32898,360)
@views for angle=1:360
    test_rod=rod(10,10,30,angle,4)
    out0[:,angle].=speedy_DHC(test_rod,fink_filter_set2)
end

h5write("./Data/j8l16w1p2.h5", "main/data", out0)

fink_filter_set2 = fink_filter_list(fink_filter_bank_fast(8,16,wd=2,pc=2))
out0 = zeros(32898,360)
@views for angle=1:360
    test_rod=rod(10,10,30,angle,4)
    out0[:,angle].=speedy_DHC(test_rod,fink_filter_set2)
end

h5write("./Data/j8l16w2p2.h5", "main/data", out0)

J=8
L=8
2+J*L+J*L*J*L+J*L*J*L

J=8
L=16
2+J*L+J*L*J*L+J*L*J*L
J*L*J*L

temp = reshape(1:100,20,5)
temp1 = reshape(temp,4,5,5)

out0[2+J*L+1:2+J*L+J*L*J*L,:]
test = reshape(out0[2+J*L+1:2+J*L+J*L*J*L,:],J,L,J,L,360)

l=2
d=2
N=360
test[:,l,:,mod(l+d,L),:]

intermed = zeros(J,J,N,L)
for d=1:L
    intermed[:,:,:,d]=test[:,l,:,mod(l+d,L)+1,:]
end

intermed

dropdims(sum(intermed,dims=4),dims=4)

iso_mat = zeros(J,J,N,L)
for l=1:L
    intermed = zeros(J,J,N,L)
    for d=1:L
        intermed[:,:,:,d].=test[:,l,:,mod(l+d,L)+1,:]
    end
    iso_mat[:,:,:,l] .= dropdims(sum(intermed,dims=4),dims=4)
end
pow_mat = dropdims(sum(iso_mat,dims=4),dims=4)

function S20_iso(data,J,L,N)
    S20 = reshape(data[2+J*L+1:2+J*L+J*L*J*L,:],J,L,J,L,N)
    iso_mat = zeros(J,J,N,L)
    for l=1:L
        intermed = zeros(J,J,N,L)
        for d=1:L
            intermed[:,:,:,d].=S20[:,l,:,mod(l+d,L)+1,:]
        end
        iso_mat[:,:,:,l] .= dropdims(sum(intermed,dims=4),dims=4)
    end
    pow_mat = zeros(J,J,N)
    pow_mat = dropdims(sum(iso_mat,dims=4),dims=4)
    return pow_mat
end

angle_iso = S20_iso(out0,8,16,360)

plot(angle_iso[3,4,:])

pdf_err = zeros(J,J)
temp = zeros(N)
for j2=1:J
    for j1=1:J
        temp = (angle_iso[j1,j2,:].-mean(angle_iso[j1,j2,:]))./mean(angle_iso[j1,j2,:])
        pdf_err[j1,j2] = maximum(temp)-minimum(temp)
     end
 end

 maximum(pdf_err[2:8,2:8])

 function err_extract(iso_out)
     pdf_err = zeros(J,J)
     temp = zeros(N)
     for j2=1:J
         for j1=1:J
             temp = (iso_out[j1,j2,:].-mean(iso_out[j1,j2,:]))./mean(iso_out[j1,j2,:])
             pdf_err[j1,j2] = maximum(temp)-minimum(temp)
          end
      end
      return maximum(pdf_err[2:7,2:7])
 end

## Let me try to run through whole thing.
out0 = h5read("./Data/j8l16w1p1.h5", "main/data")
angle_iso = S20_iso(out0,8,16,360)
err_extract(angle_iso)

out0 = h5read("./Data/j8l16w2p1.h5", "main/data")
angle_iso = S20_iso(out0,8,16,360)
err_extract(angle_iso)

out0 = h5read("./Data/j8l16w1p2.h5", "main/data")
angle_iso = S20_iso(out0,8,16,360)
err_extract(angle_iso)

out0 = h5read("./Data/j8l16w2p2.h5", "main/data")
angle_iso = S20_iso(out0,8,16,360)
err_extract(angle_iso)

out0 = h5read("./Data/j8l8w1p1.h5", "main/data")
angle_iso = S20_iso(out0,8,8,360)
err_extract(angle_iso)

out0 = h5read("./Data/j8l8w2p1.h5", "main/data")
angle_iso = S20_iso(out0,8,8,360)
err_extract(angle_iso)

out0 = h5read("./Data/j8l8w1p2.h5", "main/data")
angle_iso = S20_iso(out0,8,8,360)
err_extract(angle_iso)

out0 = h5read("./Data/j8l8w2p2.h5", "main/data")
angle_iso = S20_iso(out0,8,8,360)
err_extract(angle_iso)


## Start abs2 test. Probably want a wd 4 test...
#this might seem scarily wide. But if we want L=16/pi, maybe it is not so bad.

function speedy_DHC_abs2(image::Array{Float64,2}, filter_list)
    # Use 2 threads for FFT
    FFTW.set_num_threads(2)

    # array sizes
    (Nx, Ny)  = size(image)
    (J,L)     = size(filter_list[1])

    # allocate coeff arrays
    out_coeff = []
    S0  = zeros(Float64, 2)
    S1  = zeros(Float64, J, L)
    S20 = zeros(Float64, J, L, J, L)
    S12 = zeros(Float64, J, L, J, L)

    # allocate image arrays for internal use
    im_fdf_0_1 = zeros(Float64,           Nx, Ny, J, L)   # this must be zeroed!
    im_rd_0_1  = Array{Float64, 4}(undef, Nx, Ny, J, L)

    ## 0th Order
    S0[1]   = mean(image)
    norm_im = image.-S0[1]
    S0[2]   = sum(norm_im .* norm_im)/(Nx*Ny)
    norm_im ./= sqrt(Nx*Ny*S0[2])

    append!(out_coeff,S0[:])

    ## 1st Order
    im_fd_0 = fft(norm_im)

    # unpack filter_list
    f_ind   = filter_list[1]  # (J, L) array of filters represented as index value pairs
    f_val   = filter_list[2]

    zarr = zeros(ComplexF64, Nx, Ny)  # temporary array to fill with zvals

    # make a FFTW "plan" for an array of the given size and type
    P   = plan_ifft(im_fd_0)   # P is an operator, P*im is ifft(im)

    ## Main 1st Order and Precompute 2nd Order
    for l = 1:L
        for j = 1:J
            S1tot = 0.0
            f_i = f_ind[j,l]  # CartesianIndex list for filter
            f_v = f_val[j,l]  # Values for f_i
            # for (ind, val) in zip(f_i, f_v)   # this is slower!
            if length(f_i) > 0
                for i = 1:length(f_i)
                    ind       = f_i[i]
                    zval      = f_v[i] * im_fd_0[ind]
                    S1tot    += abs2(zval)
                    zarr[ind] = zval
                    im_fdf_0_1[ind,j,l] = abs(zval)
                end
                S1[j,l] = S1tot
                im_rd_0_1[:,:,j,l] .= abs2.(P*zarr)
                zarr[f_i] .= 0
            end
        end
    end
    append!(out_coeff, S1[:])

    # we stored the abs()^2, so take sqrt (this is faster to do all at once)
    #im_rd_0_1 .= sqrt.(im_rd_0_1) #commented out for abs2

    ## 2nd Order
    Amat = reshape(im_fdf_0_1, Nx*Ny, J*L)
    S12  = reshape(Amat' * Amat, J, L, J, L)
    Amat = reshape(im_rd_0_1, Nx*Ny, J*L)
    S20  = reshape(Amat' * Amat, J, L, J, L)

    append!(out_coeff, S20)
    append!(out_coeff, S12)

    return out_coeff
end

J=8
L=8
fink_filter_set2 = fink_filter_list(fink_filter_bank_fast(J,L,wd=1,pc=1))
out0 = zeros(2+J*L+J*L*J*L+J*L*J*L,360)
@views for angle=1:360
    test_rod=rod(10,10,30,angle,4)
    out0[:,angle].=speedy_DHC_abs2(test_rod,fink_filter_set2)
end

h5write("./Data/abs2j8l8w1p1.h5", "main/data", out0)

fink_filter_set2 = fink_filter_list(fink_filter_bank_fast(J,L,wd=2,pc=1))
out0 = zeros(2+J*L+J*L*J*L+J*L*J*L,360)
@views for angle=1:360
    test_rod=rod(10,10,30,angle,4)
    out0[:,angle].=speedy_DHC_abs2(test_rod,fink_filter_set2)
end

h5write("./Data/abs2j8l8w2p1.h5", "main/data", out0)

fink_filter_set2 = fink_filter_list(fink_filter_bank_fast(J,L,wd=1,pc=2))
out0 = zeros(2+J*L+J*L*J*L+J*L*J*L,360)
@views for angle=1:360
    test_rod=rod(10,10,30,angle,4)
    out0[:,angle].=speedy_DHC_abs2(test_rod,fink_filter_set2)
end

h5write("./Data/abs2j8l8w1p2.h5", "main/data", out0)

fink_filter_set2 = fink_filter_list(fink_filter_bank_fast(J,L,wd=2,pc=2))
out0 = zeros(2+J*L+J*L*J*L+J*L*J*L,360)
@views for angle=1:360
    test_rod=rod(10,10,30,angle,4)
    out0[:,angle].=speedy_DHC_abs2(test_rod,fink_filter_set2)
end

h5write("./Data/abs2j8l8w2p2.h5", "main/data", out0)

J=8
L=16
fink_filter_set2 = fink_filter_list(fink_filter_bank_fast(J,L,wd=1,pc=1))
out0 = zeros(2+J*L+J*L*J*L+J*L*J*L,360)
@views for angle=1:360
    test_rod=rod(10,10,30,angle,4)
    out0[:,angle].=speedy_DHC_abs2(test_rod,fink_filter_set2)
end

h5write("./Data/abs2j8l16w1p1.h5", "main/data", out0)

fink_filter_set2 = fink_filter_list(fink_filter_bank_fast(J,L,wd=2,pc=1))
out0 = zeros(2+J*L+J*L*J*L+J*L*J*L,360)
@views for angle=1:360
    test_rod=rod(10,10,30,angle,4)
    out0[:,angle].=speedy_DHC_abs2(test_rod,fink_filter_set2)
end

h5write("./Data/abs2j8l16w2p1.h5", "main/data", out0)

fink_filter_set2 = fink_filter_list(fink_filter_bank_fast(J,L,wd=1,pc=2))
out0 = zeros(2+J*L+J*L*J*L+J*L*J*L,360)
@views for angle=1:360
    test_rod=rod(10,10,30,angle,4)
    out0[:,angle].=speedy_DHC_abs2(test_rod,fink_filter_set2)
end

h5write("./Data/abs2j8l16w1p2.h5", "main/data", out0)

fink_filter_set2 = fink_filter_list(fink_filter_bank_fast(J,L,wd=2,pc=2))
out0 = zeros(2+J*L+J*L*J*L+J*L*J*L,360)
@views for angle=1:360
    test_rod=rod(10,10,30,angle,4)
    out0[:,angle].=speedy_DHC_abs2(test_rod,fink_filter_set2)
end

h5write("./Data/abs2j8l16w2p2.h5", "main/data", out0)

## Let me try to run through whole thing with abs2
out0 = h5read("./Data/abs2j8l16w1p1.h5", "main/data")
angle_iso = S20_iso(out0,8,16,360)
err_extract(angle_iso)

out0 = h5read("./Data/abs2j8l16w2p1.h5", "main/data")
angle_iso = S20_iso(out0,8,16,360)
err_extract(angle_iso)

out0 = h5read("./Data/abs2j8l16w1p2.h5", "main/data")
angle_iso = S20_iso(out0,8,16,360)
err_extract(angle_iso)

out0 = h5read("./Data/abs2j8l16w2p2.h5", "main/data")
angle_iso = S20_iso(out0,8,16,360)
err_extract(angle_iso)

out0 = h5read("./Data/abs2j8l8w1p1.h5", "main/data")
angle_iso = S20_iso(out0,8,8,360)
err_extract(angle_iso)

out0 = h5read("./Data/abs2j8l8w2p1.h5", "main/data")
angle_iso = S20_iso(out0,8,8,360)
err_extract(angle_iso)

out0 = h5read("./Data/abs2j8l8w1p2.h5", "main/data")
angle_iso = S20_iso(out0,8,8,360)
err_extract(angle_iso)

out0 = h5read("./Data/abs2j8l8w2p2.h5", "main/data")
angle_iso = S20_iso(out0,8,8,360)
err_extract(angle_iso)

## try wd = 3

J=8
L=8
fink_filter_set2 = fink_filter_list(fink_filter_bank_fast(J,L,wd=3,pc=1))
out0 = zeros(2+J*L+J*L*J*L+J*L*J*L,360)
@views for angle=1:360
    test_rod=rod(10,10,30,angle,4)
    out0[:,angle].=speedy_DHC_abs2(test_rod,fink_filter_set2)
end

h5write("./Data/abs2j8l8w3p1.h5", "main/data", out0)

fink_filter_set2 = fink_filter_list(fink_filter_bank_fast(J,L,wd=3,pc=2))
out0 = zeros(2+J*L+J*L*J*L+J*L*J*L,360)
@views for angle=1:360
    test_rod=rod(10,10,30,angle,4)
    out0[:,angle].=speedy_DHC_abs2(test_rod,fink_filter_set2)
end

h5write("./Data/abs2j8l8w3p2.h5", "main/data", out0)

fink_filter_set2 = fink_filter_list(fink_filter_bank_fast(J,L,wd=3,pc=1))
out0 = zeros(2+J*L+J*L*J*L+J*L*J*L,360)
@views for angle=1:360
    test_rod=rod(10,10,30,angle,4)
    out0[:,angle].=speedy_DHC(test_rod,fink_filter_set2)
end

h5write("./Data/j8l8w3p1.h5", "main/data", out0)

fink_filter_set2 = fink_filter_list(fink_filter_bank_fast(J,L,wd=3,pc=2))
out0 = zeros(2+J*L+J*L*J*L+J*L*J*L,360)
@views for angle=1:360
    test_rod=rod(10,10,30,angle,4)
    out0[:,angle].=speedy_DHC(test_rod,fink_filter_set2)
end

h5write("./Data/j8l8w3p2.h5", "main/data", out0)

J=8
L=16
fink_filter_set2 = fink_filter_list(fink_filter_bank_fast(J,L,wd=3,pc=1))
out0 = zeros(2+J*L+J*L*J*L+J*L*J*L,360)
@views for angle=1:360
    test_rod=rod(10,10,30,angle,4)
    out0[:,angle].=speedy_DHC(test_rod,fink_filter_set2)
end

h5write("./Data/j8l16w3p1.h5", "main/data", out0)

fink_filter_set2 = fink_filter_list(fink_filter_bank_fast(J,L,wd=3,pc=2))
out0 = zeros(2+J*L+J*L*J*L+J*L*J*L,360)
@views for angle=1:360
    test_rod=rod(10,10,30,angle,4)
    out0[:,angle].=speedy_DHC(test_rod,fink_filter_set2)
end

h5write("./Data/j8l16w3p2.h5", "main/data", out0)

fink_filter_set2 = fink_filter_list(fink_filter_bank_fast(J,L,wd=3,pc=1))
out0 = zeros(2+J*L+J*L*J*L+J*L*J*L,360)
@views for angle=1:360
    test_rod=rod(10,10,30,angle,4)
    out0[:,angle].=speedy_DHC_abs2(test_rod,fink_filter_set2)
end

h5write("./Data/abs2j8l16w3p1.h5", "main/data", out0)

J=8
L=16
fink_filter_set2 = fink_filter_list(fink_filter_bank_fast(J,L,wd=3,pc=2))
out0 = zeros(2+J*L+J*L*J*L+J*L*J*L,360)
@views for angle=1:360
    test_rod=rod(10,10,30,angle,4)
    out0[:,angle].=speedy_DHC_abs2(test_rod,fink_filter_set2)
end

h5write("./Data/abs2j8l16w3p2.h5", "main/data", out0)

## WU w3 case

out0 = h5read("./Data/abs2j8l8w3p1.h5", "main/data")
angle_iso = S20_iso(out0,8,8,360)
err_extract(angle_iso)

out0 = h5read("./Data/abs2j8l8w3p2.h5", "main/data")
angle_iso = S20_iso(out0,8,8,360)
err_extract(angle_iso)

out0 = h5read("./Data/j8l8w3p1.h5", "main/data")
angle_iso = S20_iso(out0,8,8,360)
err_extract(angle_iso)

out0 = h5read("./Data/j8l8w3p2.h5", "main/data")
angle_iso = S20_iso(out0,8,8,360)
err_extract(angle_iso)


out0 = h5read("./Data/abs2j8l16w3p1.h5", "main/data")
angle_iso = S20_iso(out0,8,16,360)
err_extract(angle_iso)

out0 = h5read("./Data/abs2j8l16w3p2.h5", "main/data")
angle_iso = S20_iso(out0,8,16,360)
err_extract(angle_iso)

out0 = h5read("./Data/j8l16w3p1.h5", "main/data")
angle_iso = S20_iso(out0,8,16,360)
err_extract(angle_iso)

out0 = h5read("./Data/j8l16w3p2.h5", "main/data")
angle_iso = S20_iso(out0,8,16,360)
err_extract(angle_iso)

## Start L=32 Insanity

J=8
L=32
fink_filter_set2 = fink_filter_list(fink_filter_bank_fast(J,L,wd=1,pc=1))
out0 = zeros(2+J*L+J*L*J*L+J*L*J*L,360)
@views for angle=1:360
    test_rod=rod(10,10,30,angle,4)
    out0[:,angle].=speedy_DHC(test_rod,fink_filter_set2)
end

h5write("./Data/j8l32w1p1.h5", "main/data", out0)

fink_filter_set2 = fink_filter_list(fink_filter_bank_fast(J,L,wd=2,pc=1))
out0 = zeros(2+J*L+J*L*J*L+J*L*J*L,360)
@views for angle=1:360
    test_rod=rod(10,10,30,angle,4)
    out0[:,angle].=speedy_DHC(test_rod,fink_filter_set2)
end

h5write("./Data/j8l32w2p1.h5", "main/data", out0)

fink_filter_set2 = fink_filter_list(fink_filter_bank_fast(J,L,wd=3,pc=1))
out0 = zeros(2+J*L+J*L*J*L+J*L*J*L,360)
@views for angle=1:360
    test_rod=rod(10,10,30,angle,4)
    out0[:,angle].=speedy_DHC(test_rod,fink_filter_set2)
end

h5write("./Data/j8l32w3p1.h5", "main/data", out0)

## abs2

fink_filter_set2 = fink_filter_list(fink_filter_bank_fast(J,L,wd=1,pc=1))
out0 = zeros(2+J*L+J*L*J*L+J*L*J*L,360)
@views for angle=1:360
    test_rod=rod(10,10,30,angle,4)
    out0[:,angle].=speedy_DHC_abs2(test_rod,fink_filter_set2)
end

h5write("./Data/abs2j8l32w1p1.h5", "main/data", out0)

fink_filter_set2 = fink_filter_list(fink_filter_bank_fast(J,L,wd=2,pc=1))
out0 = zeros(2+J*L+J*L*J*L+J*L*J*L,360)
@views for angle=1:360
    test_rod=rod(10,10,30,angle,4)
    out0[:,angle].=speedy_DHC_abs2(test_rod,fink_filter_set2)
end

h5write("./Data/abs2j8l32w2p1.h5", "main/data", out0)

fink_filter_set2 = fink_filter_list(fink_filter_bank_fast(J,L,wd=3,pc=1))
out0 = zeros(2+J*L+J*L*J*L+J*L*J*L,360)
@views for angle=1:360
    test_rod=rod(10,10,30,angle,4)
    out0[:,angle].=speedy_DHC_abs2(test_rod,fink_filter_set2)
end

h5write("./Data/abs2j8l32w3p1.h5", "main/data", out0)

## Work up

out0 = h5read("./Data/j8l32w1p1.h5", "main/data")
angle_iso = S20_iso(out0,8,32,360)
err_extract(angle_iso)

out0 = h5read("./Data/j8l32w2p1.h5", "main/data")
angle_iso = S20_iso(out0,8,32,360)
err_extract(angle_iso)

out0 = h5read("./Data/j8l32w3p1.h5", "main/data")
angle_iso = S20_iso(out0,8,32,360)
err_extract(angle_iso)


out0 = h5read("./Data/abs2j8l32w1p1.h5", "main/data")
angle_iso = S20_iso(out0,8,32,360)
err_extract(angle_iso)

out0 = h5read("./Data/abs2j8l32w2p1.h5", "main/data")
angle_iso = S20_iso(out0,8,32,360)
err_extract(angle_iso)

out0 = h5read("./Data/abs2j8l32w3p1.h5", "main/data")
angle_iso = S20_iso(out0,8,32,360)
err_extract(angle_iso)

## Try S12 analysis on the 2pi coverage cases. No rotation. Then do j1/2

function S20_iso(data,J,L,N)
    S20 = reshape(data[2+J*L+J*L*J*L+1:2+J*L+J*L*J*L+J*L*J*L,:],J,L,J,L,N)
    iso_mat = zeros(J,J,N,L)
    for l=1:L
        intermed = zeros(J,J,N,L)
        for d=1:L
            intermed[:,:,:,d].=S20[:,l,:,mod(l+d,L)+1,:]
        end
        iso_mat[:,:,:,l] .= dropdims(sum(intermed,dims=4),dims=4)
    end
    pow_mat = zeros(J,J,N)
    pow_mat = dropdims(sum(iso_mat,dims=4),dims=4)
    return pow_mat
end

angle_iso = S20_iso(out0,8,16,360)

plot(angle_iso[3,4,:])

 function err_extract(iso_out)
     pdf_err = zeros(J,J)
     temp = zeros(N)
     for j2=1:J
         for j1=1:J
             temp = (iso_out[j1,j2,:].-mean(iso_out[j1,j2,:]))./mean(iso_out[j1,j2,:])
             pdf_err[j1,j2] = maximum(temp)-minimum(temp)
          end
      end
      return maximum(pdf_err[2:7,2:7])
 end
