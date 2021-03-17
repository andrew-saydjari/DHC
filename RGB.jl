push!(LOAD_PATH, "/Users/saydjari/Dropbox/GradSchool_AKS/Doug/Projects/DHC/main")
using DHC_2DUtils
using Profile
using BenchmarkTools
using FFTW
using HDF5
using Test
using SparseArrays
using Statistics
using Plots
using LinearAlgebra
using Measures
using Plots
using MLDatasets
using Images
theme(:juno)
using ImageFiltering

filter_hash = fink_filter_hash(1, 8, nx=256, pc=1, wd=2)

train_x, train_y = CIFAR10.traindata();
test_x, test_y   = CIFAR10.testdata();

using TestImages
using ImageMetadata
using ImageAxes
using ImageDraw
using ImageView
using ImageTransformations
using ImageCore, Colors

ind = 10
s = StackedView(transpose(train_x[:,:,1,ind]),transpose(train_x[:,:,2,ind]),transpose(train_x[:,:,3,ind]));
sc = colorview(RGB, s)

image_x = train_x[:,:,:,ind]
size(train_x[:,:,:,ind])

mean(image_x)

@benchmark mean(image_x, dims=(1,2))

mean_x = dropdims(mean(image_x, dims=(1,2)),dims=(1,2))
mean_x_shaped = mean(image_x, dims=(1,2))
norm_im = image_x .- mean_x_shaped

pwr_im = sum(norm_im .* norm_im,dims=(1,2))
norm_im./sqrt.(32*32 .*pwr_im)

import AbstractFFTs

@benchmark P0 = plan_fft(norm_im,(1,2))
@benchmark im_fd_0 = fft(norm_im,(1,2))
@benchmark im_fd_0 = P0*norm_im

S1M = filter_hash["S1_iso_mat"]
Mtest = [S1M 0I 0I; 0I S1M 0I; 0I 0I S1M]
blockdiag(S1M,S1M,S1M)

1+1

function DHC_compute_RGB(image::Array{Float64}, filter_hash::Dict, filter_hash2::Dict=filter_hash;
    doS20::Bool=true, norm=true, iso=false, FFTthreads=2)
    # image        - input for WST
    # filter_hash  - filter hash from fink_filter_hash
    # filter_hash2 - filters for second order.  Default to same as first order.
    # doS2         - compute S2 coeffs
    # doS12        - compute S2 coeffs
    # doS20        - compute S2 coeffs
    # norm         - scale to mean zero, unit variance
    # iso          - sum over angles to obtain isotropic coeffs

    # Use 2 threads for FFT
    FFTW.set_num_threads(FFTthreads)

    # array sizes
    (Nx, Ny, Nc)  = size(image)
    if Nx != Ny error("Input image must be square") end
    (Nf, )    = size(filter_hash["filt_index"])
    if Nf == 0  error("filter hash corrupted") end
    @assert Nx==filter_hash["npix"] "Filter size should match npix"
    @assert Nx==filter_hash2["npix"] "Filter2 size should match npix"

    # allocate coeff arrays
    out_coeff = []
    S0  = zeros(Float64, Nc*2)
    S1  = zeros(Float64, Nc*Nf)

    if doS20 S20 = zeros(Float64, Nf, Nf) end  # real space correlatio
    # anyM2 = doS2 | doS12 | doS20
    anyrd = doS20 #| doS2             # compute real domain with iFFT

    # allocate image arrays for internal use
    mean_im = zeros(Float64,1,1,Nc)
    pwr_im = zeros(Float64,1,1,Nc)
    norm_im = zeros(Float64,Nx,Ny,Nc)
    im_fd_0 = zeros(ComplexF64, Nx, Ny, Nc)
    im_fd_0_sl = zeros(ComplexF64, Nx, Ny)

    if doS20
        Amat1 = zeros(Nx*Ny, Nf)
        Amat2 = zeros(Nx*Ny, Nf)
    end

    if anyrd im_rd_0_1  = Array{Float64, 4}(undef, Nx, Ny, Nf, Nc) end

    ## 0th Order
    mean_im = mean(image, dims=(1,2))
    S0[1:Nc]   = dropdims(mean_im,dims=(1,2))
    norm_im = image.-mean_im
    pwr_im = sum(norm_im .* norm_im,dims=(1,2))
    S0[1+Nc:end]   = dropdims(pwr_im,dims=(1,2))./(Nx*Ny)
    if norm
        norm_im ./= sqrt.(pwr_im)
    else
        norm_im = copy(image)
    end

    append!(out_coeff,S0[:])

    ## 1st Order
    im_fd_0 .= fft(norm_im,(1,2))  # total power=1.0

    # unpack filter_hash
    f_ind   = filter_hash["filt_index"]  # (J, L) array of filters represented as index value pairs
    f_val   = filter_hash["filt_value"]

    zarr = zeros(ComplexF64, Nx, Ny)  # temporary array to fill with zvals

    # make a FFTW "plan" for an array of the given size and type
    if anyrd
        P = plan_ifft(zarr) end  # P is an operator, P*im is ifft(im)

    ## Main 1st Order and Precompute 2nd Order
    for f = 1:Nf
        S1tot = 0.0
        f_i = f_ind[f]  # CartesianIndex list for filter
        f_v = f_val[f]  # Values for f_i
        # for (ind, val) in zip(f_i, f_v)   # this is slower!
        for chan = 1:Nc
            im_fd_0_sl .= im_fd_0[:,:,chan]
            if length(f_i) > 0
                for i = 1:length(f_i)
                    ind       = f_i[i]
                    zval      = f_v[i] * im_fd_0_sl[ind]
                    S1tot    += abs2(zval)
                    zarr[ind] = zval        # filter*image in Fourier domain
                end
                S1[f+(chan-1)*Nf] = S1tot/(Nx*Ny)  # image power
                if anyrd
                    im_rd_0_1[:,:,f,chan] .= abs2.(P*zarr) end
                zarr[f_i] .= 0
            end
        end
    end

    if iso
        S1M = filter_hash["S1_iso_mat"]
        M1 = blockdiag(S1M,S1M,S1M)
    end
    append!(out_coeff, iso ? M1*S1 : S1)

    # we stored the abs()^2, so take sqrt (this is faster to do all at once)
    if anyrd im_rd_0_1 .= sqrt.(im_rd_0_1) end

    # Real domain 2nd order
    if doS20
        for chan1 = 1:Nc
            for chan2 = 1:Nc
                Amat1 = reshape(im_rd_0_1[:,:,:,chan1], Nx*Ny, Nf)
                Amat2 = reshape(im_rd_0_1[:,:,:,chan2], Nx*Ny, Nf)
                S20  = Amat1' * Amat2
                append!(out_coeff, iso ? filter_hash["S2_iso_mat"]*S20[:] : S20[:])
            end
        end
    end

    return out_coeff
end

DHC_compute_RGB(train_x[:,:,:,ind],filter_hash)

using Interpolations
using DSP

function wind_2d_RGB(nx)
    dx   = nx/2-1
    filter = zeros(Float64, nx, nx)
    A = DSP.tukey(nx, 0.3)
    itp = extrapolate(interpolate(A,BSpline(Linear())),0)
    @inbounds for x = 1:nx
        sx = x-dx-1    # define sx,sy so that no fftshift() needed
        for y = 1:nx
            sy = y-dx-1
            r  = sqrt.((sx).^2 + (sy).^2) + nx/2
            filter[x,y] = itp(r)
        end
    end
    return reshape(filter,nx,nx,1)
end

wind_2d_RGB(256)

function apodizer(data,sp1,sp2,im_size)
    temp2d = data[sp1:sp1+im_size,sp2:sp2+im_size]
    datad_w = fweights(wind_2d(256));
    meanVal = mean(temp2d,datad_w)
    temp2d_a = (temp2d.-meanVal).*wind_2d(256).+meanVal
    return temp2d_a
end

function wind_2d(nx)
    dx   = nx/2-1
    filter = zeros(Float64, nx, nx)
    A = DSP.tukey(nx, 0.3)
    itp = extrapolate(interpolate(A,BSpline(Linear())),0)
    @inbounds for x = 1:nx
        sx = x-dx-1    # define sx,sy so that no fftshift() needed
        for y = 1:nx
            sy = y-dx-1
            r  = sqrt.((sx).^2 + (sy).^2) + nx/2
            filter[x,y] = itp(r)
        end
    end
    return filter
end

using StatsBase

temp = imresize(train_x[:,:,:,ind],(64,64,3))
mean(convert(Array{Float64,3},temp),dims=(1,2))
mean(temp,dims=(1,2))
convert(Array{Float64,3},temp)

test = zeros(64,64,3)
test[:,:,1] = wind_2d_RGB(64)
test[:,:,2] = wind_2d_RGB(64)
test[:,:,3] = wind_2d_RGB(64)

datad_w = fweights(wind_2d_RGB(64));
datad_w = fweights(test);
mean(temp,reshape(datad_w,64,64,3),dims=1)
mean(temp,datad_w)

size(temp)
size(reshape(datad_w,64,64,3))

function cifar_pad_RGB(im; θ=0.0)
    imbig = convert(Array{Float64,3},imresize(im,(64,64,3)))
    datad_w = fweights(wind_2d(64));
    mu_imbig = zeros(1,1,3)
    for chan = 1:3
        mu_imbig[chan] = mean(imbig[:,:,chan],datad_w)
    end
    imbig .-= mu_imbig
    imbig .*= wind_2d_RGB(64)
    impad = zeros(Float64,128,128,3)
    impad[96:-1:33,33:96,:] = imbig

    if θ != 0.0
        imrot = imrotate(impad, θ, axes(impad), Cubic(Throw(OnGrid())))
        imrot[findall(imrot .!= imrot)] .= 0.0
        return imrot .+ mu_imbig
    end

    return impad.+ mu_imbig
end

out_test = cifar_pad_RGB(train_x[:,:,:,ind])

function mnist_pad(im; θ=0.0)
    imbig = convert(Array{Float64,2},imresize(im,(64,64)))
    mu_imbig = mean(imbig)
    imbig .-= mu_imbig
    imbig .*= wind_2d(64)
    impad = zeros(Float64,128,128)
    impad[96:-1:33,33:96] = imbig

    if θ != 0.0
        imrot = imrotate(impad, θ, axes(impad), Cubic(Throw(OnGrid())))
        imrot[findall(imrot .!= imrot)] .= 0.0
        return imrot .+ mu_imbig
    end

    return impad.+ mu_imbig
end

d2_test = mnist_pad(train_x[:,:,1,ind])
@time d2_test = mnist_pad(train_x[:,:,1,ind])

s = StackedView(transpose(out_test[:,:,1]),transpose(out_test[:,:,2]),transpose(out_test[:,:,3]));
sc = colorview(RGB, s)

@time filter_hash = fink_filter_hash(1, 8, nx=128, pc=1, wd=2)

DHC_compute(d2_test,filter_hash)
DHC_compute_RGB(out_test,filter_hash)

@time DHC_compute(d2_test,filter_hash)

@time DHC_compute_RGB(out_test,filter_hash,doS20=true)


chan_img = channelview(train_x[:,:,:,ind])
cv = colorview(RGB, StackedView(train_x[:,:,1,ind],train_x[:,:,2,ind],train_x[:,:,3,ind]))
convert(RGB, HSL(270, 0.5, 0.5))
y_im = YCbCr.(cv)
channels = permutedims(channelview(float.(y_im)),(2,3,1))
