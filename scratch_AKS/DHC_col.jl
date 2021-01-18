## Preloads
using Statistics
using FFTW
using Plots
using BenchmarkTools
using Profile
using LinearAlgebra
using StaticArrays
using HybridArrays


# %%code cell

function finklet(j, l)
    # -------- set filters
    jrad = 7-j
    dθ = π/8        # 8 angular bins hardwired
    θ_l = dθ*l
    # -------- define coordinates
    nx = 256
    xbox = LinRange(-nx/2, nx/2-1 , nx)
    # make a 256x256 grid of X
    sx = xbox' .* ones(nx)
    sy = ones(nx)' .* xbox
    r  = sqrt.((sx).^2 + (sy).^2)
    θ  = mod.(atan.(sy, sx).+π .-θ_l,2*π)
    nozeros = r .> 0
    logr = log2.(r[nozeros])
    r[nozeros] = logr
    # -------- in Fourier plane, envelope of psi_j,l
    mask = (abs.(θ.-π).<= dθ) .& (abs.(r.-jrad) .<= 1)
    # -------- angular part
    ang = cos.((θ.-π).*4)
    # -------- radial part
    rad = cos.((r.-jrad).*π./2)
    psi = mask.*ang.*rad             #mask times angular part times radial part
    return psi
end

function fink_filter_bank(J,L)
    fink_filter = Array{Float64, 4}(undef, 256, 256, J, L)
    for l = 1:L
        for j = 1:J
            fink_filter[:,:,j,l]=fftshift(finklet(j-1,l-1))
        end
    end
    return fink_filter
end

function had!(A,B)
    m,n = size(A)
    @assert (m,n) == size(B)
    for j in 1:n
       for i in 1:m
         @inbounds A[i,j] *= B[i,j]
       end
    end
    return A
end

fink_filter_set = fink_filter_bank(8,8)

test_img = zeros(256,256)
copyto!(test_img,fink_filter_set[:,:,1,3])

function DHC(image, filter_set; norm_on = 1, coeff_12_on =1, coeff_20_on = 1)
    FFTW.set_num_threads(2)
    (Nx, Ny) = size(image)

    (_,_,J,L) = size(fink_filter_set)

    out_coeff = []

    ## Coeff at (1,0)
    #takes care of image normalization
    S0 = zeros(2)
    if norm_on == 1
        S0[1] = mean(image)
        norm_im = image.-S0[1]
        S0[2] = sum(abs2.(norm_im))/(Nx*Ny)
        norm_im = norm_im./sqrt(Nx*Ny*S0[2])
    else
        norm_im = image
    end

    append!(out_coeff,S0)

    ##Coeff at (1,1)
    #store the fft of images and im*filters for later
    im_fd_0 = fft(norm_im)

    S1 = zeros((J,L))
    im_fd_0_1 = Array{ComplexF64,4}(undef, Nx, Ny, J, L)
    @views for l = 1:L
        for j = 1:J
            im_fd_0_1[:,:,j,l] .= im_fd_0
        end
    end

    if coeff_20_on == 1
        im_rd_0_1 = Array{ComplexF64,4}(undef, Nx, Ny, J, L)
    end

    @views for l = 1:L
        for j = 1:J
            had!(im_fd_0_1[:,:,j,l],filter_set[:,:,j,l]) #wavelet already in fft domain not shifted
            S1[j,l]+=sum(abs2.(im_fd_0_1[:,:,j,l])) #normalization choice arb to make order unity
            if coeff_20_on == 1
                im_rd_0_1[:,:,j,l] .= ifft(im_fd_0_1[:,:,j,l])
            end
        end
    end
    append!(out_coeff,S1)

    ##Coeff at (2,0)
    if coeff_20_on == 1
        S20 = zeros((J, L, J, L))

        Atmp = HybridArray{Tuple{256,StaticArrays.Dynamic()}}(zeros(ComplexF64,Nx,Ny))
        Btmp = HybridArray{Tuple{256,StaticArrays.Dynamic()}}(zeros(ComplexF64,Nx,Ny))
        interm = 0
        @views for l2 = 1:L
            for j2 = 1:J
                copyto!(Btmp, im_rd_0_1[:,:,j2,l2])
                for l1 = 1:L
                    for j1  = 1:J
                        copyto!(Atmp, im_rd_0_1[:,:,j1,l1])
                        S20[j1,l1,j2,l2] += sum(abs.(Atmp.*Btmp))
                    end
                end
            end
        end

        append!(out_coeff,S20)
    end

    ##Coeff at (1,2) level
    if coeff_12_on == 1
        S12 = Array{Float64,4}(undef, J, J, L, L)
        for j1 = 1:J
            for l1 = 1:L
                not_rot_im = fftshift(im_fd_0_1[j1,l1,:,:])
                for l2 = 1:L
                    for j2 = 1:J
                        S12[j1,j2,l1,l2] += sum(abs.(not_rot_im.*fftshift(im_fd_0_1[j2,l2,:,:])))
                    end
                end
            end
        end
    append!(out_coeff,S12)
    end
    return out_coeff
end

function dual_conj(A,B)
    @. abs(A.*B)
end

temp = DHC(test_img,fink_filter_set,coeff_12_on =0, coeff_20_on = 1)

@time DHC(test_img,fink_filter_set,coeff_12_on =0, coeff_20_on = 1)

Profile.clear()
@profile DHC(test_img,fink_filter_set,coeff_12_on =0, coeff_20_on = 1)

Juno.profiler()

BenchmarkTools.DEFAULT_PARAMETERS.samples = 5
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 120
@benchmark DHC(test_img,fink_filter_set,coeff_12_on =0, coeff_20_on = 1)

##Trying out static speed ups

function dual_conj_pre(A,B)
    Atmp = SMatrix{256,256}(A)
    Btmp = SMatrix{256,256}(B)
    @. abs(Atmp.*Btmp)
end

@time dual_conj_pre(test_img,test_img)

test_img_s = @SMatrix rand(4,4)

function mult_wrapper(A,B)
    sum(abs.(A.*B))
end

@time mult_wrapper(test_img_s,test_img_s)

test_img_2 = rand(4,4)

@time mult_wrapper(test_img_2,test_img_2)

A = HybridArray{Tuple{256,StaticArrays.Dynamic()}}(randn(256,256))

mult_wrapper(A,A)

@benchmark mult_wrapper(A,A)

A = randn(256,256)

@benchmark mult_wrapper(A,A)

function mult_wrapper(A,B)
    A = HybridArray{Tuple{256,StaticArrays.Dynamic()}}(randn(256,256))
    B = HybridArray{Tuple{256,StaticArrays.Dynamic()}}(randn(256,256))
    sum(abs.(A.*B))
end

@benchmark mult_wrapper(A,A)

Btmp = HybridArray{Tuple{256,StaticArrays.Dynamic()}}(zeros(ComplexF64,256,256))
copyto!(Btmp, fink_filter_set[:,:,5,5])

## Suppose I was willing to reshape in DHC_ISO

function structure_test(J,L)
    S20 = zeros((J, L, J, L))
    @views for l2 = 1:L
        for j2 = 1:J
            for l1 = 1:L
                for j1  = 1:J
                    S20[j1,l1,j2,l2] = rand()
                end
            end
        end
    end
end

@benchmark structure_test(8,8)

function structure_test(J,L)
    S20 = zeros(J*L*J*L)
    @views for l2 = 1:L
        for j2 = 1:J
            for l1 = 1:L
                for j1  = 1:J
                    S20[(j1-1)*J+(l1-1)*L+(j2-1)*J+l2] = rand()
                end
            end
        end
    end
end

@benchmark structure_test(8,8)

function structure_test(J,L)
    S20 = zeros(J*L*J*L)
    Atmp = HybridArray{Tuple{256,StaticArrays.Dynamic()}}(zeros(ComplexF64,256,256))
    Btmp = HybridArray{Tuple{256,StaticArrays.Dynamic()}}(zeros(ComplexF64,256,256))
    @views for l2 = 1:L
        for j2 = 1:J
            copyto!(Btmp, fink_filter_set[:,:,j2,l2])
            for l1 = 1:L
                for j1  = 1:J
                    copyto!(Atmp, fink_filter_set[:,:,j1,l1])
                    S20[(j1-1)*J+(l1-1)*L+(j2-1)*J+l2] = sum(abs.(Atmp.*Btmp))
                end
            end
        end
    end
end

structure_test(8,8)

@time structure_test(8,8)

@benchmark structure_test(8,8)

function structure_test(J,L)
    S20 = zeros(J,L,J,L)
    Atmp = HybridArray{Tuple{256,StaticArrays.Dynamic()}}(zeros(ComplexF64,256,256))
    Btmp = HybridArray{Tuple{256,StaticArrays.Dynamic()}}(zeros(ComplexF64,256,256))
    @views for l2 = 1:L
        for j2 = 1:J
            copyto!(Btmp, fink_filter_set[:,:,j2,l2])
            for l1 = 1:L
                for j1  = 1:J
                    copyto!(Atmp, fink_filter_set[:,:,j1,l1])
                    S20[j1,l1,j2,l2] = sum(abs.(Atmp.*Btmp))
                end
            end
        end
    end
end

structure_test(8,8)

@benchmark structure_test(8,8)

## Using BLAS

function structure_test(J,L,test_img_1)
    S20 = zeros((J, L, J, L))
    @views for l2 = 1:L
        for j2 = 1:J
            for l1 = 1:L
                for j1  = 1:J
                    S20[j1,l1,j2,l2] = sum(abs.(test_img_1))
                end
            end
        end
    end
end

structure_test(8,8,rand(256,256))

@benchmark structure_test(8,8,rand(256,256))

function structure_test(J,L,test_img_1)
    S20 = zeros((J, L, J, L))
    @views for l2 = 1:L
        for j2 = 1:J
            for l1 = 1:L
                for j1  = 1:J
                    S20[j1,l1,j2,l2] = BLAS.asum(test_img_1)
                end
            end
        end
    end
end

structure_test(8,8,rand(256,256))

@benchmark structure_test(8,8,rand(256,256))

# %% codecell

function DHC(image, filter_set; norm_on = 1, coeff_12_on =1, coeff_20_on = 1)
    FFTW.set_num_threads(2)
    (Nx, Ny) = size(image)

    (_,_,J,L) = size(fink_filter_set)

    out_coeff = []

    ## Coeff at (1,0)
    #takes care of image normalization
    S0 = zeros(2)
    if norm_on == 1
        S0[1] = mean(image)
        norm_im = image.-S0[1]
        S0[2] = sum(abs2.(norm_im))/(Nx*Ny)
        norm_im = norm_im./sqrt(Nx*Ny*S0[2])
    else
        norm_im = image
    end

    append!(out_coeff,S0)

    ##Coeff at (1,1)
    #store the fft of images and im*filters for later
    im_fd_0 = fft(norm_im)

    S1 = zeros((J,L))
    im_fd_0_1 = Array{ComplexF64,4}(undef, Nx, Ny, J, L)
    @views for l = 1:L
        for j = 1:J
            im_fd_0_1[:,:,j,l] .= im_fd_0
        end
    end

    if coeff_20_on == 1
        im_rd_0_1 = Array{ComplexF64,4}(undef, Nx, Ny, J, L)
    end

    @views for l = 1:L
        for j = 1:J
            had!(im_fd_0_1[:,:,j,l],filter_set[:,:,j,l]) #wavelet already in fft domain not shifted
            S1[j,l]+=sum(abs2.(im_fd_0_1[:,:,j,l])) #normalization choice arb to make order unity
            if coeff_20_on == 1
                im_rd_0_1[:,:,j,l] .= ifft(im_fd_0_1[:,:,j,l])
            end
        end
    end
    append!(out_coeff,S1)

    ##Coeff at (2,0)
    if coeff_20_on == 1
        S20 = zeros((J, L, J, L))

        Atmp = zeros(ComplexF64,Nx,Ny)
        Btmp = zeros(ComplexF64,Nx,Ny)
        interm = 0
        @views for l2 = 1:L
            for j2 = 1:J
                copyto!(Btmp, im_rd_0_1[:,:,j2,l2])
                for l1 = 1:L
                    for j1  = 1:J
                        copyto!(Atmp, im_rd_0_1[:,:,j1,l1])
                        S20[j1,l1,j2,l2] += BLAS.asum(Atmp.*Btmp)
                    end
                end
            end
        end

        append!(out_coeff,S20)
    end

    ##Coeff at (1,2) level
    if coeff_12_on == 1
        S12 = Array{Float64,4}(undef, J, J, L, L)
        for j1 = 1:J
            for l1 = 1:L
                not_rot_im = fftshift(im_fd_0_1[j1,l1,:,:])
                for l2 = 1:L
                    for j2 = 1:J
                        S12[j1,j2,l1,l2] += sum(abs.(not_rot_im.*fftshift(im_fd_0_1[j2,l2,:,:])))
                    end
                end
            end
        end
    append!(out_coeff,S12)
    end
    return out_coeff
end

temp = DHC(test_img,fink_filter_set,coeff_12_on =0, coeff_20_on = 1)

@time DHC(test_img,fink_filter_set,coeff_12_on =0, coeff_20_on = 1)

Profile.clear()
@profile DHC(test_img,fink_filter_set,coeff_12_on =0, coeff_20_on = 1)

Juno.profiler()

BenchmarkTools.DEFAULT_PARAMETERS.samples = 5
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 120
@benchmark DHC(test_img,fink_filter_set,coeff_12_on =0, coeff_20_on = 1)

## Factor of 2 from BLAS asum
# Need to do something for .*

BLAS.dot(test_img,test_img)

dot(test_img,test_img)

At = [1 2 3; 4 5 6; 7 8 9]
Bt = [1 2 3; 4 5 6; 7 8 9]
dot(At,Bt)

1^2+2^2+3^2+4^2+5^2+6^2+7^2+8^2+9^2

## Ok, so asum was a cool find, but dot(abs) is prob better

function structure_test1(J,L,test_img)
    S1 = zeros((J,L))
    @views for l = 1:L
        for j = 1:J
            S1[j,l]+=BLAS.dot(test_img,test_img) #normalization choice arb to make order unity
        end
    end
end

structure_test1(8,8,rand(256,256))

@time structure_test1(8,8,rand(256,256))

@benchmark structure_test1(8,8,rand(256,256))

function structure_test2(J,L,test_img)
    S1 = zeros((J,L))
    @views for l = 1:L
        for j = 1:J
            S1[j,l]+=sum(abs2.(test_img))#normalization choice arb to make order unity
        end
    end
end

structure_test2(8,8,rand(256,256))

@time structure_test2(8,8,rand(256,256))

@benchmark structure_test2(8,8,rand(256,256))

## Yeah, ok, so that is much better. Can we take abs as we allocate?

function structure_test(J,L,test_img_1)
    S20 = zeros((J, L, J, L))
    Atmp = zeros(Float64,256,256)
    Btmp = zeros(Float64,256,256)
    @views for l2 = 1:L
        copyto!(Btmp, abs.(test_img_1))
        for j2 = 1:J
            for l1  = 1:L
                copyto!(Atmp, abs.(test_img_1))
                for j1  = 1:J
                    S20[j1,l1,j2,l2] = BLAS.dot(Atmp,Btmp)
                end
            end
        end
    end
end

structure_test(8,8,rand(256,256))

@time structure_test(8,8,rand(256,256))

@benchmark structure_test(8,8,rand(256,256))

function structure_test(J,L,test_img_1)
    S20 = zeros((J, L, J, L))
    Atmp = zeros(ComplexF64,256,256)
    Btmp = zeros(ComplexF64,256,256)
    @views for l2 = 1:L
        copyto!(Btmp, test_img_1)
        for j2 = 1:J
            for l1  = 1:L
                copyto!(Atmp, test_img_1)
                for j1  = 1:J
                    S20[j1,l1,j2,l2] = BLAS.asum(Atmp.*Btmp)
                end
            end
        end
    end
end

structure_test(8,8,rand(256,256))

@time structure_test(8,8,rand(256,256))

@benchmark structure_test(8,8,rand(256,256))

##
function DHC(image, filter_set; norm_on = 1, coeff_12_on =1, coeff_20_on = 1)
    FFTW.set_num_threads(2)
    (Nx, Ny) = size(image)

    (_,_,J,L) = size(fink_filter_set)

    out_coeff = []

    ## Coeff at (1,0)
    #takes care of image normalization
    S0 = zeros(2)
    if norm_on == 1
        S0[1] = mean(image)
        norm_im = image.-S0[1]
        S0[2] = BLAS.dot(norm_im,norm_im)/(Nx*Ny)
        norm_im = norm_im./sqrt(Nx*Ny*S0[2])
    else
        norm_im = image
    end

    append!(out_coeff,S0)

    ##Coeff at (1,1)
    #store the fft of images and im*filters for later
    im_fd_0 = fft(norm_im)

    S1 = zeros((J,L))
    im_fd_0_1 = Array{ComplexF64,4}(undef, Nx, Ny, J, L)
    @views for l = 1:L
        for j = 1:J
            im_fd_0_1[:,:,j,l] .= im_fd_0
        end
    end

    if coeff_20_on == 1
        im_rd_0_1 = Array{Float64,4}(undef, Nx, Ny, J, L)
    end

    Atmp = zeros(ComplexF64,Nx,Ny)
    Btmp = zeros(Float64,Nx,Ny)
    Ctmp = zeros(Float64,Nx,Ny)
    @views for l = 1:L
        for j = 1:J
            had!(im_fd_0_1[:,:,j,l],filter_set[:,:,j,l]) #wavelet already in fft domain not shifted
            Btmp .= abs.(im_fd_0_1[:,:,j,l])
            S1[j,l]+=BLAS.dot(Btmp,Btmp) #normalization choice arb to make order unity
            if coeff_20_on == 1
                Atmp .= ifft(im_fd_0_1[:,:,j,l])
                im_rd_0_1[:,:,j,l] .= abs.(Atmp)
            end
        end
    end
    append!(out_coeff,S1)

    ##Coeff at (2,0)
    if coeff_20_on == 1
        S20 = zeros(Float64,J, L, J, L)
        @views for l2 = 1:L
            for j2 = 1:J
                copyto!(Btmp, im_rd_0_1[:,:,j2,l2])
                for l1 = 1:L
                    for j1  = 1:J
                        copyto!(Ctmp, im_rd_0_1[:,:,j1,l1])
                        S20[j1,l1,j2,l2] += BLAS.dot(Btmp,Ctmp)
                    end
                end
            end
        end

        append!(out_coeff,S20)
    end

    ##Coeff at (1,2) level
    if coeff_12_on == 1
        S12 = zeros(Float64,J, L, J, L)
        for l2 = 1:L
            for j2 = 1:J
                not_rot_im = fftshift(im_fd_0_1[:,:,j1,l1])
                for l1 = 1:L
                    for j1 = 1:J
                        S12[j1,l1,j2,l2] += sum(abs.(not_rot_im.*fftshift(im_fd_0_1[:,:,j2,l2])))
                    end
                end
            end
        end
    append!(out_coeff,S12)
    end
    return out_coeff
end

temp = DHC(test_img,fink_filter_set,coeff_12_on =0, coeff_20_on = 1)

@time DHC(test_img,fink_filter_set,coeff_12_on =0, coeff_20_on = 1)

Profile.clear()
@profile DHC(test_img,fink_filter_set,coeff_12_on =0, coeff_20_on = 1)

Juno.profiler()

BenchmarkTools.DEFAULT_PARAMETERS.samples = 5
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 120
@benchmark DHC(test_img,fink_filter_set,coeff_12_on =0, coeff_20_on = 1)

## Starting to test the S12 coeff

function fftshift_wrap()
    A = rand(256,256)
    fftshift(A)
end

@time fftshift_wrap()

@benchmark fftshift_wrap()

function fftshift_wrap()
    A = rand(256,256)
end

@time fftshift_wrap()

@benchmark fftshift_wrap()
