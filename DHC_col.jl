## Preloads
using Statistics
using FFTW
using Plots
using BenchmarkTools
using Profile
using LinearAlgebra
using StaticArrays

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

test_img = fink_filter_set[:,:,1,3]

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
        S20 = zeros((L, J, J, L))

        Atmp = zeros(ComplexF64,Nx,Ny)
        Btmp = zeros(ComplexF64,Nx,Ny)
        interm = 0
        @views for j1 = 1:J
            for j2 = 1:J
                for l1 = 1:L
                    for l2  = 1:L
                        copyto!(Atmp, im_rd_0_1[:,:,j1,l1])
                        copyto!(Btmp, im_rd_0_1[:,:,j2,l2])
                        S20[l1,j1,j2,l2] = sum(abs.(Atmp.*Btmp))
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
                        S12[j1,j2,l1,l2] = sum(
                            abs.(
                                not_rot_im.*
                                conj(fftshift(im_fd_0_1[j2,l2,:,:]))
                            )
                        )
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

test_img_s = SMatrix{256,256}(test_img)
