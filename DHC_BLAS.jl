## Preloads
using Statistics
using FFTW
using Plots
using BenchmarkTools
using Profile
using LinearAlgebra
using StaticArrays
using HybridArrays

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
