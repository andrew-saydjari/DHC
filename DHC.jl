using Statistics
using FFTW

function DHC(image, filter_set; norm_on = 1, coeff_12_on =1, coeff_20_on = 1)
    (Nx, Ny) = size(image)

    (J,L,_,_) = size(fink_filter_set)

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
    im_fd_0_1 = Array{ComplexF64,4}(undef, J, L, Nx, Ny)

    for j = 1:J
        for l = 1:L
            im_fd_0_1[j,l,:,:] = im_fd_0.*filter_set[j,l,:,:] #wavelet already in fft domain not shifted
            S1[j,l]=sum(abs2.(im_fd_0_1[j,l,:,:])) #normalization choice arb to make order unity
        end
    end
    append!(out_coeff,S1)

    ##Coeff at (2,0)
    if coeff_20_on == 1
        S20 = Array{Float64,4}(undef, J, J, L, L)

        im_rd_0_1 = Array{ComplexF64,4}(undef, J, L, Nx, Ny)
        for j1 = 1:J
            for l1 = 1:L
                im_rd_0_1[j1,l1,:,:] = ifft(im_fd_0_1[j1,l1,:,:])
            end
        end

        for j1 = 1:J
            for l1 = 1:L
                for l2 = 1:L
                    for j2  = 1:J
                        S20[j1,j2,l1,l2] = sum(
                            abs.(
                                im_rd_0_1[j1,l1,:,:].*
                                conj.(im_rd_0_1[j2,l2,:,:])
                            )
                        )
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
end

using Plots

##
heatmap(fftshift(fink_filter))

##
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
    fink_filter = Array{Float64, 4}(undef, J, L, 256, 256)
    for j = 1:J-1
        for l = 1:L
            fink_filter[j,l,:,:]=fftshift(finklet(j-1,l-1))
        end
    end
    return fink_filter
end

fink_filter_set = fink_filter_bank(8,8)

size(fink_filter_set)

##
using BenchmarkTools

DHC(fink_filter,fink_filter_set)

##No FFT
@benchmark DHC(fink_filter,fink_filter_set)

##FFT
@benchmark DHC(fink_filter,fink_filter_set)

##FFT + Zeros
DHC(fink_filter,fink_filter_set)

@benchmark DHC(fink_filter,fink_filter_set)

##FFT S1 Simplest mat mult
DHC(fink_filter,fink_filter_set)

@benchmark DHC(fink_filter,fink_filter_set)


## Need to switch to 8,8

##FFT S1 S20
DHC(fink_filter_set[1,3,:,:],fink_filter_set)

20.8/(8*8*8*8+8*8+2)
8/1857

@benchmark DHC(fink_filter_set[1,3,:,:],fink_filter_set)

##FFT S1 S20 S21
DHC(fink_filter_set[1,3,:,:],fink_filter_set)

@benchmark DHC(fink_filter_set[1,3,:,:],fink_filter_set)

39.348/(2*8*8*8*8+8*8+2)

## Let me benchmark on S1 only first.

@benchmark DHC(fink_filter_set[1,3,:,:],fink_filter_set,coeff_12_on =0, coeff_20_on = 0)

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

function DHC(image, filter_set; norm_on = 1, coeff_12_on =1, coeff_20_on = 1)
    (Nx, Ny) = size(image)

    (J,L,_,_) = size(fink_filter_set)

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
    im_fd_0_1 = repeat(im_fd_0,inner=[1,1,J,L])

    for j = 1:J
        for l = 1:L
            had!(im_fd_0_1[:,:,j,l],filter_set[j,l,:,:]) #wavelet already in fft domain not shifted
            S1[j,l]=sum(abs2.(im_fd_0_1[:,:,j,l])) #normalization choice arb to make order unity
        end
    end
    append!(out_coeff,S1)

    ##Coeff at (2,0)
    if coeff_20_on == 1
        S20 = Array{Float64,4}(undef, J, J, L, L)

        im_rd_0_1 = Array{ComplexF64,4}(undef, J, L, Nx, Ny)
        for j1 = 1:J
            for l1 = 1:L
                im_rd_0_1[j1,l1,:,:] = ifft(im_fd_0_1[:,:,j1,l1])
            end
        end

        # for j1 = 1:J
        #     for l1 = 1:L
        #         for l2 = 1:L
        #             for j2  = 1:J
        #                 S20[j1,j2,l1,l2] = sum(
        #                     abs.(
        #                         im_rd_0_1[j1,l1,:,:].*
        #                         conj.(im_rd_0_1[j2,l2,:,:])
        #                     )
        #                 )
        #             end
        #         end
        #     end
        # end

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

temp = repeat(fink_filter_set[1,3,:,:],inner=[1,1,8,8])

@benchmark had!(fink_filter_set[1,3,:,:],fink_filter_set[1,3,:,:])

function wraper(A,B)
    A.*B
end

@benchmark wraper(fink_filter_set[1,3,:,:],fink_filter_set[1,3,:,:])

## Slightly better inplace S1

@benchmark DHC(fink_filter_set[1,3,:,:],fink_filter_set,coeff_12_on =0, coeff_20_on = 0)

## Start checking the IFFT step
out1 = DHC(fink_filter_set[1,3,:,:],fink_filter_set,coeff_12_on =0, coeff_20_on = 1)

@benchmark DHC(fink_filter_set[1,3,:,:],fink_filter_set,coeff_12_on =0, coeff_20_on = 1)

## Quick FFT

using LinearAlgebra

x = rand(Complex{Float64}, 256,256)
FFTW.set_num_threads(1)
p1 = plan_fft(x, flags=FFTW.MEASURE)
FFTW.set_num_threads(2)
p2 = plan_fft(x, flags=FFTW.MEASURE)
y = similar(x)

@time mul!(y, p1, x)

@time mul!(y, p2, x)

## FFT plan
function DHC(image, filter_set; norm_on = 1, coeff_12_on =1, coeff_20_on = 1)
    (Nx, Ny) = size(image)

    (J,L,_,_) = size(fink_filter_set)

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
    im_fd_0_1 = repeat(im_fd_0,inner=[1,1,J,L])

    for j = 1:J
        for l = 1:L
            had!(im_fd_0_1[:,:,j,l],filter_set[j,l,:,:]) #wavelet already in fft domain not shifted
            S1[j,l]=sum(abs2.(im_fd_0_1[:,:,j,l])) #normalization choice arb to make order unity
        end
    end
    append!(out_coeff,S1)

    ##Coeff at (2,0)
    if coeff_20_on == 1
        x = rand(Complex{Float64}, 256,256)
        FFTW.set_num_threads(2)
        p2 = plan_ifft(x, flags=FFTW.MEASURE)

        S20 = Array{Float64,4}(undef, J, J, L, L)

        im_rd_0_1 = Array{ComplexF64,4}(undef, J, L, Nx, Ny)
        for j1 = 1:J
            for l1 = 1:L
                im_rd_0_1[j1,l1,:,:] = mul!(im_rd_0_1[j1,l1,:,:], p2, im_fd_0_1[:,:,j1,l1])
            end
        end

        # for j1 = 1:J
        #     for l1 = 1:L
        #         for l2 = 1:L
        #             for j2  = 1:J
        #                 S20[j1,j2,l1,l2] = sum(
        #                     abs.(
        #                         im_rd_0_1[j1,l1,:,:].*
        #                         conj.(im_rd_0_1[j2,l2,:,:])
        #                     )
        #                 )
        #             end
        #         end
        #     end
        # end

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

## checking the IFFT step

out2 = DHC(fink_filter_set[1,3,:,:],fink_filter_set,coeff_12_on =0, coeff_20_on = 1)

@benchmark DHC(fink_filter_set[1,3,:,:],fink_filter_set,coeff_12_on =0, coeff_20_on = 1)

size(fink_filter_set)

A = out2-out1

minimum(out2)
maximum(out2)
minimum(out1)
maximum(out1)

## Ok, something is wrong with the fft preallocation. Seems like we are getting overflows.
# out 1 also seems to have some big numbers though... since I got a factor of 10 speed up using 2 cores..
# seems paralleizing the native version might be the way to go

function DHC(image, filter_set; norm_on = 1, coeff_12_on =1, coeff_20_on = 1)
    FFTW.set_num_threads(2)
    (Nx, Ny) = size(image)

    (J,L,_,_) = size(fink_filter_set)

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
    im_fd_0_1 = repeat(im_fd_0,inner=[1,1,J,L])
    if coeff_20_on == 1
        im_rd_0_1 = Array{ComplexF64,4}(undef, J, L, Nx, Ny)
    end

    @views for j = 1:J
        for l = 1:L
            had!(im_fd_0_1[:,:,j,l],filter_set[j,l,:,:]) #wavelet already in fft domain not shifted
            S1[j,l]+=sum(abs2.(im_fd_0_1[:,:,j,l])) #normalization choice arb to make order unity
            if coeff_20_on == 1
                im_rd_0_1[j1,l1,:,:] = ifft(im_fd_0_1[:,:,j1,l1])
            end
        end
    end
    append!(out_coeff,S1)

    ##Coeff at (2,0)
    if coeff_20_on == 1
        S20 = Array{Float64,4}(undef, J, J, L, L)

        # for j1 = 1:J
        #     for l1 = 1:L
        #         for l2 = 1:L
        #             for j2  = 1:J
        #                 S20[j1,j2,l1,l2] = sum(
        #                     abs.(
        #                         im_rd_0_1[j1,l1,:,:].*
        #                         conj.(im_rd_0_1[j2,l2,:,:])
        #                     )
        #                 )
        #             end
        #         end
        #     end
        # end

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

## 2 core native FFT

out = DHC(fink_filter_set[1,3,:,:],fink_filter_set,coeff_12_on =0, coeff_20_on = 1)

@benchmark DHC(fink_filter_set[1,3,:,:],fink_filter_set,coeff_12_on =0, coeff_20_on = 1)

function DHC(image, filter_set; norm_on = 1, coeff_12_on =1, coeff_20_on = 1)
    (Nx, Ny) = size(image)

    (J,L,_,_) = size(fink_filter_set)

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
    im_fd_0_1 = repeat(im_fd_0,inner=[1,1,J,L])

    for j = 1:J
        for l = 1:L
            had!(im_fd_0_1[:,:,j,l],filter_set[j,l,:,:]) #wavelet already in fft domain not shifted
            S1[j,l]=sum(abs2.(im_fd_0_1[:,:,j,l])) #normalization choice arb to make order unity
        end
    end
    append!(out_coeff,S1)

    ##Coeff at (2,0)
    if coeff_20_on == 1
        S20 = Array{Float64,4}(undef, J, J, L, L)

        FFTW.set_num_threads(1)
        im_rd_0_1 = Array{ComplexF64,4}(undef, J, L, Nx, Ny)
        for j1 = 1:J
            for l1 = 1:L
                im_rd_0_1[j1,l1,:,:] = ifft(im_fd_0_1[:,:,j1,l1])
            end
        end

        # for j1 = 1:J
        #     for l1 = 1:L
        #         for l2 = 1:L
        #             for j2  = 1:J
        #                 S20[j1,j2,l1,l2] = sum(
        #                     abs.(
        #                         im_rd_0_1[j1,l1,:,:].*
        #                         conj.(im_rd_0_1[j2,l2,:,:])
        #                     )
        #                 )
        #             end
        #         end
        #     end
        # end

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

out = DHC(fink_filter_set[1,3,:,:],fink_filter_set,coeff_12_on =0, coeff_20_on = 1)

@benchmark DHC(fink_filter_set[1,3,:,:],fink_filter_set,coeff_12_on =0, coeff_20_on = 1)


FFTW.set_num_threads(1)
@time ifft(fink_filter_set[1,3,:,:])
FFTW.set_num_threads(2)
@time ifft(fink_filter_set[1,3,:,:])

##
using Profile

DHC(fink_filter_set[1,3,:,:],fink_filter_set,coeff_12_on =0, coeff_20_on = 1)
@time DHC(fink_filter_set[1,3,:,:],fink_filter_set,coeff_12_on =0, coeff_20_on = 1)

Profile.clear()
@profile DHC(fink_filter_set[1,3,:,:],fink_filter_set,coeff_12_on =0, coeff_20_on = 1)

Juno.profiler()

##
using Profile

function DHC(image, filter_set; norm_on = 1, coeff_12_on =1, coeff_20_on = 1)
    FFTW.set_num_threads(2)
    (Nx, Ny) = size(image)

    (J,L,_,_) = size(fink_filter_set)

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
    im_fd_0_1 = repeat(im_fd_0,inner=[1,1,J,L])
    if coeff_20_on == 1
        im_rd_0_1 = Array{ComplexF64,4}(undef, J, L, Nx, Ny)
    end

    @views for j = 1:J
        for l = 1:L
            had!(im_fd_0_1[:,:,j,l],filter_set[j,l,:,:]) #wavelet already in fft domain not shifted
            S1[j,l]+=sum(abs2.(im_fd_0_1[:,:,j,l])) #normalization choice arb to make order unity
            if coeff_20_on == 1
                im_rd_0_1[j,l,:,:] = ifft(im_fd_0_1[:,:,j,l])
            end
        end
    end
    append!(out_coeff,S1)

    ##Coeff at (2,0)
    if coeff_20_on == 1
        S20 = Array{Float64,4}(undef, J, J, L, L)

        # for j1 = 1:J
        #     for l1 = 1:L
        #         for l2 = 1:L
        #             for j2  = 1:J
        #                 S20[j1,j2,l1,l2] = sum(
        #                     abs.(
        #                         im_rd_0_1[j1,l1,:,:].*
        #                         conj.(im_rd_0_1[j2,l2,:,:])
        #                     )
        #                 )
        #             end
        #         end
        #     end
        # end

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


DHC(fink_filter_set[1,3,:,:],fink_filter_set,coeff_12_on =0, coeff_20_on = 1)
@time DHC(fink_filter_set[1,3,:,:],fink_filter_set,coeff_12_on =0, coeff_20_on = 1)

Profile.clear()
@profile DHC(fink_filter_set[1,3,:,:],fink_filter_set,coeff_12_on =0, coeff_20_on = 1)

Juno.profiler()

@benchmark DHC(fink_filter_set[1,3,:,:],fink_filter_set,coeff_12_on =0, coeff_20_on = 1)


##
function DHC(image, filter_set; norm_on = 1, coeff_12_on =1, coeff_20_on = 1)
    FFTW.set_num_threads(2)
    (Nx, Ny) = size(image)

    (J,L,_,_) = size(fink_filter_set)

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
    im_fd_0_1 = repeat(im_fd_0,inner=[1,1,J,L])
    if coeff_20_on == 1
        im_rd_0_1 = Array{ComplexF64,4}(undef, J, L, Nx, Ny)
    end

    p2 = plan_ifft(im_fd_0_1[:,:,1,1], flags=FFTW.MEASURE)

    for j = 1:J
        for l = 1:L
            had!(im_fd_0_1[:,:,j,l],filter_set[j,l,:,:]) #wavelet already in fft domain not shifted
            S1[j,l]+=sum(abs2.(im_fd_0_1[:,:,j,l])) #normalization choice arb to make order unity
            if coeff_20_on == 1
                mul!(im_rd_0_1[j,l,:,:], p2, im_fd_0_1[:,:,j,l])
            end
        end
    end
    append!(out_coeff,S1)

    ##Coeff at (2,0)
    if coeff_20_on == 1
        S20 = Array{Float64,4}(undef, J, J, L, L)

        # for j1 = 1:J
        #     for l1 = 1:L
        #         for l2 = 1:L
        #             for j2  = 1:J
        #                 S20[j1,j2,l1,l2] = sum(
        #                     abs.(
        #                         im_rd_0_1[j1,l1,:,:].*
        #                         conj.(im_rd_0_1[j2,l2,:,:])
        #                     )
        #                 )
        #             end
        #         end
        #     end
        # end

        append!(out_coeff,S20)
    end

    ##Coeff at (1,2) level
    if coeff_12_on == 1
        S12 = Array{Float64,4}(undef, J, J, L, L)
        for j1 = 1:J
            for l1 = 1:L
                not_rot_im .= fftshift(im_fd_0_1[j1,l1,:,:])
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


DHC(fink_filter_set[1,3,:,:],fink_filter_set,coeff_12_on =0, coeff_20_on = 1)
@time DHC(fink_filter_set[1,3,:,:],fink_filter_set,coeff_12_on =0, coeff_20_on = 1)

Profile.clear()
@profile DHC(fink_filter_set[1,3,:,:],fink_filter_set,coeff_12_on =0, coeff_20_on = 1)

Juno.profiler()

@benchmark DHC(fink_filter_set[1,3,:,:],fink_filter_set,coeff_12_on =0, coeff_20_on = 1)

##

function DHC(image, filter_set; norm_on = 1, coeff_12_on =1, coeff_20_on = 1)
    FFTW.set_num_threads(2)
    (Nx, Ny) = size(image)

    (J,L,_,_) = size(fink_filter_set)

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
    im_fd_0_1 = Array{ComplexF64,4}(undef, J, L, Nx, Ny)
    @views for j = 1:J
        for l = 1:L
            im_fd_0_1[j,l,:,:] .= im_fd_0
        end
    end

    if coeff_20_on == 1
        im_rd_0_1 = Array{ComplexF64,4}(undef, J, L, Nx, Ny)
    end

    @views for j = 1:J
        for l = 1:L
            had!(im_fd_0_1[j,l,:,:],filter_set[j,l,:,:]) #wavelet already in fft domain not shifted
            S1[j,l]+=sum(abs2.(im_fd_0_1[j,l,:,:])) #normalization choice arb to make order unity
            if coeff_20_on == 1
                im_rd_0_1[j,l,:,:] .= ifft(im_fd_0_1[j,l,:,:])
            end
        end
    end
    append!(out_coeff,S1)

    ##Coeff at (2,0)
    if coeff_20_on == 1
        S20 = Array{Float64,4}(undef, J, J, L, L)

        # for j1 = 1:J
        #     for l1 = 1:L
        #         for l2 = 1:L
        #             for j2  = 1:J
        #                 S20[j1,j2,l1,l2] = sum(
        #                     abs.(
        #                         im_rd_0_1[j1,l1,:,:].*
        #                         conj.(im_rd_0_1[j2,l2,:,:])
        #                     )
        #                 )
        #             end
        #         end
        #     end
        # end

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

DHC(fink_filter_set[1,3,:,:],fink_filter_set,coeff_12_on =0, coeff_20_on = 1)
@time DHC(fink_filter_set[1,3,:,:],fink_filter_set,coeff_12_on =0, coeff_20_on = 1)

Profile.clear()
@profile DHC(fink_filter_set[1,3,:,:],fink_filter_set,coeff_12_on =0, coeff_20_on = 1)

Juno.profiler()

@benchmark DHC(fink_filter_set[1,3,:,:],fink_filter_set,coeff_12_on =0, coeff_20_on = 1)


## testing my own repeat
J=8
L=8

test_img = fink_filter_set[1,3,:,:]

test = Array{ComplexF64,4}(undef, J, L, 256, 256)
@views for j = 1:J
    for l = 1:L
        test[j,l,:,:] = test_img
    end
end
