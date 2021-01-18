## Preloads
using Statistics
using FFTW
using Plots
using BenchmarkTools
using Profile
using LinearAlgebra

# %% Code Cell
function fink_filter_bank_fast(J, L)

    # -------- set parameters
    dθ   = π/8        # 8 angular bins hardwired
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
                θ_good = abs(θ_pix-π) <= dθ

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
        F_angular = cos.((θ[angmask].-π).*4)

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

fink_filter_set = fink_filter_bank_fast(8,8)
test_img = ones(256,256)
copyto!(test_img,fink_filter_set[:,:,1,3])

# %% Code Cell

function speedy_DHC(image::Array{Float64,2}, filter_set::Array{Float64,4})
    FFTW.set_num_threads(2)

    # array sizes
    (Nx, Ny)  = size(image)
    (_,_,J,L) = size(fink_filter_set)

    out_coeff = []
    # allocate coeff arrays
    S0  = zeros(2)
    S1  = zeros(J, L)
    S20 = zeros(Float64, J*L, J*L)
    S12 = zeros(Float64, J*L, J*L)
    im_rd_0_1  = zeros(Float64, Nx, Ny, J, L)
    im_fd_0_1 = zeros(Float64, Nx, Ny, J, L)

    ## 0th Order
    S0[1]   = mean(image)
    norm_im = image.-S0[1]
    S0[2]   = BLAS.dot(norm_im,norm_im)/(Nx*Ny)
    norm_im ./= sqrt(Nx*Ny*S0[2])

    append!(out_coeff,S0[:])

    ## 1st Order
    im_fd_0 = fft(norm_im)

    Atmp = zeros(ComplexF64,Nx,Ny)
    Btmp = zeros(Float64,Nx,Ny)
    ## Main 1st Order and Precompute 2nd Order
    @views for l = 1:L
        for j = 1:J
            @inbounds copyto!(Atmp, im_fd_0 .* filter_set[:,:,j,l])
            Btmp .= abs.(Atmp)
            @inbounds im_fd_0_1[:,:,j,l] .= Btmp
            @inbounds S1[j,l] +=BLAS.dot(Btmp,Btmp)
            ifft!(Atmp)
            @inbounds im_rd_0_1[:,:,j,l] .= abs.(Atmp)
        end
    end
    append!(out_coeff,S1[:])

    #DPF version of 2nd Order
    Amat    = reshape(im_fd_0_1, Nx*Ny, J*L)
    mul!(S12,Amat',Amat)

    Amat    = reshape(im_rd_0_1, Nx*Ny, J*L)
    mul!(S20,Amat',Amat)

    append!(out_coeff,S20[:])
    append!(out_coeff,S12[:])

    return out_coeff
end

temp = speedy_DHC(test_img,fink_filter_set)

@time speedy_DHC(test_img,fink_filter_set)

Profile.clear()
@profile speedy_DHC(test_img,fink_filter_set)

Juno.profiler()

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 30
@benchmark speedy_DHC(test_img,fink_filter_set)

function speedy_DHC_old(image::Array{Float64,2}, filter_set::Array{Float64,4})
    FFTW.set_num_threads(2)

    # array sizes
    (Nx, Ny)  = size(image)
    (_,_,J,L) = size(fink_filter_set)

    out_coeff = []
    # allocate coeff arrays
    S0  = zeros(2)
    S1  = zeros(J, L)
    S20 = zeros(Float64, J, L, J, L)
    S12 = zeros(Float64, J, L, J, L)
    im_rd_0_1  = zeros(Float64,    Nx, Ny, J, L)
    im_fdf_0_1 = zeros(Float64,    Nx, Ny, J, L)
    #im_fd_0_1  = zeros(ComplexF64, Nx, Ny, J, L)

    ## 0th Order
    S0[1]   = mean(image)
    norm_im = image.-S0[1]
    S0[2]   = sum(norm_im .* norm_im)/(Nx*Ny)
    norm_im ./= sqrt(Nx*Ny*S0[2])

    # Was this intentional to set this back to image?
    norm_im = image

    append!(out_coeff,S0[:])

    ## 1st Order
    im_fd_0 = fft(norm_im)

    foo = zeros(ComplexF64,Nx,Ny)
    Btmp = zeros(Float64,Nx,Ny)
    ## Main 1st Order and Precompute 2nd Order
    for l = 1:L
        for j = 1:J
            #@inbounds had!(im_fd_0_1[:,:,j,l],filter_set[:,:,j,l]) #wavelet already in fft domain not shifted
            # @inbounds Btmp .= abs.(im_fd_0_1[:,:,j,l])
            # We don't need an fftshift()
            # @inbounds im_fdf_0_1[:,:,j,l] .= fftshift(Btmp)
            copyto!(foo, im_fd_0 .* filter_set[:,:,j,l])
            Btmp .= abs.(foo)
            @inbounds im_fdf_0_1[:,:,j,l] .= Btmp
            # I think you are doing a 256x256 matrix multiplication here
            #@inbounds S1[j,l]+=BLAS.dot(Btmp,Btmp) #normalization choice arb to make order unity
            # You mean this:  (and BLAS won't speed this up much)
            # S1[j,l] = sum(Btmp.*Btmp)
            S1[j,l] = sum([x^2 for x in Btmp])  #slightly faster? (no malloc)
            #@inbounds Atmp .= ifft(foo)
            @inbounds im_rd_0_1[:,:,j,l] .= abs.(ifft(foo))

        end
    end
    append!(out_coeff,S1[:])


    #DPF version of 2nd Order
    Amat    = reshape(im_fdf_0_1, Nx*Ny, J*L)
    S12 = reshape(Amat' * Amat, J, L, J, L)

    Amat    = reshape(im_rd_0_1, Nx*Ny, J*L)
    S20 = reshape(Amat' * Amat, J, L, J, L)

    append!(out_coeff,S20)
    append!(out_coeff,S12)


    return out_coeff
end

temp = speedy_DHC_old(test_img,fink_filter_set)

@time speedy_DHC_old(test_img,fink_filter_set)

Profile.clear()
@profile speedy_DHC_old(test_img,fink_filter_set)

Juno.profiler()

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 30
@benchmark speedy_DHC_old(test_img,fink_filter_set)
