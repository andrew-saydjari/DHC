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
            @inbounds fink_filter[:,:,j,l]=fftshift(finklet(j-1,l-1))
        end
    end
    return fink_filter
end


# Faster filter bank generation
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

fink_filter_set  = fink_filter_bank(8,8)
fink_filter_set2 = fink_filter_bank_fast(8,8)
print(maximum(abs.(fink_filter_set - fink_filter_set2)))

test_img = zeros(256,256)
copyto!(test_img,fink_filter_set[:,:,1,3])

function DHC(image::Array{Float64,2}, filter_set::Array{Float64,4}; norm_on = 1, coeff_12_on =1, coeff_20_on = 1)
    FFTW.set_num_threads(2)

    #Sizing
    (Nx, Ny) = size(image)
    (_,_,J,L) = size(fink_filter_set)

    out_coeff = []

    #Preallocate Coeff Arrays
    S0 = zeros(2)
    S1 = zeros(J,L)
    if coeff_20_on == 1
        S20 = zeros(Float64,J, L, J, L)
    end
    if coeff_12_on == 1
        S12 = zeros(Float64,J, L, J, L)
    end

    if coeff_20_on == 1
        im_rd_0_1 = zeros(Float64,Nx, Ny, J, L)
    end
    if coeff_12_on == 1
        im_fdf_0_1 = zeros(Float64,Nx, Ny, J, L)
    end

    im_fd_0_1 = zeros(ComplexF64,Nx, Ny, J, L)

    Atmp = zeros(ComplexF64,Nx,Ny)
    Btmp = zeros(Float64,Nx,Ny)
    Ctmp = zeros(Float64,Nx,Ny)
    Dtmp = zeros(Float64,Nx,Ny)
    Etmp = zeros(Float64,Nx,Ny)

    ## 0th Order
    if norm_on == 1
        @inbounds S0[1] = mean(image)
        @inbounds norm_im = image.-S0[1]
        @inbounds S0[2] = BLAS.dot(norm_im,norm_im)/(Nx*Ny)
        @inbounds norm_im = norm_im./sqrt(Nx*Ny*S0[2])
    else
        norm_im = image
    end

    append!(out_coeff,S0)

    ## 1st Order
    im_fd_0 = fft(norm_im)

    @views for l = 1:L
        for j = 1:J
            @inbounds im_fd_0_1[:,:,j,l] .= im_fd_0
        end
    end

    #Precompute loop for 2nd Order
    @views for l = 1:L
        for j = 1:J
            @inbounds had!(im_fd_0_1[:,:,j,l],filter_set[:,:,j,l]) #wavelet already in fft domain not shifted
            @inbounds Btmp .= abs.(im_fd_0_1[:,:,j,l])
            @inbounds S1[j,l]+=BLAS.dot(Btmp,Btmp) #normalization choice arb to make order unity
            if coeff_20_on == 1
                @inbounds Atmp .= ifft(im_fd_0_1[:,:,j,l])
                @inbounds im_rd_0_1[:,:,j,l] .= abs.(Atmp)
            end
            if coeff_12_on == 1
                @inbounds im_fdf_0_1[:,:,j,l] .= abs.(im_fd_0_1[:,:,j,l])
                @inbounds im_fdf_0_1[:,:,j,l] .= fftshift(im_fdf_0_1[:,:,j,l])
            end
        end
    end
    append!(out_coeff,S1)

    ## 2nd Order
    @views for l2 = 1:L
        for j2 = 1:J
            if coeff_20_on == 1
                @inbounds copyto!(Btmp, im_rd_0_1[:,:,j2,l2])
            end
            if coeff_12_on == 1
                @inbounds copyto!(Dtmp, im_fdf_0_1[:,:,j2,l2])
            end
            for l1 = 1:L
                for j1  = 1:J
                    if coeff_20_on == 1
                        @inbounds copyto!(Ctmp, im_rd_0_1[:,:,j1,l1])
                        @inbounds S20[j1,l1,j2,l2] += BLAS.dot(Btmp,Ctmp)
                    end
                    if coeff_12_on == 1
                        @inbounds copyto!(Etmp, im_fdf_0_1[:,:,j1,l1])
                        @inbounds S12[j1,l1,j2,l2] += BLAS.dot(Dtmp,Etmp)
                    end
                end
            end
        end
    end
    if coeff_20_on == 1
        append!(out_coeff,S20)
    end
    if coeff_12_on == 1
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


temp = DHC(test_img,fink_filter_set,coeff_12_on =1, coeff_20_on = 1)

@time DHC(test_img,fink_filter_set,coeff_12_on =1, coeff_20_on = 1)

Profile.clear()
@profile DHC(test_img,fink_filter_set,coeff_12_on =1, coeff_20_on = 1)

Juno.profiler()

BenchmarkTools.DEFAULT_PARAMETERS.samples = 5
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 120
@benchmark DHC(test_img,fink_filter_set,coeff_12_on =1, coeff_20_on = 1)

## No if ands or buts about it...

function DHC(image::Array{Float64,2}, filter_set::Array{Float64,4})
    FFTW.set_num_threads(2)

    #Sizing
    (Nx, Ny)  = size(image)
    (_,_,J,L) = size(fink_filter_set)

    out_coeff = []
    #Preallocate Coeff Arrays
    S0  = zeros(2)
    S1  = zeros(J,L)
    S20 = zeros(Float64,J, L, J, L)
    S12 = zeros(Float64,J, L, J, L)
    im_rd_0_1  = zeros(Float64,Nx, Ny, J, L)
    im_fdf_0_1 = zeros(Float64,Nx, Ny, J, L)
    im_fd_0_1  = zeros(ComplexF64,Nx, Ny, J, L)

    Atmp = zeros(ComplexF64,Nx,Ny)
    Btmp = zeros(Float64,Nx,Ny)
    Ctmp = zeros(Float64,Nx,Ny)
    Dtmp = zeros(Float64,Nx,Ny)
    Etmp = zeros(Float64,Nx,Ny)

    ## 0th Order
    @inbounds S0[1] = mean(image)
    @inbounds norm_im = image.-S0[1]
    # make this a vector before computing variance with BLAS.dot
    # (This was a bug!)
    norm_im_vec = reshape(norm_im, Nx*Ny)
    @inbounds S0[2] = BLAS.dot(norm_im_vec,norm_im_vec)/(Nx*Ny)
    @inbounds norm_im = norm_im./sqrt(Nx*Ny*S0[2])

    # Was this intentional to set this back to image?
    norm_im = image

    append!(out_coeff,S0[:])

    ## 1st Order
    im_fd_0 = fft(norm_im)

    @views for l = 1:L
        for j = 1:J
            @inbounds im_fd_0_1[:,:,j,l] .= im_fd_0
        end
    end

    ## Main 1st Order and Precompute 2nd Order
    @views for l = 1:L
        for j = 1:J
            @inbounds had!(im_fd_0_1[:,:,j,l],filter_set[:,:,j,l]) #wavelet already in fft domain not shifted
            @inbounds Btmp .= abs.(im_fd_0_1[:,:,j,l])
            @inbounds im_fdf_0_1[:,:,j,l] .= fftshift(Btmp)
            # I think you are doing a 256x256 matrix multiplication here
            #@inbounds S1[j,l]+=BLAS.dot(Btmp,Btmp) #normalization choice arb to make order unity
            # You mean this:  (and BLAS won't speed this up much)
            S1[j,l] = sum(Btmp.*Btmp)
            @inbounds Atmp .= ifft(im_fd_0_1[:,:,j,l])
            @inbounds im_rd_0_1[:,:,j,l] .= abs.(Atmp)

        end
    end
    append!(out_coeff,S1[:])

    ## 2nd Order
    @views for l2 = 1:L
        for j2 = 1:J
                @inbounds copyto!(Btmp, im_rd_0_1[:,:,j2,l2])
                @inbounds copyto!(Dtmp, im_fdf_0_1[:,:,j2,l2])
            for l1 = 1:L
                for j1  = 1:J
                        @inbounds copyto!(Ctmp, im_rd_0_1[:,:,j1,l1])
                        @inbounds copyto!(Etmp, im_fdf_0_1[:,:,j1,l1])
                        @inbounds S20[j1,l1,j2,l2] += BLAS.dot(Btmp,Ctmp)
                        @inbounds S12[j1,l1,j2,l2] += BLAS.dot(Dtmp,Etmp)
                    end
                end
            end
        end
    append!(out_coeff,S20)
    append!(out_coeff,S12)

    #DPF version of 2nd Order
    Amat    = reshape(im_fdf_0_1, Nx*Ny, J*L)
    FastS12 = reshape(Amat' * Amat, J, L, J, L)
    diffS12 = S12-FastS12
    println("minmax(diffS12)", minimum(diffS12), "   ", maximum(diffS12))

    Amat    = reshape(im_rd_0_1, Nx*Ny, J*L)
    FastS20 = reshape(Amat' * Amat, J, L, J, L)
    diffS20 = S20-FastS20
    println("minmax(diffS20)", minimum(diffS20), "   ", maximum(diffS20))


    return out_coeff
end

temp = DHC(test_img,fink_filter_set)

@time DHC(test_img,fink_filter_set)

Profile.clear()
@profile DHC(test_img,fink_filter_set)

Juno.profiler()

BenchmarkTools.DEFAULT_PARAMETERS.samples = 10000
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 120
@benchmark DHC(test_img,fink_filter_set,coeff_12_on =1, coeff_20_on = 1)

1.682/8258
8/1857

##Seems the order in memory changes performance a bit...
#Let me just check for a faster abs.() by hand and then call this good
#for a bit

function man_abs_dot!(A,B)
    m,n = size(A)
    @assert (m,n) == size(B)
    for j in 1:n
       for i in 1:m
         @inbounds A[i,j] = abs(B[i,j])
       end
    end
    return A
end


function wrapper(A,B)
    A .= abs.(B)
end

@time man_abs_dot!(zeros(256,256),test_img)

@time wrapper(zeros(256,256),test_img)

compare_img = zeros(256,256)

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 10
@benchmark wrapper($compare_img,test_img)

@benchmark man_abs_dot!($compare_img,test_img)

## Nope, not all that much faster... ok. Good night.
function DHC(image::Array{Float64,2}, filter_set::Array{Float64,4})
    FFTW.set_num_threads(2)
    #Sizing
    (Nx, Ny) = size(image)
    (_,_,J,L) = size(fink_filter_set)
    out_coeff = []
    #Preallocate Coeff Arrays
    S0 = zeros(2)
    S1 = zeros(J,L)
    S20 = zeros(Float64,J, L, J, L)
    S12 = zeros(Float64,J, L, J, L)
    im_rd_0_1 = zeros(Float64,Nx, Ny, J, L)
    im_fdf_0_1 = zeros(Float64,Nx, Ny, J, L)
    im_fd_0_1 = zeros(ComplexF64,Nx, Ny, J, L)
    Atmp = zeros(ComplexF64,Nx,Ny)
    Btmp = zeros(Float64,Nx,Ny)
    Ctmp = zeros(Float64,Nx,Ny)
    Dtmp = zeros(Float64,Nx,Ny)
    Etmp = zeros(Float64,Nx,Ny)
    ## 0th Order
    @inbounds S0[1] = mean(image)
    @inbounds norm_im = image.-S0[1]
    @inbounds S0[2] = BLAS.dot(norm_im,norm_im)/(Nx*Ny)
    @inbounds norm_im = norm_im./sqrt(Nx*Ny*S0[2])
    norm_im = image
    append!(out_coeff,S0[:])
    ## 1st Order
    im_fd_0 = fft(norm_im)
    @views for l = 1:L
        for j = 1:J
            @inbounds im_fd_0_1[:,:,j,l] .= im_fd_0
        end
    end
    ## Main 1st Order and Precompute 2nd Order
    @views for l = 1:L
        for j = 1:J
            @inbounds had!(im_fd_0_1[:,:,j,l],filter_set[:,:,j,l]) #wavelet already in fft domain not shifted
            @inbounds Btmp .= abs.(im_fd_0_1[:,:,j,l])
            @inbounds im_fdf_0_1[:,:,j,l] .= fftshift(Btmp)
            @inbounds S1[j,l]+=BLAS.dot(Btmp,Btmp) #normalization choice arb to make order unity
            @inbounds Atmp .= ifft(im_fd_0_1[:,:,j,l])
            @inbounds im_rd_0_1[:,:,j,l] .= abs.(Atmp)
        end
    end
    append!(out_coeff,S1[:])
    ## 2nd Order
    @views for l2 = 1:L
        for j2 = 1:J
                @inbounds copyto!(Btmp, im_rd_0_1[:,:,j2,l2])
                @inbounds copyto!(Dtmp, im_fdf_0_1[:,:,j2,l2])
            for l1 = 1:L
                for j1  = 1:J
                        @inbounds copyto!(Ctmp, im_rd_0_1[:,:,j1,l1])
                        @inbounds copyto!(Etmp, im_fdf_0_1[:,:,j1,l1])
                        @inbounds S20[j1,l1,j2,l2] += BLAS.dot(Btmp,Ctmp)
                        @inbounds S12[j1,l1,j2,l2] += BLAS.dot(Dtmp,Etmp)
                    end
                end
            end
        end
    append!(out_coeff,S20)
    append!(out_coeff,S12)
    return out_coeff
end

## No if ands or buts about it...

function speedy_DHC(image::Array{Float64,2}, filter_set::Array{Float64,4})
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


temp1 = DHC(test_img,fink_filter_set)
temp2 = speedy_DHC(test_img,fink_filter_set)


@time speedy_DHC(test_img,fink_filter_set)

Profile.clear()
@profile speedy_DHC(test_img,fink_filter_set)

Juno.profiler()
