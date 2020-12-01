## Preloads
using Statistics
using FFTW
using Plots
using BenchmarkTools
using Profile
using LinearAlgebra
using StaticArrays
using HybridArrays

# put the cwd on the path so Julia can find the module
push!(LOAD_PATH, pwd())
using DHC_Utils

##

# create filter bank
@time fink_filter_set  = fink_filter_bank(8,8)

# condense this to a filter list of (index, value) pairs.
@time filter_list = fink_filter_list(fink_filter_set)

# Generate phi from an L=16 filter bank
donut = fink_filter_bank(8,16)
phi   = sqrt.(reshape(sum(donut[:,:,8,:].^2,dims=[3,4]),256,256))
phi_r = real(ifft(phi))
# check that the real-space function sums to 1 as expected.
print(sum(phi_r))

# really we should never use j=7 finklets, only use phi.

test_img = rand(256,256)

temp2 = speedy_DHC(test_img, filter_list)


@time speedy_DHC(test_img, filter_list)
@benchmark speedy_DHC(test_img, filter_list)
# mean time:        194.513 ms (8.08% GC)

for l=1:8 filter_list[1][1,l] = [] end  # j=0 is useless and expensive!
for l=1:8 filter_list[2][1,l] = [] end
@benchmark speedy_DHC(test_img, filter_list)
# mean time:        156.590 ms (8.67% GC)

for l=1:8 filter_list[1][8,l] = [] end  # and so is j=7
for l=1:8 filter_list[2][8,l] = [] end
@benchmark speedy_DHC(test_img, filter_list)
# mean time:        148.648 ms (8.75% GC)


Profile.clear()
@profile speedy_DHC(test_img, filter_list)
Juno.profiler()






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

# Todo list
# Check if 2 threads really help FFT when computer is busy


#test_img = rand(256,256)
#temp1 = DHC(test_img,fink_filter_set)
temp2 = speedy_DHC(test_img, filter_list)


@time speedy_DHC(test_img, filter_list);

Profile.clear()
@profile speedy_DHC(test_img, filter_list)

Juno.profiler()
