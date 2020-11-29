
##
using Plots
plotlyjs()

theme(:juno)

##
display(1)

print("hello doug")

##
plot([1,2,1])

function f(x)
    return x
end

f(3)

3+3

## Simple testing suite

## Preloads
using Statistics
using FFTW
using Plots
using BenchmarkTools
using Profile
using LinearAlgebra

## BLAS.dot

test_img = rand(256,256)

function variance_BLAS(A)
    (Nx, Ny)  = size(A)
    norm_im_vec = reshape(A, Nx*Ny)
    BLAS.dot(norm_im_vec,norm_im_vec)/(Nx*Ny)
end

function variance(A)
    (Nx, Ny)  = size(A)
    sum(A .* A)/(Nx*Ny)
end

function variance_BLAS_sq(A)
    (Nx, Ny)  = size(A)
    BLAS.dot(A,A)/(Nx*Ny)
end

@time variance(test_img)

@benchmark variance(test_img)

@time variance_BLAS(test_img)

@benchmark variance_BLAS(test_img)

#BLAS dot 30x faster; Was the unflattened version really wrong

@time variance_BLAS_sq(test_img)

@benchmark variance_BLAS_sq(test_img)

function variance_lc(A)
    (Nx, Ny)  = size(A)
    sum([x^2 for x in A])/(Nx*Ny)
end

@time variance_lc(test_img)

@benchmark variance_lc(test_img)

function variance_lc_abs2(A)
    (Nx, Ny)  = size(A)
    sum([abs2(x) for x in A])/(Nx*Ny)
end

@time variance_lc_abs2(test_img)

@benchmark variance_lc_abs2(test_img)

## Test the * versus mul!

function Doug_mul(A)
    B_test = zeros(256,256)
    B_test     .= A' * A
end

@time Doug_mul(test_img)

@benchmark Doug_mul(test_img)

function noraml_mul(A)
    B_test = zeros(256,256)
    mul!(B_test,A',A)
end

@time noraml_mul(test_img)

@benchmark noraml_mul(test_img)

BLAS.syrk!(test_img,B_test)
