## Derivatives
function wst_S1_deriv(image::Array{Float64,2}, filter_hash::Dict, FFTthreads::Int=1)

    # Use nthread threads for FFT -- but for Nx<512 nthread=1 is fastest.  Overhead?
    FFTW.set_num_threads(FFTthreads)

    # array sizes
    (Nx, Ny)  = size(image)
    (Nf, )    = size(filter_hash["filt_index"])

    # allocate output array
    dS1dα  = zeros(Float64, Nx, Nx, Nf)

    ## 1st Order
    im_fd = fft(image)

    # unpack filter_hash
    f_ind   = filter_hash["filt_index"]  # (J, L) array of filters represented as index value pairs
    f_val   = filter_hash["filt_value"]
    zarr = zeros(ComplexF64, Nx, Nx)  # temporary array to fill with zvals

    # make a FFTW "plan" for an array of the given size and type
    P = plan_ifft(im_fd)  # P is an operator, P*im is ifft(im)

    # Loop over filters
    for f = 1:Nf
        f_i = f_ind[f]  # CartesianIndex list for filter
        f_v = f_val[f]  # Values for f_i

        zarr[f_i] = (f_v.*f_v) .* im_fd[f_i]
        dS1dα[:,:,f] = 2 .* real.(P*zarr)

        zarr[f_i] .= 0   # reset zarr for next loop
    end
    return dS1dα
end

function wst_S20_deriv(image::Array{Float64,2}, filter_hash::Dict, FFTthreads::Int=1)

    # Use nthread threads for FFT -- but for Nx<512 nthread=1 is fastest.  Overhead?
    FFTW.set_num_threads(FFTthreads)

    # array sizes
    (Nx, Ny)  = size(image)
    (Nf, )    = size(filter_hash["filt_index"])

    # allocate output array
    dS20dα  = zeros(Float64, Nx, Nx, Nf, Nf)

    # allocate image arrays for internal use
    im_rdc = Array{ComplexF64, 3}(undef, Nx, Ny, Nf)  # real domain, complex
    im_rd  = Array{Float64, 3}(undef, Nx, Ny, Nf)  # real domain, complex
    im_fd  = fft(image)

    # unpack filter_hash
    f_ind   = filter_hash["filt_index"]  # (J, L) array of filters represented as index value pairs
    f_val   = filter_hash["filt_value"]
    zarr = zeros(ComplexF64, Nx, Nx)  # temporary array to fill with zvals

    # make a FFTW "plan" a complex array, both forward and inverse transform
    P_fft  = plan_fft(im_fd)   # P_fft is an operator,  P_fft*im is fft(im)
    P_ifft = plan_ifft(im_fd)  # P_ifft is an operator, P_ifft*im is ifft(im)

    # Loop over filters
    for f = 1:Nf
        f_i = f_ind[f]  # CartesianIndex list for filter
        f_v = f_val[f]  # Values for f_i

        zarr[f_i] = f_v .* im_fd[f_i]
        Z_λ = P_ifft*zarr  # complex valued ifft of zarr
        zarr[f_i] .= 0   # reset zarr for next loop
        im_rdc[:,:,f] = Z_λ
        im_rd[:,:,f]  = abs.(Z_λ)
    end

    zarr = zeros(ComplexF64, Nx, Nx)  # temporary array to fill with zvals
    for f2 = 1:Nf
        f_i = f_ind[f2]  # CartesianIndex list for filter
        f_v = f_val[f2]  # Values for f_i
        uvec = im_rdc[:,:,f2] ./ im_rd[:,:,f2]
        for f1 = 1:Nf
            temp = P_fft*(im_rd[:,:,f1].*uvec)
            zarr[f_i] = f_v .* temp[f_i]

            Z1dZ2 = real.(P_ifft*zarr)
            #  It is possible to do this with rifft, but it is not much faster...
            #   Z1dZ2 = myrealifft(zarr)
            dS20dα[:,:,f1,f2] += Z1dZ2
            dS20dα[:,:,f2,f1] += Z1dZ2
            zarr[f_i] .= 0   # reset zarr for next loop
        end
    end
    return dS20dα
end

function wst_S20_deriv_sum(image::Array{Float64,2}, filter_hash::Dict, wt::Array{Float64}, FFTthreads::Int=1)
    # Sum over (f1,f2) filter pairs for S20 derivative.  This is much faster
    #   than calling wst_S20_deriv() because the sum can be moved inside the FFT.
    # Use FFTthreads threads for FFT -- but for Nx<512 FFTthreads=1 is fastest.  Overhead?
    # On Cascade Lake box, 4 is good for 2D, 8 or 16 for 3D FFTs
    FFTW.set_num_threads(FFTthreads)

    # array sizes
    (Nx, Ny)  = size(image)
    (Nf, )    = size(filter_hash["filt_index"])

    # allocate image arrays for internal use
    Uvec   = Array{ComplexF64, 3}(undef, Nx, Ny, Nf)  # real domain, complex
    im_rd  = Array{Float64, 3}(undef, Nx, Ny, Nf)  # real domain, complex
    im_fd  = fft(image)

    # unpack filter_hash
    f_ind   = filter_hash["filt_index"]  # (J, L) array of filters represented as index value pairs
    f_val   = filter_hash["filt_value"]
    zarr    = zeros(ComplexF64, Nx, Nx)  # temporary array to fill with zvals

    # make a FFTW "plan" a complex array, both forward and inverse transform
    P_fft  = plan_fft(im_fd)   # P_fft is an operator,  P_fft*im is fft(im)
    P_ifft = plan_ifft(im_fd)  # P_ifft is an operator, P_ifft*im is ifft(im)

    # Loop over filters
    for f = 1:Nf
        f_i = f_ind[f]  # CartesianIndex list for filter
        f_v = f_val[f]  # Values for f_i

        zarr[f_i] = f_v .* im_fd[f_i]
        Z_λ = P_ifft*zarr  # complex valued ifft of zarr
        zarr[f_i] .= 0     # reset zarr for next loop
        im_rd[:,:,f] = abs.(Z_λ)
        Uvec[:,:,f]  = Z_λ ./ im_rd[:,:,f]
    end

    zarr = zeros(ComplexF64, Nx, Nx)  # temporary array to fill with zvals
    for f2 = 1:Nf
        f_i = f_ind[f2]  # CartesianIndex list for filter
        f_v = f_val[f2]  # Values for f_i

        Wtot = reshape( reshape(im_rd,Nx*Nx,Nf)*wt[:,f2], Nx, Nx)
        temp = P_fft*(Wtot.*Uvec[:,:,f2])
        zarr[f_i] .+= f_v .* temp[f_i]
    end
    ΣdS20dα = real.(P_ifft*zarr)

    return ΣdS20dα
end
