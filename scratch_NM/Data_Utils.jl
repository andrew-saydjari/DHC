module Data_Utils
    using Statistics
    using Plots
    using FFTW
    using Statistics
    using Optim
    using Images, FileIO, ImageIO
    using Distributions
    using Printf
    using SparseArrays
    using LinearAlgebra
    # put the cwd on the path so Julia can find the module
    push!(LOAD_PATH, pwd()*"/main")
    using DHC_2DUtils

    export readdust
    export S2_uniweights
    export S2_whitenoiseweights

    function readdust(Nx)
        RGBA_img = load(pwd()*"/scratch_DF/t115_clean_small.png")
        img = reinterpret(UInt8,rawview(channelview(RGBA_img)))
        return imresize(Float64.(img[1,:,:])[1:256, 1:256], (Nx, Nx))
    end


    function S2_uniweights(im, fhash; high=25, Nsam=10, iso=false, norm=true, smooth=false, smoothval=0.8)
        #=
        Noise model: uniform[-high, high]
        =#

        (Nx, Ny)  = size(im)
        if Nx != Ny error("Input image must be square") end
        (N1iso, Nf)    = size(fhash["S1_iso_mat"])

        # fhash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1)

        S2   = DHC_compute(im, fhash, doS2=true, doS20=false, norm=norm, iso=iso)
        Ns    = length(S2)
        S2arr = zeros(Float64, Ns, Nsam)
        println("Ns", Ns)
        for j=1:Nsam
            noise = rand(Nx,Nx).*(2*high) .- high
            init = im+noise
            if smooth init = imfilter(init, Kernel.gaussian(smoothval)) end
            S2arr[:,j] = DHC_2DUtils.DHC_compute(init, fhash, doS2=true, doS20=false, norm=norm, iso=iso)
        end
        wt = zeros(Float64, Ns)
        for i=1:Ns
            wt[i] = std(S2arr[i,:])
        end
        msub = S2arr .- mean(S2arr, dims=2)
        cov = (msub * msub')./(Nsam-1)

        return wt, cov
    end


    function S2_whitenoiseweights(im, fhash; loc=0.0, sig=1.0, Nsam=10, iso=false, norm=true, smooth=false, smoothval=0.8)
        #=
        Noise model: uniform[-high, high]
        =#

        (Nx, Ny)  = size(im)
        if Nx != Ny error("Input image must be square") end
        (N1iso, Nf)    = size(fhash["S1_iso_mat"])

        # fhash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1)

        S2   = DHC_compute(im, fhash, doS2=true, doS20=false, norm=norm, iso=iso)
        Ns    = length(S2)
        S2arr = zeros(Float64, Ns, Nsam)
        println("Ns", Ns)
        for j=1:Nsam
            noise = reshape(rand(Normal(loc, sig),Nx^2), (Nx, Nx))
            init = im+noise
            if smooth init = imfilter(init, Kernel.gaussian(smoothval)) end
            S2arr[:,j] = DHC_2DUtils.DHC_compute(init, fhash, doS2=true, doS20=false, norm=norm, iso=iso)
        end
        wt = zeros(Float64, Ns)
        for i=1:Ns
            wt[i] = std(S2arr[i,:])
        end
        msub = S2arr .- mean(S2arr, dims=2)
        cov = (msub * msub')./(Nsam-1)

        return wt, cov
    end


    function S20_whitenoiseweights(im, fhash; loc=0.0, sig=1.0, Nsam=10, iso=false, norm=true, smooth=false, smoothval=0.8, coeff_choice="S20")
        #=
        Noise model: N(0, std(I))
        =#

        (Nx, Ny)  = size(im)
        if Nx != Ny error("Input image must be square") end
        (N1iso, Nf)    = size(fhash["S1_iso_mat"])

        # fhash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1)
        if coeff_choice=="S12"
            S20   = DHC_compute(im, fhash, doS2=false, doS20=false, doS12=true, norm=norm, iso=iso)
        else
            S20   = DHC_compute(im, fhash, doS2=false, doS20=true, norm=norm, iso=iso)
        end
        Ns    = length(S20)
        S2arr = zeros(Float64, Ns, Nsam)
        println("Ns", Ns)
        for j=1:Nsam
            noise = reshape(rand(Normal(loc, sig),Nx^2), (Nx, Nx))
            init = im+noise
            if smooth init = imfilter(init, Kernel.gaussian(smoothval)) end
            if coeff_choice=="S12"
                S2arr[:,j] = DHC_2DUtils.DHC_compute(init, fhash, doS2=false, doS20=false, doS12=true, norm=norm, iso=iso)
            else
                S2arr[:,j] = DHC_2DUtils.DHC_compute(init, fhash, doS2=false, doS20=true, norm=norm, iso=iso)
            end
        end
        wt = zeros(Float64, Ns)
        for i=1:Ns
            wt[i] = std(S2arr[i,:])
        end
        msub = S2arr .- mean(S2arr, dims=2)
        cov = (msub * msub')./(Nsam-1)

        return wt, cov
    end
end
