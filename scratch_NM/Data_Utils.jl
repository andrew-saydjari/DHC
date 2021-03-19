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
    using FITSIO
    # put the cwd on the path so Julia can find the module
    push!(LOAD_PATH, pwd()*"/main")
    using DHC_2DUtils

    export readdust
    export S2_uniweights
    export S2_whitenoiseweights
    export invert_covmat
    export whitenoiseweights_forlog
    export mldust
    export dbn_coeffs_calc

    function readdust(Nx)
        RGBA_img = load(pwd()*"/scratch_DF/t115_clean_small.png")
        img = reinterpret(UInt8,rawview(channelview(RGBA_img)))
        return imresize(Float64.(img[1,:,:])[1:256, 1:256], (Nx, Nx))
    end

    #Functions that add simulated noise to the true image and return covariance and std^2 for the simulations
    function S2_uniweights(im, fhash; high=25, Nsam=1000, iso=false, norm=true, smooth=false, smoothval=0.8, coeff_mask=nothing)
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
        if coeff_mask!=nothing
            cov = cov[coeff_mask, coeff_mask]
            wt = wt[coeff_mask]
        end
        println("Condition number of diag cov", cond(Diagonal(wt.^(2))))
        println("Output of S2weights: Condition Number of covariance", cond(cov))
        return wt.^(2), cov
    end


    function S2_whitenoiseweights(im, fhash; loc=0.0, sig=1.0, Nsam=1000, iso=false, norm=false, smooth=false, smoothval=0.8, coeff_mask=nothing)
        #=
        Noise model: N(loc, std(sig))
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
        if coeff_mask!=nothing
            cov = cov[coeff_mask, coeff_mask]
            wt = wt[coeff_mask]
        end
        println("Output of S2 weights: Condition number of diag cov", cond(Diagonal(wt.^(2))))
        println("Condition Number of covariance", cond(cov))
        return wt.^(2), cov
    end


    function S20_whitenoiseweights_old(im, fhash; loc=0.0, sig=1.0, Nsam=1000, iso=false, norm=false, smooth=false, smoothval=0.8, coeff_choice="S20", coeff_mask=nothing)
        #=
        Noise model: N(0, std(I))
        Output: std^2 vector, full covariance matrix
        =#
        if coeff_mask==nothing println("Warning: Running without coeff_mask") end

        (Nx, Ny)  = size(im)
        if Nx != Ny error("Input image must be square") end
        (N1iso, Nf)    = size(fhash["S1_iso_mat"])

        # fhash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1)
        if coeff_choice=="S12"
            S20   = DHC_compute(im, fhash, doS2=false, doS20=false, doS12=true, norm=norm, iso=iso)
        else #S20
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
            else #S20
                S2arr[:,j] = DHC_2DUtils.DHC_compute(init, fhash, doS2=false, doS20=true, norm=norm, iso=iso)
            end
        end
        wt = zeros(Float64, Ns)
        for i=1:Ns
            wt[i] = std(S2arr[i,:])
        end
        msub = S2arr .- mean(S2arr, dims=2)
        cov = (msub * msub')./(Nsam-1)
        if coeff_mask!=nothing
            cov = cov[coeff_mask, coeff_mask]
            wt = wt[coeff_mask]
        end
        println("Output of S20weights: Condition number of diag cov", cond(Diagonal(wt.^(2))))
        println("Condition Number of covariance", cond(cov))
        return wt.^(2), cov
    end

    function S20_whitenoiseweights(im, fhash, coeff_choice; loc=0.0, sig=1.0, Nsam=1000, iso=false, norm=false, smooth=false, smoothval=0.8, coeff_mask=nothing)
        #=
        Noise model: N(0, std(I))
        Output: std^2 vector, full covariance matrix
        coeff_choice is now a dictionary of args to pass to DHC_compute
        =#
        if coeff_mask==nothing println("Warning: Running without coeff_mask") end

        (Nx, Ny)  = size(im)
        if Nx != Ny error("Input image must be square") end
        (N1iso, Nf)    = size(fhash["S1_iso_mat"])

        # fhash = fink_filter_hash(1, 8, nx=Nx, pc=1, wd=1)
        S2   = DHC_compute(im, fhash, norm=norm, iso=iso, coeff_choice)

        Ns    = length(S20)
        S2arr = zeros(Float64, Ns, Nsam)
        println("Ns", Ns)
        for j=1:Nsam
            noise = reshape(rand(Normal(loc, sig),Nx^2), (Nx, Nx))
            init = im+noise
            if smooth init = imfilter(init, Kernel.gaussian(smoothval)) end
            S2arr[:,j] = DHC_2DUtils.DHC_compute(init, fhash, coeff_choice, norm=norm, iso=iso)
        end
        wt = zeros(Float64, Ns)
        for i=1:Ns
            wt[i] = std(S2arr[i,:])
        end
        msub = S2arr .- mean(S2arr, dims=2)
        cov = (msub * msub')./(Nsam-1)
        if coeff_mask!=nothing
            cov = cov[coeff_mask, coeff_mask]
            wt = wt[coeff_mask]
        end
        println("Output of S20weights: Condition number of diag cov", cond(Diagonal(wt.^(2))))
        println("Condition Number of covariance", cond(cov))
        return wt.^(2), cov
    end
    function whitenoiseweights_forlog(oriim, fhash, coeff_choice; loc=0.0, sig=1.0, Nsam=10, iso=false, norm=false, smooth=false, smoothval=0.8, coeff_mask=nothing) #Complete
        (Nx, Ny)  = size(oriim)
        if Nx != Ny error("Input image must be square") end
        (N1iso, Nf)    = size(fhash["S1_iso_mat"])
        logim = log.(oriim)
        if coeff_choice=="S2"
            S2   = DHC_compute(logim, fhash, doS2=true, doS20=false, norm=norm, iso=iso)
        elseif coeff_choice=="S20"
            S2   = DHC_compute(logim, fhash, doS2=false, doS20=true, norm=norm, iso=iso)
        else
            if coeff_choice=="S12" S2 = DHC_compute(logim, fhash, doS2=false, doS20=false, doS12=true, norm=norm, iso=iso)
            else
                error("Invalid coeff choice")
            end
        end
        Ns    = length(S2)
        S2arr = zeros(Float64, Ns, Nsam)
        println("Ns", Ns)
        for j=1:Nsam
            noise = reshape(rand(Normal(loc, sig),Nx^2), (Nx, Nx))
            init = oriim+noise
            loginit = log.(init)
            if smooth init = imfilter(init, Kernel.gaussian(smoothval)) end
            if coeff_choice=="S2"
                S2arr[:, j]   = DHC_compute(loginit, fhash, doS2=true, doS20=false, norm=norm, iso=iso)
            elseif coeff_choice=="S20"
                S2arr[:, j]   = DHC_compute(loginit, fhash, doS2=false, doS20=true, norm=norm, iso=iso)
            else
                if coeff_choice=="S12" S2arr[:, j] = DHC_compute(loginit, fhash, doS2=false, doS20=false, doS12=true, norm=norm, iso=iso)
                else
                    error("Invalid coeff choice")
                end
            end
        end
        wt = zeros(Float64, Ns)
        for i=1:Ns
            wt[i] = std(S2arr[i,:])
        end
        msub = S2arr .- mean(S2arr, dims=2)
        cov = (msub * msub')./(Nsam-1)
        if coeff_mask!=nothing
            cov = cov[coeff_mask, coeff_mask]
            wt = wt[coeff_mask]
        end
        println("Output of S2 weights: Condition number of diag cov", cond(Diagonal(wt.^(2))))
        println("Condition Number of covariance", cond(cov))
        return wt.^(2), cov
    end

    function invert_covmat(cov, epsilon=nothing)
        #=
        cov: Vector if Diagonal matrix | Covariance
        Computes inverse after adding epsilon for stabilization if requested.
        =#
        isdiag = ndims(cov)==1
        dstb = zeros(size(cov)[1])
        if epsilon!=nothing
            dstb = fill(epsilon, size(cov)[1])
        end

        if isdiag #Vector
            cov = cov + dstb
            println("Cond No before inversion", cond(Diagonal(cov)))
            icov = Diagonal(cov.^(-1))
            println("Numerical error wrt Id", mean(abs.((Diagonal(cov) * icov) - I)))
        else #Matrix
            cov = cov + Diagonal(dstb)
            println("Cond No before inversion", cond(cov))
            icov = inv(cov)
            println("Numerical error wrt Id", mean(abs.((cov * icov) - I)))
        end

        return icov
    end

    function mldust(nx; logbool=true)

        # read FITS file with images
        # file in /n/fink2/dfink/mldust/dust10000.fits
        #     OR  /n/fink2/dfink/mldust/dust100000.fits

        fname = "scratch_NM/data/dust10000.fits"
        f = FITS(fname, "r")
        big = read(f[1])

        (_,__,Nslice) = size(big)
        println(Nslice, " slices")

        #nx = 96
        #im = Float64.(big[11:11+nx-1, 11:11+nx-1, :])
        #nx = 48

        # rebin it to something smaller and take the log
        if logbool
            im = log.(imresize(Float64.(big), nx, nx, Nslice)) #Log of SFD images
        else
            im = imresize(Float64.(big), nx, nx, Nslice)
        end

        covar = dust_covar_matrix(im)

        #writefits, 'covar48_new.fits', covar

        # Cholesky factorization; chol.U and chol.L will be upper,lower triangular matrix
        println("Cholesky")
        chol = cholesky(covar)

        # generate some mock maps
        Nmock = 800
        ran = rand(Normal(), nx*nx, Nmock)

        recon = reshape(chol.L*ran, nx, nx, Nmock)

        return im, covar
    end

    function dust_covar_matrix(im)
        # derive covariance matrix from [Nx,Nx,Nimage] array

        (nx,__,Nslice) = size(im)
        eps=1E-6

        # make it mean zero with eps noise, otherwise covar is singular
        println("Mean zero")
        for i=1:Nslice  im[:, :, i] .-= (mean(im[:, :, i])+(rand()-0.5)*eps) end

        #println("Unit variance")
        #for i=1:Nslice  im[:, :, i] ./= std(im[:, :, i]) end

        dat = reshape(im, nx*nx, Nslice)

        # covariance matrix
        println("covariance")
        covar = (dat * dat') ./ Nslice

        # Check condition number
        println("Condition number  ", cond(covar))
        return covar

    end

    function dbn_coeffs_calc(dbnimg, fhash, coeff_choice, coeff_mask, settings)
        #dbnimg: Either a set of aimges from a dbn or their log
        (Nf,) = size(fhash["filt_index"])
        Ncovsamp = size(dbnimg)[3]
        s20_dbn = zeros(Float64, Ncovsamp, 2+Nf+Nf^2)
        for idx=1:Ncovsamp
            if coeff_choice=="S20"
                s20_dbn[idx, :] = DHC_compute(dbnimg[:, :, idx], fhash, doS2=false, doS20=true, norm=false, iso=false)
            elseif coeff_choice=="S2"
                s20_dbn[idx, :] = DHC_compute(dbnimg[:, :, idx], fhash, doS2=true, doS20=false, norm=false, iso=false)
            else
                if coeff_choice!="S12" error("Invalid coeff_choice") end
                s20_dbn[idx, :] = DHC_compute(dbnimg[:, :, idx], fhash, doS2=false, doS20=false, doS12=true, norm=false, iso=false)
            end
        end

        s_targ_mean = mean(s20_dbn, dims=1)
        scov  = (s20_dbn .- s_targ_mean)' * (s20_dbn .- s_targ_mean) ./(Ncovsamp-1)
        return s_targ_mean[coeff_mask], diag(scov)[coeff_mask], scov[coeff_mask, coeff_mask]
    end


end
