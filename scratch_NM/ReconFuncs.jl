module ReconFuncs
    using Statistics
    using LinearAlgebra
    using FileIO
    using Optim
    using Flux

    using DHC_2DUtils
    using Data_Utils
    using Deriv_Utils_New


    export meancov_generator
    export reconstruction_wrapper
    export Loss3Gaussian
    export dLoss3Gaussian!
    export image_recon_derivsum_custom

    #Basic Gaussian Loss Function
    #=
    function LossGaussianS20(img_curr, filter_hash, ori_input, coeff_mask, s_targ_mean, s_invcov, lambda)
        s_curr = DHC_compute(reshape(img_curr, (Nx, Nx)),  filter_hash, doS2=false, doS12=false, doS20=true, norm=false)[coeff_mask]
        regterm =  0.5*lambda*sum((img_curr - ori_input).^2)
        lnlik = ( 0.5 .* (s_curr - s_targ_mean)' * s_targ_invcov * (s_curr - s_targ_mean))
        #println("Lnlik size | Reg Size", size(lnlik), size(regterm))
        neglogloss = lnlik[1] + regterm
        return neglogloss
    end

    function dLossGaussianS20(storage_grad, img_curr, filter_hash, ori_input, coeff_mask, s_targ_mean, s_invcov, lambda)
        s_curr = DHC_compute(reshape(img_curr, (Nx, Nx)), filter_hash, doS2=false, doS12=false, doS20=true, norm=tonorm)[coeff_mask]
        diff = s_curr - s_targ_mean
        wt = reshape(convert(Array{Float64, 2}, transpose(diff) * s_targ_invcov), (Nf, Nf))
        dterm = wst_S20_deriv_sum(reshape(img_curr, (Nx, Nx)), filter_hash, wt, FFTthreads=FFTthreads) + lambda.*(img_curr - ori_input)
        storage_grad .= reshape(dterm, (Nx^2, 1))
        storage_grad[pixmask, 1] .= 0 # better way to do this by taking pixmask as an argument wst_s2_deriv_sum?
        return
    end
    =#
    function meancov_generator(true_img, fhash, dhc_args, coeff_mask, settings; safety=nothing)
        #generates targmean and covariance for Gaussian loss func
        (Nf,) = size(fhash["filt_index"])
        if dhc_args[:iso]
            @assert length(coeff_mask)==(fhash["num_iso_coeff"]) "Coeff_mask must have length Num_iso_coeff = 2+ S1_iso + S2_iso"
        else
            @assert length(coeff_mask)==(2+Nf+Nf^2) "Coeff_mask must have length 2+Nf+Nf^2"
        end

        (Nx, Ny) = size(true_img)
        preproc_img = copy(true_img)
        if settings["log"] preproc_img = log.(true_img) end

        ##Get S_Targ using true image
        if settings["target_type"]=="ground_truth" #if starg is based on true coeff
            s2targ = convert(Array{Float64,1}, DHC_compute_wrapper(preproc_img, fhash, norm=false; dhc_args...)[coeff_mask]) #Iso
            #=if coeff_type=="S2" #BUGFix: Wrap coeff_type into a dictionary and pass to DHC_COMPUTE
                s2targ = DHC_compute(preproc_img, fhash, doS2=true, doS20=false, norm=false, iso=false)[coeff_mask]
            elseif coeff_type=="S20"
                s2targ = DHC_compute(preproc_img, fhash, doS2=false, doS20=true, norm=false, iso=false)[coeff_mask]
            else
                if coeff_type!="S12" error("Not implemented") end
                s2targ = DHC_compute(preproc_img, fhash, doS2=false, doS20=false, doS12=true, norm=false, iso=false)[coeff_mask]
            end=#
        end
        if safety!=nothing
            println("Mean abs Safety check ", mean(abs.(s2targ .- safety)))
        end
        ##Get covariance based on true+noise #BUG:???
        if settings["covar_type"]=="white_noise" #if covariance is based on true+noise simulated coeff
            if !settings["log"]  #Confirm that this handles both the apd and nonapd cases appropriately
                sig2, cov = S2_whitenoiseweights(true_img, fhash, dhc_args, coeff_mask=coeff_mask; settings["white_noise_args"]...)
            #Messy here because I'm adding noise in true image space but calculating the coeffs of the log of the image. Cant just pass the log image.
            else #if settings["log"]
                sig2, cov = whitenoiseweights_forlog(true_img, fhash, dhc_args, coeff_mask=coeff_mask; settings["white_noise_args"]...)
            end
        end

        ##Get S_Targ and covariance based on sfd_dbn
        if settings["target_type"]=="sfd_dbn" #if starg is based on sfd_dbn,
            sfdimg = readsfd(Nx, logbool=settings["log"]) #Return sfd log or regular images, and PIXEL covariance, BUGFix:DON'T ADD APD HERE. APD IS TO BE ADDED EXCLUSIVELY IN DHC_COMPUTE.
            s2targ, _, _ = dbn_coeffs_calc(sfdimg, fhash, dhc_args, coeff_mask)
            if (settings["covar_type"]!="sfd_dbn") & (settings["covar_type"]!="white_noise") error("Invalid covar_type") end
        end

        if settings["covar_type"]=="sfd_dbn"
            sfdimg = Data_Utils.readsfd(Nx, logbool=settings["log"]) #Return sfd log or regular images, and PIXEL covariance, BUGFix:DON'T ADD APD HERE. APD IS TO BE ADDED EXCLUSIVELY IN DHC_COMPUTE.
            _, sig2, cov = dbn_coeffs_calc(sfdimg, fhash, dhc_args, coeff_mask)
        end

        #At this point you have an s2targ and a sig2 / cov from either the true images coeffs or a dbn
        #Invcov_matrix:
        if settings["Invcov_matrix"]=="Diagonal"
            s2icov = invert_covmat(sig2)
        elseif settings["Invcov_matrix"]=="Diagonal+Eps"
            s2icov = invert_covmat(sig2, settings["epsvalue"]) #DEBUG
        elseif settings["Invcov_matrix"]=="Full+Eps"
            s2icov = invert_covmat(cov, settings["epsvalue"])
        else#s2icov
            s2icov = invert_covmat(cov)
        end
        if safety!=nothing
            println("End ", mean(abs.(s2targ .- safety)))
        end
        return s2targ, s2icov
    end

    function reconstruction_wrapper(true_img, noisy_init, fhash, dhc_args, coeff_mask, settings)
        #=
        true_img: true_img, even if settings['log']=true, this is the non-log image
        noisy_init: initial image. Either Image+Noise or Smooth(Image+Noise)
        dhc_args: passed to dhc everywhere
        settings: Dictionary with
            'target_type': 'ground truth' | 'sfd_dbn'
            'log': Boolean
            'GaussianLoss': Boolean
            'Invcov_matrix': 'Diagonal' | 'Diagonal+Eps' | 'Full' | 'Full+Eps'
            'Covar_type': 'white_noise' | 'sfd_dbn'
            'apd': 'non_apd'
            'white_noise_args': args for white_noise sims if GaussianLoss and Covar_type='white_noise'. :loc...
            'optim_settings': args for optimization related stuff
            "lambda": regularization

        =#

        #Add logging utility to save: S_targ, s_cov, all settings, true_img, noisy_init, coeff_mask
        (Nf,) = size(fhash["filt_index"])
        (Nx, Ny) = size(true_img)
        if dhc_args[:iso]
            @assert length(coeff_mask)==(fhash["num_iso_coeff"]) "Coeff_mask must have length Num_iso_coeff = 2+ S1_iso + S2_iso"
        else
            @assert length(coeff_mask)==(2+Nf+Nf^2) "Coeff_mask must have length 2+Nf+Nf^2"
        end

        if settings["GaussianLoss"]
            starg, sinvcov = settings["s_targ_mean"], settings["s_invcov"] #meancov_generator(true_img, fhash, coeff_type, coeff_mask, settings) call this outside function with the same dict

            if !settings["log"]
                res, recon_img = Deriv_Utils_New.image_recon_derivsum_regularized(noisy_init, fhash, starg, sinvcov, falses(Nx, Nx), dhc_args, optim_settings=settings["optim_settings"], coeff_mask=coeff_mask, lambda=settings["lambda"])
            elseif settings["log"]
                res, recon_img = Deriv_Utils_New.image_recon_derivsum_regularized(log.(noisy_init), fhash, Float64.(starg), sinvcov, falses(Nx, Nx), dhc_args, coeff_mask=coeff_mask, optim_settings=settings["optim_settings"], lambda=settings["lambda"])
                recon_img = exp.(recon_img)
            else
                error("??")
            end
        elseif settings["TransformedGaussianLoss"]
            fstarg, fsinvcov = settings["fs_targ_mean"], settings["fs_invcov"]
            #HERE!
            if !settings["log"]
                res, recon_img = Deriv_Utils_New.image_recon_derivsum_regularized_transformed_gaussian(noisy_init, fhash, fstarg, fsinvcov, falses(Nx, Nx), dhc_args, settings["transform_func"], settings["transform_dfunc"], optim_settings=settings["optim_settings"], coeff_mask=coeff_mask, lambda=settings["lambda"])
            elseif settings["log"]
                res, recon_img = Deriv_Utils_New.image_recon_derivsum_regularized_transformed_gaussian(log.(noisy_init), fhash, Float64.(fstarg), fsinvcov, falses(Nx, Nx), dhc_args, settings["transform_func"], settings["transform_dfunc"], coeff_mask=coeff_mask, optim_settings=settings["optim_settings"], lambda=settings["lambda"])
                recon_img = exp.(recon_img)
            else
                error("??")
            end
        elseif settings["RandomInit"]
            fstarg, fsinvcov = settings["s_targ_mean"], settings["s_invcov"]
            #HERE!
            if !settings["log"]
                res, recon_img = Deriv_Utils_New.image_recon_derivsum_separate(noisy_init, settings["iteration_init"], fhash, fstarg, fsinvcov, falses(Nx, Nx), dhc_args, optim_settings=settings["optim_settings"], coeff_mask=coeff_mask, lambda=settings["lambda"])
            elseif settings["log"]
                res, recon_img = Deriv_Utils_New.image_recon_derivsum_separate(log.(noisy_init), log.(settings["iteration_init"]), fhash, Float64.(fstarg), fsinvcov, falses(Nx, Nx), dhc_args, coeff_mask=coeff_mask, optim_settings=settings["optim_settings"], lambda=settings["lambda"])
                recon_img = exp.(recon_img)
            else
                error("??")
            end

        else error("Non Gaussian not implemented yet") #NonGaussian
        end

        #Log and save everything
        if isfile(settings["fname_save"]) & (settings["safemode"])
            error("Overwriting file")
        else
            if !settings["safemode"] println("Overwriting existing files") end
            save(settings["fname_save"], Dict("true_img"=>true_img, "init"=>noisy_init, "recon"=>recon_img, "dict"=>settings, "trace"=>Optim.trace(res), "coeff_mask"=>coeff_mask, "fhash"=>fhash, "dhc_args"=>dhc_args))
        end
        return res, recon_img

    end


    #Custom Loss Functions######################################################
    #Auxiliary functions
    function adaptive_apodizer(input_image::Array{Float64, 2}, dhc_args)
        if dhc_args[:apodize]
            apd_image = apodizer(input_image)
        else
            apd_image = input_image
        end
        return apd_image
    end


    function augment_weights_S20(inpwt::Array{Float64}, filter_hash, dhc_args, coeff_mask)
        Nf = size(filter_hash["filt_index"])[1]
        if (dhc_args[:doS20]) & (dhc_args[:iso])
            iso2nf2mask = zeros(Int64, Nf^2)
            M20 = filter_hash["S2_iso_mat"]
            for id=1:Nf^2 iso2nf2mask[M20.colptr[id]] = M20.rowval[id] end
            coeff_masks20 = coeff_mask[3+N1iso:end]
            #wt (Nc) -> |S2_iso| -> |Nf^2|
            w_s2iso = zeros(size(filter_hash["S2_iso_mat"])[1])
            w_s2iso[coeff_masks20] .= inpwt
            w_nf2 = zeros(Nf^2)
            w_nf2 .= w_s2iso[iso2nf2mask]
            return reshape(w_nf2, (Nf, Nf))

        else# (dhc_args[:doS20]) & (!dhc_args[:iso])
            #if (dhc_args[:doS20]) & (!dhc_args[:iso])
            coeff_masks20 = coeff_mask[3+Nf:end]
            w_nf2 = zeros(Nf^2)
            w_nf2[coeff_masks20] .= inpwt
            return reshape(w_nf2, (Nf, Nf))
        end
    end

    function get_dApodizer(Nx, dhc_args)
        #For the Jacobian dA*P/dP
        if dhc_args[:apodize]
            Ap = wind_2d(Nx)
            cA = sum(Ap)
            Apflat = reshape(Ap, Nx^2)
            od_nz_idx = findall(!iszero, Apflat) #findall((x->((x!=0) & (x!=1))), Apflat)
            #od_zidx = findall(iszero, Apflat)
            avals = Apflat[od_nz_idx]
            dA = zeros(Nx^2, Nx^2)
            dA[:, od_nz_idx] .= ((1.0 .- Apflat) * avals')./cA
            dA[diagind(dA)] += Apflat
        else
            dA = I
        end
        return dA
    end

    #Loss Functions
    function Loss3Gaussian(img_curr, filter_hash, dhc_args, coeff_mask1, target1, invcov1, lambda; reg_input=nothing, coeff_mask2=nothing, target2=nothing, invcov2=nothing)
        #func_specific_params should contain: coeff_mask2, target2, invcov2
        s_curr = DHC_compute_wrapper(img_curr, filter_hash, norm=false; dhc_args...)
        s_curr1 = s_curr[coeff_mask1]
        s_curr2 = s_curr[coeff_mask2]
        regterm =  0.5*lambda*sum((adaptive_apodizer(img_curr, dhc_args) - adaptive_apodizer(reg_input, dhc_args)).^2)
        lnlik_sfd = ( 0.5 .* (s_curr1 - target1)' * invcov1 * (s_curr1 - target1))
        s_curr2 = s_curr[coeff_mask2]
        lnlik_init = ( 0.5 .* (s_curr2 - target2)' * invcov2 * (s_curr2 - target2))
        neglogloss = lnlik_sfd[1] + lnlik_init[1] + regterm
    end

    function dLoss3Gaussian!(storage_grad, img_curr, filter_hash, dhc_args, coeff_mask1, target1, invcov1, lambda; reg_input=nothing, coeff_mask2=nothing, target2=nothing, invcov2=nothing, dA=nothing)
        #func_specific_params should contain: coeff_mask2, target2, invcov2
        s_curr = DHC_compute_wrapper(img_curr, filter_hash, norm=false; dhc_args...)
        s_curr1 = s_curr[coeff_mask1]
        s_curr2 = s_curr[coeff_mask2]

        diff1 = s_curr1 - target1
        diff2 = s_curr2 - target2
        #Add branches here
        wt1 = augment_weights_S20(convert(Array{Float64,1}, reshape(transpose(diff1) * invcov1, (length(diff1),))), filter_hash, dhc_args, coeff_mask1)
        wt2 = augment_weights_S20(convert(Array{Float64,1}, reshape(transpose(diff2) * invcov2, (length(diff2),))), filter_hash, dhc_args, coeff_mask2)
        apdimg_curr = adaptive_apodizer(img_curr, dhc_args)
        dsumterms = wst_S20_deriv_sum(apdimg_curr, filter_hash, wt1)' + wst_S20_deriv_sum(apdimg_curr, filter_hash, wt2)' + reshape(lambda.*(apdimg_curr - adaptive_apodizer(reg_input, dhc_args)), (1, Nx^2))
        storage_grad .= reshape(dsumterms * dA, (Nx, Nx))
    end


    function Loss3Gaussian_transformed(img_curr, filter_hash, dhc_args, coeff_mask1, target1, invcov1, lambda; reg_input=nothing, coeff_mask2=nothing, target2=nothing, invcov2=nothing, func=nothing, dfunc=nothing)
        #func_specific_params should contain: coeff_mask2, target2, invcov2
        s_curr = DHC_compute_wrapper(img_curr, filter_hash, norm=false; dhc_args...)
        s_curr1 = s_curr[coeff_mask1]
        s_curr2 = s_curr[coeff_mask2]
        regterm =  0.5*lambda*sum((adaptive_apodizer(img_curr, dhc_args) - adaptive_apodizer(reg_input, dhc_args)).^2)
        diff1 = func(s_curr1) - target1
        diff2 = func(s_curr2) - target2
        lnlik_sfd = ( 0.5 .* (diff1)' * invcov1 * (diff1))
        s_curr2 = s_curr[coeff_mask2]
        lnlik_init = ( 0.5 .* (diff2)' * invcov2 * (diff2))
        neglogloss = lnlik_sfd[1] + lnlik_init[1] + regterm
    end

    function dLoss3Gaussian_transformed!(storage_grad, img_curr, filter_hash, dhc_args, coeff_mask1, target1, invcov1, lambda; reg_input=nothing, coeff_mask2=nothing, target2=nothing, invcov2=nothing, dA=nothing, func=nothing, dfunc=nothing)
        #func_specific_params should contain: coeff_mask2, target2, invcov2
        Nx = size(img_curr)[1]
        #println(Nx)
        s_curr = DHC_compute_wrapper(img_curr, filter_hash, norm=false; dhc_args...)
        s_curr1 = s_curr[coeff_mask1]
        s_curr2 = s_curr[coeff_mask2]

        diff1 = func(s_curr1) - target1
        diff2 = func(s_curr2) - target2
        #Add branches here
        wt1 = augment_weights_S20(convert(Array{Float64,1}, reshape(transpose(diff1) * invcov1, (length(diff1),))) .* dfunc(s_curr1), filter_hash, dhc_args, coeff_mask1)
        wt2 = augment_weights_S20(convert(Array{Float64,1}, reshape(transpose(diff2) * invcov2, (length(diff2),))) .* dfunc(s_curr2), filter_hash, dhc_args, coeff_mask2)
        apdimg_curr = adaptive_apodizer(img_curr, dhc_args)

        term1 = Deriv_Utils_New.wst_S20_deriv_sum(apdimg_curr, filter_hash, wt1)'
        term2 = Deriv_Utils_New.wst_S20_deriv_sum(apdimg_curr, filter_hash, wt2)'
        term3= reshape(lambda.*(apdimg_curr - adaptive_apodizer(reg_input, dhc_args)),(1, Nx^2))
        #println(size(term1), size(term2), size(term3))
        dsumterms = term1 + term2 + term3
        storage_grad .= reshape(dsumterms * dA, (Nx, Nx))
    end

    function image_recon_derivsum_custom(input::Array{Float64, 2}, filter_hash::Dict, s_targ_mean::Array{Float64, 1}, s_targ_invcov, coeff_mask, dhc_args, LossFunc, dLossFunc;
        FFTthreads::Int=1, optim_settings=Dict([("iterations", 10)]), lambda=0.001, func_specific_params=nothing) #add iso here and a check that returns error if both coeffmask is not nothing and iso is present.
        #=
        func_specific_params: All the functions, target vectors, covariance matrices etc that go into the loss and its derivative

        Cases to be handled by this function:
        S20
        Apodized or not
        Iso
        Log or Regular: If log then input, targ and cov should have been constructed using log
        Select pixels masked or all floating--not using for masked currently
        Select coefficients optimized wrt
        Lambda: regularization

        Cases NOT handled by this function:
        Only S1
        Non-Gaussian loss function
        Using the S0 params (mean and variance) in the loss function
        DHC_compute with norm=true
        =#
        #@assert !haskey(dhc_args, :coeff_mask) #remove because you're gonna have two different masks right?
        (N1iso, Nf) = size(filter_hash["S1_iso_mat"])
        println("Coeff mask:", (coeff_mask!=nothing))

        (Nx, Ny)  = size(input)
        if Nx != Ny error("Input image must be square") end
        if !dhc_args[:doS20] error("Not implemented") end
        if Nf == 0 error("filter hash corrupted") end

        #pixmask = pixmask[:] #flattened: Nx^2s #DEB
        ori_input = reshape(copy(input), (Nx^2, 1))

        #=
        if coeff_mask!=nothing
            if (!dhc_args[:iso]) & (count((i->i==true), coeff_mask[1:Nf+2])!=0) error("Code assumes S1 coeffs and S0 are masked out") end
            if (dhc_args[:iso]) & (count((i->i==true), coeff_mask[1:N1iso+2])!=0) error("Code assumes S1iso coeffs and S0 are masked out") end
            if size(s_targ_mean)[1]!=count((i->(i==true)), coeff_mask) error("s_targ_mean should only contain coeffs to be optimized") end #remove?
            if size(s_targ_invcov)[1]!=count((i->(i==true)), coeff_mask) error("s_targ_invcov should only have coeffs to be optimized") end
        else
            error("You must supply a coeff_mask")
        end
        =#

        #After this all cases have a coeffmask, and s_targ_mean and s_targ_invcov have the shapes of the coefficients that we want to select.
        #Optimization Hyperparameters
        tonorm = false
        numitns_dict = get(optim_settings, "iterations", 100)
        minmethod = get(optim_settings, "minmethod", ConjugateGradient())

        dA = get_dApodizer(Nx, dhc_args)

        function loss_func20(img_curr::Array{Float64, 2})
            neglogloss = LossFunc(img_curr, filter_hash, dhc_args, coeff_mask, s_targ_mean, s_targ_invcov, lambda; func_specific_params...) #tonorm
            return neglogloss
        end
        #TODO: Need to replace the deriv_sums20 with a deriv_sum wrapper that handles all cases (S12, S20, S2)

        function dloss20(storage_grad::Array{Float64, 2}, img_curr::Array{Float64, 2})
            dLossFunc(storage_grad, img_curr, filter_hash, dhc_args, coeff_mask, s_targ_mean, s_targ_invcov, lambda; dA=dA, func_specific_params...)
            return
        end


        println("Diff check")
        eps = zeros(size(input))
        row, col = 24, 18 #convert(Int8, Nx/2), convert(Int8, Nx/2)+3
        epsmag = 1e-4
        eps[row, col] = epsmag
        chisq1 = loss_func20(input+eps./2) #DEB
        chisq0 = loss_func20(input-eps./2) #DEB
        brute  = (chisq1-chisq0)/epsmag
        #df_brute = DHC_compute(reshape(input, (Nx, Nx)), filter_hash, doS2=true, doS12=false, doS20=false, norm=false)[coeff_mask] - s_targ_mean
        clever = zeros(size(input)) #DEB

        _bar = dloss20(clever, input) #DEB
        println("Chisq Derve Check")
        println("Brute:  ",brute)
        println("Clever: ",clever[row, col], " Difference: ", brute - clever[row, col]) #DEB
        println("Initial Loss ", loss_func20(input))
        if typeof(minmethod)==ADAM
            result_img = copy(input)
            buffer_grad = zeros(size(input))
            function dLoss_adam(img_curr)
                dLossFunc(buffer_grad, img_curr, filter_hash, dhc_args, coeff_mask, s_targ_mean, s_targ_invcov, lambda; dA=dA, func_specific_params...)
                return buffer_grad
            end

            for itn=1:numitns_dict
                Flux.update!(minmethod, result_img, dLoss_adam)
            end
        else
            res = optimize(loss_func20, dloss20, input, minmethod, Optim.Options(iterations = numitns_dict, store_trace = true, show_trace = true))
            result_img = Optim.minimizer(res)
        end
        println("Final Loss ", loss_func20(result_img))
        return res, reshape(result_img, (Nx, Nx))
    end



end
