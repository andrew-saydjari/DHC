module ReconFuncs
    using Statistics
    using LinearAlgebra
    using FileIO
    using Optim

    using DHC_2DUtils
    using Data_Utils
    using Deriv_Utils_New

    export LossGaussianS20
    export dLossGaussianS20
    export meancov_generator
    export reconstruction_wrapper

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
    function meancov_generator(true_img, fhash, dhc_args, coeff_mask, settings)
        #generates targmean and covariance for Gaussian loss func
        (Nx, Ny) = size(true_img)
        preproc_img = copy(true_img)
        if settings["log"] preproc_img = log.(true_img) end
        #Add apd case above

        ##Get S_Targ using true image
        if settings["target_type"]=="ground_truth" #if starg is based on true coeff
            s2targ = convert(Array{Float64,1}, DHC_compute_apd(preproc_img, fhash, norm=false, iso=false; dhc_args...)[coeff_mask])
            #=if coeff_type=="S2" #BUGFix: Wrap coeff_type into a dictionary and pass to DHC_COMPUTE
                s2targ = DHC_compute(preproc_img, fhash, doS2=true, doS20=false, norm=false, iso=false)[coeff_mask]
            elseif coeff_type=="S20"
                s2targ = DHC_compute(preproc_img, fhash, doS2=false, doS20=true, norm=false, iso=false)[coeff_mask]
            else
                if coeff_type!="S12" error("Not implemented") end
                s2targ = DHC_compute(preproc_img, fhash, doS2=false, doS20=false, doS12=true, norm=false, iso=false)[coeff_mask]
            end=#
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
            s2targ, sig2, cov = dbn_coeffs_calc(sfdimg, fhash, dhc_args, coeff_mask)
        end

        #At this point you have an s2targ and a sig2 / cov from either the true images coeffs or a dbn
        #Invcov_matrix:
        if settings["Invcov_matrix"]=="Diagonal"
            s2icov = invert_covmat(sig2)
        elseif settings["Invcov_matrix"]=="Diagonal+Eps"
            s2icov = invert_covmat(sig2, 1e-10) #DEBUG
        elseif settings["Invcov_matrix"]=="Full+Eps"
            s2icov = invert_covmat(cov, 1e-10)
        else#s2icov
            s2icov = invert_covmat(cov)
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
        if size(coeff_mask)[1]!=(2+Nf+Nf^2) error("Wrong length mask") end

        #s2targ = DHC_compute(img, fhash, doS2=true, norm=norm, iso=false)
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
        else error("Non Gaussian not implemented yet") #NonGaussian
        end

        #Log and save everything
        if isfile(settings["fname_save"]) & (settings["safemode"])
            error("Overwriting file")
        else
            if !settings["safemode"] println("Overwriting existing files") end
            save(settings["fname_save"], Dict("true_img"=>true_img, "init"=>noisy_init, "recon"=>recon_img, "dict"=>settings, "trace"=>Optim.trace(res)))
        end
        return res, recon_img

    end

end
