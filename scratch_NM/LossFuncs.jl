module LossFuncs
    using Statistics
    using LinearAlgebra

    using DHC_2DUtils
    using Data_Utils

    export LossGaussianS20
    export dLossGaussianS20
    export meancov_generator

    #Basic Gaussian Loss Function
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

    function meancov_generator(true_img, fhash, coeff_type, coeff_mask, settings)
        #generates targmean and covariance for Gaussian loss func
        (Nx, Ny) = size(true_img)
        preproc_img = copy(true_img)
        if settings["log"] preproc_img = log.(true_img) end
        #Add apd case above

        ##Calculating S_Targ
        if settings["target_type"]=="ground truth" #if starg is based on true coeff
            s2targ = DHC_compute(preproc_img, fhash, norm=false, iso=false, coeff_type)[coeff_mask]
            #=if coeff_type=="S2" #BUGFix: Wrap coeff_type into a dictionary and pass to DHC_COMPUTE
                s2targ = DHC_compute(preproc_img, fhash, doS2=true, doS20=false, norm=false, iso=false)[coeff_mask]
            elseif coeff_type=="S20"
                s2targ = DHC_compute(preproc_img, fhash, doS2=false, doS20=true, norm=false, iso=false)[coeff_mask]
            else
                if coeff_type!="S12" error("Not implemented") end
                s2targ = DHC_compute(preproc_img, fhash, doS2=false, doS20=false, doS12=true, norm=false, iso=false)[coeff_mask]
            end=#
        end
        if settings["target_type"]=="sfd_dbn" #if starg is based on sfd_dbn,
            sfdimg, _ = mldust(Nx, logbool=settings["log"]) #Return sfd log or not log images, and PIXEL covariance, BUG:Add apd here

            #Calculate sfdimg coeffs covariance
            #if settings["covar_type"]=="sfd_dbn"
            if (settings["apd"]=="non_apd")
                s2targ, sig2, cov = dbn_coeffs_calc(sfdimg, fhash, coeff_type, coeff_mask, settings)

            elseif  settings["log"] & (settings["apd"]=="apd")
                error("Not Implemented yet")
            else  #!settings['log'] & settings['apd']=='apd'
                error("Not Implemented yet")
            end
        end

        ##Get covariance, #BUG:
        if settings["covar_type"]=="white_noise" #if covariance is based on true+noise simulated coeff
            if !settings["log"] & (settings["apd"]=="non_apd")
                sig2, cov = S20_whitenoiseweights(true_img, fhash, coeff_type, coeff_mask=coeff_mask, settings["white_noise_args"]...)

            elseif settings["log"] & (settings["apd"]=="non_apd")
                sig2, cov = whitenoiseweights_forlog(true_img, fhash, coeff_type, coeff_mask=coeff_mask, settings["white_noise_args"]...) #check if semi colon here
            elseif  settings["log"] & (settings["apd"]=="apd")
                error("Not Implemented yet")
            else  #!settings['log'] & settings['apd']=='apd'
                error("Not Implemented yet")
            end
        end

        #At this point you have an s2targ and a sig2 / cov from either the true images coeffs or a dbn
        #Invcov_matrix
        if settings["Invcov_matrix"]=="Diagonal"
            s2icov = invert_covmat(sig2)
        elseif settings["Invcov_matrix"]=="Diagonal+Eps"
            s2icov = invert_covmat(sig2, 1e-10)
        elseif settings["Invcov_matrix"]=="Full+Eps"
            s2icov = invert_covmat(cov, 1e-10)
        else#s2icov
            s2icov = invert_covmat(cov)
        end

        return s2targ, s2icov
    end


end
