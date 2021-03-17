## Preloads
module DHC_2DUtils

    using FFTW
    using LinearAlgebra
    using SparseArrays
    using Statistics
    using Test

    import CUDA

    export fink_filter_bank
    export fink_filter_list
    export fink_filter_hash
    export wst_S1_deriv
    export wst_S20_deriv
    export wst_S20_deriv_sum
    export DHC_compute
    export fink_filter_bank_3dizer
    export DHC_compute_3d
    export S1_iso_matrix3d
    export S2_iso_matrix3d
    export transformMaker
    export DHC_compute_RGB

## Filter hash construct core

    function fink_filter_hash(c, L; nx=256, wd=1, pc=1, shift=false, Omega=false, safety_on=true)
        # -------- compute the filter bank
        filt, hash = fink_filter_bank(c, L; nx=nx, wd=wd, pc=pc, shift=shift, Omega=Omega, safety_on=safety_on)

        # -------- list of non-zero pixels
        flist = fink_filter_list(filt)

        # -------- pack everything you need into the info structure
        hash["filt_index"] = flist[1]
        hash["filt_value"] = flist[2]

        # -------- compute matrix that projects iso coeffs, add to hash
        S1_iso_mat = S1_iso_matrix(hash)
        hash["S1_iso_mat"] = S1_iso_mat
        S2_iso_mat = S2_iso_matrix(hash)
        hash["S2_iso_mat"] = S2_iso_mat
        hash["num_iso_coeff"] = size(S1_iso_mat)[1] + size(S2_iso_mat)[1] + 2
        hash["num_coeff"] = size(S1_iso_mat)[2] + size(S2_iso_mat)[2] + 2

        return hash
    end

    function fink_filter_hash_gpu(c, L; nx=256, wd=1, pc=1, shift=false, Omega=false, safety_on=true,threeD=false,cz=1,nz=256)

        # -------- compute the filter bank
        filt, hash = fink_filter_bank(c, L; nx=nx, wd=wd, pc=pc, shift=shift, Omega=Omega, safety_on=safety_on)

        flist = fink_filter_list(filt)
        # -------- pack everything you need into the info structure
        hash["filt_index"] = flist[1]
        hash["filt_value"] = flist[2]

        if threeD
            hash=fink_filter_bank_3dizer(hash,cz,nz=nz)
        end

        hash=list_to_box(hash,dim=threeD ? 3 : 2)

        # send to gpu
        CUDA.reclaim()
        hash["filt_vals"] = [CUDA.CuArray(val) for val in hash["filt_vals"]]
        CUDA.reclaim()

        #Not implemented
        # -------- compute matrix that projects iso coeffs, add to hash
        #S1_iso_mat = S1_iso_matrix(hash)
        hash["S1_iso_mat"] = nothing
        #S2_iso_mat = S2_iso_matrix(hash)
        hash["S2_iso_mat"] = nothing

        return hash
    end

    function fink_filter_hash_gpu_fake(c, L; nx=256, wd=1, pc=1, shift=false, Omega=false, safety_on=true,threeD=false,cz=1,nz=256)

        # -------- compute the filter bank
        filt, hash = fink_filter_bank(c, L; nx=nx, wd=wd, pc=pc, shift=shift, Omega=Omega, safety_on=safety_on)

        flist = fink_filter_list(filt)
        # -------- pack everything you need into the info structure
        hash["filt_index"] = flist[1]
        hash["filt_value"] = flist[2]

        if threeD
            hash=fink_filter_bank_3dizer(hash,cz,nz=nz)
        end

        hash=list_to_box(hash,dim=threeD ? 3 : 2)


        #Not implemented
        # -------- compute matrix that projects iso coeffs, add to hash
        #S1_iso_mat = S1_iso_matrix(hash)
        hash["S1_iso_mat"] = nothing
        #S2_iso_mat = S2_iso_matrix(hash)
        hash["S2_iso_mat"] = nothing

        return hash
    end

    function fink_filter_bank_3dizer(hash, cz; nz=256)
        # -------- set parameters
        nx = convert(Int64,hash["npix"])
        n2d = size(hash["filt_value"])[1]
        dz = nz/2-1
        J = size(hash["j_value"])[1]
        L = size(hash["theta_value"])[1]

        im_scale = convert(Int8,log2(nz))
        # -------- number of bins in radial direction (size scales)
        K = (im_scale-2)*cz

        # -------- allocate output array of zeros
        J_L_K     = zeros(Int32, n2d*K,3)
        psi_index = zeros(Int32, J+1, L, K)
        k_value   = zeros(Float64, K)
        logr      = zeros(Float64,nz)
        filt_temp = zeros(Float64,nx,nx,nz)
        F_z       = zeros(Float64,nx,nx,nz)
        #F_p       = zeros(Float64,nx,nx,nz)

        filtind = fill(CartesianIndex{3}[], K*n2d)
        filtval = fill(Float64[], K*n2d)

        for z = 1:convert(Int64,dz+1)
            sz = mod(z+dz,nz)-dz-1
            z2 = sz^2
            logr[z] = 0.5*log2(max(1,z2))
        end

        @inbounds for k_ind = 1:K
            k = k_ind/cz
            k_value[k_ind] = k  # store for later
            krad  = im_scale-k-1
            Δk    = abs.(logr.-krad)
            kmask = (Δk .<= 1/cz)
            k_count = count(kmask)
            k_vals = findall(kmask)

            # -------- radial part
            #I have somehow dropped a factor of sqrt(2) which has been reinserted as a 0.5... fix my brain
            @views F_z[:,:,kmask].= reshape(1/sqrt(2).*cos.(Δk[kmask] .* (cz*π/2)),1,1,k_count)
            @inbounds for index = 1:n2d
                p_ind = hash["filt_index"][index]
                p_filt = hash["filt_value"][index]
                #F_p[p_ind,:] .= p_filt
                f_ind    = k_ind + (index-1)*K
                @views filt_tmp = p_filt.*F_z[p_ind,kmask]
                #temp_ind = findall((F_p .> 1E-13) .& (F_z .> 1E-13))
                #@views filt_tmp = F_p[temp_ind].*F_z[temp_ind]
                ind = findall(filt_tmp .> 1E-13)
                @views filtind[f_ind] = map(x->CartesianIndex(p_ind[x[1]][1],p_ind[x[1]][2],k_vals[x[2]]),ind)
                #filtind[f_ind] = temp_ind[ind]
                @views filtval[f_ind] = filt_tmp[ind]
                @views J_L_K[f_ind,:] = [hash["J_L"][index,1],hash["J_L"][index,2],k_ind-1]
                psi_index[hash["J_L"][index,1]+1,hash["J_L"][index,2]+1,k_ind] = f_ind

            end
        end

        # -------- metadata dictionary
        info=Dict()
        info["npix"]         = hash["npix"]
        info["j_value"]      = hash["j_value"]
        info["theta_value"]  = hash["theta_value"]
        info["2d_psi_index"]    = hash["psi_index"]
        info["2d_phi_index"]    = hash["phi_index"]
        info["2d_J_L"]          = hash["J_L"]
        info["2d_pc"]           = hash["pc"]
        info["2d_wd"]           = hash["wd"]
        info["2d_fs_center_r"]  = hash["fs_center_r"]
        info["2d_filt_index"]   = hash["filt_index"]
        info["2d_filt_value"]   = hash["filt_value"]

        info["nz"]              = nz
        info["cz"]              = cz
        info["J_L_K"]           = psi_index
        info["psi_index"]       = psi_index
        info["k_value"]         = k_value
        info["filt_index"]      = filtind
        info["filt_value"]      = filtval

        # -------- compute matrix that projects iso coeffs, add to hash
        S1_iso_mat = S1_iso_matrix3d(info)
        info["S1_iso_mat"] = S1_iso_mat
        S2_iso_mat = S2_iso_matrix3d(info)
        info["S2_iso_mat"] = S2_iso_mat

        #

        return info
    end

## Matrix for transformations
    function S1_iso_matrix(fhash)
        # fhash is the filter hash output by fink_filter_hash
        # The output matrix converts an S1 coeff vector to S1iso by
        #   summing over l
        # Matrix is stored in sparse CSC format using SparseArrays.
        # DPF 2021-Feb-18

        # Does hash contain Omega filter?
        Omega   = haskey(fhash, "Omega_index")
        if Omega Ω_ind = fhash["Omega_index"] end

        # unpack fhash
        Nl      = length(fhash["theta_value"])
        Nj      = length(fhash["j_value"])
        Nf      = length(fhash["filt_value"])
        ψ_ind   = fhash["psi_index"]
        ϕ_ind   = fhash["phi_index"]

        # number of iso coefficients
        Niso    = Omega ? Nj+2 : Nj+1
        Mat     = zeros(Int32, Niso, Nf)

        # first J elements of iso
        for j = 1:Nj
            for l = 1:Nl
                λ = ψ_ind[j,l]
                Mat[j, λ] = 1
            end
        end

        # Next elements are ϕ, Ω
        I0     = Nj+1
        Mat[I0, ϕ_ind] = 1
        if Omega Mat[I0+1, Ω_ind] = 1 end

        return sparse(Mat)
    end

    function S2_iso_matrix(fhash)
        # fhash is the filter hash output by fink_filter_hash
        # The output matrix converts an S2 coeff vector to S2iso by
        #   summing over l1,l2 and fixed Δl.
        # Matrix is stored in sparse CSC format using SparseArrays.
        # DPF 2021-Feb-18

        # Does hash contain Omega filter?
        Omega   = haskey(fhash, "Omega_index")
        if Omega Ω_ind = fhash["Omega_index"] end

        # unpack fhash
        Nl      = length(fhash["theta_value"])
        Nj      = length(fhash["j_value"])
        Nf      = length(fhash["filt_value"])
        ψ_ind   = fhash["psi_index"]
        ϕ_ind   = fhash["phi_index"]

        # number of iso coefficients
        Niso    = Omega ? Nj*Nj*Nl+4*Nj+4 : Nj*Nj*Nl+2*Nj+1
        Mat     = zeros(Int32, Niso, Nf*Nf)

        # first J*J*L elements of iso
        for j1 = 1:Nj
            for j2 = 1:Nj
                for l1 = 1:Nl
                    for l2 = 1:Nl
                        DeltaL = mod(l1-l2, Nl)
                        λ1     = ψ_ind[j1,l1]
                        λ2     = ψ_ind[j2,l2]

                        Iiso   = j1+Nj*((j2-1)+Nj*DeltaL)
                        Icoeff = λ1+Nf*(λ2-1)
                        Mat[Iiso, Icoeff] = 1
                    end
                end
            end
        end

        # Next J elements are λϕ, then J elements ϕλ
        for j = 1:Nj
            for l = 1:Nl
                λ      = ψ_ind[j,l]
                Iiso   = Nj*Nj*Nl+j
                Icoeff = λ+Nf*(ϕ_ind-1)  # λϕ
                Mat[Iiso, Icoeff] = 1

                Iiso   = Nj*Nj*Nl+Nj+j
                Icoeff = ϕ_ind+Nf*(λ-1)  # ϕλ
                Mat[Iiso, Icoeff] = 1
            end
        end

        # Next 1 element is ϕϕ
        I0     = Nj*Nj*Nl+Nj+Nj+1
        Icoeff = ϕ_ind+Nf*(ϕ_ind-1)
        Mat[I0, Icoeff] = 1

        # If the Omega filter exists, add more terms
        if Omega
            # Next J elements are λΩ, then J elements Ωλ
            for j = 1:Nj
                for l = 1:Nl
                    λ      = ψ_ind[j,l]
                    Iiso   = I0+j
                    Icoeff = λ+Nf*(Ω_ind-1)  # λΩ
                    Mat[Iiso, Icoeff] = 1

                    Iiso   = I0+Nj+j
                    Icoeff = Ω_ind+Nf*(λ-1)  # Ωλ
                    Mat[Iiso, Icoeff] = 1
                end
            end
            # Next 3 elements are ϕΩ, Ωϕ, ΩΩ
            Iiso   = I0+Nj+Nj
            Mat[Iiso+1, ϕ_ind+Nf*(Ω_ind-1)] = 1
            Mat[Iiso+2, Ω_ind+Nf*(ϕ_ind-1)] = 1
            Mat[Iiso+3, Ω_ind+Nf*(Ω_ind-1)] = 1
        end

        return sparse(Mat)
    end

    function S1_iso_matrix3d(fhash)
        # fhash is the filter hash output by fink_filter_hash
        # The output matrix converts an S1 coeff vector to S1iso by
        #   summing over l
        # Matrix is stored in sparse CSC format using SparseArrays.
        # DPF 2021-Feb-18
        # AKS 2021-Feb-22

        # Does hash contain Omega filter?
        #Omega   = haskey(fhash, "Omega_index")
        #if Omega Ω_ind = fhash["Omega_index"] end

        # unpack fhash
        Nl      = length(fhash["theta_value"])
        Nj      = length(fhash["j_value"])
        Nk      = length(fhash["k_value"])
        Nf      = length(fhash["filt_value"])
        ψ_ind   = fhash["psi_index"]
        #ϕ_ind   = fhash["phi_index"]

        # number of iso coefficients
        #Niso    = Omega ? Nj+2 : Nj+1
        Niso    = (Nj+1)*Nk
        Mat     = zeros(Int32, Niso, Nf)

        # first J elements of iso
        for j = 1:Nj
            for l = 1:Nl
                for k=1:Nk
                    λ = ψ_ind[j,l,k]
                    Mat[j+(k-1)*Nj, λ] = 1
                end
            end
        end

        j = Nj+1
        l = 1
        for k=1:Nk
            λ = ψ_ind[j,l,k]
            Mat[j+(k-1)*Nj, λ] = 1
        end

        return sparse(Mat)
    end

    function S2_iso_matrix3d(fhash)
        # fhash is the filter hash output by fink_filter_hash
        # The output matrix converts an S2 coeff vector to S2iso by
        #   summing over l1,l2 and fixed Δl.
        # Matrix is stored in sparse CSC format using SparseArrays.
        # DPF 2021-Feb-18
        # AKS 2021-Feb-22

        # Does hash contain Omega filter?
        #Omega   = haskey(fhash, "Omega_index")
        #if Omega Ω_ind = fhash["Omega_index"] end

        # unpack fhash
        Nl      = length(fhash["theta_value"])
        Nj      = length(fhash["j_value"])
        Nk      = length(fhash["k_value"])
        Nf      = length(fhash["filt_value"])
        ψ_ind   = fhash["psi_index"]
        #ϕ_ind   = fhash["phi_index"]

        # number of iso coefficients
        #Niso    = Omega ? Nj*Nj*Nl+4*Nj+4 : Nj*Nj*Nl+2*Nj+1
        Niso    = Nj*Nj*Nk*Nk*Nl+2*Nj*Nk*Nk+Nk*Nk
        Mat     = zeros(Int32, Niso, Nf*Nf)

        #should probably think about reformatting to (Nj+1) so phi
        #phi cross terms are in the right place

        # first J*J*K*K*L elements of iso
        for j1 = 1:Nj
            for j2 = 1:Nj
                for l1 = 1:Nl
                    for l2 = 1:Nl
                        for k1 = 1:Nk
                            for k2 = 1:Nk
                                DeltaL = mod(l1-l2, Nl)
                                λ1     = ψ_ind[j1,l1,k1]
                                λ2     = ψ_ind[j2,l2,k2]

                                Iiso   = k1+Nk*((k2-1)+Nk*((j1-1)+Nj*((j2-1)+Nj*DeltaL)))
                                Icoeff = λ1+Nf*(λ2-1)
                                Mat[Iiso, Icoeff] = 1
                            end
                        end
                    end
                end
            end
        end

        # Next J elements are λϕ, then J elements ϕλ
        for j = 1:Nj
            for l = 1:Nl
                for k1 = 1:Nk
                    for k2 =1:Nk
                        λ      = ψ_ind[j,l,k1]
                        Iiso   = Nj*Nj*Nk*Nk*Nl+k1+Nk*((k2-1)+Nk*(j-1))
                        Icoeff = λ+Nf*(ψ_ind[Nj+1,1,k2]-1)  # λϕ
                        Mat[Iiso, Icoeff] = 1

                        Iiso   = Nj*Nj*Nk*Nk*Nl+Nj*Nk*Nk+k1+Nk*((k2-1)+Nk*(j-1))
                        Icoeff = (ψ_ind[Nj+1,1,k2]-1)+Nf*(λ-1)  # ϕλ
                        Mat[Iiso, Icoeff] = 1
                    end
                end
            end
        end

        # Next Nk*Nk elements are ϕ(k)ϕ(k)
        j = Nj+1
        l = 1
        for k1=1:Nk
            for k2=1:Nk
                λ1 = ψ_ind[j,l,k1]
                λ2 = ψ_ind[j,l,k2]

                Iiso = Nj*Nj*Nk*Nk*Nl+2*Nj*Nk*Nk+k1+Nk*(k2-1)
                Icoeff = λ1+Nf*(λ2-1)
                Mat[Iiso, Icoeff] = 1
            end
        end

        # # If the Omega filter exists, add more terms
        # if Omega
        #     # Next J elements are λΩ, then J elements Ωλ
        #     for j = 1:Nj
        #         for l = 1:Nl
        #             λ      = ψ_ind[j,l]
        #             Iiso   = I0+j
        #             Icoeff = λ+Nf*(Ω_ind-1)  # λΩ
        #             Mat[Iiso, Icoeff] = 1
        #
        #             Iiso   = I0+Nj+j
        #             Icoeff = Ω_ind+Nf*(λ-1)  # Ωλ
        #             Mat[Iiso, Icoeff] = 1
        #         end
        #     end
        #     # Next 3 elements are ϕΩ, Ωϕ, ΩΩ
        #     Iiso   = I0+Nj+Nj
        #     Mat[Iiso+1, ϕ_ind+Nf*(Ω_ind-1)] = 1
        #     Mat[Iiso+2, Ω_ind+Nf*(ϕ_ind-1)] = 1
        #     Mat[Iiso+3, Ω_ind+Nf*(Ω_ind-1)] = 1
        # end
        #
        return sparse(Mat)
    end

    function S1_equiv_matrix(fhash,l_shift)
            # fhash is the filter hash output by fink_filter_hash
            # The output matrix converts an S1 coeff vector to S1iso by
            #   summing over l
            # Matrix is stored in sparse CSC format using SparseArrays.
            # DPF 2021-Feb-18

            # Does hash contain Omega filter?
            Omega   = haskey(fhash, "Omega_index")
            if Omega Ω_ind = fhash["Omega_index"] end

            # unpack fhash
            Nl      = length(fhash["theta_value"])
            Nj      = length(fhash["j_value"])
            Nf      = length(fhash["filt_value"])
            ψ_ind   = fhash["psi_index"]
            ϕ_ind   = fhash["phi_index"]

            # number of iso coefficients
            Niso    = Omega ? Nj+2 : Nj+1
            Mat     = zeros(Int32, Nf, Nf)

            # first J elements of iso
            for j = 1:Nj
                for l = 1:Nl
                    λ = ψ_ind[j,l]
                    λ1 = ψ_ind[j,mod1(l+l_shift,Nl)]
                    Mat[λ1, λ] = 1
                end
            end

            # Next elements are ϕ, Ω
            Mat[ϕ_ind, ϕ_ind] = 1
            if Omega Mat[Ω_ind, Ω_ind] = 1 end

            return sparse(Mat)
    end

    function S2_equiv_matrix(fhash,l_shift)
        # fhash is the filter hash output by fink_filter_hash
        # The output matrix converts an S2 coeff vector to S2iso by
        #   summing over l1,l2 and fixed Δl.
        # Matrix is stored in sparse CSC format using SparseArrays.
        # DPF 2021-Feb-18

        # Does hash contain Omega filter?
        Omega   = haskey(fhash, "Omega_index")
        if Omega Ω_ind = fhash["Omega_index"] end

        # unpack fhash
        Nl      = length(fhash["theta_value"])
        Nj      = length(fhash["j_value"])
        Nf      = length(fhash["filt_value"])
        ψ_ind   = fhash["psi_index"]
        ϕ_ind   = fhash["phi_index"]

        # number of iso coefficients
        Niso    = Omega ? Nj*Nj*Nl+4*Nj+4 : Nj*Nj*Nl+2*Nj+1
        Mat     = zeros(Int32, Nf*Nf, Nf*Nf)

        # first J*J*L elements of iso
        for j1 = 1:Nj
            for j2 = 1:Nj
                for l1 = 1:Nl
                    for l2 = 1:Nl
                        λ1     = ψ_ind[j1,l1]
                        λ2     = ψ_ind[j2,l2]
                        λ1_new     = ψ_ind[j1,mod1(l1+l_shift,Nl)]
                        λ2_new     = ψ_ind[j2,mod1(l2+l_shift,Nl)]

                        Icoeff = λ1+Nf*(λ2-1)
                        Icoeff_new = λ1_new+Nf*(λ2_new-1)
                        Mat[Icoeff_new, Icoeff] = 1
                    end
                end
            end
        end

        # Next J elements are λϕ, then J elements ϕλ
        for j = 1:Nj
            for l = 1:Nl
                λ      = ψ_ind[j,l]
                Icoeff = λ+Nf*(ϕ_ind-1)  # λϕ

                λ_new      = ψ_ind[j,mod1(l+l_shift,Nl)]
                Icoeff_new = λ_new+Nf*(ϕ_ind-1)  # λϕ

                Mat[Icoeff_new, Icoeff] = 1

                Icoeff = ϕ_ind+Nf*(λ-1)  # ϕλ
                Icoeff_new = ϕ_ind+Nf*(λ_new-1)  # ϕλ
                Mat[Icoeff_new, Icoeff] = 1
            end
        end

        # Next 1 element is ϕϕ
        Icoeff = ϕ_ind+Nf*(ϕ_ind-1)
        Mat[Icoeff, Icoeff] = 1

        # If the Omega filter exists, add more terms
        if Omega
            # Next J elements are λΩ, then J elements Ωλ
            for j = 1:Nj
                for l = 1:Nl
                    λ      = ψ_ind[j,l]
                    λ_new      = ψ_ind[j,mod1(l+l_shift,Nl)]
                    Icoeff = λ+Nf*(Ω_ind-1)  # λΩ
                    Icoeff_new = λ_new+Nf*(Ω_ind-1)  # λΩ
                    Mat[Icoeff_new, Icoeff] = 1

                    Iiso   = I0+Nj+j
                    Icoeff = Ω_ind+Nf*(λ-1)  # Ωλ
                    Icoeff_new = Ω_ind+Nf*(λ_new-1)  # Ωλ
                    Mat[Icoeff_new, Icoeff] = 1
                end
            end
            # Next 3 elements are ϕΩ, Ωϕ, ΩΩ
            Mat[ϕ_ind+Nf*(Ω_ind-1), ϕ_ind+Nf*(Ω_ind-1)] = 1
            Mat[Ω_ind+Nf*(ϕ_ind-1), Ω_ind+Nf*(ϕ_ind-1)] = 1
            Mat[Ω_ind+Nf*(Ω_ind-1), Ω_ind+Nf*(Ω_ind-1)] = 1
        end

        return sparse(Mat)
    end

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

## Core compute function

    function DHC_compute(image::Array{Float64,2}, filter_hash::Dict, filter_hash2::Dict=filter_hash;
        doS2::Bool=true, doS12::Bool=false, doS20::Bool=false, norm=true, iso=false, FFTthreads=2)
        # image        - input for WST
        # filter_hash  - filter hash from fink_filter_hash
        # filter_hash2 - filters for second order.  Default to same as first order.
        # doS2         - compute S2 coeffs
        # doS12        - compute S2 coeffs
        # doS20        - compute S2 coeffs
        # norm         - scale to mean zero, unit variance
        # iso          - sum over angles to obtain isotropic coeffs

        # Use 2 threads for FFT
        FFTW.set_num_threads(FFTthreads)

        # array sizes
        (Nx, Ny)  = size(image)
        if Nx != Ny error("Input image must be square") end
        (Nf, )    = size(filter_hash["filt_index"])
        if Nf == 0  error("filter hash corrupted") end
        @assert Nx==filter_hash["npix"] "Filter size should match npix"
        @assert Nx==filter_hash2["npix"] "Filter2 size should match npix"

        # allocate coeff arrays
        out_coeff = []
        S0  = zeros(Float64, 2)
        S1  = zeros(Float64, Nf)
        if doS2  S2  = zeros(Float64, Nf, Nf) end  # traditional 2nd order
        if doS12 S12 = zeros(Float64, Nf, Nf) end  # Fourier correlation
        if doS20 S20 = zeros(Float64, Nf, Nf) end  # real space correlation
        anyM2 = doS2 | doS12 | doS20
        anyrd = doS2 | doS20             # compute real domain with iFFT

        # allocate image arrays for internal use
        if doS12 im_fdf_0_1 = zeros(Float64,           Nx, Ny, Nf) end   # this must be zeroed!
        if anyrd im_rd_0_1  = Array{Float64, 3}(undef, Nx, Ny, Nf) end

        ## 0th Order
        S0[1]   = mean(image)
        norm_im = image.-S0[1]
        S0[2]   = sum(norm_im .* norm_im)/(Nx*Ny)
        if norm
            norm_im ./= sqrt(Nx*Ny*S0[2])
        else
            norm_im = copy(image)
        end

        append!(out_coeff,S0[:])

        ## 1st Order
        im_fd_0 = fft(norm_im)  # total power=1.0

        # unpack filter_hash
        f_ind   = filter_hash["filt_index"]  # (J, L) array of filters represented as index value pairs
        f_val   = filter_hash["filt_value"]

        zarr = zeros(ComplexF64, Nx, Ny)  # temporary array to fill with zvals

        # make a FFTW "plan" for an array of the given size and type
        if anyrd
            P = plan_ifft(im_fd_0) end  # P is an operator, P*im is ifft(im)

        ## Main 1st Order and Precompute 2nd Order
        for f = 1:Nf
            S1tot = 0.0
            f_i = f_ind[f]  # CartesianIndex list for filter
            f_v = f_val[f]  # Values for f_i
            # for (ind, val) in zip(f_i, f_v)   # this is slower!
            if length(f_i) > 0
                for i = 1:length(f_i)
                    ind       = f_i[i]
                    zval      = f_v[i] * im_fd_0[ind]
                    S1tot    += abs2(zval)
                    zarr[ind] = zval        # filter*image in Fourier domain
                    if doS12 im_fdf_0_1[ind,f] = abs(zval) end
                end
                S1[f] = S1tot/(Nx*Ny)  # image power
                if anyrd
                    im_rd_0_1[:,:,f] .= abs2.(P*zarr) end
                zarr[f_i] .= 0
            end
        end

        append!(out_coeff, iso ? filter_hash["S1_iso_mat"]*S1 : S1)


        # we stored the abs()^2, so take sqrt (this is faster to do all at once)
        if anyrd im_rd_0_1 .= sqrt.(im_rd_0_1) end

        Mat2 = filter_hash["S2_iso_mat"]
        if doS2
            f_ind2   = filter_hash2["filt_index"]  # (J, L) array of filters represented as index value pairs
            f_val2   = filter_hash2["filt_value"]

            ## Traditional second order
            for f1 = 1:Nf
                thisim = fft(im_rd_0_1[:,:,f1])  # Could try rfft here
                # println("  f1",f1,"  sum(fft):",sum(abs2.(thisim))/Nx^2, "  sum(im): ",sum(abs2.(im_rd_0_1[:,:,f1])))
                # Loop over f2 and do second-order convolution
                for f2 = 1:Nf
                    f_i = f_ind2[f2]  # CartesianIndex list for filter
                    f_v = f_val2[f2]  # Values for f_i
                    # sum im^2 = sum(|fft|^2/npix)
                    S2[f1,f2] = sum(abs2.(f_v .* thisim[f_i]))/(Nx*Ny)
                end
            end
            append!(out_coeff, iso ? Mat2*S2[:] : S2[:])
        end

        # Fourier domain 2nd order
        if doS12
            Amat = reshape(im_fdf_0_1, Nx*Ny, Nf)
            S12  = Amat' * Amat
            append!(out_coeff, iso ? Mat2*S12[:] : S12[:])
        end

        # Real domain 2nd order
        if doS20
            Amat = reshape(im_rd_0_1, Nx*Ny, Nf)
            S20  = Amat' * Amat
            append!(out_coeff, iso ? Mat2*S20[:] : S20[:])
        end

        return out_coeff
    end

    function DHC_compute_RGB(image::Array{Float64}, filter_hash::Dict, filter_hash2::Dict=filter_hash;
        doS20::Bool=true, norm=true, iso=false, FFTthreads=2)
        # image        - input for WST
        # filter_hash  - filter hash from fink_filter_hash
        # filter_hash2 - filters for second order.  Default to same as first order.
        # doS2         - compute S2 coeffs
        # doS12        - compute S2 coeffs
        # doS20        - compute S2 coeffs
        # norm         - scale to mean zero, unit variance
        # iso          - sum over angles to obtain isotropic coeffs

        # Use 2 threads for FFT
        FFTW.set_num_threads(FFTthreads)

        # array sizes
        (Nx, Ny, Nc)  = size(image)
        if Nx != Ny error("Input image must be square") end
        (Nf, )    = size(filter_hash["filt_index"])
        if Nf == 0  error("filter hash corrupted") end
        @assert Nx==filter_hash["npix"] "Filter size should match npix"
        @assert Nx==filter_hash2["npix"] "Filter2 size should match npix"

        # allocate coeff arrays
        out_coeff = []
        S0  = zeros(Float64, Nc*2)
        S1  = zeros(Float64, Nc*Nf)

        if doS20 S20 = zeros(Float64, Nf, Nf) end  # real space correlatio
        # anyM2 = doS2 | doS12 | doS20
        anyrd = doS20 #| doS2             # compute real domain with iFFT

        # allocate image arrays for internal use
        mean_im = zeros(Float64,1,1,Nc)
        pwr_im = zeros(Float64,1,1,Nc)
        norm_im = zeros(Float64,Nx,Ny,Nc)
        im_fd_0 = zeros(ComplexF64, Nx, Ny, Nc)
        im_fd_0_sl = zeros(ComplexF64, Nx, Ny)

        if doS20
            Amat1 = zeros(Nx*Ny, Nf)
            Amat2 = zeros(Nx*Ny, Nf)
        end

        if anyrd im_rd_0_1  = Array{Float64, 4}(undef, Nx, Ny, Nf, Nc) end

        ## 0th Order
        mean_im = mean(image, dims=(1,2))
        S0[1:Nc]   = dropdims(mean_im,dims=(1,2))
        norm_im = image.-mean_im
        pwr_im = sum(norm_im .* norm_im,dims=(1,2))
        S0[1+Nc:end]   = dropdims(pwr_im,dims=(1,2))./(Nx*Ny)
        if norm
            norm_im ./= sqrt.(pwr_im)
        else
            norm_im = copy(image)
        end

        append!(out_coeff,S0[:])

        ## 1st Order
        im_fd_0 .= fft(norm_im,(1,2))  # total power=1.0

        # unpack filter_hash
        f_ind   = filter_hash["filt_index"]  # (J, L) array of filters represented as index value pairs
        f_val   = filter_hash["filt_value"]

        zarr = zeros(ComplexF64, Nx, Ny)  # temporary array to fill with zvals

        # make a FFTW "plan" for an array of the given size and type
        if anyrd
            P = plan_ifft(zarr) end  # P is an operator, P*im is ifft(im)

        ## Main 1st Order and Precompute 2nd Order
        for f = 1:Nf
            S1tot = 0.0
            f_i = f_ind[f]  # CartesianIndex list for filter
            f_v = f_val[f]  # Values for f_i
            # for (ind, val) in zip(f_i, f_v)   # this is slower!
            for chan = 1:Nc
                im_fd_0_sl .= im_fd_0[:,:,chan]
                if length(f_i) > 0
                    for i = 1:length(f_i)
                        ind       = f_i[i]
                        zval      = f_v[i] * im_fd_0_sl[ind]
                        S1tot    += abs2(zval)
                        zarr[ind] = zval        # filter*image in Fourier domain
                    end
                    S1[f+(chan-1)*Nf] = S1tot/(Nx*Ny)  # image power
                    if anyrd
                        im_rd_0_1[:,:,f,chan] .= abs2.(P*zarr) end
                    zarr[f_i] .= 0
                end
            end
        end

        if iso
            S1M = filter_hash["S1_iso_mat"]
            M1 = blockdiag(S1M,S1M,S1M)
        end
        append!(out_coeff, iso ? M1*S1 : S1)

        # we stored the abs()^2, so take sqrt (this is faster to do all at once)
        if anyrd im_rd_0_1 .= sqrt.(im_rd_0_1) end

        # Real domain 2nd order
        if doS20
            for chan1 = 1:Nc
                for chan2 = 1:Nc
                    Amat1 = reshape(im_rd_0_1[:,:,:,chan1], Nx*Ny, Nf)
                    Amat2 = reshape(im_rd_0_1[:,:,:,chan2], Nx*Ny, Nf)
                    S20  = Amat1' * Amat2
                    append!(out_coeff, iso ? filter_hash["S2_iso_mat"]*S20[:] : S20[:])
                end
            end
        end

        return out_coeff
    end

    function DHC_compute_gpu_fake(image::Array{Float64,2}, filter_hash::Dict, filter_hash2::Dict=filter_hash;
        doS2::Bool=true, doS12::Bool=false, doS20::Bool=false, norm=true, iso=false,batch_mode::Bool=false,opt_memory::Bool=false,max_memory::Int64=0)

        @assert !iso "Isotropic is not implemented on GPU"
        @assert !doS20 "S20 is not supported on GPU"
        # image        - input for WST
        # filter_hash  - filter hash from fink_filter_hash
        # filter_hash2 - filters for second order.  Default to same as first order.
        # doS2         - compute S2 coeffs Similar to Mallat(2014), Abs in real space
        # doS12        - compute S2 coeffs Abs in Fourier space
        # doS20        - compute S2 coeffs Square in real space
        # norm         - scale to mean zero, unit variance
        # iso          - sum over angles to obtain isotropic coeffs


        # array sizes
        (Nx, Ny)  = size(image)
        if Nx != Ny error("Input image must be square") end
        (Nf, )    = size(filter_hash["filt_vals"])
        if Nf == 0  error("filter hash corrupted") end
        @assert Nx==filter_hash["npix"] "Filter size should match npix"
        @assert Nx==filter_hash2["npix"] "Filter2 size should match npix"

        # allocate coeff arrays
        out_coeff = []
        S0  = zeros(Float64, 2)
        S1  = zeros(Float64, Nf)
        if doS2  S2  = zeros(Float64, Nf, Nf) end  # traditional 2nd order
        if doS12 S12 = zeros(Float64, Nf, Nf) end  # Fourier correlation
        if doS20 S20 = zeros(Float64, Nf, Nf) end  # real space correlation
        anyM2 = doS2 | doS12 | doS20
        anyrd = doS2 | doS20             # compute real domain with iFFT

        # allocate image arrays for internal use
        if doS12 im_fdf_0_1 = zeros(Float64,           Nx, Ny) end   # this must be zeroed!
        if anyrd im_rd_0_1  = Array{Float64, 2}(undef, Nx, Ny) end

        image= image
        ## 0th Order
        S0[1]   = mean(image)
        norm_im = image.-S0[1]
        S0[2]   = sum(norm_im .* norm_im)/(Nx*Ny)
        if norm
            norm_im./= sqrt(Nx*Ny*S0[2])
        else
            norm_im = image
        end


        ## 1st Order
        #REview: We fft shift!
        im_fd_0 = FFTW.fftshift(FFTW.fft(norm_im))  # total power=1.0

        # unpack filter_hash
        f_mms   = filter_hash["filt_mms"]  # (J, L) array of filters represented as index value pairs
        f_vals   = filter_hash["filt_vals"]

        zarr = zeros(ComplexF64, Nx, Ny)  # temporary array to fill with zvals

        # make a FFTW "plan" for an array of the given size and type
        if anyrd
            P = FFTW.plan_ifft(im_fd_0) end  # P is an operator, P*im is ifft(im)

        ## Main 1st Order and Precompute 2nd Order
        for f1 = 1:Nf
            f_mm1 = f_mms[:,:,f1]  # Min,Max
            f_val1 = f_vals[f1]  # values

            #Review: Removes length check-> empty filter case???, I think this should be prevented beforehands
            zv=f_val1.*im_fd_0[f_mm1[1,1]:f_mm1[1,2],f_mm1[2,1]:f_mm1[2,2]] #dense multiplication for the filter region
            zarr[f_mm1[1,1]:f_mm1[1,2],f_mm1[2,1]:f_mm1[2,2]]=zv

            if doS12 im_fdf_0_1[f_mm1[1,1]:f_mm1[1,2],f_mm1[2,1]:f_mm1[2,2]] .= abs.(zv) end

            S1[f1] = sum(abs2.(zv))/(Nx*Ny)  # image power
            if anyrd im_rd_0_1.= abs2.(P*FFTW.ifftshift(zarr)) end
            zarr[f_mm1[1,1]:f_mm1[1,2],f_mm1[2,1]:f_mm1[2,2]] .= 0


            im_rd_0_1 .= sqrt.(im_rd_0_1)
            im_fdf_1_1 = FFTW.fftshift(FFTW.fft(im_rd_0_1))
            for f2 = 1:Nf
                f_mm2 = f_mms[:,:,f2]
                f_val2 = f_vals[f2]

                S2[f1,f2] = sum(abs2.(f_val2 .* im_fdf_1_1[f_mm2[1,1]:f_mm2[1,2],f_mm2[2,1]:f_mm2[2,2]]))/(Nx*Ny)
                if !doS12
                    continue
                end

                if f2>f1
                    S12[f1,f2]=NaN
                    continue
                end

                if f_mm1[1,1]>f_mm2[1,2] || f_mm2[1,1]>f_mm1[1,2] || f_mm1[2,1]>f_mm2[2,2] || f_mm2[2,1]>f_mm1[2,2]
                    S12[f1,f2]=NaN
                    continue
                end

                mx=max(f_mm1[1,1],f_mm2[1,1])
                my=max(f_mm1[2,1],f_mm2[2,1])
                Mx=min(f_mm1[1,2],f_mm2[1,2])
                My=min(f_mm1[2,2],f_mm2[2,2])
                sfx=Mx-mx
                sfy=My-my
                st1x=mx-f_mm1[1,1]+1
                st1y=my-f_mm1[2,1]+1
                st2x=mx-f_mm2[1,1]+1
                st2y=my-f_mm2[2,1]+1

                S12[f1,f2]=sum(f_val1[st1x:st1x+sfx,st1y:st1y+sfy].*f_val2[st2x:st2x+sfx,st2y:st2y+sfy].*abs2.(im_fd_0[mx:Mx,my:My]))
            end
        end
        append!(out_coeff,S0)
        append!(out_coeff, iso ? filter_hash["S1_iso_mat"]*S1 : S1)
        if doS2 append!(out_coeff, iso ? filter_hash["S1_iso_mat"]*S2 : S2) end
        if doS12 append!(out_coeff, iso ? filter_hash["S1_iso_mat"]*S12 : S12) end
        return out_coeff
    end

    function DHC_compute_gpu(image::Array{Float64,2}, filter_hash::Dict, filter_hash2::Dict=filter_hash;
        doS2::Bool=true, doS12::Bool=false, doS20::Bool=false, norm=true, iso=false,batch_mode::Bool=false,opt_memory::Bool=false,max_memory::Int64=0)

        @assert !iso "Isotropic is not implemented on GPU"
        @assert !doS20 "S20 is not supported on GPU"
        # image        - input for WST
        # filter_hash  - filter hash from fink_filter_hash
        # filter_hash2 - filters for second order.  Default to same as first order.
        # doS2         - compute S2 coeffs Similar to Mallat(2014), Abs in real space
        # doS12        - compute S2 coeffs Abs in Fourier space
        # doS20        - compute S2 coeffs Square in real space
        # norm         - scale to mean zero, unit variance
        # iso          - sum over angles to obtain isotropic coeffs


        # array sizes
        (Nx, Ny)  = size(image)
        if Nx != Ny error("Input image must be square") end
        (Nf, )    = size(filter_hash["filt_vals"])
        if Nf == 0  error("filter hash corrupted") end
        @assert Nx==filter_hash["npix"] "Filter size should match npix"
        @assert Nx==filter_hash2["npix"] "Filter2 size should match npix"

        # allocate coeff arrays
        out_coeff = []
        S0  = zeros(Float64, 2)
        S1  = zeros(Float64, Nf)
        if doS2  S2  = zeros(Float64, Nf, Nf) end  # traditional 2nd order
        if doS12 S12 = zeros(Float64, Nf, Nf) end  # Fourier correlation
        if doS20 S20 = zeros(Float64, Nf, Nf) end  # real space correlation
        anyM2 = doS2 | doS12 | doS20
        anyrd = doS2 | doS20             # compute real domain with iFFT

        # allocate image arrays for internal use
        if doS12 im_fdf_0_1 = CUDA.zeros(Float64,           Nx, Ny) end   # this must be zeroed!
        if anyrd im_rd_0_1  = CUDA.CuArray(Array{Float64, 2}(undef, Nx, Ny)) end

        image= CUDA.CuArray(image)
        ## 0th Order
        S0[1]   = mean(image)
        norm_im = image.-S0[1]
        S0[2]   = sum(norm_im .* norm_im)/(Nx*Ny)
        if norm
            norm_im./= sqrt(Nx*Ny*S0[2])
        else
            norm_im = image
        end


        ## 1st Order
        #REview: We fft shift!
        im_fd_0 = CUDA.CUFFT.fftshift(CUDA.CUFFT.fft(norm_im))  # total power=1.0

        # unpack filter_hash
        f_mms   = filter_hash["filt_mms"]  # (J, L) array of filters represented as index value pairs
        f_vals   = filter_hash["filt_vals"]

        zarr = CUDA.zeros(ComplexF64, Nx, Ny)  # temporary array to fill with zvals

        # make a FFTW "plan" for an array of the given size and type
        if anyrd
            P = CUDA.CUFFT.plan_ifft(im_fd_0) end  # P is an operator, P*im is ifft(im)

        ## Main 1st Order and Precompute 2nd Order
        for f1 = 1:Nf
            f_mm1 = f_mms[:,:,f1]  # Min,Max
            f_val1 = f_vals[f1]  # values

            #Review: Removes length check-> empty filter case???, I think this should be prevented beforehands
            zv=f_val1.*im_fd_0[f_mm1[1,1]:f_mm1[1,2],f_mm1[2,1]:f_mm1[2,2]] #dense multiplication for the filter region
            zarr[f_mm1[1,1]:f_mm1[1,2],f_mm1[2,1]:f_mm1[2,2]]=zv

            if doS12 im_fdf_0_1[f_mm1[1,1]:f_mm1[1,2],f_mm1[2,1]:f_mm1[2,2]] .= abs.(zv) end

            S1[f1] = sum(abs2.(zv))/(Nx*Ny)  # image power
            if anyrd im_rd_0_1.= abs2.(P*CUDA.CUFFT.ifftshift(zarr)) end
            zarr[f_mm1[1,1]:f_mm1[1,2],f_mm1[2,1]:f_mm1[2,2]] .= 0


            im_rd_0_1 .= sqrt.(im_rd_0_1)
            im_fdf_1_1 = CUDA.CUFFT.fftshift(CUDA.CUFFT.fft(im_rd_0_1))
            for f2 = 1:Nf
                f_mm2 = f_mms[:,:,f2]
                f_val2 = f_vals[f2]

                S2[f1,f2] = sum(abs2.(f_val2 .* im_fdf_1_1[f_mm2[1,1]:f_mm2[1,2],f_mm2[2,1]:f_mm2[2,2]]))/(Nx*Ny)
                if !doS12
                    continue
                end

                if f2>f1
                    S12[f1,f2]=NaN
                    continue
                end

                if f_mm1[1,1]>f_mm2[1,2] || f_mm2[1,1]>f_mm1[1,2] || f_mm1[2,1]>f_mm2[2,2] || f_mm2[2,1]>f_mm1[2,2]
                    S12[f1,f2]=NaN
                    continue
                end

                mx=max(f_mm1[1,1],f_mm2[1,1])
                my=max(f_mm1[2,1],f_mm2[2,1])
                Mx=min(f_mm1[1,2],f_mm2[1,2])
                My=min(f_mm1[2,2],f_mm2[2,2])
                sfx=Mx-mx
                sfy=My-my
                st1x=mx-f_mm1[1,1]+1
                st1y=my-f_mm1[2,1]+1
                st2x=mx-f_mm2[1,1]+1
                st2y=my-f_mm2[2,1]+1

                S12[f1,f2]=sum(f_val1[st1x:st1x+sfx,st1y:st1y+sfy].*f_val2[st2x:st2x+sfx,st2y:st2y+sfy].*abs2.(im_fd_0[mx:Mx,my:My]))
            end
        end
        append!(out_coeff,S0)
        append!(out_coeff, iso ? filter_hash["S1_iso_mat"]*S1 : S1)
        if doS2 append!(out_coeff, iso ? filter_hash["S1_iso_mat"]*S2 : S2) end
        if doS12 append!(out_coeff, iso ? filter_hash["S1_iso_mat"]*S12 : S12) end
        return out_coeff
    end

    function DHC_compute_3d(image::Array{Float64,3}, filter_hash; FFTthreads=2)
        # Use 2 threads for FFT
        FFTW.set_num_threads(FFTthreads)

        # array sizes
        (Nx, Ny, Nz)  = size(image)
        (Nf, )    = size(filter_hash["filt_index"])

        # allocate coeff arrays
        out_coeff = []
        S0  = zeros(Float64, 2)
        S1  = zeros(Float64, Nf)
        S20 = zeros(Float64, Nf, Nf)
        S12 = zeros(Float64, Nf, Nf)
        S2  = zeros(Float64, Nf, Nf)  # traditional 2nd order

        # allocate image arrays for internal use
        im_fdf_0_1 = zeros(Float64,           Nx, Ny, Nz, Nf)   # this must be zeroed!
        im_rd_0_1  = Array{Float64, 4}(undef, Nx, Ny, Nz, Nf)

        ## 0th Order
        S0[1]   = mean(image)
        norm_im = image.-S0[1]
        S0[2]   = sum(norm_im .* norm_im)/(Nx*Ny*Nz)
        norm_im ./= sqrt(Nx*Ny*Nz*S0[2])

        append!(out_coeff,S0[:])

        ## 1st Order
        im_fd_0 = fft(norm_im)  # total power=1.0

        # unpack filter_hash
        f_ind   = filter_hash["filt_index"]  # (J, L) array of filters represented as index value pairs
        f_val   = filter_hash["filt_value"]

        #f_ind2   = filter_hash2["filt_index"]  # (J, L) array of filters represented as index value pairs
        #f_val2   = filter_hash2["filt_value"]

        zarr = zeros(ComplexF64, Nx, Ny, Nz)  # temporary array to fill with zvals

        # make a FFTW "plan" for an array of the given size and type
        P   = plan_ifft(im_fd_0)   # P is an operator, P*im is ifft(im)

        ## Main 1st Order and Precompute 2nd Order
        for f = 1:Nf
            S1tot = 0.0
            f_i = f_ind[f]  # CartesianIndex list for filter
            f_v = f_val[f]  # Values for f_i
            # for (ind, val) in zip(f_i, f_v)   # this is slower!
            if length(f_i) > 0
                for i = 1:length(f_i)
                    ind       = f_i[i]
                    zval      = f_v[i] * im_fd_0[ind]
                    S1tot    += abs2(zval)
                    zarr[ind] = zval
                    im_fdf_0_1[ind,f] = abs(zval)
                end
                S1[f] = S1tot/(Nx*Ny*Nz)  # image power ###why do we need to normalize again?
                im_rd_0_1[:,:,:,f] .= abs2.(P*zarr)
                zarr[f_i] .= 0
            end
        end
        append!(out_coeff, S1[:])

        # we stored the abs()^2, so take sqrt (this is faster to do all at once)
        im_rd_0_1 .= sqrt.(im_rd_0_1)

        ## 2nd Order
        #Amat = reshape(im_fdf_0_1, Nx*Ny, Nf)
        #S12  = Amat' * Amat
        #Amat = reshape(im_rd_0_1, Nx*Ny, Nf)
        #S20  = Amat' * Amat

        #append!(out_coeff, S20)
        #append!(out_coeff, S12)

        ## Traditional second order
        for f1 = 1:Nf
            thisim = fft(im_rd_0_1[:,:,:,f1])  # Could try rfft here
            # println("  f1",f1,"  sum(fft):",sum(abs2.(thisim))/Nx^2, "  sum(im): ",sum(abs2.(im_rd_0_1[:,:,f1])))
            # Loop over f2 and do second-order convolution
            for f2 = 1:Nf
                f_i = f_ind[f2]  # CartesianIndex list for filter
                f_v = f_val[f2]  # Values for f_i
                # sum im^2 = sum(|fft|^2/npix)
                S2[f1,f2] = sum(abs2.(f_v .* thisim[f_i]))/(Nx*Ny*Nz)
            end
        end
        append!(out_coeff, S2)

        return out_coeff
    end

    function DHC_compute_3d_gpu(image::Array{Float64,3}, filter_hash;doS2::Bool=true, doS12::Bool=false, norm=true)
        CUDA.reclaim()
        # array sizes
        (Nx, Ny, Nz)  = size(image)
        (Nf, )    = size(filter_hash["filt_vals"])

        # allocate coeff arrays
        out_coeff = []
        S0  = zeros(Float64, 2)
        S1  = zeros(Float64, Nf)
        #S20 = zeros(Float64, Nf, Nf)
        if doS12 S12 = zeros(Float64, Nf, Nf) end
        if doS2 S2  = zeros(Float64, Nf, Nf)  end # traditional 2nd order

        # allocate image arrays for internal use
        im_fdf_0_1 = CUDA.CuArray(Array{Float64, 3}(undef, Nx, Ny, Nz))
        im_rd_0_1  = CUDA.CuArray(Array{Float64, 3}(undef, Nx, Ny, Nz))

        image=CUDA.CuArray(image)

        ## 0th Order
        S0[1]   = mean(image)
        norm_im = image.-S0[1]
        S0[2]   = sum(norm_im .* norm_im)/(Nx*Ny*Nz)
        if norm
            norm_im ./= sqrt(Nx*Ny*Nz*S0[2])
        else
            norm_im=image
        end

        ## 1st Order
        im_fd_0 = CUDA.CUFFT.fftshift(CUDA.CUFFT.fft(norm_im))  # total power=1.0

        # unpack filter_hash
        f_mms   = filter_hash["filt_mms"]  # (J, L) array of filters represented as index value pairs
        f_vals   = filter_hash["filt_vals"]

        #f_ind2   = filter_hash2["filt_index"]  # (J, L) array of filters represented as index value pairs
        #f_val2   = filter_hash2["filt_value"]

        zarr = CUDA.zeros(ComplexF64, Nx, Ny, Nz)  # temporary array to fill with zvals

        # make a FFTW "plan" for an array of the given size and type
        P   = CUDA.CUFFT.plan_ifft(im_fd_0)   # P is an operator, P*im is ifft(im)

        #depth first search
        for f1 = 1:Nf
            im_fdf_0_1.=0.
            f_mm1 = f_mms[:,:,f1]  # CartesianIndex list for filter
            f_val1 = f_vals[f1]  # Values for f_i
            # for (ind, val) in zip(f_i, f_v)   # this is slower!

            zv=f_val1.*im_fd_0[f_mm1[1,1]:f_mm1[1,2],f_mm1[2,1]:f_mm1[2,2],f_mm1[3,1]:f_mm1[3,2]]
            S1[f1] = sum(abs2.(zv))/(Nx*Ny*Nz)

            zarr[f_mm1[1,1]:f_mm1[1,2],f_mm1[2,1]:f_mm1[2,2],f_mm1[3,1]:f_mm1[3,2]]=zv

            im_fdf_0_1[f_mm1[1,1]:f_mm1[1,2],f_mm1[2,1]:f_mm1[2,2],f_mm1[3,1]:f_mm1[3,2]] .= abs.(zv)


            im_rd_0_1=abs2.(P*CUDA.CUFFT.ifftshift(zarr))

            zarr[f_mm1[1,1]:f_mm1[1,2],f_mm1[2,1]:f_mm1[2,2],f_mm1[3,1]:f_mm1[3,2]] .= 0.

            im_rd_0_1 .= sqrt.(im_rd_0_1)
            im_fdf_1_1 = CUDA.CUFFT.fftshift(CUDA.CUFFT.fft(im_rd_0_1))
            for f2 = 1:Nf
                f_mm2 = f_mms[:,:,f2]
                f_val2 = f_vals[f2]

                S2[f1,f2] = sum(abs2.(f_val2 .* im_fdf_1_1[f_mm2[1,1]:f_mm2[1,2],f_mm2[2,1]:f_mm2[2,2],f_mm2[3,1]:f_mm2[3,2]] ))/(Nx*Ny*Nz)

                if !doS12
                    continue
                end

                if f2>f1
                    S12[f1,f2]=NaN
                    continue
                end

                if f_mm1[1,1]>f_mm2[1,2] || f_mm2[1,1]>f_mm1[1,2] || f_mm1[2,1]>f_mm2[2,2] || f_mm2[2,1]>f_mm1[2,2] || f_mm1[3,1]>f_mm2[3,2] || f_mm2[3,1]>f_mm1[3,2]
                    S12[f1,f2]=NaN
                    continue
                end

                mx=max(f_mm1[1,1],f_mm2[1,1])
                my=max(f_mm1[2,1],f_mm2[2,1])
                mz=max(f_mm1[3,1],f_mm2[3,1])
                Mx=min(f_mm1[1,2],f_mm2[1,2])
                My=min(f_mm1[2,2],f_mm2[2,2])
                Mz=min(f_mm1[3,2],f_mm2[3,2])
                sfx=Mx-mx
                sfy=My-my
                sfz=Mz-mz
                st1x=mx-f_mm1[1,1]+1
                st1y=my-f_mm1[2,1]+1
                st1z=mz-f_mm1[3,1]+1
                st2x=mx-f_mm2[1,1]+1
                st2y=my-f_mm2[2,1]+1
                st2z=mz-f_mm2[3,1]+1

                S12[f1,f2]=sum(f_val1[st1x:st1x+sfx,st1y:st1y+sfy,st1z:st1z+sfz].*f_val2[st2x:st2x+sfx,st2y:st2y+sfy,st2z:st2z+sfz].*abs2.(im_fd_0[mx:Mx,my:My,mz:Mz]))
                CUDA.reclaim()
            end
            CUDA.reclaim()
        end
        append!(out_coeff,S0)
        append!(out_coeff, S1)
        if doS2 append!(out_coeff, S2) end
        if doS12 append!(out_coeff, S12) end
        return out_coeff
    end

    function DHC_compute_3d_gpu_fake(image::Array{Float64,3}, filter_hash;doS2::Bool=true, doS12::Bool=false, norm=true)
        # array sizes
        (Nx, Ny, Nz)  = size(image)
        (Nf, )    = size(filter_hash["filt_vals"])

        # allocate coeff arrays
        out_coeff = []
        S0  = zeros(Float64, 2)
        S1  = zeros(Float64, Nf)
        #S20 = zeros(Float64, Nf, Nf)
        if doS12 S12 = zeros(Float64, Nf, Nf) end
        if doS2 S2  = zeros(Float64, Nf, Nf)  end # traditional 2nd order

        # allocate image arrays for internal use
        im_fdf_0_1 = Array{Float64, 3}(undef, Nx, Ny, Nz)
        im_rd_0_1  = Array{Float64, 3}(undef, Nx, Ny, Nz)


        ## 0th Order
        S0[1]   = mean(image)
        norm_im = image.-S0[1]
        S0[2]   = sum(norm_im .* norm_im)/(Nx*Ny*Nz)
        if norm
            norm_im ./= sqrt(Nx*Ny*Nz*S0[2])
        else
            norm_im=image
        end

        ## 1st Order
        im_fd_0 = FFTW.fftshift(FFTW.fft(norm_im))  # total power=1.0

        # unpack filter_hash
        f_mms   = filter_hash["filt_mms"]  # (J, L) array of filters represented as index value pairs
        f_vals   = filter_hash["filt_vals"]

        #f_ind2   = filter_hash2["filt_index"]  # (J, L) array of filters represented as index value pairs
        #f_val2   = filter_hash2["filt_value"]

        zarr = zeros(ComplexF64, Nx, Ny, Nz)  # temporary array to fill with zvals

        # make a FFTW "plan" for an array of the given size and type
        P   = FFTW.plan_ifft(im_fd_0)   # P is an operator, P*im is ifft(im)

        #depth first search
        for f1 = 1:Nf
            im_fdf_0_1.=0.
            f_mm1 = f_mms[:,:,f1]  # CartesianIndex list for filter
            f_val1 = f_vals[f1]  # Values for f_i
            # for (ind, val) in zip(f_i, f_v)   # this is slower!

            zv=f_val1.*im_fd_0[f_mm1[1,1]:f_mm1[1,2],f_mm1[2,1]:f_mm1[2,2],f_mm1[3,1]:f_mm1[3,2]]
            S1[f1] = sum(abs2.(zv))/(Nx*Ny*Nz)

            zarr[f_mm1[1,1]:f_mm1[1,2],f_mm1[2,1]:f_mm1[2,2],f_mm1[3,1]:f_mm1[3,2]]=zv

            im_fdf_0_1[f_mm1[1,1]:f_mm1[1,2],f_mm1[2,1]:f_mm1[2,2],f_mm1[3,1]:f_mm1[3,2]] .= abs.(zv)


            im_rd_0_1=abs2.(P*FFTW.ifftshift(zarr))

            zarr[f_mm1[1,1]:f_mm1[1,2],f_mm1[2,1]:f_mm1[2,2],f_mm1[3,1]:f_mm1[3,2]] .= 0.

            im_rd_0_1 .= sqrt.(im_rd_0_1)
            im_fdf_1_1 = FFTW.fftshift(FFTW.fft(im_rd_0_1))
            for f2 = 1:Nf
                f_mm2 = f_mms[:,:,f2]
                f_val2 = f_vals[f2]

                S2[f1,f2] = sum(abs2.(f_val2 .* im_fdf_1_1[f_mm2[1,1]:f_mm2[1,2],f_mm2[2,1]:f_mm2[2,2],f_mm2[3,1]:f_mm2[3,2]] ))/(Nx*Ny*Nz)

                if !doS12
                    continue
                end

                if f2>f1
                    S12[f1,f2]=NaN
                    continue
                end

                if f_mm1[1,1]>f_mm2[1,2] || f_mm2[1,1]>f_mm1[1,2] || f_mm1[2,1]>f_mm2[2,2] || f_mm2[2,1]>f_mm1[2,2] || f_mm1[3,1]>f_mm2[3,2] || f_mm2[3,1]>f_mm1[3,2]
                    S12[f1,f2]=NaN
                    continue
                end

                mx=max(f_mm1[1,1],f_mm2[1,1])
                my=max(f_mm1[2,1],f_mm2[2,1])
                mz=max(f_mm1[3,1],f_mm2[3,1])
                Mx=min(f_mm1[1,2],f_mm2[1,2])
                My=min(f_mm1[2,2],f_mm2[2,2])
                Mz=min(f_mm1[3,2],f_mm2[3,2])
                sfx=Mx-mx
                sfy=My-my
                sfz=Mz-mz
                st1x=mx-f_mm1[1,1]+1
                st1y=my-f_mm1[2,1]+1
                st1z=mz-f_mm1[3,1]+1
                st2x=mx-f_mm2[1,1]+1
                st2y=my-f_mm2[2,1]+1
                st2z=mz-f_mm2[3,1]+1

                S12[f1,f2]=sum(f_val1[st1x:st1x+sfx,st1y:st1y+sfy,st1z:st1z+sfz].*f_val2[st2x:st2x+sfx,st2y:st2y+sfy,st2z:st2z+sfz].*abs2.(im_fd_0[mx:Mx,my:My,mz:Mz]))
            end
        end
        append!(out_coeff,S0)
        append!(out_coeff, S1)
        if doS2 append!(out_coeff, S2) end
        if doS12 append!(out_coeff, S12) end
        return out_coeff
    end

## Post processing

    function transformMaker(coeff, S1Mat, S2Mat)
        NS1 = size(S1Mat)[2]
        NS2 = size(S2Mat)[2]
        S1iso = transpose(S1Mat*transpose(coeff[:,2+1:2+NS1]))
        S2iso = transpose(S2Mat*transpose(coeff[:,2+NS1+1:2+NS1+NS2]))
        return hcat(coeff[:,1:2],S1iso,S2iso)
    end

## Filter bank utilities
    function fink_filter_bank(c, L; nx=256, wd=1, pc=1, shift=false, Omega=false, safety_on=true)
        #c     - sets the scale sampling rate (1 is dyadic, 2 is half dyadic)
        #L     - number of angular bins (usually 8*pc or 16*pc)
        #wd    - width of the wavelets (default 1, wd=2 for a double covering)
        #pc    - plane coverage (default 1, full 2pi 2)
        #shift - shift in θ by 1/2 of the θ spacing
        #Omega - true= append Omega filter (all power beyond Nyquist) so the sum of filters is 1.0

        # -------- assertion errors to make sure arguments are reasonable
        @test wd <= L/2

        # -------- set parameters
        dθ   = pc*π/L
        θ_sh = shift ? dθ/2 : 0.0
        dx   = nx/2-1

        im_scale = convert(Int8,log2(nx))
        # -------- number of bins in radial direction (size scales)
        J = (im_scale-2)*c

        # -------- allocate output array of zeros
        filt      = zeros(nx, nx, J*L+(Omega ? 2 : 1))
        psi_index = zeros(Int32, J, L)
        psi_ind_in= zeros(Int32, J*L+(Omega ? 2 : 1), 2)
        theta     = zeros(Float64, L)
        j_value   = zeros(Float64, J)

        # -------- compute the required wd
        j_rad_exp = zeros(J)
        for j_ind = 1:J
            j = j_ind/c
            jrad  = im_scale-j-1
            j_rad_exp[j_ind] = 2^(jrad)
        end

        wd_j = max.(ceil.(L./(pc.*π.*j_rad_exp)),wd)

        if !safety_on
            wd_j.=wd
        end

        # loop over wd from small to large
        ## there is some uneeded redundancy added by doing this esp in l loop
        for wd in sort(unique(wd_j))
            # -------- allocate theta and logr arrays
            θ    = zeros(nx, nx)
            logr = zeros(nx, nx)

            wdθ  = wd*dθ
            norm = 1.0/sqrt(wd)
            # -------- loop over l
            for l = 0:L-1
                θ_l        = dθ*l+θ_sh
                theta[l+1] = θ_l

            # -------- allocate anggood BitArray
                anggood = falses(nx, nx)

            # -------- loop over pixels
                for x = 1:nx
                    sx = mod(x+dx,nx)-dx -1    # define sx,sy so that no fftshift() needed
                    for y = 1:nx
                        sy = mod(y+dx,nx)-dx -1
                        θ_pix  = mod(atan(sy, sx)+π -θ_l, 2*π)
                        θ_good = abs(θ_pix-π) <= wdθ

                        # If this is a pixel we might use, calculate log2(r)
                        if θ_good
                            anggood[y, x] = θ_good
                            θ[y, x]       = θ_pix
                            r2            = sx^2 + sy^2
                            logr[y, x]    = 0.5*log2(max(1,r2))
                        end
                    end
                end
                angmask = findall(anggood)
            # -------- compute the wavelet in the Fourier domain
            # -------- the angular factor is the same for all j
                F_angular = norm .* cos.((θ[angmask].-π).*(L/(2*wd*pc)))

            # -------- loop over j for the radial part
            #    for (j_ind, j) in enumerate(1/c:1/c:im_scale-2)
                j_ind_w_wd = findall(wd_j.==wd)
                for j_ind = j_ind_w_wd
                    j = j_ind/c
                    j_value[j_ind] = j  # store for later
                    jrad  = im_scale-j-1
                    Δj    = abs.(logr[angmask].-jrad)
                    rmask = (Δj .<= 1/c)

            # -------- radial part
                    F_radial = cos.(Δj[rmask] .* (c*π/2))
                    ind      = angmask[rmask]
            #      Let's have these be (J,L) if you reshape...
            #        f_ind    = (j_ind-1)*L+l+1
                    f_ind    = j_ind + l*J
                    filt[ind, f_ind] = F_radial .* F_angular[rmask]
                    psi_index[j_ind,l+1] = f_ind
                    psi_ind_in[f_ind,:] = [j_ind-1,l]
                end
            end
        end

        # -------- phi contains power near k=0 not yet accounted for
        filter_power = (sum(filt.*filt, dims=3))[:,:,1]

        # -------- for plane half-covered (pc=1), add other half-plane
        if pc == 1
            filter_power .+= circshift(filter_power[end:-1:1,end:-1:1],(1,1))
        end

        # -------- compute power required to sum to 1.0
        i0 = round(Int16,nx/2-2)
        i1 = round(Int16,nx/2+4)
        center_power = 1.0 .- fftshift(filter_power)[i0:i1,i0:i1]
        zind = findall(center_power .< 1E-15)
        center_power[zind] .= 0.0  # set small numbers to zero
        phi_cen = zeros(nx, nx)
        phi_cen[i0:i1,i0:i1] = sqrt.(center_power)

        # -------- before adding ϕ to filter bank, renormalize ψ if pc=1
        if pc==1 filt .*= sqrt(2.0) end  # double power for half coverage

        # -------- add result to filter array
        phi_index  = J*L+1
        filt[:,:,phi_index] .= fftshift(phi_cen)
        psi_ind_in[phi_index,:] = [J,0]

        # -------- metadata dictionary
        info=Dict{String,Any}()
        info["npix"]         = nx
        info["j_value"]      = j_value
        info["theta_value"]  = theta
        info["psi_index"]    = psi_index
        info["phi_index"]    = phi_index
        info["J_L"]          = psi_ind_in
        info["pc"]           = pc
        info["wd"]           = wd_j
        info["fs_center_r"]  = j_rad_exp

        if Omega     # append a filter containing the rest (outside Nyquist)
            filter_power += filt[:,:,phi_index].^2
            edge_power    = 1.0 .- filter_power
            zind          = findall(edge_power .< 1E-15)
            edge_power[zind]     .= 0.0  # set small numbers to zero
            Omega_index           = J*L+2
            info["Omega_index"]   = Omega_index
            filt[:,:,Omega_index] = sqrt.(edge_power)
        end


        return filt, info
    end

    ## Make a list of non-zero pixels for the (Fourier plane) filters
    function fink_filter_list(filt)
        (ny,nx,Nf) = size(filt)

        # Allocate output arrays
        filtind = fill(CartesianIndex{2}[], Nf)
        filtval = fill(Float64[], Nf)

        # Loop over filters and record non-zero values
        for l=1:Nf
            f = @view filt[:,:,l]
            ind = findall(f .> 1E-13)
            val = f[ind]
            filtind[l] = ind
            filtval[l] = val
        end
        return [filtind, filtval]
    end

    function fink_filter_box(filt)
        (ny,nx,Nf) = size(filt)

        # Allocate output arrays
        filtmms = Array{Int64}(undef,2,2,Nf)
        filtvals = Array{Any}(undef,Nf)#is this a terrible way to allocate memory?


        # Loop over filters and record non-zero values
        for l=1:Nf
            f = FFTW.fftshift(filt[:,:,l])
            ind = findall(f .> 1E-13)
            ind=getindex.(ind,[1 2])
            mins=minimum(ind,dims=1)
            maxs=maximum(ind,dims=1)

            #val = f[ind]
            #filtind[l] = ind
            filtmms[:,1,l] = mins
            filtmms[:,2,l] = maxs
            filtvals[l]=f[mins[1]:maxs[1],mins[2]:maxs[2]]

        end
        return [filtmms, filtvals]
    end

    function list_to_box(hash;dim=2)#this modifies the hash
        (Nf,) = size(hash["filt_index"])
        nx=hash["npix"]
        if dim==3 nz=hash["nz"] end

        # Allocate output arrays
        filtmms = Array{Int64}(undef,dim,2,Nf)
        filtvals = Array{Any}(undef,Nf)#is this a terrible way to allocate memory?
        if dim==2
            filt=zeros(Float64,nx,nx)
        else
            filt=zeros(Float64,nx,nx,nz)
        end
        # Loop over filters and record non-zero values
        for l=1:Nf
            filt.=0.
            f_i = hash["filt_index"][l]
            v_i=hash["filt_value"][l]
            for i = 1:length(f_i)
                filt[f_i[i]]= v_i[i]
            end
            filt = FFTW.fftshift(filt)

            ind = findall(filt .> 1E-13)
            if dim==2
                ind=getindex.(ind,[1 2])
            else
                ind=getindex.(ind,[1 2 3])
            end
            mins=minimum(ind,dims=1)
            maxs=maximum(ind,dims=1)


            filtmms[:,1,l] = mins
            filtmms[:,2,l] = maxs
            if dim==2
                filtvals[l]=filt[mins[1]:maxs[1],mins[2]:maxs[2]]
            else
                filtvals[l]=filt[mins[1]:maxs[1],mins[2]:maxs[2],mins[3]:maxs[3]]
            end
        end

        hash["filt_index"] = nothing
        hash["filt_value"] = nothing

        hash["filt_mms"] = filtmms
        hash["filt_vals"]=filtvals
        return hash
    end

end # of module
