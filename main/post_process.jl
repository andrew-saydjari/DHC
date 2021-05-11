## Matrix for transformations
function transformMaker(coeff, S1Mat, S2Mat; Nc=1)
    NS1 = size(S1Mat)[2]
    NS2 = size(S2Mat)[2]
    if Nc==1
        S0iso = coeff[:,1:2]
        S1iso = transpose(S1Mat*transpose(coeff[:,2+1:2+NS1]))
        S2iso = transpose(S2Mat*transpose(coeff[:,2+NS1+1:2+NS1+NS2]))
    else
        S0iso = coeff[:,1:2*Nc]
        S1MatChan = blockdiag(collect(Iterators.repeated(S1Mat,Nc))...)
        S2MatChan = blockdiag(collect(Iterators.repeated(S2Mat,Nc*Nc))...)
        S1iso = transpose(S1MatChan*transpose(coeff[:,2*Nc+1:2*Nc+Nc*NS1]))
        S2iso = transpose(S2MatChan*transpose(coeff[:,2*Nc+Nc*NS1+1:end]))
    end
    return hcat(S0iso,S1iso,S2iso)
end

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
    Omega   = haskey(fhash, "Omega_index") & fhash["Omega3d"]

    # unpack fhash
    Nl      = length(fhash["theta_value"])
    Nj      = length(fhash["j_value"])
    Nk      = length(fhash["k_value"])
    Nf      = length(fhash["filt_value"])
    ψ_ind   = fhash["psi_index"]

    # number of iso coefficients
    Niso    = Omega ? (Nj+2)*Nk+3 : (Nj+1)*Nk+1
    Nstep   = Omega ? Nj+2 : Nj+1
    Mat     = zeros(Int32, Niso, Nf)

    # first J elements of iso
    for j = 1:Nj
        for l = 1:Nl
            for k=1:Nk
                λ = ψ_ind[j,l,k]
                Mat[j+(k-1)*Nstep, λ] = 1
            end
        end
    end

    #this is currently how I am storing the phi_k, omega_k
    j = Nj+1
    l = 1
    for k=1:Nk
        λ = ψ_ind[j,l,k]
        Mat[j+(k-1)*Nstep, λ] = 1
    end

    if Omega
        j = Nj+1
        l = 2
        for k=1:Nk
            λ = ψ_ind[j,l,k]
            Mat[j+1+(k-1)*Nstep, λ] = 1
        end
    end

    ### these are the 3d globals

    #phi0
    λ = ψ_ind[Nj+1,1,Nk+1]
    if Omega
        Mat[Niso-2, λ] = 1
    else
        Mat[Niso, λ] = 1
    end

    if Omega
        #omega_0
        λ = ψ_ind[Nj+1,2,Nk+1]
        Mat[Niso-1, λ] = 1

        # omega_3d
        λ = ψ_ind[Nj+1,3,Nk+1]
        Mat[Niso, λ] = 1
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
    Omega   = haskey(fhash, "Omega_index") & fhash["Omega3d"]

    # unpack fhash
    L      = length(fhash["theta_value"])
    J      = length(fhash["j_value"])
    K      = length(fhash["k_value"])
    Nf     = length(fhash["filt_value"])
    ψ_ind   = fhash["psi_index"]

    # number of iso coefficients
    NisoL   = Omega ? J*J*K*K*L+2*J*K*K+2*J*K*K+6*J*K : J*J*K*K*L+2*J*K*K+2*J*K
    NnoL    = Omega ? 2*K+3 : K+1
    Mat     = zeros(Int32, NisoL+NnoL^2, Nf*Nf)

    indexes = 1:Nf
    mask = fhash["psi_ind_L"].==0
    NnoLind = indexes[mask]

    #should probably think about reformatting to (Nj+1) so phi
    #phi cross terms are in the right place

    # first J*J*K*K*L elements of iso
    ref = 0
    for j1 = 1:J
        for j2 = 1:J
            for l1 = 1:L
                for l2 = 1:L
                    for k1 = 1:K
                        for k2 = 1:K
                            DeltaL = mod(l1-l2, L)
                            λ1     = ψ_ind[j1,l1,k1]
                            λ2     = ψ_ind[j2,l2,k2]

                            Iiso   = k1+K*((k2-1)+K*((j1-1)+J*((j2-1)+J*DeltaL)))
                            Icoeff = λ1+Nf*(λ2-1)
                            Mat[Iiso, Icoeff] = 1
                        end
                    end
                end
            end
        end
    end
    ref+=J*J*K*K*L

    # Next JKK elements are λϕ, then JKK elements ϕλ

    for j = 1:J
        for l = 1:L
            for k1 = 1:K
                for k2 =1:K
                    λ      = ψ_ind[j,l,k1]
                    Iiso   = ref + k1+K*((k2-1)+K*(j-1))
                    Icoeff = λ+Nf*(ψ_ind[J+1,1,k2]-1)  # λϕ
                    Mat[Iiso, Icoeff] = 1

                    Iiso   = ref+J*K*K+k1+K*((k2-1)+K*(j-1))
                    Icoeff = ψ_ind[J+1,1,k2]+Nf*(λ-1)  # ϕλ
                    Mat[Iiso, Icoeff] = 1
                end
            end
        end
    end

    ref += 2*J*K*K

    # Next JKK elements are λΩ, then JKK elements Ωλ
    if Omega
        for j = 1:J
            for l = 1:L
                for k1 = 1:K
                    for k2 =1:K
                        λ      = ψ_ind[j,l,k1]
                        Iiso   = ref + k1+K*((k2-1)+K*(j-1))
                        Icoeff = λ+Nf*(ψ_ind[J+1,2,k2]-1)  # λϕ
                        Mat[Iiso, Icoeff] = 1

                        Iiso   = ref+J*K*K+k1+K*((k2-1)+K*(j-1))
                        Icoeff = ψ_ind[J+1,2,k2]+Nf*(λ-1)  # ϕλ
                        Mat[Iiso, Icoeff] = 1
                    end
                end
            end
        end
        ref += 2*J*K*K
    end

    # Next 2*J*K elements are ψϕ0 and ϕ0ψ

    ϕ0 = ψ_ind[J+1,1,K+1]
    for j1=1:J
        for l1 = 1:L
            for k1=1:K
                λ1 = ψ_ind[j1,l1,k1]

                Iiso = ref+k1+K*(j1-1)
                Icoeff = λ1+Nf*(ϕ0-1)
                Mat[Iiso, Icoeff] = 1

                Iiso = ref+J*K+k1+K*(j1-1)
                Icoeff = ϕ0+Nf*(λ1-1)
                Mat[Iiso, Icoeff] = 1
            end
        end
    end
    ref += 2*J*K

    if Omega
        Ω0 = ψ_ind[J+1,2,K+1]
        for j1=1:J
            for l1 = 1:L
                for k1=1:K
                    λ1 = ψ_ind[j1,l1,k1]

                    Iiso = ref+k1+K*(j1-1)
                    Icoeff = λ1+Nf*(Ω0-1)
                    Mat[Iiso, Icoeff] = 1

                    Iiso = ref+J*K+k1+K*(j1-1)
                    Icoeff = Ω0+Nf*(λ1-1)
                    Mat[Iiso, Icoeff] = 1
                end
            end
        end
        ref += 2*J*K

        Ω3 = ψ_ind[J+1,3,K+1]
        for j1=1:J
            for l1 = 1:L
                for k1=1:K
                    λ1 = ψ_ind[j1,l1,k1]

                    Iiso = ref+k1+K*(j1-1)
                    Icoeff = λ1+Nf*(Ω3-1)
                    Mat[Iiso, Icoeff] = 1

                    Iiso = ref+J*K+k1+K*(j1-1)
                    Icoeff = Ω3+Nf*(λ1-1)
                    Mat[Iiso, Icoeff] = 1
                end
            end
        end
        ref += 2*J*K
    end

    # take care of the L independent subblocks
    for m1=1:NnoL
        for m2=1:NnoL
            Iiso = NisoL + m1 + NnoL*(m2-1)
            λ1 = NnoLind[m1]
            λ2 = NnoLind[m2]
            Icoeff = λ1+Nf*(λ2-1)
            Mat[Iiso, Icoeff] = 1
        end
    end

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

## Assist in convmap

function pre_subdivide(coeff_list; sub_block = 0)
    Nc, Ns = size(coeff_list)
    Nx, Ny = size(coeff_list[3])
    out_maps = zeros(Nx,Ny,Nc-2,Ns)
    for i=1:Ns
        out_maps[:,:,:,i] .= cat(coeff_list[3:end,i]...,dims=3)
    end

    out = zeros(Int(2^(sub_block)),Int(2^(sub_block)),size(out_maps)[3],size(out_maps)[4])
    red = Nx/(2^(sub_block))
    for i=1:Nx
        for j=1:Ny
            out[ceil(Int,i/red),ceil(Int,j/red),:,:] .+=  out_maps[i,j,:,:]
        end
    end
    return out
end

function post_subdivide(coeff_maps; sub_block = 0)
    Nx, Ny, Nc, Ns = size(coeff_maps)

    out = zeros(Int(2^(sub_block)),Int(2^(sub_block)),size(out_maps)[3],size(out_maps)[4])
    red = Nx/(2^(sub_block))
    for i=1:Nx
        for j=1:Ny
            out[ceil(Int,i/red),ceil(Int,j/red),:,:] .+=  out_maps[i,j,:,:]
        end
    end
    return out
end

function sub_strip(coeff_maps, lab; sub_bl_size=64)
    Nx, Ny, Nc, Ns  = size(coeff_maps)
    Nl = size(lab)[1]
    out_stripped = zeros(Nc,Nx*Ny*Ns)
    new_lab = zeros(Nl,Nx*Ny*Ns)
    for k=1:Ns
        for i=1:Nx
            for j=1:Ny
                new_lab[:,i+Nx*(j-1)+Nx*Ny*(k-1)] .= lab[:,k] .+ [0; 0; 0; 0; 0; sub_bl_size*(i-1); sub_bl_size*(j-1)]
                out_stripped[:,i+Nx*(j-1)+Nx*Ny*(k-1)] .= out[i,j,:,k]
            end
        end
    end
    return out_stripped, new_lab
end

function renorm(coeff_mats,filter_hash)
    Nc, Ns  = size(coeff_maps)
    Nf = size(filter_hash["filt_index"])[1]

    S1normed = coeff_mats[1:Nf,:]./sum(coeff_mats[1:Nf,:],dims=1)
    S2normed = coeff_mats[Nf+1:end,:]./sum(coeff_mats[Nf+1:end,:],dims=1)

    return vcat(S1normed,S2normed)
end
