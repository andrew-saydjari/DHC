push!(LOAD_PATH, "/Users/saydjari/Dropbox/GradSchool_AKS/Doug/Projects/DHC/main")
using DHC_2DUtils
using Profile
using BenchmarkTools
using FFTW
using HDF5
using Test
using SparseArrays

filter_hash = fink_filter_hash(1, 16, nx=64, pc=2, wd=2)
filt_3d = fink_filter_bank_3dizer(filter_hash, 1, nz=128)

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
    Nl      = length(fhash["2d_theta_value"])
    Nj      = length(fhash["2d_j_value"])
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

S1_iso_matrix3d(filt_3d)

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
    Nl      = length(fhash["2d_theta_value"])
    Nj      = length(fhash["2d_j_value"])
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

S2_iso_matrix3d(filt_3d)

Nj = 4
Nk = 5
Nl = 16
Nj*Nj*Nk*Nk*Nl+2*Nj*Nk*Nk+Nk*Nk

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

S1_iso_matrix(filter_hash)
S2_iso_matrix(filter_hash)
