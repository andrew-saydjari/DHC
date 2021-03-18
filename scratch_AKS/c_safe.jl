push!(LOAD_PATH, "/Users/saydjari/Dropbox/GradSchool_AKS/Doug/Projects/DHC/main")
using DHC_2DUtils
using DHC_tests
using Profile
using BenchmarkTools
using FFTW
using HDF5
using Test
using SparseArrays
using Plots
theme(:dark)

function fink_filter_bank_csafe(c, L; nx=256, wd=1, pc=1, shift=false, Omega=false, safety_on=true)
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
    J = (im_scale-3)*c + 1

    # -------- allocate output array of zeros
    filt      = zeros(nx, nx, J*L+(Omega ? 2 : 1))
    psi_index = zeros(Int32, J, L)
    psi_ind_in= zeros(Int32, J*L+(Omega ? 2 : 1), 2)
    theta     = zeros(Float64, L)
    j_value   = zeros(Float64, J)

    # -------- compute the required wd
    j_rad_exp = zeros(J)
    for j_ind = 1:J
        j = (j_ind-1)/c
        jrad  = im_scale-j-2
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
            for j_ind in j_ind_w_wd
                j = (j_ind-1)/c
                j_value[j_ind] = 1+j  # store for later
                jrad  = im_scale-j-2
                Δj    = abs.(logr[angmask].-jrad)
                rmask = (Δj .<= 1) #deprecating the 1/c to 1, constant width

        # -------- radial part
                F_radial = cos.(Δj[rmask] .* (π/2)) #deprecating c*π/2 to π/2
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

function fink_filter_hash_csafe(c, L; nx=256, wd=1, pc=1, shift=false, Omega=false, safety_on=true)
    # -------- compute the filter bank
    filt, hash = fink_filter_bank_csafe(c, L; nx=nx, wd=wd, pc=pc, shift=shift, Omega=Omega, safety_on=safety_on)

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

fhash = fink_filter_hash_csafe(1,8,wd=2,pc=1)
filt_test = fink_filter_bank_csafe(1,8,wd=2,pc=1)[1]

plot_filter_bank_QA(filt_test, fhash; fname="/Users/saydjari/Dropbox/GradSchool_AKS/Doug/Projects/DHC/scratch_AKS/images/filter_bank_QA_csafe.png")

fhash = fink_filter_hash_csafe(2,8,wd=2,pc=1)
filt_test = fink_filter_bank_csafe(2,8,wd=2,pc=1)[1]

plot_filter_bank_QA(filt_test, fhash; fname="/Users/saydjari/Dropbox/GradSchool_AKS/Doug/Projects/DHC/scratch_AKS/images/filter_bank_QA_csafe_C2.png")

fhash = fink_filter_hash_csafe(2,8,wd=1,pc=1)

fhash = fink_filter_hash(2,8,wd=1,pc=1)

#In both formulations, only the smallest FS wavelet gets widened by our original criteria, even with c=2

fhash = fink_filter_hash_csafe(2,8,wd=1,pc=1)

function fink_filter_bank_csafe(c, L; nx=256, wd=1, pc=1, shift=false, Omega=false, safety_on=true, wd_cutoff=2)
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
    J = (im_scale-3)*c + 1

    # -------- allocate output array of zeros
    filt      = zeros(nx, nx, J*L+(Omega ? 2 : 1))
    psi_index = zeros(Int32, J, L)
    psi_ind_in= zeros(Int32, J*L+(Omega ? 2 : 1), 2)
    theta     = zeros(Float64, L)
    j_value   = zeros(Float64, J)

    # -------- compute the required wd
    j_rad_exp = zeros(J)
    for j_ind = 1:J
        j = (j_ind-1)/c
        jrad  = im_scale-j-2
        j_rad_exp[j_ind] = 2^(jrad)
    end

    wd_j = max.(ceil.(wd_cutoff.*L./(pc.*π.*j_rad_exp)),wd)

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
            for j_ind in j_ind_w_wd
                j = (j_ind-1)/c
                j_value[j_ind] = 1+j  # store for later
                jrad  = im_scale-j-2
                Δj    = abs.(logr[angmask].-jrad)
                rmask = (Δj .<= 1) #deprecating the 1/c to 1, constant width

        # -------- radial part
                F_radial = cos.(Δj[rmask] .* (π/2)) #deprecating c*π/2 to π/2
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
    info["wd_cutoff"]    = wd_cutoff
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

fhash = fink_filter_hash_csafe(2,8,wd=2,pc=1)
filt_test = fink_filter_bank_csafe(2,8,wd=2,pc=1)[1]

plot_filter_bank_QA(filt_test, fhash; fname="/Users/saydjari/Dropbox/GradSchool_AKS/Doug/Projects/DHC/scratch_AKS/images/filter_bank_QA_csafe_C2_wide.png")

#Now, only the smallest FS wavelet with wd=2 is bumped up to 3... visually seems a good choice.

fhash = fink_filter_hash_csafe(1,8,wd=2,pc=1)
filt_test = fink_filter_bank_csafe(1,8,wd=2,pc=1)[1]

plot_filter_bank_QA(filt_test, fhash; fname="/Users/saydjari/Dropbox/GradSchool_AKS/Doug/Projects/DHC/scratch_AKS/images/filter_bank_QA_csafe_C1_wide.png")

fhash = fink_filter_hash_csafe(1,8,wd=1,pc=1)
filt_test = fink_filter_bank_csafe(1,8,wd=1,pc=1)[1]

plot_filter_bank_QA(filt_test, fhash; fname="/Users/saydjari/Dropbox/GradSchool_AKS/Doug/Projects/DHC/scratch_AKS/images/filter_bank_QA_csafe_wd1_C1_wide.png")

# with wd=1, smallest is widened to 3, next to 2, rest to 1
