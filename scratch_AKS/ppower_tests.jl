using Statistics
using FFTW
using Plots
using BenchmarkTools
using Profile
using LinearAlgebra
using Measures
using ProgressMeter
using Plots
theme(:dark)
using Revise
push!(LOAD_PATH, pwd()*"/main")
using DHC_2DUtils
using DHC_tests

function fink_filter_bank_p(c, L; nx=256, wd=2, t=1, shift=false, Omega=false, safety_on=true, wd_cutoff=1, p=2)
    #c     - sets the scale sampling rate (1 is dyadic, 2 is half dyadic)
    #L     - number of angular bins (usually 8*t or 16*t)
    #wd    - width of the wavelets (default 1, wd=2 for a double covering)
    #t    - plane coverage (default 1, full 2pi 2)
    #shift - shift in θ by 1/2 of the θ spacing
    #Omega - true= append Omega filter (all power beyond Nyquist) so the sum of filters is 1.0

    # -------- assertion errors to make sure arguments are reasonable
    #@test wd <= L/2

    # -------- set parameters
    dθ   = t*π/L
    θ_sh = shift ? dθ/2 : 0.0
    dx   = nx/2-1

    im_scale = convert(Int8,log2(nx))
    # -------- number of bins in radial direction (size scales)
    J = (im_scale-3)*c + 1
    normj = 1/sqrt(c)

    # -------- allocate output array of zeros
    filt      = zeros(nx, nx, J*L+(Omega ? 2 : 1))
    psi_index = zeros(Int32, J, L)
    psi_ind_in= zeros(Int32, J*L+(Omega ? 2 : 1), 2)
    psi_ind_L = zeros(Int32, J*L+(Omega ? 2 : 1))
    theta     = zeros(Float64, L)
    j_value   = zeros(Float64, J)
    info=Dict{String,Any}()

    # -------- compute the required wd
    j_rad_exp = zeros(J)
    for j_ind = 1:J
        j = (j_ind-1)/c
        jrad  = im_scale-j-2
        j_rad_exp[j_ind] = 2^(jrad)
    end

    wd_j = max.(ceil.(wd_cutoff.*L./(t.*π.*j_rad_exp)),wd)

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
        norm = 1.0/((wd).^(1/p))
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
            F_angular = norm .* cos.((θ[angmask].-π).*(L/(2*wd*t))).^(2/p)

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
                F_radial = normj .* cos.(Δj[rmask] .* (π/2)).^(2/p) #deprecating c*π/2 to π/2
                ind      = angmask[rmask]
        #      Let's have these be (J,L) if you reshape...
        #        f_ind    = (j_ind-1)*L+l+1
                f_ind    = j_ind + l*J
                filt[ind, f_ind] = F_radial .* F_angular[rmask]
                psi_index[j_ind,l+1] = f_ind
                psi_ind_in[f_ind,:] = [j_ind-1,l]
                psi_ind_L[f_ind] = 1
            end
        end
    end

    # -------- phi contains power near k=0 not yet accounted for
    filter_power = (sum(filt.^p, dims=3))[:,:,1]

    # -------- for plane half-covered (t=1), add other half-plane
    if t == 1
        filter_power .+= circshift(filter_power[end:-1:1,end:-1:1],(1,1))
    end

    # -------- compute power required to sum to 1.0
    i0 = round(Int16,nx/2-2)
    i1 = round(Int16,nx/2+4)
    center_power = 1.0 .- fftshift(filter_power)[i0:i1,i0:i1]
    zind = findall(center_power .< 1E-15)
    center_power[zind] .= 0.0  # set small numbers to zero
    phi_cen = zeros(nx, nx)
    phi_cen[i0:i1,i0:i1] = (center_power).^(1/p)

    # -------- before adding ϕ to filter bank, renormalize ψ if t=1
    if t==1 filt .*= 2.0^(1/p) end  # double power for half coverage

    # -------- add result to filter array
    phi_index  = J*L+1
    filt[:,:,phi_index] .= fftshift(phi_cen)
    psi_ind_in[phi_index,:] = [J,0]
    psi_ind_L[phi_index] = 0

    if Omega     # append a filter containing the rest (outside Nyquist)
        filter_power += filt[:,:,phi_index].^p
        edge_power    = 1.0 .- filter_power
        zind          = findall(edge_power .< 1E-15)
        edge_power[zind]     .= 0.0  # set small numbers to zero
        Omega_index           = J*L+2
        info["Omega_index"]   = Omega_index
        filt[:,:,Omega_index] = (edge_power).^(1/p)
        psi_ind_in[Omega_index,:] = [J,1]
        psi_ind_L[Omega_index] = 0
    end

    # -------- metadata dictionary
    info["npix"]         = nx
    info["j_value"]      = j_value
    info["theta_value"]  = theta
    info["psi_index"]    = psi_index
    info["phi_index"]    = phi_index
    info["J_L"]          = psi_ind_in
    info["t"]            = t
    info["wd"]           = wd_j
    info["wd_cutoff"]    = wd_cutoff
    info["fs_center_r"]  = j_rad_exp
    info["psi_ind_L"]    = psi_ind_L
    info["p"]            = p

    return filt, info
end

testtest_p = fink_filter_bank_p(1,8)
test_bank = fink_filter_bank(1,8)
test_bank .== test_p

using PyCall
using PyPlot
plt.matplotlib.style.use("dark_background")

plot_filter_bank_QA(test_p[1], test_p[2]; fname="/Users/saydjari/Dropbox/GradSchool_AKS/Doug/Projects/DHC/scratch_AKS/images/filter_bank_QA_p2.png")

test_p1 = fink_filter_bank_p(1,8,p=1)

disc = plot_filter_bank_QA(test_p1[1], test_p1[2]; fname="/Users/saydjari/Dropbox/GradSchool_AKS/Doug/Projects/DHC/scratch_AKS/images/filter_bank_QA_p1.png",p=1)

heatmap(disc)

test_pp5 = fink_filter_bank_p(1,8,p=0.5)

disc = plot_filter_bank_QA(test_pp5[1], test_pp5[2]; fname="/Users/saydjari/Dropbox/GradSchool_AKS/Doug/Projects/DHC/scratch_AKS/images/filter_bank_QA_pp5.png",p=0.5)

heatmap(disc)

test_p3 = fink_filter_bank_p(1,8,p=3)

disc = plot_filter_bank_QA(test_p3[1], test_p3[2]; fname="/Users/saydjari/Dropbox/GradSchool_AKS/Doug/Projects/DHC/scratch_AKS/images/filter_bank_QA_p3.png",p=3)

heatmap(disc)

function DHC_compute_p(image::Array{Float64,2}, filter_hash::Dict, filter_hash2::Dict=filter_hash;
    doS2::Bool=true, doS20::Bool=false, norm=true, iso=false, FFTthreads=2, normS1::Bool=false, normS1iso::Bool=false,
    p=2)
    # image        - input for WST
    # filter_hash  - filter hash from fink_filter_hash
    # filter_hash2 - filters for second order.  Default to same as first order.
    # doS2         - compute S2 coeffs
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
    @assert (normS1 && normS1iso) != 1 "normS1 and normS1iso are mutually exclusive"

    # allocate coeff arrays
    out_coeff = []
    S0  = zeros(Float64, 2)
    S1  = zeros(Float64, Nf)
    if doS2  S2  = zeros(Float64, Nf, Nf) end  # traditional 2nd order
    if doS20 S20 = zeros(Float64, Nf, Nf) end  # real space correlation
    anyM2 = doS2 | doS20

    # allocate image arrays for internal use
    im_rd_0_1  = Array{Float64, 3}(undef, Nx, Ny, Nf)

    ## 0th Order
    S0[1]   = mean(image)
    norm_im = image.-S0[1]
    S0[2]   = sum(abs.(norm_im).^p)/(Nx*Ny)
    if norm
        norm_im ./= (Nx*Ny*S0[2]).^(1/p)
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
    P = plan_ifft(im_fd_0)  # P is an operator, P*im is ifft(im)

    ## Main 1st Order and Precompute 2nd Order
    for f = 1:Nf
        f_i = f_ind[f]  # CartesianIndex list for filter
        f_v = f_val[f]  # Values for f_i
        if length(f_i) > 0
            zarr[f_i] = f_v.*im_fd_0[f_i]
            im_rd_0_1[:,:,f] .= abs.(P*zarr)
            S1[f] = sum(im_rd_0_1[:,:,f].^(p))  # image power
            #zero out the intermediate arrays
            zarr[f_i] .= 0
        end
    end

    append!(out_coeff, iso ? filter_hash["S1_iso_mat"]*S1 : S1)

    if normS1iso
        S1iso = vec(reshape(filter_hash["S1_iso_mat"]*S1,(1,:))*filter_hash["S1_iso_mat"])
    end

    Mat2 = filter_hash["S2_iso_mat"]
    if doS2
        f_ind2   = filter_hash2["filt_index"]  # (J, L) array of filters represented as index value pairs
        f_val2   = filter_hash2["filt_value"]

        ## Traditional second order
        for f1 = 1:Nf
            thisim = fft(im_rd_0_1[:,:,f1])  # Could try rfft here
            # Loop over f2 and do second-order convolution
            if normS1
                normS1pwr = S1[f1]
            elseif normS1iso
                normS1pwr = S1iso[f1]
            else
                normS1pwr = 1
            end
            for f2 = 1:Nf
                f_i = f_ind2[f2]  # CartesianIndex list for filter
                f_v = f_val2[f2]  # Values for f_i
                zarr[f_i] = f_v .* thisim[f_i]
                S2[f1,f2] = sum(abs.(P*zarr).^(p))/normS1pwr
                #zero out the intermediate arrays
                zarr[f_i] .= 0
            end
        end
        append!(out_coeff, iso ? Mat2*S2[:] : S2[:])
    end

    # Real domain 2nd order
    if doS20
        Amat = reshape(im_rd_0_1, Nx*Ny, Nf)
        S20  = Amat' * Amat
        append!(out_coeff, iso ? Mat2*S20[:] : S20[:])
    end

    return out_coeff
end

test = zeros(16,16,6)
for i=1:6
    a = rand(16,16)
    test[:,:,i] .= a
end

test

filter_hash = fink_filter_hash(1,8)
f_ind   = filter_hash["filt_index"]  # (J, L) array of filters represented as index value pairs
f_val   = filter_hash["filt_value"]

temp_mat = zeros(Float64, 256, 256)
f_i = f_ind[10]  # CartesianIndex list for filter
f_v = f_val[10]  # Values for f_i

temp_mat[f_i] = f_v
temp_mat
@benchmark temp_mat[f_i] = f_v

@benchmark for i = 1:length(f_i)
    ind       = f_i[i]
    temp_mat[ind] = f_v[i]        # filter*image in Fourier domain
end

a = rand(256,256)
temp_mat[f_i] = f_v
temp_mat
@benchmark temp_mat[f_i] = f_v.*a[f_i]

@benchmark for i = 1:length(f_i)
    ind       = f_i[i]
    temp_mat[ind] = f_v[i]*a[ind]       # filter*image in Fourier domain
end

a = rand(256,256)

original = DHC_compute(a,filter_hash)

new = DHC_compute_p(a,filter_hash)

@benchmark DHC_compute_p(a,filter_hash)

@benchmark DHC_compute(a,filter_hash)

6/.178

# so we are slower by slightly less than a factor of Nf... not sure we can help that

function fink_filter_hash_p(c, L; nx=256, wd=2, t=1, shift=false, Omega=false, safety_on=true, wd_cutoff=1, p=2)
    # -------- compute the filter bank
    filt, hash = fink_filter_bank_p(c, L; nx=nx, wd=wd, t=t, shift=shift, Omega=Omega, safety_on=safety_on, wd_cutoff=wd_cutoff, p=p)

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

filter_hash = fink_filter_hash(1,8)
filter_hashp2 = fink_filter_hash_p(1,8,p=2)

filter_hashp2 == filter_hash

filter_hashp1 = fink_filter_hash_p(1,8,p=1)

@benchmark DHC_compute_p(a,filter_hashp1,p=1)

filter_hashp3 = fink_filter_hash_p(1,8,p=3)

@benchmark DHC_compute_p(a,filter_hashp3,p=3)

filter_hashpp5 = fink_filter_hash_p(1,8,p=0.5)

@benchmark DHC_compute_p(a,filter_hashpp5,p=0.5)

b = [-1, 2, 3, 4]

norm(b)

norm(b,1)

norm(b,3)

norm(b,1/3)

sum(abs.(b).^1)

sqrt(sum(abs.(b).^2))

@benchmark sum(abs.(b).^1)

@benchmark norm(b,1)

@benchmark sum(abs.(b).^2)

@benchmark norm(b,2)^2

Profile.clear()
@profile DHC_compute_p(a,filter_hashp3,p=3)

Juno.profiler()

b = rand(ComplexF64,256,256)

sum(abs.(b).^3)

@benchmark sum(abs.(b).^3)

@benchmark norm(b,3)^3

@benchmark sum(abs2.(b).^(3/2))

sum(abs2.(b).^(3/2))

ref = fink_filter_bank(1,8)
test = fink_filter_bank_p(1,8)

ref == test


a = rand(256,256)

original = DHC_compute(a,filter_hash)

new = DHC_compute_p(a,filter_hash)

original[2:5]./new[2:5]

256*256

original == new

original ≈ new

nearlysame(x, y) = x ≈ y || (isnan(x) & isnan(y))
nearlysame(A::AbstractArray, B::AbstractArray) = all(map(nearlysame, A, B))

nearlysame(original[1:6*8+1],new[1:6*8+1])

maximum(abs.(new.-original))

original[100:105]./new[100:105]

new[100:105]

sum(original[3:end])

sum(original[3:6*8+1])

function rod_image(xcen, ycen, length, pa, fwhm; nx=256)
    # returns image of a rod with some (x,y) position, length,
    #   position angle, and FWHM in an (nx,nx) image.
    rodimage = zeros(nx,nx)

    x=0
    y=0
    sig = fwhm/2.355
    dtor = π/180
    # -------- define a unit vector in direction of rod at position angle pa
    ux = sin(pa*dtor)   #  0 deg is up
    uy = cos(pa*dtor)   # 90 deg to the right

    for i=1:nx
        for j=1:nx
            x=i-nx/2+xcen
            y=j-nx/2+ycen

            # -------- distance parallel and perpendicular to
            dpara =  ux*x + uy*y
            dperp = -uy*x + ux*y

            if abs(dpara)-length <0
                dpara= 0
            end
            dpara = abs(dpara)
            dpara = min(abs(dpara-length),dpara)

            rodimage[i,j] = exp(-(dperp^2+dpara^2)/(2*sig^2))
        end
    end
    rodimage ./= sqrt(sum(rodimage.^2))
    return rodimage
end

rod_test = rod_image(1,1,30,35,8)

heatmap(rod_test)

original = DHC_compute(rod_test,filter_hash)

new = DHC_compute_p(rod_test,filter_hash)

original ≈ new

sum(original[3:end])

sum(new[3:end])

sum(new[3:end])-2 < 1e-6
sum(original[3:end])-2 < 1e-6
