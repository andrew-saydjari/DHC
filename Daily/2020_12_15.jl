## Preloads
using Statistics
using FFTW
using Plots
using BenchmarkTools
using Profile
using LinearAlgebra
using Measures
using HDF5

##
function fink_filter_bank_fast(J, L;wd=1,pc=1)
    #plane coverage (default 1, full 2Pi 2)
    #width of the wavelets (default 1, wide 2)

    # -------- set parameters
    dθ   = pc*π/L
    wdθ  = wd*dθ
    nx   = 256
    dx   = nx/2-1

    # -------- allocate output array of zeros
    filt = zeros(256, 256, J, L)

    # -------- allocate theta and logr arrays
    logr = zeros(nx, nx)
    θ    = zeros(nx, nx)

    for l = 0:L-1
        θ_l = dθ*l

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
                    r = sqrt(sx^2 + sy^2)
                    logr[y, x] = log2(max(1,r))
                end
            end
        end
        angmask = findall(anggood)
    # -------- compute the wavelet in the Fourier domain
    # -------- the angular factor is the same for all j
        F_angular = cos.((θ[angmask].-π).*L./(2 .*wd .*pc))

    # -------- loop over j for the radial part
        for j = 0:J-1
            jrad  = 7-j
            Δj    = abs.(logr[angmask].-jrad)
            rmask = (Δj .<= 1)

    # -------- radial part
            F_radial = cos.(Δj[rmask] .* (π/2))
            ind      = angmask[rmask]
            filt[ind,j+1,l+1] = F_radial .* F_angular[rmask]
        end
    end
    return filt
end

@time fink_filter_set = fink_filter_bank_fast(8,16,wd=1)

findall(test[:,:,6] .== maximum(test[:,:,6]))

theme(:juno)

plot(Plots.heatmap(fftshift(fink_filter_set[:,:,2,1]),
    xlims=(1,256),
    ylims=(1,256),
    aspectratio=1,
    axis=nothing,
    border=:none,
    c=:cubehelix,
    legend = :none,
    margin=0mm))

plot(Plots.heatmap(fftshift(real.(ifft(fink_filter_set[:,:,7,2]))),
    xlims=(1,256),
    ylims=(1,256),
    aspectratio=1,
    axis=nothing,
    border=:none,
    c=:twilight,
    legend = :none,
    margin=0mm))

ifft(fink_filter_set[:,:,8,1])

nx = 256
c=2
im_scale = convert(Int8,log2(nx))
J = (im_scale-2)*c

println("start")
for i = 1/c:1/c:im_scale-2
    println(i)
end

## dev

function fink_filter_bank_fast(c, L; nx=256,wd=1,pc=1)
    #plane coverage (default 1, full 2Pi 2)
    #width of the wavelets (default 1, wide 2)
    #c sets the scale sampling rate (1 is dyadic, 2 is half dyadic)

    # -------- set parameters
    dθ   = pc*π/L
    wdθ  = wd*dθ
    dx   = nx/2-1

    im_scale = convert(Int8,log2(nx))
    J = (im_scale-2)*c

    # -------- allocate output array of zeros
    filt = zeros(nx, nx, J*L+1)

    # -------- allocate theta and logr arrays
    logr = zeros(nx, nx)
    θ    = zeros(nx, nx)

    # -------- allocate temp phi building zeros
    phi_b = zeros(nx, nx)

    for l = 0:L-1
        θ_l = dθ*l

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
                    r = sqrt(sx^2 + sy^2)
                    logr[y, x] = log2(max(1,r))
                end
            end
        end
        angmask = findall(anggood)
    # -------- compute the wavelet in the Fourier domain
    # -------- the angular factor is the same for all j
        F_angular = cos.((θ[angmask].-π).*L./(2 .*wd .*pc))

    # -------- loop over j for the radial part
        for (j_ind, j) in enumerate(1/c:1/c:im_scale-2)
            jrad  = 7-j
            Δj    = abs.(logr[angmask].-jrad)
            rmask = (Δj .<= 1/c)

    # -------- radial part
            F_radial = cos.(Δj[rmask] .* (c*π/2))
            ind      = angmask[rmask]
            filt[ind,(j_ind-1)*L+l+1] = F_radial .* F_angular[rmask]
        end
    # -------- handle the phi case (jrad=0, j=7)
        for (j_ind, j) in enumerate(im_scale-2+1/c:1/c:im_scale-1)
            jrad  = 7-j
            Δj    = abs.(logr[angmask].-jrad)
            rmask = (Δj .<= 1/c)

            # -------- radial part
            F_radial = cos.(Δj[rmask] .* (c*π/2))
            ind      = angmask[rmask]
            phi_b[ind] .+= (F_radial .* F_angular[rmask]).^2
        end
    end

    # -------- normalize the phi case correctly
    if pc == 1
        phi_b .+= circshift(phi_b[end:-1:1,end:-1:1],(1,1))
        phi_b[1,1] /= 2
    end

    phi_b .= sqrt.(phi_b)

    filt[:,:,J*L+1] .= phi_b
    return filt
end

test = fink_filter_bank_fast(1,8,pc=1)
h5write("./Data/filter_bank_1_8_1.h5", "main/data", test)
test = fink_filter_bank_fast(1,16,pc=2)
h5write("./Data/filter_bank_1_16_2.h5", "main/data", test)
test = fink_filter_bank_fast(2,8,pc=1)
h5write("./Data/filter_bank_2_8_1.h5", "main/data", test)
test = fink_filter_bank_fast(2,16,pc=2)
h5write("./Data/filter_bank_2_16_2.h5", "main/data", test)

push!(LOAD_PATH, pwd())
using DHC_Utils

# Generate phi from an L=16 filter bank
donut = fink_filter_bank(8,8)
phi   = sqrt.(reshape(sum(donut[:,:,8,:].^2,dims=[3,4]),256,256))
phi_r = real(ifft(phi))
# check that the real-space function sums to 1 as expected.
print(sum(phi_r))

plot(Plots.heatmap(fftshift(phi_r),
    xlims=(1,256),
    ylims=(1,256),
    aspectratio=1,
    axis=nothing,
    border=:none,
    c=:cubehelix,
    legend = :none,
    margin=0mm))

plot(Plots.heatmap(fftshift(phi),
    xlims=(1,256),
    ylims=(1,256),
    aspectratio=1,
    axis=nothing,
    border=:none,
    c=:cubehelix,
    legend = :none,
    margin=0mm))

phi_shift = fftshift(phi)

plot(Plots.heatmap(phi_shift.+phi_shift[end:-1:1,end:-1:1],
    xlims=(1,256),
    ylims=(1,256),
    aspectratio=1,
    axis=nothing,
    border=:none,
    c=:cubehelix,
    legend = :none,
    margin=0mm))

plot(Plots.heatmap(fftshift(donut[:,:,8,2].^2),
    xlims=(1,256),
    ylims=(1,256),
    aspectratio=1,
    axis=nothing,
    border=:none,
    c=:cubehelix,
    legend = :none,
    margin=0mm))

X = fftshift(donut[:,:,8,2].^2)

plot(Plots.heatmap(X[end:-1:1,end:-1:1],
    xlims=(1,256),
    ylims=(1,256),
    aspectratio=1,
    axis=nothing,
    border=:none,
    c=:cubehelix,
    legend = :none,
    margin=0mm))

# Generate phi from an L=16 filter bank
donut = fink_filter_bank(8,8)
phi   = sqrt.(reshape(sum(donut[:,:,8,:].^2,dims=[3,4]),256,256))
phi_r = real(ifft(phi))
# check that the real-space function sums to 1 as expected.
print(sum(phi_r))

plot(Plots.heatmap(fftshift(donut[:,:,8,8].^2),
    xlims=(1,256),
    ylims=(1,256),
    aspectratio=1,
    axis=nothing,
    border=:none,
    c=:cubehelix,
    legend = :none,
    margin=0mm))

# Generate phi from an L=16 filter bank
donut = fink_filter_bank(8,8)
phi   = sqrt.(reshape(sum(donut[:,:,8,:].^2,dims=[3,4]),256,256))
phi_r = real(ifft(phi))
# check that the real-space function sums to 1 as expected.
print(sum(phi_r))

plot(Plots.heatmap(fftshift(phi),
    xlims=(1,256),
    ylims=(1,256),
    aspectratio=1,
    axis=nothing,
    border=:none,
    c=:cubehelix,
    legend = :none,
    margin=0mm))

phi_shift2 = fftshift(phi)
phi_shift3 = phi_shift2.+circshift(phi_shift2[end:-1:1,end:-1:1],(1,1))
phi_shift = fftshift(phi)

plot(Plots.heatmap(phi_shift.+phi_shift[end:-1:1,end:-1:1],
    xlims=(1,256),
    ylims=(1,256),
    aspectratio=1,
    axis=nothing,
    border=:none,
    c=:cubehelix,
    legend = :none,
    margin=0mm))

donut = fink_filter_bank(8,16)
phi   = sqrt.(reshape(sum(donut[:,:,8,:].^2,dims=[3,4]),256,256))
phi_r = real(ifft(phi))
# check that the real-space function sums to 1 as expected.
print(sum(phi_r))

## check run straight through...
push!(LOAD_PATH, pwd())
using DHC_2DUtils

fink_filter_set = fink_filter_list(fink_filter_bank(2,8,wd=1,pc=1))
DHC_compute(rand(256,256),fink_filter_set)
