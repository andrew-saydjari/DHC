##
using Plots
backend(:plotly)
theme(:juno)

##
function finklet(j, l)
    # -------- set filters
    jrad = 7-j
    dθ = π/8        # 8 angular bins hardwired
    θ_l = dθ*l
    # -------- define coordinates
    nx = 256
    xbox = LinRange(-nx/2, nx/2-1 , nx)
    # make a 256x256 grid of X
    sx = xbox' .* ones(nx)
    sy = ones(nx)' .* xbox
    r  = sqrt.((sx).^2 + (sy).^2)
    θ  = mod.(atan.(sy, sx).+π .-θ_l,2*π)
    nozeros = r .> 0
    logr = log2.(r[nozeros])
    r[nozeros] = logr
    # -------- in Fourier plane, envelope of psi_j,l
    mask = (abs.(θ.-π).<= dθ) .& (abs.(r.-jrad) .<= 1)
    # -------- angular part
    ang = cos.((θ.-π).*4)
    # -------- radial part
    rad = cos.((r.-jrad).*π./2)
    psi = mask.*ang.*rad             #mask times angular part times radial part
    return psi
end

##
temp = finklet(1,0)
Plots.heatmap(temp, aspectratio=1)

##
using FFTW

##
J = 8
L = 16

fink_filter = Array{Float64, 4}(undef, J, L, 256, 256)
for j = 1:J
    for l = 1:L
        fink_filter[j,l,:,:]=fftshift(finklet(j-1,l-1))
    end
end

Plots.heatmap(fftshift(fink_filter[3,16,:,:]), aspectratio=1)

fink_filter_sum = zeros(256, 256)
for j = 1:J
    for l = 1:L
        fink_filter_sum+=(finklet(j-1,l-1)).^2
    end
end

Plots.heatmap(fink_filter_sum, aspectratio=1)

fink_filter_sum[127:129,127:129]

##
Plots.heatmap(fftshift(fink_filter[8,4,:,:]), aspectratio=1)

(fftshift(fink_filter[8,4,:,:]))[126:130,126:130]

fink_filter_1 = Array{Float64, 4}(undef, J, L, 256, 256)
for j = 1:J
    for l = 1:L
        fink_filter_1[j,l,:,:]=finklet(j-1,l-1)
    end
end

fink_filter_1[8,3,127:131,127:131]

##
temp2 = [1 2; 3 4]
fftshift(temp2)

blarg = rand(4,4)
fftshift(blarg)

fink_filter_sum = zeros(256, 256)
for j = J
    for l = 1:L
        fink_filter_sum+=(fink_filter_1[j,l,:,:]).^2
    end
end

##
Plots.heatmap(fink_filter_sum, aspectratio=1)

fink_filter_sum[127:131,127:131]

phi = sqrt.(fink_filter_sum)

##
phi_real = ifft(fftshift(phi))

Plots.heatmap(phi, aspectratio=1)

Plots.heatmap(abs.(phi_real), aspectratio=1)

Plots.heatmap(real.(phi_real), aspectratio=1)

Plots.heatmap(imag.(phi_real), aspectratio=1)

function fink_filter_bank(J,L)
    fink_filter = Array{Float64, 4}(undef, J, L, 256, 256)
    for j = 1:J-1
        for l = 1:L
            fink_filter[j,l,:,:]=fftshift(finklet(j-1,l-1))
        end
    end
end

@time fink_filter_bank(8,16)

@time fink_filter_bank(8,16)

##
using ColorSchemes

Plots.heatmap(abs.(phi_real), aspectratio=1,c=:cubehelix)

colorschemes[:cubehelix]

findcolorscheme("cube")

##

Plots.heatmap(abs.(phi_real),
    xlims=(1,256),
    ylims=(1,256),
    aspectratio=1,
    axis=nothing,
    border=:none,
    c=:cubehelix)
