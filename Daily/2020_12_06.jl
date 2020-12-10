# %% Preloads
using Statistics
using FFTW
using Plots
using BenchmarkTools
using Profile
using LinearAlgebra
using Measures
using HDF5

#%%
function err_extract(iso_out,J,N)
    pdf_err = zeros(J,J)
    temp = zeros(N)
    for j2=1:J
        for j1=1:J
            temp = (iso_out[j1,j2,:].-mean(iso_out[j1,j2,:]))./mean(iso_out[j1,j2,:])
            pdf_err[j1,j2] = maximum(temp)-minimum(temp)
         end
     end
     return maximum(pdf_err[2:7,2:7])
end

function S20_iso(data,J,L,N)
    S20 = reshape(data[2+J*L+1:2+J*L+J*L*J*L,:],J,L,J,L,N)
    iso_mat = zeros(J,J,N,L)
    for l=1:L
        intermed = zeros(J,J,N,L)
        for d=1:L
            intermed[:,:,:,d].=S20[:,l,:,mod(l+d,L)+1,:]
        end
        iso_mat[:,:,:,l] .= dropdims(sum(intermed,dims=4),dims=4)
    end
    pow_mat = zeros(J,J,N)
    pow_mat = dropdims(sum(iso_mat,dims=4),dims=4)
    return pow_mat
end

# %% Plat with some data
out0 = h5read("./Data/abs2j8l16w1p2.h5", "main/data")
angle_iso = S20_iso(out0,8,16,360)
err_extract(angle_iso,8,360)

# %%
J = 8
L= 16

plot(angle_iso[3,4,:])

out0[2+J*L+1:2+J*L+J*L*J*L,:]
test = reshape(out0[2+J*L+1:2+J*L+J*L*J*L,:],J,L,J,L,360)

plot(1:360,test[3,2,3,2,:])
plot!(1:360,test[3,4,3,4,:])


plot(1:360,test[3,1,3,16,:])
plot!(1:360,test[3,1,3,2,:])

plot!(1:360,test[3,2,3,1,:])
plot!(1:360,test[3,2,3,3,:])

plot!(1:360,test[3,3,3,2,:])
plot!(1:360,test[3,3,3,4,:])

plot!(1:360,test[3,4,3,3,:])
plot!(1:360,test[3,4,3,5,:])

fig = plot()
for i=1:L
    plot!(fig,1:360,test[3,i,3,mod1(i-1,L),:],show=true)
    plot!(fig,1:360,test[3,i,3,mod1(i+1,L),:],show=true)
end
fig

# %% Retry understanding mod... but actually, do not need to for simplest case.

function S20_iso(data,J,L,N)
    S20 = reshape(data[2+J*L+1:2+J*L+J*L*J*L,:],J,L,J,L,N)
    iso_mat0 = zeros(J,L,J,N)
    iso_mat1 = zeros(J,J,N)

    iso_mat0 .= dropdims(sum(S20,dims=4),dims=4)
    iso_mat1 .= dropdims(sum(iso_mat0,dims=2),dims=2)

    return iso_mat1
end

# %% Plot with some data
out0 = h5read("./Data/abs2j8l16w1p2.h5", "main/data")
angle_iso = S20_iso(out0,8,16,360)
err_extract(angle_iso,8,360)

plotlyjs()

plot(angle_iso[4,4,:])

test = reshape(out0[2+J*L+1:2+J*L+J*L*J*L,:],J,L,J,L,360)

plot(test[4,4,4,4,:])

plot(angle_iso[3,:,:])

# %%
function S20_iso(data,J,L,N)
    S20 = reshape(data[2+J*L+1:2+J*L+J*L*J*L,:],J,L,J,L,N)
    iso_mat = zeros(J,J,N,L)
    for l=1:L
        intermed = zeros(J,J,N,L)
        for d=1:L
            intermed[:,:,:,d].=S20[:,l,:,mod1(l+d,L)+1,:]
        end
        iso_mat[:,:,:,l] .= dropdims(sum(intermed,dims=4),dims=4)
    end
    pow_mat = zeros(J,J,N)
    pow_mat = dropdims(sum(iso_mat,dims=4),dims=4)
    return pow_mat
end

fig = plot()
for i=3
    plot!(fig,1:360,test[3,i,3,mod1(i-1,L),:],show=true)
    plot!(fig,1:360,test[3,i,3,mod1(i+1,L),:],show=true)

    plot!(fig,1:360,test[3,i-1,3,mod1(i,L),:],show=true)
    plot!(fig,1:360,test[3,i+1,3,mod1(i,L),:],show=true)
end
fig

L=8
J=8
out0 = h5read("./Data/abs2j8l8w1p1.h5", "main/data")
out0[2+J*L+1:2+J*L+J*L*J*L,:]
test = reshape(out0[2+J*L+1:2+J*L+J*L*J*L,:],J,L,J,L,360)

h5write("./Data/abs2j8l8w1p1_forpy.h5", "main/data", test)

L=16
J=8
out0 = h5read("./Data/abs2j8l16w1p2.h5", "main/data")
out0[2+J*L+1:2+J*L+J*L*J*L,:]
test = reshape(out0[2+J*L+1:2+J*L+J*L*J*L,:],J,L,J,L,360)

h5write("./Data/abs2j8l16w1p2_forpy.h5", "main/data", test)

##

UpperTriangular(test[3,:,3,:,1])

function upper_tri(A,n)
    v = zeros(Int(n*(n+1)/2),360)
    k = 0
    for i in 1:n
        for j in 1:i
            v[k+j,:] = A[j,i,:]
        end
        k+=i
    end
    v
end

temp_v = upper_tri(test[3,:,3,:,:],16)

u = 2 .*mod.(1:136,2).-1

dot_prod = dropdims(sum(temp_v .* u,dims=1),dims=1)

plot(1:360,dot_prod)

u2 = 2 .*mod.(1:16,2).-1

function upper_tri_1d(A,n)
    v = zeros(Int(n*(n+1)/2))
    k = 0
    for i in 1:n
        for j in 1:i
            v[k+j] = A[j,i]
        end
        k+=i
    end
    v
end

u3 = upper_tri_1d(ones(16,16).*transpose(u2),16)

dot_prod = dropdims(sum(temp_v .* u3,dims=1),dims=1)

plot(1:360,dot_prod)

u4 = upper_tri_1d(u2.*transpose(u2),16)

dot_prod = dropdims(sum(temp_v .* u4,dims=1),dims=1)

plot(1:360,dot_prod)

maximum(dot_prod)
minimum(dot_prod)

# %% Redoing without the S1 contribution

function S20_iso(data,J,L,N)
    S20 = reshape(data[2+J*L+1:2+J*L+J*L*J*L,:],J,L,J,L,N)
    iso_mat = zeros(Int(L*(L+1)/2),J,J,N)
    k = 0
    @inbounds for i in 1:L
        for j in 1:i
            iso_mat[k+j,:,:,:] = S20[:,j,:,i,:]
        end
        k+=i
    end
    pow_mat = dropdims(sum(iso_mat,dims=1),dims=1)
    return pow_mat
end

function err_extract(iso_out,J,N)
    pdf_err = zeros(J,J)
    temp = zeros(N)
    for j2=1:J
        for j1=1:J
            temp = (iso_out[j1,j2,:].-mean(iso_out[j1,j2,:]))./mean(iso_out[j1,j2,:])
            pdf_err[j1,j2] = maximum(temp)-minimum(temp)
         end
     end
     return maximum(pdf_err[2:7,2:7])
end

out0 = h5read("./Data/abs2j8l16w1p1.h5", "main/data")
angle_iso = S20_iso(out0,8,16,360)
err_extract(angle_iso,8,360)

out0 = h5read("./Data/abs2j8l16w2p1.h5", "main/data")
angle_iso = S20_iso(out0,8,16,360)
err_extract(angle_iso,8,360)

out0 = h5read("./Data/abs2j8l16w3p1.h5", "main/data")
angle_iso = S20_iso(out0,8,16,360)
err_extract(angle_iso,8,360)

out0 = h5read("./Data/abs2j8l16w1p2.h5", "main/data")
angle_iso = S20_iso(out0,8,16,360)
err_extract(angle_iso,8,360)

out0 = h5read("./Data/abs2j8l16w2p2.h5", "main/data")
angle_iso = S20_iso(out0,8,16,360)
err_extract(angle_iso,8,360)

out0 = h5read("./Data/abs2j8l16w3p2.h5", "main/data")
angle_iso = S20_iso(out0,8,16,360)
err_extract(angle_iso,8,360)

out0 = h5read("./Data/abs2j8l8w1p1.h5", "main/data")
angle_iso = S20_iso(out0,8,8,360)
err_extract(angle_iso,8,360)

out0 = h5read("./Data/abs2j8l8w2p1.h5", "main/data")
angle_iso = S20_iso(out0,8,8,360)
err_extract(angle_iso,8,360)

out0 = h5read("./Data/abs2j8l8w3p1.h5", "main/data")
angle_iso = S20_iso(out0,8,8,360)
err_extract(angle_iso,8,360)

out0 = h5read("./Data/abs2j8l8w1p2.h5", "main/data")
angle_iso = S20_iso(out0,8,8,360)
err_extract(angle_iso,8,360)

out0 = h5read("./Data/abs2j8l8w2p2.h5", "main/data")
angle_iso = S20_iso(out0,8,8,360)
err_extract(angle_iso,8,360)

out0 = h5read("./Data/abs2j8l8w3p2.h5", "main/data")
angle_iso = S20_iso(out0,8,8,360)
err_extract(angle_iso,8,360)

out0 = h5read("./Data/abs2j8l32w1p1.h5", "main/data")
angle_iso = S20_iso(out0,8,32,360)
err_extract(angle_iso,8,360)

out0 = h5read("./Data/abs2j8l32w2p1.h5", "main/data")
angle_iso = S20_iso(out0,8,32,360)
err_extract(angle_iso,8,360)

out0 = h5read("./Data/abs2j8l32w3p1.h5", "main/data")
angle_iso = S20_iso(out0,8,32,360)
err_extract(angle_iso,8,360)



out0 = h5read("./Data/j8l16w1p1.h5", "main/data")
angle_iso = S20_iso(out0,8,16,360)
err_extract(angle_iso,8,360)

out0 = h5read("./Data/j8l16w2p1.h5", "main/data")
angle_iso = S20_iso(out0,8,16,360)
err_extract(angle_iso,8,360)

out0 = h5read("./Data/j8l16w3p1.h5", "main/data")
angle_iso = S20_iso(out0,8,16,360)
err_extract(angle_iso,8,360)

out0 = h5read("./Data/j8l16w1p2.h5", "main/data")
angle_iso = S20_iso(out0,8,16,360)
err_extract(angle_iso,8,360)

out0 = h5read("./Data/j8l16w2p2.h5", "main/data")
angle_iso = S20_iso(out0,8,16,360)
err_extract(angle_iso,8,360)

out0 = h5read("./Data/j8l16w3p2.h5", "main/data")
angle_iso = S20_iso(out0,8,16,360)
err_extract(angle_iso,8,360)

out0 = h5read("./Data/j8l8w1p1.h5", "main/data")
angle_iso = S20_iso(out0,8,8,360)
err_extract(angle_iso,8,360)

out0 = h5read("./Data/j8l8w2p1.h5", "main/data")
angle_iso = S20_iso(out0,8,8,360)
err_extract(angle_iso,8,360)

out0 = h5read("./Data/j8l8w3p1.h5", "main/data")
angle_iso = S20_iso(out0,8,8,360)
err_extract(angle_iso,8,360)

out0 = h5read("./Data/j8l8w1p2.h5", "main/data")
angle_iso = S20_iso(out0,8,8,360)
err_extract(angle_iso,8,360)

out0 = h5read("./Data/j8l8w2p2.h5", "main/data")
angle_iso = S20_iso(out0,8,8,360)
err_extract(angle_iso,8,360)

out0 = h5read("./Data/j8l8w3p2.h5", "main/data")
angle_iso = S20_iso(out0,8,8,360)
err_extract(angle_iso,8,360)

out0 = h5read("./Data/j8l32w1p1.h5", "main/data")
angle_iso = S20_iso(out0,8,32,360)
err_extract(angle_iso,8,360)

out0 = h5read("./Data/j8l32w2p1.h5", "main/data")
angle_iso = S20_iso(out0,8,32,360)
err_extract(angle_iso,8,360)

out0 = h5read("./Data/j8l32w3p1.h5", "main/data")
angle_iso = S20_iso(out0,8,32,360)
err_extract(angle_iso,8,360)

# %%
function S12_iso(data,J,L,N)
    S20 = reshape(data[2+J*L+J*L*J*L+1:2+J*L+J*L*J*L+J*L*J*L,:],J,L,J,L,N)
    iso_mat = zeros(Int(L*(L+1)/2),J,J,N)
    k = 0
    @inbounds for i in 1:L
        for j in 1:i
            iso_mat[k+j,:,:,:] = S20[:,j,:,i,:]
        end
        k+=i
    end
    pow_mat = dropdims(sum(iso_mat,dims=1),dims=1)
    return pow_mat
end

function err_extract(iso_out,J,N)
     pdf_err = zeros(J,J)
     temp = zeros(N)
     for j2=1:J
         for j1=1:J
             temp = (iso_out[j1,j2,:].-mean(iso_out[j1,j2,:]))./mean(iso_out[j1,j2,:])
             pdf_err[j1,j2] = maximum(temp)-minimum(temp)
          end
      end
      pdf_err[isnan.(pdf_err)] .= 0
      return maximum(pdf_err[2:7,2:7])
 end

out0 = h5read("./Data/j8l8w1p2.h5", "main/data")
angle_iso = S12_iso(out0,8,8,360)
val = err_extract(angle_iso,8,360)

out0 = h5read("./Data/j8l8w2p2.h5", "main/data")
angle_iso = S12_iso(out0,8,8,360)
val = err_extract(angle_iso,8,360)

out0 = h5read("./Data/j8l8w3p2.h5", "main/data")
angle_iso = S12_iso(out0,8,8,360)
val = err_extract(angle_iso,8,360)


out0 = h5read("./Data/j8l16w1p2.h5", "main/data")
angle_iso = S12_iso(out0,8,16,360)
val = err_extract(angle_iso,8,360)

out0 = h5read("./Data/j8l16w2p2.h5", "main/data")
angle_iso = S12_iso(out0,8,16,360)
val = err_extract(angle_iso,8,360)

out0 = h5read("./Data/j8l16w3p2.h5", "main/data")
angle_iso = S12_iso(out0,8,16,360)
val = err_extract(angle_iso,8,360)


out0 = h5read("./Data/abs2j8l8w1p2.h5", "main/data")
angle_iso = S12_iso(out0,8,8,360)
val = err_extract(angle_iso,8,360)

out0 = h5read("./Data/abs2j8l8w2p2.h5", "main/data")
angle_iso = S12_iso(out0,8,8,360)
val = err_extract(angle_iso,8,360)

out0 = h5read("./Data/j8l8w3p2.h5", "main/data")
angle_iso = S12_iso(out0,8,8,360)
val = err_extract(angle_iso,8,360)


out0 = h5read("./Data/abs2j8l16w1p2.h5", "main/data")
angle_iso = S12_iso(out0,8,16,360)
val = err_extract(angle_iso,8,360)

out0 = h5read("./Data/abs2j8l16w2p2.h5", "main/data")
angle_iso = S12_iso(out0,8,16,360)
val = err_extract(angle_iso,8,360)

out0 = h5read("./Data/abs2j8l16w3p2.h5", "main/data")
angle_iso = S12_iso(out0,8,16,360)
val = err_extract(angle_iso,8,360)

# %%
x = 1:360
plot(x,sin.(π/180*x) .+ cos.(π/180*x))
plot!(x,abs.(sin.(π/180*x)) .+ abs.(cos.(π/180*x)))
plot!(x,abs.(sin.(π/180*x)).^2 .+ abs.(cos.(π/180*x)).^2)
plot!(x,abs.(sin.(π/180*x)).^4 .+ abs.(cos.(π/180*x)).^4)
plot!(x,abs.(sin.(π/180*x)).^6 .+ abs.(cos.(π/180*x)).^6)
plot!(x,abs.(sin.(π/180*x)).^(.5) .+ abs.(cos.(π/180*x)).^(.5))

out0 = h5read("./Data/abs2j8l16w1p2.h5", "main/data")
test = reshape(out0[2+J*L+1:2+J*L+J*L*J*L,:],J,L,J,L,360)
angle_iso = S20_iso(out0,8,16,360)

plot(angle_iso[4,4,:])

out0 = h5read("./Data/j8l16w1p2.h5", "main/data")
test = reshape(out0[2+J*L+1:2+J*L+J*L*J*L,:],J,L,J,L,360)
angle_iso = S20_iso(out0,8,16,360)

plot!(angle_iso[4,4,:])


function S20_iso(data,J,L,N)
    S20 = reshape(data[2+J*L+1:2+J*L+J*L*J*L,:],J,L,J,L,N)
    iso_mat = zeros(Int(L*(L+1)/2),J,J,N)
    k = 0
    @inbounds for i in 1:L
        for j in 1:i
            iso_mat[k+j,:,:,:] = sqrt.(S20[:,j,:,i,:])
        end
        k+=i
    end
    pow_mat = dropdims(sum(iso_mat,dims=1),dims=1)
    return pow_mat
end

out0 = h5read("./Data/abs2j8l16w1p2.h5", "main/data")
test = reshape(out0[2+J*L+1:2+J*L+J*L*J*L,:],J,L,J,L,360)
angle_iso = S20_iso(out0,8,16,360)

plot(angle_iso[4,4,:])

function S20_iso(data,J,L,N)
    S20 = reshape(data[2+J*L+1:2+J*L+J*L*J*L,:],J,L,J,L,N)
    iso_mat = zeros(Int(L*(L+1)/2),J,J,N)
    k = 0
    @inbounds for i in 1:L
        for j in 1:i
            iso_mat[k+j,:,:,:] = S20[:,j,:,i,:]
        end
        k+=i
    end
    pow_mat = dropdims(sum(iso_mat.*iso_mat,dims=1),dims=1)
    return pow_mat
end

out0 = h5read("./Data/abs2j8l16w1p2.h5", "main/data")
test = reshape(out0[2+J*L+1:2+J*L+J*L*J*L,:],J,L,J,L,360)
angle_iso = S20_iso(out0,8,16,360)

plot(angle_iso[4,4,:])

x = 1:360
plot(x,sin.(π/180*x) .+ 10)
plot(x,(sin.(π/180*x) .+ 10).^2)
plot!(x,(cos.(π/180*x) .+ 10).^2)
plot(x,(sin.(π/180*x) .+ 10).^2+(cos.(π/180*x) .+ 10).^2)
