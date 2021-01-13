using Statistics
using Plots
using BenchmarkTools
#using Profile
using LinearAlgebra

# put the cwd on the path so Julia can find the module
push!(LOAD_PATH, pwd())
using DHC_2DUtils
using MLDatasets
using Images


# read the MNIST training data, x=image, y=class
train_x, train_y = MNIST.traindata()
test_x, test_y   = MNIST.testdata()
# To view, heatmap(train_x[:,:,6]',yaxis=(:flip),aspect_ratio=1)

# transpose and flip, pad to 32, rebin to 64
function mnist_pad(im; θ=0.0)
    impad = zeros(Float64,32,32)
    impad[30:-1:3,3:30] = im'
    imbig = imresize(impad,(64,64))
    if θ != 0.0
        print("Rotating")
        imrot = imrotate(imbig, θ, axes(imbig), Constant())
        return imrot
    end
    return imbig
end


# call DHC WST transform
function mnist_DHC(x, y)
    # filter bank
    filter_hash = fink_filter_hash(1,8,nx=64,wd=2)

    # get transform size}
    Nimage = size(x)[end]
    image  = mnist_pad(x[:,:,1])
    test   = DHC_compute(image, filter_hash)
    Ncoeff = length(test)
    # allocate output arrays
    WST    = zeros(Ncoeff, Nimage)
    for i=1:Nimage
        image    = mnist_pad(x[:,:,i])
        WST[:,i] = DHC_compute(image, filter_hash)
    end
    return WST
end

nuse = 20000   # this takes 8.5 mins for 20000
@time wst = mnist_DHC(train_x[:,:,1:nuse],train_y[1:nuse])

nuse = 40000   # this takes 17 mins for 40000
@time wst4 = mnist_DHC(train_x[:,:,1:nuse],train_y[1:nuse])

nfilt = 33
S1  = wst4[1:2+nfilt,:]
S20 = wst4[3+nfilt:2+nfilt+nfilt^2,:]
S2  = wst4[3+nfilt+nfilt^2:end,:]


# Compute the mean and covariance of a set of vectors, x
function get_covar(x; cut=0.0)
    (Nd,Nx) = size(x)
    xx = x
    if cut != 0.0
        xmean, cov = get_covar(x)
        chi2 = get_chi2(x, xmean, cov)
        ind = findall(chi2 .< cut)
        println(Nd)
        xx = x[:,ind]
    end

    (Nd,Nx) = size(xx)
    if Nx < Nd println("Warning: Matrix not full rank") end
    x̄     = mean(xx,dims=2)[:,1]
    xsub  = xx.-x̄
    cov   = (xsub * xsub') ./ Nx
    return x̄, cov
end


# Compute chi2 of x given mean and covariance
function get_chi2(x, x̄, cov)
    Δx = x.-x̄
    χ2 = sum(Δx .* (inv(cov)*Δx), dims=1)[1,:]
end


# Arrange a stack of images in a grid
function imgrid(dat)
    (Ny,Nx,Nd) = size(dat)
    Ng = ceil(Int32,sqrt(Nd))
    grid = zeros(Float32,Ng*Ny,Ng*Nx)
    for i=1:Ng
        x0 = (i-1)*Nx
        for j=1:Ng
            y0 = (j-1)*Ny
            ii = (i-1)*Ng+j
            #println("i $i j $j x0=$x0 y0=$y0")
            if ii <= Nd grid[y0+1:y0+Ny, x0+1:x0+Nx] = dat[:,:,ii] end
        end
    end
    return grid
end


function mdisp(x, ind)
    dat = x[:,:,ind]
    grid = imgrid(dat)
    heatmap(grid',yaxis=(:flip))  # transpose and flip
end


function get_all_covar(train_x, train_y, wst; noise=1e-7)
    # select a class
    println("bar ",size(wst))
    (nk, Nd) = size(wst)
    #cut = 1.5*nk # ?????????
    cut=0

    clist = []
    mlist = []
    for i = 1:10
        class = mod(i, 10)
        ind   = findall(train_y[1:Nd] .== class)
        println("Class: ",class,"  ",length(ind))

        mn,cov = get_covar(wst[:,ind]+noise.*randn(nk,length(ind)),cut=cut)
        push!(clist, cov)
        push!(mlist, mn)
    end
    return mlist, clist
end

# mdisp(train_x,1:100)

function mnist_2class_plot(y, wst, mlist, clist, classes)
    # select a class
    Nd = (size(wst))[end]
    class1 = classes[1]
    class2 = classes[2]

    i1 = mod1(class1,10)
    i2 = mod1(class2,10)
    χ2_1   = get_chi2(wst, mlist[i1], clist[i1])
    χ2_2   = get_chi2(wst, mlist[i2], clist[i2])

    ind1 = findall(y[1:Nd] .== class1)
    ind2 = findall(y[1:Nd] .== class2)
    scatter(χ2_1[ind1],χ2_2[ind1],lims=(0,1000),msize=2,label=class1)
    scatter!(χ2_1[ind2],χ2_2[ind2],msize=2,label=class2)
    plot!([0,1000],[0,1000],label=nothing)
end


function make_iso(filter_hash,wst)
    (nk, Nd) = size(wst)
    Nf      = length(filter_hash["filt_value"])
    Nj      = length(filter_hash["j_value"])
    Nl      = length(filter_hash["theta_value"])

    jlind   = filter_hash["psi_index"]

    S       = reshape(S2, Nf, Nf, Nd)[jlind,jlind,:]
    Siso    = zeros(Nj, Nj, Nl, Nd)

    for j1 = 1:Nj
        for j2 = 1:Nj
            for l1 = 1:Nl
                for l2 = 1:Nl
                    DeltaL = mod(l1-l2, Nl)
                    Siso[j1,j2,DeltaL+1,:] += S[j1,l1,j2,l2,:]
                end
            end
        end
    end
    # Need to add phi part as well!!
    return Siso
end



mlist, clist = get_all_covar(train_x, train_y, S2)
mnist_2class_plot(train_y, S2, mlist, clist, [3,4])

S2iso = reshape(make_iso(filter_hash, S2), length(S2iso)÷Nd, Nd)
mlist, clist = get_all_covar(train_x, train_y, S2iso)
mnist_2class_plot(train_y, S2iso, mlist, clist, [3,4])

S20iso = reshape(make_iso(filter_hash, S20), length(S20iso)÷Nd, Nd)
mlist, clist = get_all_covar(train_x, train_y, S20iso)
mnist_2class_plot(train_y, S20iso, mlist, clist, [3,4])

Siso = [S20iso;S2iso]
mlist, clist = get_all_covar(train_x, train_y, Siso)
mnist_2class_plot(train_y, Siso, mlist, clist, [3,4])

Siso = log.([S20iso;S2iso])
mlist, clist = get_all_covar(train_x, train_y, Siso)
mnist_2class_plot(train_y, Siso, mlist, clist, [3,4])

Siso = [S20iso; log.(S20iso)]
mlist, clist = get_all_covar(train_x, train_y, Siso)
mnist_2class_plot(train_y, Siso, mlist, clist, [3,4])




Nd = 20000
class1 = 4
nk = 600
ind1 = findall(train_y[1:Nd] .== class1)
mn1,cov1 = get_covar(wst[200+1:200+nk,ind1]+1e-5 .*randn(nk,length(ind1)),cut=1000)
c1 = get_chi2(wst[200+1:200+nk,:], mn1, cov1)
histogram((c1[ind1]./nk))
println("Median: ",median(c1[ind1]), "   Mean: ",mean(c1[ind1]))

ind2 = findall((train_y[1:Nd] .== class1) .& (c1 .< 1.5*nk))
println("Keeping ",100*length(ind2)/length(ind1), " %")
mn2,cov2 = get_covar(wst[200+1:200+nk,ind2]+1e-5 .*randn(nk,length(ind2)))
c2 = get_chi2(wst[200+1:200+nk,:], mn2, cov2)
histogram((c2[ind2]./nk))
println("Median: ",median(c2[ind2]), "   Mean: ",mean(c2[ind2]))




cc = get_chi2(wst4[200+1:200+nk,20001:40000], mn1, cov1)
indc = findall(train_y[20001:20000+Nd] .== class1)

dc2 = c1-c2
histogram(c1-c2,xlims=(0,1500),bins=1000)

histogram(foo2,xlims=(0,1500))
histogram!(foo,bins=1000)
