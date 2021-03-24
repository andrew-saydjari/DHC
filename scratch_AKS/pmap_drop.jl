a = 1:1:20
b = 21:1:40

iter_prod = Iterators.product(a,b)

Iterators.take(iter_prod,100)

size(iter_prod)

for i=1:10
    subset = Iterators.take(iter_prod,100)
    for x in subset
        println(x)
    end
    iter_prod = Iterators.drop(iter_prod,100)
end

length(iter_prod)

n_batch = 10
batch_size = length(iter_prod)÷n_batch

for i=1:n_batch
    subset = Iterators.take(iter_prod,batch_size)
    for x in subset
        println(x)
    end
    iter_prod = Iterators.drop(iter_prod,batch_size)
end

iter_prod = Iterators.product(a,b)
n_batch = 10
batch_size = length(iter_prod)÷n_batch
fid = h5open("../DHC/scratch_AKS/data/test.h5","w")
dset = create_dataset(fid, "A", datatype(Float64), dataspace(2,400), chunk=(2,40))
close(fid)
for i=1:n_batch
    subset = Iterators.take(iter_prod,batch_size)
    temp = zeros(2,batch_size)
    for (ind,x) in enumerate(subset)
        temp[:,ind] .= x[:]
    end
    fid = h5open("../DHC/scratch_AKS/data/test.h5","r+")
    data = open_dataset(fid, "A")
    data[:,(1:batch_size).+(i-1)*batch_size] = temp
    close(fid)
    iter_prod = Iterators.drop(iter_prod,batch_size)
end

A = h5read("../DHC/scratch_AKS/data/test.h5", "A")

iter_prod = Iterators.product(a,b)
n_batch = 10
batch_size = length(iter_prod)÷n_batch
fid = h5open("../DHC/scratch_AKS/data/testc.h5","w")
dset = create_dataset(fid, "A", datatype(Float64), dataspace(2,400), chunk=(2,40),compress=3)
close(fid)
for i=1:n_batch
    subset = Iterators.take(iter_prod,batch_size)
    temp = zeros(2,batch_size)
    for (ind,x) in enumerate(subset)
        temp[:,ind] .= x[:]
    end
    fid = h5open("../DHC/scratch_AKS/data/testc.h5","r+")
    data = open_dataset(fid, "A")
    data[:,(1:batch_size).+(i-1)*batch_size] = temp
    close(fid)
    iter_prod = Iterators.drop(iter_prod,batch_size)
end

A = h5read("../DHC/scratch_AKS/data/testc.h5", "A")


iter_prod = Iterators.product(a,b)
n_batch = 10
batch_size = length(iter_prod)÷n_batch
fid = h5open("../DHC/scratch_AKS/data/testc2.h5","w")
dset = create_dataset(fid, "A", datatype(Float64), dataspace(2,400), chunk=(2,40),compress=6)
close(fid)
for i=1:n_batch
    subset = Iterators.take(iter_prod,batch_size)
    temp = zeros(2,batch_size)
    for (ind,x) in enumerate(subset)
        temp[:,ind] .= x[:]
    end
    fid = h5open("../DHC/scratch_AKS/data/testc2.h5","r+")
    data = open_dataset(fid, "A")
    data[:,(1:batch_size).+(i-1)*batch_size] = temp
    close(fid)
    iter_prod = Iterators.drop(iter_prod,batch_size)
end

A = h5read("../DHC/scratch_AKS/data/testc2.h5", "A")

iter_prod = Iterators.product(a,b)
n_batch = 10
batch_size = length(iter_prod)÷n_batch
fid = h5open("../DHC/scratch_AKS/data/testc2.h5","w")
dset = create_dataset(fid, "A", datatype(Float64), dataspace(2,400), chunk=(2,40),suhffle=(),deflate=3)
close(fid)
for i=1:n_batch
    subset = Iterators.take(iter_prod,batch_size)
    temp = zeros(2,batch_size)
    for (ind,x) in enumerate(subset)
        temp[:,ind] .= x[:]
    end
    fid = h5open("../DHC/scratch_AKS/data/testc2.h5","r+")
    data = open_dataset(fid, "A")
    data[:,(1:batch_size).+(i-1)*batch_size] = temp
    close(fid)
    iter_prod = Iterators.drop(iter_prod,batch_size)
end

A = h5read("../DHC/scratch_AKS/data/testc2.h5", "A")

## Ok, so the summary is to just batch it and save with some showprogress on the outer on and deflate
# by = 3 (need to better understand what that number is) I think it is related to window width of
# zlib... tbd

iter_prod = Iterators.product(a,b)
n_batch = 10
batch_size = length(iter_prod)÷n_batch
fid = h5open("../DHC/scratch_AKS/data/testc2.h5","w")
dset = create_dataset(fid, "A", datatype(Float64), dataspace(2,400), chunk=(2,40),shuffle=(),deflate=3)
close(fid)
for i=1:n_batch
    subset = Iterators.take(iter_prod,batch_size)
    temp = zeros(2,batch_size)
    for (ind,x) in enumerate(subset)
        temp[:,ind] .= x[:]
    end
    fid = h5open("../DHC/scratch_AKS/data/testc2.h5","r+")
    data = open_dataset(fid, "A")
    data[:,(1:batch_size).+(i-1)*batch_size] = temp
    close(fid)
    iter_prod = Iterators.drop(iter_prod,batch_size)
end

A = h5read("../DHC/scratch_AKS/data/testc2.h5", "A")
