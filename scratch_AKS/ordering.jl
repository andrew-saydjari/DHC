using Random

temp = zeros(Int64, 3, 2, 4)

for i in 1:3
    for j in 1:2
        for k in 1:4
            temp[i,j,k] = i*j*k
        end
    end
end

temp

using HDF5

temp[:]

h5write("../DHC/scratch_AKS/order_test.h5", "main/data", temp[:])

for
