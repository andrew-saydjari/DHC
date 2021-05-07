function apodizer(data::Array{Float64, 2})
    (Nx, Ny) = size(data)
    Amat = wind_2d(Nx)
    datad_w = fweights(Amat)
    meanVal = mean(data, datad_w) #<AF>
    temp2d_a = (data.-meanVal).*Amat.+meanVal #A(F-μ) + μ
    return temp2d_a
end

function wind_2d(nx)
    dx   = nx/2-1
    filter = zeros(Float64, nx, nx)
    A = DSP.tukey(nx, 0.3)
    itp = extrapolate(interpolate(A,BSpline(Linear())),0)
    @inbounds for x = 1:nx
        sx = x-dx-1    # define sx,sy so that no fftshift() needed
        for y = 1:nx
            sy = y-dx-1
            r  = sqrt.((sx).^2 + (sy).^2) + nx/2
            filter[x,y] = itp(r)
        end
    end
    return filter
end
