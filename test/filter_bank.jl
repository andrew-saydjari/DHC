@testset "p_power" begin
    ref = fink_filter_bank(1,8)
    test = fink_filter_bank_p(1,8)
    @test ref == test

    ref = fink_filter_hash(1,8)
    test = fink_filter_hash_p(1,8)
    @test ref == test

    ref = fink_filter_hash(1,8)
    test = fink_filter_hash_p(1,8)
    @test ref == test

    a = rand(256,256)
    orig = DHC_compute(a,ref)
    new = DHC_compute_p(a,test)
    @test orig ≈ new
end

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

@testset "compute" begin
    rod_test = rod_image(1,1,30,35,8)
    filter_hash = fink_filter_hash(1,8)
    test = DHC_compute(rod_test,filter_hash)
    @test sum(test[3:end])-2 < 1e-6 #check well sampled images sum layer by layer to 1
end
