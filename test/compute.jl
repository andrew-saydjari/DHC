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

@testset "std_use" begin
    a = rand(256,256)
    ref = fink_filter_hash(1,8)
    orig = eqws_compute(a,ref)
    @test size(orig)[1] == 2452

    rod_test = rod_image(1,1,30,35,8)
    filter_hash = fink_filter_hash(1,8)
    test = eqws_compute(rod_test,filter_hash)
    #check well sampled images sum layer by layer to 1
    @test abs(sum(test[3:end])-2) < 1e-6
end

@testset "p_power" begin
    filter_hash = fink_filter_hash(1,8)
    test_hash = fink_filter_hash_p(1,8)

    a = rand(256,256)
    orig = eqws_compute(a,filter_hash)
    new = eqws_compute_p(a,test_hash)
    @test orig ≈ new
end

@testset "conv_maps" begin
    a = rand(256,256)
    filter_hash = fink_filter_hash(1,8)
    d = eqws_compute_convmap(a,filter_hash)
    d2 = sum.(d)
    d1 = eqws_compute(a,filter_hash)
    @test maximum(abs.(d1.-d2)) < 1e-15
end
