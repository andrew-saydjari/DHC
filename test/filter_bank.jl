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
end
