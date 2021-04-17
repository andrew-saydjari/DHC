@testset "p_power" begin
    test_p = fink_filter_bank_p(1,8)
    test_bank = fink_filter_bank(1,8)
    test_bank .== test_p
end
