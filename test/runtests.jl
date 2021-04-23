using Test, ReferenceTests

push!(LOAD_PATH, pwd()*"/main")
using DHC_2DUtils

tests = [
    "filter_bank.jl",
]

@testset "EqWS" begin
for t in tests
    @testset "$t" begin
        include(t)
    end
end
end
