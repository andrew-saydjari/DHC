using Revise, Test, ReferenceTests

push!(LOAD_PATH, pwd()*"/main")
using eqws

tests = [
    "filter_bank.jl",
    "compute.jl"
]

@testset "eqws" begin
for t in tests
    @testset "$t" begin
        include(t)
    end
end
end
