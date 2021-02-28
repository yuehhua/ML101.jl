using ML101
using Test

using DataFrames

@testset "ML101.jl" begin
    @testset "linear_regression.jl" begin
        include("linear_regression.jl")
    end
end