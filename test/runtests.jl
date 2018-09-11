using ReinforcementLearning, Test

# include("buffer.jl")
# include("traces.jl")
# include("epsilongreedypolicies.jl")
# include("policies.jl")
# include("tdlearning.jl")
# include("losses.jl")
# include("learnmdp.jl")

@testset "linear function approximation" begin include("linfuncapprox.jl") end
# @testset "preprocessor" begin include("preprocessors.jl") end
# @testset "monte carlo" begin include("montecarlo.jl") end
# @testset "small backups" begin include("smallbackups.jl") end
