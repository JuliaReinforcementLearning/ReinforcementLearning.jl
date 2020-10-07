export normlogpdf

using Distributions: DiscreteNonParametric, support, probs

# watch https://github.com/JuliaStats/Distributions.jl/issues/1183

"""
GPU automatic differentiable version for the logpdf function of normal distributions.
Adding an epsilon value to guarantee numeric stability if sigma is exactly zero
(e.g. if relu is used in output layer).
"""
function normlogpdf(μ, σ, x; ϵ = 1.0f-8)
    z = (x .- μ) ./ (σ .+ ϵ)
    -(z .^ 2 .+ log(2.0f0π)) / 2.0f0 .- log.(σ .+ ϵ)
end
