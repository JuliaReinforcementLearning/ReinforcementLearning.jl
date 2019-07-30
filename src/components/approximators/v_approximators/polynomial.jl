export PolynomialV, update!

"""
    struct PolynomialV <: AbstractVApproximator{Int}
        weights::Vector{Float64}
    end

See more details at Section (9.5.1) on Page 210 of the book *Sutton, Richard S., and Andrew G. Barto. Reinforcement learning: An introduction. MIT press, 2018.*
"""
struct PolynomialV <: AbstractVApproximator{Int}
    weights::Vector{Float64}
end

PolynomialV(order::Int) = PolynomialV(zeros(Float64, order+1))

function (p::PolynomialV)(s)
    sum(w * s^(i-1) for (i, w) in enumerate(p.weights))
end

function update!(p::PolynomialV, s, e)
    for i in 1:length(p.weights)
        p.weights[i] += e * s^(i-1)
    end
end
