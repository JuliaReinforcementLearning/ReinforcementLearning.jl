export Preprocessor, FourierPreprocessor, PolynomialPreprocessor, TilingPreprocessor

using .Utils:Tiling, encode

#####
# Preprocessor
#####

struct Preprocessor{T<:Function}
    f::T
end

(p::Preprocessor)(s) = p.f(s)

#####
# FourierPreprocessor
#####

struct FourierPreprocessor
    order::Int
end

(p::FourierPreprocessor)(s) = [cos(i * π * s) for i in 0:p.order]

#####
# PolynomialPreprocessor
#####

struct PolynomialPreprocessor
    order::Int
end

(p::PolynomialPreprocessor)(s) = [s^i for i in 0:p.order]

#####
# TilingPreprocessor
#####

struct TilingPreprocessor{Tt<:Tiling}
    tilings::Vector{Tt}
end

(p::TilingPreprocessor)(s) = [encode(t, s) for t in p.tilings]