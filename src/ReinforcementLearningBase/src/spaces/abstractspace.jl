abstract type AbstractSpace end

"""
    sample(s::AbstractSpace)

Get a random sample from `s`.
"""
function sample end

"""
    occursin(x, s::AbstractSpace)

Return wheather `x` is a valid sample in space `s`.
"""
function occursin end