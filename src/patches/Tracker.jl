import Flux.Tracker:TrackedReal

Base.typemin(::Type{TrackedReal{T}}) where T = typemin(T)