export AbstractApproximator,
    ApproximatorStyle, Q_APPROXIMATOR, QApproximator, V_APPROXIMATOR, VApproximator

"""
    (app::AbstractApproximator)(env)

An approximator is a functional object for value estimation.
It serves as a black box to provides an abstraction over different 
kinds of approximate methods (for example DNN provided by Flux or Knet).
"""
abstract type AbstractApproximator end

"""
    update!(a::AbstractApproximator, correction)

Usually the `correction` is the gradient of inner parameters.
"""
function RLBase.update!(a::AbstractApproximator, correction) end

#####
# traits
#####

abstract type AbstractApproximatorStyle end

"""
Used to detect what an [`AbstractApproximator`](@ref) is approximating.
"""
function ApproximatorStyle(::AbstractApproximator) end

struct QApproximator <: AbstractApproximatorStyle end

const Q_APPROXIMATOR = QApproximator()

struct VApproximator <: AbstractApproximatorStyle end

const V_APPROXIMATOR = VApproximator()
