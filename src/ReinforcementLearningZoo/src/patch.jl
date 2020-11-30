using ReinforcementLearningCore

(app::NeuralNetworkApproximator)(args...; kwargs...) = app.model(args...; kwargs...)

using AbstractTrees
using TensorBoardLogger: TBLogger

AbstractTrees.children(t::StructTree{T}) where {T<:Union{TBLogger}} = ()
