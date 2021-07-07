include("MultiArmBanditsEnv.jl")
include("RandomWalk1D.jl")
include("TigerProblemEnv.jl")
include("MontyHallEnv.jl")
include("RockPaperScissorsEnv.jl")
include("TicTacToeEnv.jl")
include("TinyHanabiEnv.jl")
include("PigEnv.jl")
include("KuhnPokerEnv.jl")
include("CartPoleEnv.jl")
include("MountainCarEnv.jl")
include("PendulumEnv.jl")
include("BitFlippingEnv.jl")

# checking the state of env is enough?
Base.:(==)(env1::AbstractEnv, env2::AbstractEnv) = state(env1) == state(env2)
Base.hash(env::AbstractEnv, h::UInt) = hash(state(env), h)