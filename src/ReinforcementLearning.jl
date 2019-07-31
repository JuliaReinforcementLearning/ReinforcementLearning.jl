module ReinforcementLearning
    export RL
    const RL = ReinforcementLearning

    include("Utils/Utils.jl")

    import .Utils:capacity
    include("components/components.jl")
end