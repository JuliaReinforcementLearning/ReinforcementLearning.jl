module ReinforcementLearning
    export RL
    const RL = ReinforcementLearning

    include("patches/patches.jl")

    using Reexport
    include("Utils/Utils.jl")
    include("components/components.jl")
    include("glue/glue.jl")
end