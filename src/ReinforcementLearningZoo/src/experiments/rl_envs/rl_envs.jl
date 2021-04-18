using .ReinforcementLearningEnvironments

for f in readdir(@__DIR__)
    if f != splitdir(@__FILE__)[2]
        include(f)
    end
end

# Build Dueling Network
function build_dueling_network(network::Chain)
    lm = length(network)
    if !(network[lm] isa Dense) || !(network[lm-1] isa Dense) 
        error("The Qnetwork provided is incompatible with dueling.")
    end
    base = Chain([deepcopy(network[i]) for i=1:lm-2]...)
    last_layer_dims = size(network[lm].W, 2)
    val = Chain(deepcopy(network[lm-1]), Dense(last_layer_dims, 1))
    adv = Chain([deepcopy(network[i]) for i=lm-1:lm]...)
    return DuelingNetwork(base, val, adv)
end