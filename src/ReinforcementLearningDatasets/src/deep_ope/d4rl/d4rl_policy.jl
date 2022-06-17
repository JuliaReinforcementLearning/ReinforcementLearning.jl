using Pickle
using Flux
using Setfield
using Pipe: @pipe

export D4RLGaussianNetwork
export d4rl_policy

"""
    D4RLGaussianNetwork

Returns `action` and `μ` when called.
"""
Base.@kwdef struct D4RLGaussianNetwork{P,U,S}
    pre::P = identity
    μ::U
    logσ::S
end

Flux.@functor D4RLGaussianNetwork

function (model::D4RLGaussianNetwork)(
    state::AbstractArray;
    rng::AbstractRNG=MersenneTwister(123), 
    noisy::Bool=true
)
    x = model.pre(state)
    μ, logσ = model.μ(x), model.logσ(x)
    if noisy
        a = μ + exp.(logσ) .* Float32.(randn(rng, size(μ)))
    else
        a = μ + exp.(logσ)
    end
    a, μ
end 
"""
    d4rl_policy(env, agent, epoch)

Return a [`D4RLGaussianNetwork`](@ref) from [deep_ope](https://github.com/google-research/deep_ope) with preloaded weights.
Check [deep_ope](https://github.com/google-research/deep_ope) with preloaded weights for more info. Check out d4rl_policy_params() for more info on arguments.

# Arguments

- `env::String`: name of the `env`.
- `agent::String`: can be `dapg` or `online`.
- `epoch::Int`: can be in `0:10`.
"""
function d4rl_policy(
        env::String,
        agent::String, 
        epoch::Int)
    
    folder_prefix = "deep-ope-d4rl"
    try
        @datadep_str "$(folder_prefix)-$(env)_$(agent)_$(epoch)"
    catch x
        if isa(x, KeyError)
            error("invalid params, checkout d4rl_policy_params() for more info")
        end
    end
    policy_folder = @datadep_str "$(folder_prefix)-$(env)_$(agent)_$(epoch)"
    policy_file = "$(policy_folder)/$(readdir(policy_folder)[1])"
    
    model_params = Pickle.npyload(policy_file)
    @pipe parse_network_params(model_params) |> build_model(_...)
end

function parse_network_params(model_params::Dict)
    size_dict = Dict{String, Tuple}()
    nonlinearity = nothing
    output_transformation = nothing
    for param in model_params
        param_name, param_value = param[1], param[2]
        if typeof(param_value) <: AbstractArray
            size_dict[param_name] = reverse(size(param_value))
        else
            if param_name == "nonlinearity"
                if param_value == "relu"
                    nonlinearity = relu
                else
                    nonlinearity = tanh
                end
            else
                if param_value == "tanh_gaussian" 
                    output_transformation = tanh
                else
                    output_transformation = identity
                end
            end
        end
    end
    model_params, size_dict, nonlinearity, output_transformation
end

function build_model(model_params::Dict, size_dict::Dict, nonlinearity::Function, output_transformation::Function)
    fc_0 = Dense(size_dict["fc0/weight"]..., nonlinearity)
    fc_0 = @set fc_0.weight = model_params["fc0/weight"]
    fc_0 = @set fc_0.bias = model_params["fc0/bias"]
    
    fc_1 = Dense(size_dict["fc1/weight"]..., nonlinearity)
    fc_1 = @set fc_1.weight = model_params["fc1/weight"]
    fc_1 = @set fc_1.bias = model_params["fc1/bias"]
    
    μ_fc = Dense(size_dict["last_fc/weight"]...)
    μ_fc = @set μ_fc.weight = model_params["last_fc/weight"]
    μ_fc = @set μ_fc.bias = model_params["last_fc/bias"]
    
    log_σ_fc = Dense(size_dict["last_fc_log_std/weight"]...)
    log_σ_fc = @set log_σ_fc.weight = model_params["last_fc_log_std/weight"]
    log_σ_fc = @set log_σ_fc.bias = model_params["last_fc_log_std/bias"]
    
    pre = Chain(
        fc_0,
        fc_1
    )
    μ = Chain(μ_fc)
    log_σ = Chain(log_σ_fc)
        
    D4RLGaussianNetwork(pre, μ, log_σ)
end