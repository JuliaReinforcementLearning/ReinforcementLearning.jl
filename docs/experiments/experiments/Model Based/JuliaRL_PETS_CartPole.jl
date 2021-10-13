using ReinforcementLearning
using StableRNGs
using Flux
using IntervalSets

# Make wrapper to redefine reward and also allow for a query 
# about reward and termination based on some state/action/steps.
# PETS needs both `is_terminated` and `reward` to support specific keywords in this fashion.
struct CartPoleWrapper{E<:AbstractEnv} <: AbstractEnvWrapper
    env::E
end

function RLBase.is_terminated(env::CartPoleWrapper; current_state=state(env[!]), future_steps=0)
    x, xdot, theta, thetadot = current_state
    done = abs(x) > env[!].params.xthreshold ||
        abs(theta) > env[!].params.thetathreshold ||
        env[!].t + future_steps > env[!].params.max_steps
    return done
end

function RLBase.reward(env::CartPoleWrapper; last_action=env[!].last_action, current_state=state(env[!]))
    arm_length = 2 * env[!].params.halflength
    x, xdot, theta, thetadot = current_state
    target_dist = (x - arm_length * sin(theta))^2 + (arm_length + arm_length * cos(theta))^2
    cost_pos = exp(-target_dist)
    cost_act = 0.01 * sum(abs2, last_action)
    return -(cost_pos + cost_act)
end

function RL.Experiment(
    ::Val{:JuliaRL},
    ::Val{:PETS},
    ::Val{:CartPole},
    ::Nothing;
    save_dir = nothing,
    seed = 123,
)
    rng = StableRNG(seed)

    inner_env = CartPoleEnv(T = Float32, continuous = true, rng = rng)
    A = action_space(inner_env)
    low = A.left
    high = A.right

    env = StateTransformedEnv(
        ActionTransformedEnv(
            CartPoleWrapper(inner_env);
            action_mapping = x -> low + (x[1] + 1) * 0.5 * (high - low),
        );
        state_mapping = x -> [x[1:2]; sin(x[3]); cos(x[3]); x[4]], # TODO: this does not seem very efficient?
    )

    ns = length(state(env))
    na = 1

    init = glorot_uniform(rng)

    T = 5000
    hidden = 200

    agent = Agent(
        policy = PETSPolicy(
            optimizer = CEMTrajectoryOptimizer(
                lower_bound = [-1.0],
                upper_bound = [1.0],
                population = 500,
                elite_ratio = 0.1,
                iterations = 5,
                horizon = 15,
                α = 0.1,
                rng = rng,
            ),
            ensamble = [
                NeuralNetworkApproximator(
                    model = GaussianNetwork(
                        pre = Chain(
                            Dense(ns + na, hidden, leakyrelu, init = init), 
                            Dense(hidden, hidden, leakyrelu, init = init),
                            # Dense(200, 200, leakyrelu, init = init),
                        ),
                        μ = Chain(Dense(hidden, ns, init = init)),
                        logσ = Chain(Dense(hidden, ns, init = init)),
                        clampfun = softclamp,
                        min_σ = 1f-5,
                        max_σ = 1f2, 
                    ),
                    optimizer = ADAM(7.5e-4),
                ) for _ in 1:5
            ],
            batch_size = 256, # Can this be larger than update_after?
            start_steps = 200,
            start_policy = RandomPolicy(Space([-1.0..1.0 for _ in 1:na]); rng = rng),
            update_after = 200,
            update_freq = 50,
            predict_reward = false,
            rng = rng,
        ),
        trajectory = CircularArraySARTTrajectory(
            capacity = T,
            state = Vector{Float32} => (ns,),
            action = Vector{Float32} => (na,),
        ),
    )

    stop_condition = StopAfterStep(T, is_show_progress=!haskey(ENV, "CI"))
    hook = ComposedHook(
        TotalRewardPerEpisode(),
    )
    Experiment(agent, env, stop_condition, hook, "# Play CartPole with PETS")
end

#################################################################
#################################################################
#################################################################
using Plots, Statistics

ex = E`JuliaRL_PETS_CartPole`

mutable struct DataCollectionHook <: AbstractHook
    states::Vector{Vector{Float64}}
    actions::Vector{Float64}
    losses::Vector{Float64}
end
DataCollectionHook() = DataCollectionHook(Vector{Vector{Float64}}(undef, 0), Vector{Float64}(undef, 0), Vector{Float64}(undef, 0))

function (hook::DataCollectionHook)(::PreActStage, agent, env, action)
    push!(hook.states, state(env[!]))
    push!(hook.actions, Float64(action[1]))
    push!(hook.losses, Float64(mean(agent.policy.model_loss)))
end

ex.hook = ComposedHook(ex.hook.hooks..., DataCollectionHook(), StepsPerEpisode())

run(ex)

plot(ex.hook.hooks[end].steps)

actions = ex.hook.hooks[end-1].actions
plot(actions)

states = hcat(ex.hook.hooks[end-1].states...)
plot(states')

losses = ex.hook.hooks[end-1].losses;
plot(losses)
#################################################################
#################################################################
#################################################################

#+ tangle=false
using Plots
pyplot() #hide
ex = E`JuliaRL_PETS_CartPole`
run(ex)
plot(ex.hook.hooks[1].rewards)
savefig("assets/JuliaRL_PETS_CartPole.png") #hide

# ![](assets/JuliaRL_PETS_CartPole.png)