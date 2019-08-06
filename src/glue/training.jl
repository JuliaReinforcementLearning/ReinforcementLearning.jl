export train

using ReinforcementLearningEnvironments
using TensorBoardLogger, Logging

mutable struct RuntimeInfo
    step::Int
    episode::Int
    logger::TBLogger
    meta::Dict{Any,Any}
end

inc_episode(rt::RuntimeInfo) = rt.episode += 1
inc_step(rt::RuntimeInfo) = rt.step += 1
Logging.with_logger(f::Function, rt::RuntimeInfo) = with_logger(f, rt.logger)

function train(agent::AbstractAgent, env::AbstractEnv, stop_condition::Function, runtime_info)
    while true
        stop_condition(agent, env, runtime_info) && break

        obs = observe(env)

        if is_terminal(obs)
            post_episode(agent, env, runtime_info)
            pre_episode(agent, env, runtime_info)
            obs = observe(env)
        end

        action = agent(obs, runtime_info)
        pre_act(agent, env, runtime_info)
        env(action)
        post_act(agent, env, runtime_info)
    end
end

is_terminal(obs) = obs.isdone

pre_episode(agent, env, runtime_info) = reset!(env)
post_episode(agent, env, runtime_info) = inc_episode(runtime_info)

pre_act(agent, env, runtime_info) = nothing
post_act(agent, env, runtime_info) = inc_step(runtime_info)