using BenchmarkTools
using BenchmarkTools:prettytime, prettymemory
using Statistics

using ReinforcementLearningEnvironments
using ArcadeLearningEnvironment
using POMDPModels
using ViZDoom
using PyCall
using Hanabi

const N_STEPS = 1000

function run_random_actions(env, n=N_STEPS)
    reset!(env)
    as = action_space(env)
    for _ in 1:n
        a = rand(as)
        obs, reward, isdone = interact!(env, a)
        if isdone
            reset!(env)
        end
    end
end

function run_random_actions(env::HanabiEnv, n=N_STEPS)
    reset!(env)
    for _ in 1:n
        a = rand(legal_actions(env))
        obs, reward, isdone = interact!(env, a)
        if isdone
            reset!(env)
        end
    end
end

const gym_env_names = filter(
    x -> x != "KellyCoinflipGeneralized-v0",
    ReinforcementLearningEnvironments.list_gym_env_names(modules=[
        "gym.envs.algorithmic",
        "gym.envs.classic_control",
        "gym.envs.toy_text",
        "gym.envs.unittest"]))  # mujoco, box2d, robotics are not tested here

const atari_env_names = filter(x -> x in ["pong"], ReinforcementLearningEnvironments.list_atari_rom_names())  # many atari games will have segment fault

function write_benchmark_file()
    f = open(joinpath(@__DIR__, splitext(@__FILE__)[1] * ".md"), "w")
    println(
        f,
        """# Benchmarks of the runtime for different environments

        Each environment is estimated to run **$N_STEPS** steps.

        | Environment | mean time | median time | memory | allocs |
        | :---------- | --------: | ----------: | -----: | -----: |"""
    )
    for env_exp in [
        :(HanabiEnv()),
        :(basic_ViZDoom_env()),
        :(CartPoleEnv()),
        :(MountainCarEnv()),
        :(PendulumEnv()),
        :(MDPEnv(LegacyGridWorld())),
        :(POMDPEnv(TigerPOMDP())),
        :(SimpleMDPEnv()),
        :(deterministic_MDP()),
        :(absorbing_deterministic_tree_MDP()),
        :(stochastic_MDP()),
        :(stochastic_tree_MDP()),
        :(deterministic_tree_MDP_with_rand_reward()),
        :(deterministic_tree_MDP()),
        :(deterministic_MDP()),
        (:(AtariEnv($x)) for x in atari_env_names)...,
        (:(GymEnv($x)) for x in gym_env_names)...
        ]
        b = @benchmark run_random_actions($(eval(env_exp)))
        @info "Benchmark of $env_exp:\n$b"
        b_mean = mean(b)
        b_median = median(b)
        println(f, "|", join([env_exp, prettytime(b_mean.time), prettytime(b_median.time), prettymemory(b.memory), b.allocs], "|"), "|")
    end
    close(f)
end