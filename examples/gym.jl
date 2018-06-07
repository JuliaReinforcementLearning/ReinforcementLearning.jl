using PyCall, ReinforcementLearning, Flux
@pyimport gym
# @pyimport roboschool

import ReinforcementLearning.interact!,
ReinforcementLearning.getstate,
ReinforcementLearning.reset!

function getspace(space)
    if pyisinstance(space, gym.spaces[:box][:Box])
        ReinforcementLearning.Box(space[:low], space[:high])
    elseif pyisinstance(space, gym.spaces[:discrete][:Discrete])
        1:space[:n]
    else
        error("Don't know how to convert $(pytypeof(space)).")
    end
end
mutable struct GymEnvState
    done::Bool
end
struct GymEnv{TObject, TObsSpace, TActionSpace}
    pyobj::TObject
    observation_space::TObsSpace
    action_space::TActionSpace
    state::GymEnvState
end
function GymEnv(name::String)
    pyenv = gym.make(name)
    obsspace = getspace(pyenv[:observation_space])
    actspace = getspace(pyenv[:action_space])
    env = GymEnv(pyenv, obsspace, actspace, GymEnvState(false))
    reset!(env)
    env
end

function interactgym!(action, env)
    if env.state.done 
        s = reset!(env)
        r = 0
        d = false
    else
        s, r, d = env.pyobj[:step](action)
    end
    env.state.done = d
    s, r, d
end
interact!(action, env::GymEnv) = interactgym!(action, env)
interact!(action::Int64, env::GymEnv) = interactgym!(action - 1, env)
reset!(env::GymEnv) = env.pyobj[:reset]()
getstate(env::GymEnv) = (env.pyobj[:env][:state], false) # doesn't work for all envs


# List all envs

gym.envs[:registry][:all]()


# CartPole example

env = GymEnv("CartPole-v0")
learner = DQN(Chain(Dense(4, 48, relu), Dense(48, 24, relu), Dense(24, 2)),
                  updateevery = 1, updatetargetevery = 100,
                  startlearningat = 50, minibatchsize = 32,
                  doubledqn = false, replaysize = 10^3, 
                  opttype = x -> ADAM(x, .0005))
x = RLSetup(learner, env, ConstantNumberEpisodes(500),
            callbacks = [Progress(), EvaluationPerEpisode(TimeSteps())])
@time learn!(x)

xvis = RLSetup(learner, env, ConstantNumberSteps(1))
for _ in 1:500
    env.pyobj[:render]()
    run!(xvis, fillbuffer = true)
end
env.pyobj[:close]()


