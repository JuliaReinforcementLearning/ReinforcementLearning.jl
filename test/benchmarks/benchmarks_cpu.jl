using ReinforcementLearning, ReinforcementLearningEnvironmentDiscrete,
ReinforcementLearningEnvironmentClassicControl, ReinforcementLearningEnvironmentGym, 
ReinforcementLearningEnvironmentAtari, BenchmarkTools, Flux, Pkg, InteractiveUtils

versioninfo()
println()
Pkg.status()

env = CartPole()
rlsetup = RLSetup(ActorCriticPolicyGradient(ns = 4, na = 2, α = .02, 
                                            nsteps = 25), 
                  env, ConstantNumberSteps(400))
println("\n\nCartPole ActorCriticPolicyGradient")
show(stdout, "text/plain", @benchmark(learn!($rlsetup)))

env = GymEnv("CartPole-v0")
rlsetup = RLSetup(ActorCriticPolicyGradient(ns = 4, na = 2, α = .02, 
                                            nsteps = 25), 
                  env, ConstantNumberSteps(400))
println("\n\nCartPoleGym ActorCriticPolicyGradient")
show(stdout, "text/plain", @benchmark(learn!($rlsetup)))

learner = DQN(Chain(Dense(4, 16, relu), Dense(16, 32, relu), Dense(32, 2)),
              updateevery = 1, updatetargetevery = 200, startlearningat = 200,
              minibatchsize = 16, replaysize = 10^3)
rlsetup = RLSetup(learner, env, ConstantNumberSteps(400))
println("\n\nCartPole DQN")
show(stdout, "text/plain", @benchmark(learn!($rlsetup)))

env = MountainCar(maxsteps = 10^4)
preprocessor = TilingStateAggregator(StateAggregator([-1.2, -.07], [.5, .07], [8, 8]), 8)
rlsetup = RLSetup(Sarsa(ns = 8*8^2, na = 3, α = 1/8, λ = .96, γ = 1.), 
                  env, ConstantNumberSteps(400),
                  preprocessor = preprocessor)
rlsetup.policy.ϵ = 0
println("\n\nMountainCar TilingSarsa")
show(stdout, "text/plain", @benchmark(learn!($rlsetup)))

env = DiscreteMaze(ngoals = 5)
ns = length(env.mdp.isterminal)
rlsetup = RLSetup(SmallBackups(na = 4, ns = ns, γ = .99), 
                  env, ConstantNumberSteps(400))
println("\n\nDiscreteMaze SmallBackups")
show(stdout, "text/plain", @benchmark(learn!($rlsetup)))

env = AtariEnv("breakout")
model = Chain(x -> x./Float64(255), Conv((8, 8), 4 => 16, relu, stride = (4, 4)), 
              x -> reshape(x, :, size(x, 4)),
              Dense(6400, length(env.actions)));
learner = DQN(model, opttype = x -> Flux.SGD(x, .0001), 
              loss = huberloss, doubledqn = true,
              updatetargetevery = 2500, nsteps = 10, minibatchsize = 8,
              updateevery = 4, replaysize = 10^5, nmarkov = 4,
              startlearningat = 200);
preprocessor = ImagePreprocessor((160, 210), [ImageResizeNearestNeighbour((84, 84)),
                                              x -> reshape(x, (84, 84, 1))])
rlsetup = RLSetup(learner, env, ConstantNumberSteps(400), preprocessor = preprocessor)
println("\n\nAtari DQN")
show(stdout, "text/plain", @benchmark(learn!($rlsetup), samples = 10, seconds = 30))
