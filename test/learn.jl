import ReinforcementLearningBase: reset!
function testlearn()
    mdp = MDP()
    learner = Sarsa()
    x = RLSetup(
        learner = Sarsa(),
        environment = mdp,
        callbacks = [TotalReward(), RecordAll()],
        stoppingcriterion = ConstantNumberSteps(10),
    )
    seed!(13452)
    reset!(mdp)
    learn!(x)
    learn!(x)

    x2 = RLSetup(
        learner = Sarsa(),
        environment = mdp,
        callbacks = [TotalReward(), RecordAll()],
        stoppingcriterion = ConstantNumberSteps(20),
    )
    seed!(13452)
    reset!(mdp)
    learn!(x2)
    @test x.learner.params == x2.learner.params
end
testlearn()
