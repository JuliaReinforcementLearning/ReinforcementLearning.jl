import ReinforcementLearning: Buffer, pushstateaction!, pushreturn!
function testsmallbackups()
    episode = [(0., false, 1, 1), (0., false, 2, 1), (0., false, 3, 1), # r, done, nexts, nexta
               (0., false, 4, 1), (0., false, 2, 1), (0., false, 5, 1), 
               (0., false, 2, 2), (0., false, 6, 1), (2., true, 1, 1), (0., false, 2, 2)]
    na = 2; ns = 6; γ = .9
    learner = SmallBackups(na = na, ns = ns, γ = γ, maxcount = 10)
    buffer = Buffer()
    pushstateaction!(buffer, episode[1][3:4]...)
    for (r, done, s, a) in episode[2:end]
        pushreturn!(buffer, r, done)
        pushstateaction!(buffer, s, a)
        update!(learner, buffer)
    end
    Nsa = zeros(Int64, na, ns)
    for (r, done, s, a) in episode[1:end-1]; Nsa[a, s] += 1; end
    @test Nsa == learner.Nsa
    @test learner.Ns1a0s0[1] == Dict()
    @test learner.Ns1a0s0[2] == Dict((1,1) => 2, (1, 4) => 1, (1, 5) => 1)
    @test 2 * [γ^2; γ; γ^3; γ^2; γ^2; 1] == learner.V

    mdp = DetTreeMDP(na = 2, depth = 2)
    seed!(123)
    x = RLSetup(SmallBackups(ns = mdp.observationspace.n, na = mdp.actionspace.n, γ = .99, initvalue = 0),
                mdp, ConstantNumberSteps(10^4));
    learn!(x)
    reset!(mdp)
    seed!(123)
    y = RLSetup(SmallBackups(ns = mdp.observationspace.n, na = mdp.actionspace.n, γ = .99, initvalue = Inf64),
                mdp, ConstantNumberSteps(10^4));
    learn!(y)
    @test y.learner.Q[:, 1:end-1] ≈ x.learner.Q[:, 1:end-1]
end
testsmallbackups()
