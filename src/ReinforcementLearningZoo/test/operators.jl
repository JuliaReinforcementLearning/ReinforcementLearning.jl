import ReinforcementLearningCore
@testset "retrace" begin
    batch = (state= [[1 2 3], [10 11 12]], 
        action = [[1 2 3],[10 11 12]], 
        action_log_problog_prob = [log.([0.2,0.2,0.2]), log.([0.1,0.1,0.1])],
        reward = [[1f0,2f0,3f0],[10f0,11f0,12f0]], 
        terminal= [[0,0,1], [0,0,0]], 
        next_state = [[2 3 4],[11 12 13]])

    #define a fake policy where a = x and that returns the same log probabilities always
    policy(x; is_sampling = true, is_return_log_prob = false) = identity(x)
    policy(s,a) = log.([0.1/2,0.1/3,0.1/4])#both samples have the same current logprobs
    qnetwork(x, args...) = x[1, :] #the value of a state is its number
    λ, γ = 0.9, 0.99
    ReinforcementLearningCore.target(qnetwork) = qnetwork
    ops = retrace_operator(qnetwork, policy, batch, γ, λ)
    #handmade calculation of the correct ops
    op1 = 1*0.9*1/4*(1+0.99*2-1) + 0.99*0.9^2*1/4*1/6*(2+0.99*3-2) + 0.99^2*0.9^3*1/4*1/6*1/8*(3+0.99*4*1-3) 
    op2 = 1*0.9*0.5*(10+0.99*11-10) + 0.99*0.9^2*0.5*1/3*(11+0.99*12-11) + 0.99^2*0.9^3*0.5*0.33*0.25*(12+0.99*13*0-12)
    println("test")
    @test ops == [op1, op2]
end