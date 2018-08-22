import ReinforcementLearning: ArrayCircularBuffer, ArrayStateBuffer, 
pushstateaction!, pushreturn!, preprocessstate, nmarkovgetindex

struct MyPreprocessor
    N::Int64
end
preprocessstate(p::MyPreprocessor, s) = reshape(Int64[s == i for i in 1:p.N], p.N, 1)
function testbuffers()
    a = ArrayCircularBuffer(Array, Int64, (1), 8)
    for i in 1:5 push!(a, [i]) end
    @test a[1] == [1]
    @test a[lastindex(a)] == [5]
    @test lastindex(a) == 5
    @test nmarkovgetindex(a, 5, 3)[:] == collect(3:5)
    @test nmarkovgetindex(a, lastindex(a), 4)[:] == collect(2:5)
    for i in 6:12 push!(a, [i]) end
    @test a[1] == [5]
    @test a[lastindex(a)] == [12]
    @test nmarkovgetindex(a, lastindex(a), 4)[:] == collect(9:12)

    a = ArrayStateBuffer(capacity = 7)
    for i in 1:4
        pushstateaction!(a, [i], i)
        pushreturn!(a, i, false)
    end
    pushstateaction!(a, [5], 5)
    @test a.states[3][1] == a.actions[3] == a.rewards[3]
    pushreturn!(a, 5, false)
    for i in 6:9
        pushstateaction!(a, [i], i)
        pushreturn!(a, i, false)
    end
    pushstateaction!(a, [10], 10)
    @test a.states[3][1] == a.actions[3] == a.rewards[3]

    for T in [7, 97]
        p = ForcedPolicy(rand(1:4, 100))
        ends = rand(1:100, 10)
        env = ForcedEpisode([[rand(1:10)] for _ in 1:100], 
                            [i in ends ? true : false for i in 1:100], rand(100))
        x = RLSetup(1, env, ConstantNumberSteps(T), policy = p, 
                    buffer = ArrayStateBuffer(capacity = 10, datatype = Int64), 
                    callbacks = [RecordAll()],
                    fillbuffer = true, islearning = false)
        learn!(x)
        @test x.buffer.actions[end-5:end] == x.callbacks[1].actions[end-5:end]
        @test x.buffer.done[end-5:end] == x.callbacks[1].done[end-5:end]
        @test x.buffer.states[end-5:end][:] == vcat(x.callbacks[1].states[end-5:end]...)
        @test x.buffer.rewards[end-4:end] == x.callbacks[1].rewards[end-4:end]
    end

    x = RLSetup(DQN(x -> mean(x, dims = 2)[1:4], nmarkov = 4, na = 4), MDP(), 
                ConstantNumberSteps(100),
                preprocessor = MyPreprocessor(10),
                callbacks = [RecordAll()])
    learn!(x)
    i = 77
    @test findall(x -> x != 0, nmarkovgetindex(x.buffer.states, i, 4)[:]) .% 10 == x.callbacks[1].states[i-4:i-1] .% 10
    @test x.buffer.actions[i-3:i+3] == x.callbacks[1].actions[i-4:i+2]
    @test lastindex(x.buffer.states) == lastindex(x.buffer.actions) == 1 + lastindex(x.callbacks[1].states)
    @test nmarkovgetindex(x.buffer.states, lastindex(x.buffer.states), 4) == nmarkovgetindex(x.policy.buffer, lastindex(x.policy.buffer), 4)
    @test findall(x -> x != 0, nmarkovgetindex(x.buffer.states, lastindex(x.buffer.states),4)[:]) .% 10 == x.callbacks[1].states[end-3:end] .% 10

    a1 = ArrayCircularBuffer(Array, Int64, (1), 8)
    a2 = ArrayCircularBuffer(Array, Int64, (1), 5)
    a3 = ArrayCircularBuffer(Array, Int64, (1), 4)
    is = collect(1:9)
    for i in is
        push!(a1, [i])
        push!(a2, [i])
        push!(a3, [i])
    end
    @test nmarkovgetindex(a1, lastindex(a1), 4) == nmarkovgetindex(a2, lastindex(a2), 4) == nmarkovgetindex(a3, lastindex(a3), 4)
end
testbuffers()
