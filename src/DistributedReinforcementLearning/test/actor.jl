@testset "basic tests" begin

    Base.@kwdef mutable struct TestActor
        state::Union{Nothing,Int} = nothing
    end

    struct CurrentStateMsg <: AbstractMessage
        state::Any
    end

    Base.@kwdef struct ReadStateMsg <: AbstractMessage
        from = self()
    end

    struct IncMsg <: AbstractMessage end
    struct DecMsg <: AbstractMessage end

    (x::TestActor)(msg::StartMsg{Tuple{Int}}) = x.state = msg.args[1]
    (x::TestActor)(msg::StopMsg) = x.state = nothing
    (x::TestActor)(::IncMsg) = x.state += 1
    (x::TestActor)(::DecMsg) = x.state -= 1
    (x::TestActor)(msg::ReadStateMsg) = put!(msg.from, CurrentStateMsg(x.state))

    x = actor(TestActor())
    put!(x, StartMsg(0))

    put!(x, ReadStateMsg())
    @test take!(self()).state == 0

    @sync begin
        for _ in 1:100
            Threads.@spawn put!(x, IncMsg())
            Threads.@spawn put!(x, DecMsg())
        end
        for _ in 1:10
            for _ in 1:10
                Threads.@spawn put!(x, IncMsg())
            end
            for _ in 1:10
                Threads.@spawn put!(x, DecMsg())
            end
        end
    end

    put!(x, ReadStateMsg())
    @test take!(self()).state == 0

    y = actor(TestActor())
    put!(x, ProxyMsg(; to = y, msg = StartMsg(0)))
    put!(x, ProxyMsg(; to = y, msg = ReadStateMsg()))
    @test take!(self()).state == 0

end
