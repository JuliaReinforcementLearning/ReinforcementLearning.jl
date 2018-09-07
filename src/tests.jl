function test_envinterface(env)
    @eval Main begin
        @testset "$(typeof($env)) interface test" begin
            res = reset!($env)
            @test typeof(res) == NamedTuple{(:observation,), 
                                            Tuple{typeof(res[:observation])}}
            res = interact!($env, sample(actionspace($env)))
            @test typeof(res) == NamedTuple{(:observation, :reward, :isdone), 
                                            Tuple{typeof(res[:observation]), 
                                                  typeof(res[:reward]), Bool}}
            res = getstate($env)
            @test typeof(res) == NamedTuple{(:observation, :isdone), 
                                            Tuple{typeof(res[:observation]), Bool}}
        end
    end
end
