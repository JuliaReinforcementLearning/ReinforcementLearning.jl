using Base.Threads
@testset "rl_unplugged_dm" begin
    @testset "dm_control_suite" begin
        ds = rl_unplugged_dm_dataset(
            "fish_swim",
            [1, 2];
            type="dm_control_suite",
            is_shuffle = true,
            shuffle_buffer_size=10_000,
            tf_reader_bufsize=10_000,
            tf_reader_sz=10_000,
            batch_size=256,
            n_preallocations=nthreads()*12
        )

        @test typeof(ds)<:RingBuffer

        data = take!(ds)
        
        batch_size = 256
        feature_size = ReinforcementLearningDatasets.DM_CONTROL_SUITE_SIZE["fish_swim"]
        
        @test typeof(data.state) <: NamedTuple
        @test typeof(data.next_state) <: NamedTuple
        
        for feature in keys(feature_size)
            if split(feature, "/")[1] != "observation"
                if feature != "step_type"
                    ob_key = Symbol(feature)
                    @test size(getfield(data, ob_key)) == (feature_size[feature]..., batch_size,)
                end
            else
                state = data.state
                next_state = data.next_state
                ob_key = Symbol(chop(feature, head=length("observation")+1, tail=0))
                @test size(getfield(state, ob_key)) == (feature_size[feature]...,batch_size)
                @test size(getfield(next_state, ob_key)) == (feature_size[feature]..., batch_size,)
            end
        end
    end

    @testset "dm_locomotion_humanoid" begin
        ds = rl_unplugged_dm_dataset(
            "humanoid_corridor",
            [1, 2];
            type="dm_locomotion_humanoid",
            is_shuffle = true,
            shuffle_buffer_size=10_000,
            tf_reader_bufsize=10_000,
            tf_reader_sz=10_000,
            batch_size=256,
            n_preallocations=nthreads()*12
        )

        @test typeof(ds)<:RingBuffer

        data = take!(ds)
        
        batch_size = 256
        feature_size = ReinforcementLearningDatasets.DM_LOCOMOTION_HUMANOID_SIZE
        
        @test typeof(data.state) <: NamedTuple
        @test typeof(data.next_state) <: NamedTuple
        
        for feature in keys(feature_size)
            if split(feature, "/")[1] != "observation"
                if feature != "step_type"
                    ob_key = Symbol(feature)
                    @test size(getfield(data, ob_key)) == (feature_size[feature]..., batch_size,)
                end
            else
                state = data.state
                next_state = data.next_state
                ob_key = Symbol(chop(feature, head=length("observation")+1, tail=0))
                @test size(getfield(state, ob_key)) == (feature_size[feature]..., batch_size,)
                @test size(getfield(next_state, ob_key)) == (feature_size[feature]..., batch_size,)
            end
        end
    end

    @testset "dm_locomotion_rodent" begin
        ds = rl_unplugged_dm_dataset(
            "rodent_escape",
            [1, 2];
            type="dm_locomotion_rodent",
            is_shuffle = true,
            shuffle_buffer_size=10_000,
            tf_reader_bufsize=10_000,
            tf_reader_sz=10_000,
            batch_size=256,
            n_preallocations=nthreads()*12
        )

        @test typeof(ds)<:RingBuffer

        data = take!(ds)
        
        batch_size = 256
        feature_size = ReinforcementLearningDatasets.DM_LOCOMOTION_RODENT_SIZE
        
        @test typeof(data.state) <: NamedTuple
        @test typeof(data.next_state) <: NamedTuple
        
        for feature in keys(feature_size)
            if split(feature, "/")[1] != "observation"
                if feature != "step_type"
                    ob_key = Symbol(feature)
                    @test size(getfield(data, ob_key)) == (feature_size[feature]..., batch_size,)
                end
            else
                state = data.state
                next_state = data.next_state
                ob_key = Symbol(chop(feature, head=length("observation")+1, tail=0))
                @test size(getfield(state, ob_key)) == (feature_size[feature]..., batch_size,)
                @test size(getfield(next_state, ob_key)) == (feature_size[feature]..., batch_size,)
            end
        end
    end
end