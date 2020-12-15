#####
# printing
#####

function env_traits()
    [eval(x) for x in RLBase.ENV_API if endswith(String(x), "Style")]
end

Base.show(io::IO, t::MIME"text/plain", env::AbstractEnv) =
    show(io, MIME"text/markdown"(), env)

function Base.show(io::IO, t::MIME"text/markdown", env::AbstractEnv)
    show(io, t, Markdown.parse("""
    # $(nameof(env))

    ## Traits
    | Trait Type | Value |
    |:---------- | ----- |
    $(join(["|$(string(f))|$(f(env))|" for f in env_traits()], "\n"))

    ## Action Space
    `$(action_space(env))`

    ## State Space
    `$(state_space(env))`

    """))

    if NumAgentStyle(env) !== SINGLE_AGENT
        show(io, t, Markdown.parse("""
            ## Players
            $(join(["- `$p`" for p in players(env)], "\n"))

            ## Current Player
            `$(current_player(env))`
            """))
    end

    show(io, t, Markdown.parse("""
        ## Is Environment Terminated?
        $(is_terminated(env) ? "Yes" : "No")

        ## Current State

        ```
        $(state(env))
        ```
        """))
end

#####
# testing
#####

using Test

"""
Call this function after writing your customized environment to make sure that
all the necessary interfaces are implemented correctly and consistently.
"""
function test_interfaces(env)
    env = copy(env)  # make sure we don't touch the original environment

    rng = Random.MersenneTwister(666)

    @info "testing $(nameof(env)), you need to manually check these traits to make sure they are implemented correctly!" NumAgentStyle(
        env,
    ) DynamicStyle(env) ActionStyle(env) InformationStyle(env) StateStyle(env) RewardStyle(
        env,
    ) UtilityStyle(env) ChanceStyle(env)

    reset!(env)

    @testset "copy" begin
        old_env = env
        env = copy(env)

        if ChanceStyle(env) ∉ (DETERMINISTIC, EXPLICIT_STOCHASTIC)
            s = 888
            Random.seed!(env, s)
            Random.seed!(old_env, s)
        end

        @test env !== old_env

        @test state(env) == state(old_env)
        @test action_space(env) == action_space(old_env)
        @test reward(env) == reward(old_env)
        @test is_terminated(env) == is_terminated(old_env)

        while !is_terminated(env)
            A, A′ = legal_action_space(old_env), legal_action_space(env)
            @test A == A′
            a = rand(rng, A)
            env(a)
            old_env(a)
            @test state(env) == state(old_env)
            @test reward(env) == reward(old_env)
            @test is_terminated(env) == is_terminated(old_env)
        end
    end

    reset!(env)

    @testset "SingleAgent" begin
        if NumAgentStyle(env) === SINGLE_AGENT
            total_reward = 0.0
            while !is_terminated(env)
                if StateStyle(env) isa Tuple
                    for ss in StateStyle(env)
                        @test state(env, ss) ∈ state_space(env, ss)
                    end
                end

                A = legal_action_space(env)
                if ActionStyle(env) === MINIMAL_ACTION_SET
                    action_space(env) == legal_action_space
                elseif ActionStyle(env) === FULL_ACTION_SET
                    @test legal_action_space(env) ==
                          action_space(env)[legal_action_space_mask(env)]
                else
                    @error "TODO:"
                end

                a = rand(rng, A)
                @test a ∈ action_space(env)
                env(a)
                @test state(env) ∈ state_space(env)

                total_reward += reward(env)
            end
            r = reward(env)  # make sure we can still get the reward no matter what the RewardStyle of the env is.
            if RewardStyle(env) === TERMINAL_REWARD
                @test total_reward == r
            end
        end
    end

    @testset "MultiAgent" begin
        if NumAgentStyle(env) isa MultiAgent
            reset!(env)
            rewards = [0.0 for p in players(env)]
            while !is_terminated(env)
                if InformationStyle(env) === PERFECT_INFORMATION
                    for p in players(env)
                        @test state(env) == state(env, p)
                    end
                end
                a = rand(rng, legal_action_space(env))
                env(a)
                for (i, p) in enumerate(players(env))
                    @test state(env, p) ∈ state_space(env, p)
                    rewards[i] += reward(env, p)
                end
            end
            # even the game is already terminated
            # make sure each player can still get some necessary info
            for p in players(env)
                state(env, p)
                reward(env, p)
                @test is_terminated(env)
                # we don't need to check legal_action_space when the game is
                # already over
                # @test isempty(legal_action_space(env, p))
            end
            if RewardStyle(env) === TERMINAL_REWARD
                for (p, r) in zip(players(env), rewards)
                    @test r == reward(env, p)
                end
            end

            if UtilityStyle(env) === ZERO_SUM
                @test sum(rewards) == 0
            elseif UtilityStyle(env) == IDENTICAL_UTILITY
                @test all(rewards[1] .== rewards)
            end
        end
    end
end

#####
# Generate README
#####

gen_traits_table(envs) = gen_traits_table(stdout, envs)

function gen_traits_table(io, envs)
    trait_dict = Dict()
    for f in env_traits()
        for env in envs
            if !haskey(trait_dict, f)
                trait_dict[f] = Set()
            end
            t = f(env)
            if f == StateStyle
                if t isa Tuple
                    for x in t
                        push!(trait_dict[f], nameof(typeof(x)))
                    end
                else
                    push!(trait_dict[f], nameof(typeof(t)))
                end
            else
                push!(trait_dict[f], nameof(typeof(t)))
            end
        end
    end

    println(io, "<table>")

    print(io, "<th colspan=\"2\">Traits</th>")
    for i in 1:length(envs)
        print(io, "<th> $(i) </th>")
    end

    for k in sort(collect(keys(trait_dict)), by = nameof)
        vs = trait_dict[k]
        print(io, "<tr> <th rowspan=\"$(length(vs))\"> $(nameof(k)) </th>")
        for (i, v) in enumerate(vs)
            if i != 1
                print(io, "<tr> ")
            end
            print(io, "<th> $(v) </th>")
            for env in envs
                if k == StateStyle && k(env) isa Tuple
                    ss = k(env)
                    if v in map(x -> nameof(typeof(x)), ss)
                        print(io, "<td> ✔ </td>")
                    else
                        print(io, "<td> </td> ")
                    end
                else
                    if nameof(typeof(k(env))) == v
                        print(io, "<td> ✔ </td>")
                    else
                        print(io, "<td> </td> ")
                    end
                end
            end
            println(io, "</tr>")
        end
    end

    println(io, "</table>")

    print(io, "<ol>")
    for env in envs
        println(
            io,
            "<li> <a href=\"https://github.com/JuliaReinforcementLearning/ReinforcementLearningBase.jl/tree/master/src/examples/$(nameof(env)).jl\"> $(nameof(env)) </a></li>",
        )
    end
    print(io, "</ol>")
end

#####
# Utils
#####

using IntervalSets

"""
watch https://github.com/JuliaMath/IntervalSets.jl/issues/66
"""
function Base.in(x::AbstractArray, s::Array{<:Interval})
    size(x) == size(s) && all(x .∈ s)
end
