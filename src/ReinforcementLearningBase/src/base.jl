#####
# printing
#####

function env_traits()
    [eval(x) for x in RLBase.ENV_API if endswith(String(x), "Style")]
end

Base.show(io::IO, t::MIME"text/plain", env::AbstractEnv) =
    show(io, MIME"text/markdown"(), env)

function Base.show(io::IO, t::MIME"text/markdown", env::AbstractEnv)
    s = """
    # $(nameof(env))

    ## Traits
    | Trait Type | Value |
    |:---------- | ----- |
    $(join(["|$(string(f))|$(f(env))|" for f in env_traits()], "\n"))

    ## Is Environment Terminated?
    $(is_terminated(env) ? "Yes" : "No")

    """

    if get(io, :is_show_state_space, true)
        s *= """
        ## State Space
        `$(state_space(env))`

        """
    end

    if get(io, :is_show_action_space, true)
        s *= """
        ## Action Space
        `$(action_space(env))`

        """
    end

    if NumAgentStyle(env) !== SINGLE_AGENT
        s *= """
        ## Players
        $(join(["- `$p`" for p in players(env)], "\n"))

        ## Current Player
        `$(current_player(env))`
        """
    end

    if get(io, :is_show_state, true)
        s *= """
        ## Current State

        ```
        $(state(env))
        ```
        """
    end

    show(io, t, Markdown.parse(s))
end

#####
# testing
#####

using Test

"""
Call this function after writing your customized environment to make sure that
all the necessary interfaces are implemented correctly and consistently.
"""
function test_interfaces!(env)
    rng = Random.MersenneTwister(666)

    @info "testing $(nameof(env)), you need to manually check these traits to make sure they are implemented correctly!" NumAgentStyle(
        env,
    ) DynamicStyle(env) ActionStyle(env) InformationStyle(env) StateStyle(env) RewardStyle(
        env,
    ) UtilityStyle(env) ChanceStyle(env)

    @testset "copy" begin
        X = copy(env)
        Y = copy(env)
        reset!(X)
        reset!(Y)

        if ChanceStyle(Y) ∉ (DETERMINISTIC, EXPLICIT_STOCHASTIC)
            s = 888
            Random.seed!(Y, s)
            Random.seed!(X, s)
        end

        @test Y !== X

        @test state(Y) == state(X)
        @test action_space(Y) == action_space(X)
        @test reward(Y) == reward(X)
        @test is_terminated(Y) == is_terminated(X)

        while !is_terminated(Y)
            A, A′ = legal_action_space(X), legal_action_space(Y)
            @test A == A′
            a = rand(rng, A)
            Y(a)
            X(a)
            @test state(Y) == state(X)
            @test reward(Y) == reward(X)
            @test is_terminated(Y) == is_terminated(X)
        end
    end

    @testset "SingleAgent" begin
        if NumAgentStyle(env) === SINGLE_AGENT
            reset!(env)
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

    reset!(env)
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

Random.rand(s::Union{Interval,Array{<:Interval}}) = rand(Random.GLOBAL_RNG, s)

function Random.rand(rng::AbstractRNG, s::Interval)
    rand(rng) * (s.right - s.left) + s.left
end

#####
# WorldSpace
#####

export WorldSpace

"""
In some cases, we may not be interested in the action/state space.
One can return `WorldSpace()` to keep the interface consistent.
"""
struct WorldSpace{T} end

WorldSpace() = WorldSpace{Any}()

Base.in(x, ::WorldSpace{T}) where {T} = x isa T

#####
# ZeroTo
#####

export ZeroTo

"""
Similar to `Base.OneTo`. Useful when wrapping third-party environments.
"""
struct ZeroTo{T<:Integer} <: AbstractUnitRange{T}
    stop::T
    ZeroTo{T}(n) where {T<:Integer} = new(max(zero(T) - one(T), n))
end

ZeroTo(n::T) where {T<:Integer} = ZeroTo{T}(n)

Base.show(io::IO, r::ZeroTo) = print(io, "ZeroTo(", r.stop, ")")
Base.length(r::ZeroTo{T}) where {T} = T(r.stop + one(r.stop))
Base.first(r::ZeroTo{T}) where {T} = zero(r.stop)

function getindex(v::ZeroTo{T}, i::Integer) where {T}
    Base.@_inline_meta
    @boundscheck ((i >= 0) & (i <= v.stop)) || throw_boundserror(v, i)
    convert(T, i)
end

#####
# ActionProbPair
#####

export ActionProbPair

"""
Used in action space of chance player.
"""
struct ActionProbPair{A,P}
    action::A
    prob::P
end

"""
Directly copied from [StatsBase.jl](https://github.com/JuliaStats/StatsBase.jl/blob/0ea8e798c3d19609ed33b11311de5a2bd6ee9fd0/src/sampling.jl#L499-L510) to avoid depending on the whole package.
Here we assume `wv` sum to `1`
"""
function weighted_sample(rng::AbstractRNG, wv)
    t = rand(rng)
    cw = zero(Base.first(wv))
    for (i, w) in enumerate(wv)
        cw += w
        if cw >= t
            return i
        end
    end
end

Random.rand(rng::AbstractRNG, s::AbstractVector{<:ActionProbPair}) =
    s[weighted_sample(rng, (x.prob for x in s))]

(env::AbstractEnv)(a::ActionProbPair) = env(a.action)

#####
# Space
#####

export Space

"""
A wrapper to treat each element as a sub-space which supports `Random.rand` and `Base.in`.
"""
struct Space{T}
    s::T
end

Random.rand(s::Space) = rand(Random.GLOBAL_RNG, s)

Random.rand(rng::AbstractRNG, s::Space) =
    map(s.s) do x
        rand(rng, x)
    end

Random.rand(rng::AbstractRNG, s::Space{<:Dict}) = Dict(k => rand(rng, v) for (k, v) in s.s)

function Base.in(X, S::Space)
    if length(X) == length(S.s)
        for (x, s) in zip(X, S.s)
            if x ∉ s
                return false
            end
        end
        return true
    else
        return false
    end
end

function Base.in(X::Dict, S::Space{<:Dict})
    if keys(X) == keys(S.s)
        for k in keys(X)
            if X[k] ∉ S.s[k]
                return false
            end
        end
        return true
    else
        return false
    end
end
