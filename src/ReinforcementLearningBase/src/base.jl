"""
    run(π, env::AbstractEnv)

Run the policy `π` in `env` until the end.
"""
Base.run(π, env::AbstractEnv) = run(DynamicStyle(env), NumAgentStyle(env), π, env)

function Base.run(::Sequential, ::SingleAgent, π, env::AbstractEnv)
    while !get_terminal(env)
        action = π(env)
        env(action)
    end
end

function Base.run(::Sequential, ::MultiAgent, Π, env::AbstractEnv)
    is_terminal = false
    while !is_terminal
        for π in Π
            if get_terminal(env)
                is_terminal = true
                break
            end
            action = π(env)
            env(action)
        end
    end
end

#####
# printing
#####

function get_env_traits()
    [eval(x) for x in RLBase.ENV_API if endswith(String(x), "Style")]
end

Base.show(io::IO, t::MIME"text/plain", env::AbstractEnv) =
    show(io, MIME"text/markdown"(), env)

function Base.show(io::IO, t::MIME"text/markdown", env::AbstractEnv)
    show(io, t, Markdown.parse("""
    # $(get_name(env))

    ## Traits
    | Trait Type | Value |
    |:---------- | ----- |
    $(join(["|$(string(f))|$(f(env))|" for f in get_env_traits()], "\n"))

    ## Actions
    $(get_actions(env))

    ## Players
    $(join(["- `$p`" for p in get_players(env)], "\n"))

    ## Current Player
    `$(get_current_player(env))`

    ## Is Environment Terminated?
    $(get_terminal(env) ? "Yes" : "No")
    """))
end

#####
# helper functions
#####

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
