"""
    run(π, env::AbstractEnv)

Run the policy `π` in `env` until the end.
"""
Base.run(π, env::AbstractEnv) = run(π, env, DynamicStyle(env))

function Base.run(π, env::AbstractEnv, ::Sequential)
    obs = observe(env)
    while !get_terminal(obs)
        obs |> π |> env
        obs = observe(env)
    end
end

function Base.run(Π::Vector, env::AbstractEnv, ::Sequential)
    is_terminal = false
    while !is_terminal
        for π in Π
            obs = observe(env)
            if get_terminal(obs)
                is_terminal = true
                break
            end
            obs |> π |> env
        end
    end
end
