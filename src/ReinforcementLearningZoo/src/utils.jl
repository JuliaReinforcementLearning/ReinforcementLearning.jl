export load_policy

using Pkg.Artifacts

function load_policy(s::String, T=AbstractPolicy)
    if isfile(s)
        RLCore.load(s, T)
    elseif isdir(s)
        load_policy(joinpath(s, "policy.bson"), T)
    else
        dir = ensure_artifact_installed(s, find_artifacts_toml(@__DIR__))
        return load_policy(dir, T)
    end
end