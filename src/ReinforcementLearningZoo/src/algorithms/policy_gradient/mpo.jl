using LinearAlgebra, Flux
using Zygote: ignore, dropgrad

Base.@kwdef mutable struct MPOPolicy{P,Q,R}
    policy::P
    qnetwork::Q
    γ::Float32 = 0.99f0
    batch_size::Int #N
    action_sample_size::Int #K 
    ϵ::Float32  #KL bound on the non-parametric variational approximation to the policy
    ϵμ::Float32 #KL bound for the parametric policy training of mean estimations
    ϵΣ::Float32 #KL bound for the parametric policy training of (co)variance estimations
    n_epochs::Int
    update_freq::Int
    τ::Float32 = 1f-3 #Polyak avering parameter of target network
    rng::R = Random.GLOBAL_RNG
end


