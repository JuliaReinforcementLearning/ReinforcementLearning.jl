import ReinforcementLearning: updatetrace!

let λ = .7; γ = .9
    for (kind, res) in ((ReplacingTraces, (γ*λ)^2),
                        (AccumulatingTraces, (γ*λ)^2 + (γ*λ)^4))
        t = kind(2, 3, λ, γ)
        updatetrace!(t, 1, 1)
        updatetrace!(t, 2, 1)
        updatetrace!(t, 1, 1)
        updatetrace!(t, 2, 3)
        updatetrace!(t, 1, 2)
        @test t.trace[1,1] == res || (kind, t.trace[1,1], res)
    end
end
