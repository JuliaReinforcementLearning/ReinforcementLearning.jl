T = ReinforcementLearning

λ = .7; γ = .9
for (kind, res) in ((ReplacingTraces, (γ*λ)^2),
                    (AccumulatingTraces, (γ*λ)^2 + (γ*λ)^4))
    t = kind(2, 3, λ, γ)
    T.updatetrace!(t, 1, 1)
    T.updatetrace!(t, 2, 1)
    T.updatetrace!(t, 1, 1)
    T.updatetrace!(t, 2, 3)
    T.updatetrace!(t, 1, 2)
    @test t.trace[1,1] == res || (kind, t.trace[1,1], res)
end
