using .ReinforcementLearningEnvironments

for f in readdir(@__DIR__)
    if f != splitdir(@__FILE__)[2]
        include(f)
    end
end