using Pkg

"""
    activate_devmode!()

This will automatically dev all the packages of the RL.jl ecosystem (make sure your 
working directory is ReinforcementLearning.jl). You should do this when you 
create a new branch for a new PR and voilà, you're good to go. This function imitates 
the process in the `ci.yml` file. This means that tests that you run locally should 
work in github's CI.
"""
function activate_devmode!()
    @info "Switching to dev mode. You are now using your local versions of the RL.jl packages instead of the registered releases."
    #RLBase
    #No dependency to dev
    #RLCore
    Pkg.activate("src/ReinforcementLearningCore")
    Pkg.develop(path="src/ReinforcementLearningBase")
    #RLZoo
    Pkg.activate("src/ReinforcementLearningZoo")
    Pkg.develop(path="src/ReinforcementLearningCore")
    Pkg.develop(path="src/ReinforcementLearningBase")
    #RLEnvironments
    Pkg.activate("src/ReinforcementLearningEnvironments")
    Pkg.develop(path="src/ReinforcementLearningBase")
    # MultiAgentRL
    Pkg.activate("src/MultiAgentReinforcementLearning")
    Pkg.develop(path="src/ReinforcementLearningCore")
    Pkg.develop(path="src/ReinforcementLearningBase")
    #RLExperiments
    Pkg.activate("src/ReinforcementLearningExperiments")
    Pkg.develop(path=".")
    Pkg.develop(path="src/ReinforcementLearningZoo")
    Pkg.develop(path="src/ReinforcementLearningEnvironments")
    Pkg.develop(path="src/ReinforcementLearningCore")
    Pkg.develop(path="src/ReinforcementLearningBase")
    #RL
    Pkg.activate(".")
    Pkg.develop(path="src/ReinforcementLearningZoo")
    Pkg.develop(path="src/ReinforcementLearningEnvironments")
    Pkg.develop(path="src/ReinforcementLearningCore")
    Pkg.develop(path="src/ReinforcementLearningBase")
    Pkg.develop(path="src/MultiAgentReinforcementLearning")
end
