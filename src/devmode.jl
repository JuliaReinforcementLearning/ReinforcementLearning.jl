using Pkg

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
    #RLExperiments
    Pkg.activate("src/ReinforcementLearningExperiments")
    Pkg.develop(path=".")
    #RL
    Pkg.activate(".")
    Pkg.develop(path="src/ReinforcementLearningZoo")
    Pkg.develop(path="src/ReinforcementLearningEnvironments")
    Pkg.develop(path="src/ReinforcementLearningCore")
    Pkg.develop(path="src/ReinforcementLearningBase")
end
