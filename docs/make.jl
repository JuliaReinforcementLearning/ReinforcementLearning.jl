using Documenter, ReinforcementLearning

makedocs(modules = [ReinforcementLearning],
	     clean = false,
		 format = :html,
		 sitename = "ReinforcementLearning.jl",
 		 linkcheck = !("skiplinks" in ARGS),
		 pages = [ "Introduction" => "index.md", 
				   "Usage" => "usage.md",
                   "Tutorial" => "tutorial.md",
				   "Reference" => ["Comparison" => "comparison.md",
								   "Learning" => "learning.md",
								   "Learners" =>  "learners.md",
                                   "Buffers" => "buffers.md",
								   "Environments" => "environments.md",
								   "Stopping Criteria" =>  "stop.md",
                                   "Preprocessors" => "preprocessors.md",
								   "Policies" =>  "policies.md",
								   "Callbacks" =>  "callbacks.md",
								   "Evaluation Metrics" =>  "metrics.md",
                                  ]
                  ],
		 html_prettyurls = true
		)

deploydocs(
    repo = "github.com/JuliaReinforcementLearning/ReinforcementLearning.jl.git",
	julia = "1.0",
	target = "build",
    deps = nothing,
	make = nothing,
)
