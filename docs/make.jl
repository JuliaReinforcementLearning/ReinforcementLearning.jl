using Documenter, ReinforcementLearning

makedocs(modules = [ReinforcementLearning],
	     clean = false,
		 format = :html,
		 sitename = "Tabular Reinforcement Learning",
 		 linkcheck = !("skiplinks" in ARGS),
		 pages = [ "Introduction" => "index.md", 
				   "Usage" => "usage.md",
				   "Reference" => ["Comparison" => "comparison.md",
								   "Learning" => "learning.md",
								   "Learners" =>  "learners.md",
								   "Environments" => "environments.md",
								   "Stopping Criteria" =>  "stop.md",
                                   "Preprocessors" => "preprocessors.md",
								   "Policies" =>  "policies.md",
								   "Callbacks" =>  "callbacks.md",
								   "Evaluation Metrics" =>  "metrics.md",
                                  ],
				   "API" => "api.md"],
		 html_prettyurls = true
		)

deploydocs(
    repo = "github.com/jbrea/ReinforcementLearning.jl.git",
	julia = "0.6",
	target = "build",
    deps = nothing,
	make = nothing,
)
