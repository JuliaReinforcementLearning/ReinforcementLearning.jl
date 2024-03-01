# ---
# title: JuliaRL\_DQN\_MPESimple
# cover:
# description: DQN applied to MPE simple
# date: 2023-02-01
# author: "[Panajiotis Keßler](mailto:panajiotis@christoforidis.net)"
# ---


using Plots
ex = E`JuliaRL_DQN_MPESimple`
run(ex)
plot(ex.hook.rewards)
savefig("JuliaRL_DQN_MPESimple.png")

