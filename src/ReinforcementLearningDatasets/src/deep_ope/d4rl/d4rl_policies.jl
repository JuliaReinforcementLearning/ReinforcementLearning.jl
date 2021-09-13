export d4rl_policy_params

function d4rl_policy_params()
    d4rl_policy_paths = [split(policy["policy_path"], "/")[2] for policy in D4RL_POLICIES]
    env = Set(join.(map(x->x[1:end-2], split.(d4rl_policy_paths, "_")), "_"))
    agent = ["dapg", "online"]
    epoch = 0:10

    @info env agent epoch
end

const D4RL_POLICIES = [
    Dict(
        "policy_path" => "antmaze_large/antmaze_large_dapg_0.pkl",
        "task.task_names" => [
            "antmaze-large-play-v0",
            "antmaze-large-diverse-v0"
        ],
        "agent_name" => "BC",
        "return_mean" => 0.0,
        "return_std =>" => 0.0
    ),
    Dict(
        "policy_path" => "antmaze_large/antmaze_large_dapg_10.pkl",
        "task.task_names" => [
            "antmaze-large-play-v0",
            "antmaze-large-diverse-v0"
        ],
        "agent_name" => "BC",
        "return_mean" => 0.48,
        "return_std =>" => 0.4995998398718718
    ),
    Dict(
        "policy_path" => "antmaze_large/antmaze_large_dapg_1.pkl",
        "task.task_names" => [
            "antmaze-large-play-v0",
            "antmaze-large-diverse-v0"
        ],
        "agent_name" => "BC",
        "return_mean" => 0.0,
        "return_std =>" => 0.0
    ),
    Dict(
        "policy_path" => "antmaze_large/antmaze_large_dapg_2.pkl",
        "task.task_names" => [
            "antmaze-large-play-v0",
            "antmaze-large-diverse-v0"
        ],
        "agent_name" => "BC",
        "return_mean" => 0.0,
        "return_std =>" => 0.0
    ),
    Dict(
        "policy_path" => "antmaze_large/antmaze_large_dapg_3.pkl",
        "task.task_names" => [
            "antmaze-large-play-v0",
            "antmaze-large-diverse-v0"
        ],
        "agent_name" => "BC",
        "return_mean" => 0.0,
        "return_std =>" => 0.0
    ),
    Dict(
        "policy_path" => "antmaze_large/antmaze_large_dapg_4.pkl",
        "task.task_names" => [
            "antmaze-large-play-v0",
            "antmaze-large-diverse-v0"
        ],
        "agent_name" => "BC",
        "return_mean" => 0.01,
        "return_std =>" => 0.09949874371066199
    ),
    Dict(
        "policy_path" => "antmaze_large/antmaze_large_dapg_5.pkl",
        "task.task_names" => [
            "antmaze-large-play-v0",
            "antmaze-large-diverse-v0"
        ],
        "agent_name" => "BC",
        "return_mean" => 0.13,
        "return_std =>" => 0.33630343441600474
    ),
    Dict(
        "policy_path" => "antmaze_large/antmaze_large_dapg_6.pkl",
        "task.task_names" => [
            "antmaze-large-play-v0",
            "antmaze-large-diverse-v0"
        ],
        "agent_name" => "BC",
        "return_mean" => 0.22,
        "return_std =>" => 0.41424630354415964
    ),
    Dict(
        "policy_path" => "antmaze_large/antmaze_large_dapg_7.pkl",
        "task.task_names" => [
            "antmaze-large-play-v0",
            "antmaze-large-diverse-v0"
        ],
        "agent_name" => "BC",
        "return_mean" => 0.12,
        "return_std =>" => 0.32496153618543844
    ),
    Dict(
        "policy_path" => "antmaze_large/antmaze_large_dapg_8.pkl",
        "task.task_names" => [
            "antmaze-large-play-v0",
            "antmaze-large-diverse-v0"
        ],
        "agent_name" => "BC",
        "return_mean" => 0.39,
        "return_std =>" => 0.487749935930288
    ),
    Dict(
        "policy_path" => "antmaze_large/antmaze_large_dapg_9.pkl",
        "task.task_names" => [
            "antmaze-large-play-v0",
            "antmaze-large-diverse-v0"
        ],
        "agent_name" => "BC",
        "return_mean" => 0.49,
        "return_std =>" => 0.4998999899979995
    ),
    Dict(
        "policy_path" => "antmaze_medium/antmaze_medium_dapg_0.pkl",
        "task.task_names" => [
            "antmaze-medium-play-v0",
            "antmaze-medium-diverse-v0"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 0.66,
        "return_std =>" => 0.4737087712930805
    ),
    Dict(
        "policy_path" => "antmaze_medium/antmaze_medium_dapg_10.pkl",
        "task.task_names" => [
            "antmaze-medium-play-v0",
            "antmaze-medium-diverse-v0"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 0.12,
        "return_std =>" => 0.32496153618543844
    ),
    Dict(
        "policy_path" => "antmaze_medium/antmaze_medium_dapg_1.pkl",
        "task.task_names" => [
            "antmaze-medium-play-v0",
            "antmaze-medium-diverse-v0"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 0.53,
        "return_std =>" => 0.49909918853871116
    ),
    Dict(
        "policy_path" => "antmaze_medium/antmaze_medium_dapg_2.pkl",
        "task.task_names" => [
            "antmaze-medium-play-v0",
            "antmaze-medium-diverse-v0"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 0.66,
        "return_std =>" => 0.4737087712930805
    ),
    Dict(
        "policy_path" => "antmaze_medium/antmaze_medium_dapg_3.pkl",
        "task.task_names" => [
            "antmaze-medium-play-v0",
            "antmaze-medium-diverse-v0"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 0.57,
        "return_std =>" => 0.49507575177946245
    ),
    Dict(
        "policy_path" => "antmaze_medium/antmaze_medium_dapg_4.pkl",
        "task.task_names" => [
            "antmaze-medium-play-v0",
            "antmaze-medium-diverse-v0"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 0.58,
        "return_std =>" => 0.49355850717012273
    ),
    Dict(
        "policy_path" => "antmaze_medium/antmaze_medium_dapg_5.pkl",
        "task.task_names" => [
            "antmaze-medium-play-v0",
            "antmaze-medium-diverse-v0"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 0.42,
        "return_std =>" => 0.49355850717012273
    ),
    Dict(
        "policy_path" => "antmaze_medium/antmaze_medium_dapg_6.pkl",
        "task.task_names" => [
            "antmaze-medium-play-v0",
            "antmaze-medium-diverse-v0"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 0.45,
        "return_std =>" => 0.49749371855331004
    ),
    Dict(
        "policy_path" => "antmaze_medium/antmaze_medium_dapg_7.pkl",
        "task.task_names" => [
            "antmaze-medium-play-v0",
            "antmaze-medium-diverse-v0"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 0.27,
        "return_std =>" => 0.4439594576084623
    ),
    Dict(
        "policy_path" => "antmaze_medium/antmaze_medium_dapg_8.pkl",
        "task.task_names" => [
            "antmaze-medium-play-v0",
            "antmaze-medium-diverse-v0"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 0.1,
        "return_std =>" => 0.29999999999999993
    ),
    Dict(
        "policy_path" => "antmaze_medium/antmaze_medium_dapg_9.pkl",
        "task.task_names" => [
            "antmaze-medium-play-v0",
            "antmaze-medium-diverse-v0"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 0.15,
        "return_std =>" => 0.3570714214271425
    ),
    Dict(
        "policy_path" => "antmaze_umaze/antmaze_umaze_dapg_0.pkl",
        "task.task_names" => [
            "antmaze-umaze-v0"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 0.11,
        "return_std =>" => 0.31288975694324034
    ),
    Dict(
        "policy_path" => "antmaze_umaze/antmaze_umaze_dapg_10.pkl",
        "task.task_names" => [
            "antmaze-umaze-v0"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 0.84,
        "return_std =>" => 0.36660605559646725
    ),
    Dict(
        "policy_path" => "antmaze_umaze/antmaze_umaze_dapg_1.pkl",
        "task.task_names" => [
            "antmaze-umaze-v0"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 0.15,
        "return_std =>" => 0.3570714214271425
    ),
    Dict(
        "policy_path" => "antmaze_umaze/antmaze_umaze_dapg_2.pkl",
        "task.task_names" => [
            "antmaze-umaze-v0"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 0.08,
        "return_std =>" => 0.2712931993250107
    ),
    Dict(
        "policy_path" => "antmaze_umaze/antmaze_umaze_dapg_3.pkl",
        "task.task_names" => [
            "antmaze-umaze-v0"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 0.13,
        "return_std =>" => 0.33630343441600474
    ),
    Dict(
        "policy_path" => "antmaze_umaze/antmaze_umaze_dapg_4.pkl",
        "task.task_names" => [
            "antmaze-umaze-v0"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 0.19,
        "return_std =>" => 0.3923009049186606
    ),
    Dict(
        "policy_path" => "antmaze_umaze/antmaze_umaze_dapg_5.pkl",
        "task.task_names" => [
            "antmaze-umaze-v0"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 0.27,
        "return_std =>" => 0.4439594576084623
    ),
    Dict(
        "policy_path" => "antmaze_umaze/antmaze_umaze_dapg_6.pkl",
        "task.task_names" => [
            "antmaze-umaze-v0"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 0.41,
        "return_std =>" => 0.4918333050943175
    ),
    Dict(
        "policy_path" => "antmaze_umaze/antmaze_umaze_dapg_7.pkl",
        "task.task_names" => [
            "antmaze-umaze-v0"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 0.66,
        "return_std =>" => 0.4737087712930805
    ),
    Dict(
        "policy_path" => "antmaze_umaze/antmaze_umaze_dapg_8.pkl",
        "task.task_names" => [
            "antmaze-umaze-v0"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 0.72,
        "return_std =>" => 0.4489988864128729
    ),
    Dict(
        "policy_path" => "antmaze_umaze/antmaze_umaze_dapg_9.pkl",
        "task.task_names" => [
            "antmaze-umaze-v0"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 0.45,
        "return_std =>" => 0.49749371855331
    ),
    Dict(
        "policy_path" => "ant/ant_online_0.pkl",
        "task.task_names" => [
            "ant-medium-v0",
            "ant-random-v0",
            "ant-expert-v0",
            "ant-medium-replay-v0",
            "ant-medium-expert-v0"
        ],
        "agent_name" => "SAC",
        "return_mean" => -61.02055183305979,
        "return_std =>" => 118.86259895376526
    ),
    Dict(
        "policy_path" => "ant/ant_online_10.pkl",
        "task.task_names" => [
            "ant-medium-v0",
            "ant-random-v0",
            "ant-expert-v0",
            "ant-medium-replay-v0",
            "ant-medium-expert-v0"
        ],
        "agent_name" => "SAC",
        "return_mean" => 5226.071929273204,
        "return_std =>" => 1351.489114884685
    ),
    Dict(
        "policy_path" => "ant/ant_online_1.pkl",
        "task.task_names" => [
            "ant-medium-v0",
            "ant-random-v0",
            "ant-expert-v0",
            "ant-medium-replay-v0",
            "ant-medium-expert-v0"
        ],
        "agent_name" => "SAC",
        "return_mean" => 1128.957315814236,
        "return_std =>" => 545.9910621405912
    ),
    Dict(
        "policy_path" => "ant/ant_online_2.pkl",
        "task.task_names" => [
            "ant-medium-v0",
            "ant-random-v0",
            "ant-expert-v0",
            "ant-medium-replay-v0",
            "ant-medium-expert-v0"
        ],
        "agent_name" => "SAC",
        "return_mean" => 1874.9426222623788,
        "return_std =>" => 821.523301172575
    ),
    Dict(
        "policy_path" => "ant/ant_online_3.pkl",
        "task.task_names" => [
            "ant-medium-v0",
            "ant-random-v0",
            "ant-expert-v0",
            "ant-medium-replay-v0",
            "ant-medium-expert-v0"
        ],
        "agent_name" => "SAC",
        "return_mean" => 2694.0050365558186,
        "return_std =>" => 829.1251729756312
    ),
    Dict(
        "policy_path" => "ant/ant_online_4.pkl",
        "task.task_names" => [
            "ant-medium-v0",
            "ant-random-v0",
            "ant-expert-v0",
            "ant-medium-replay-v0",
            "ant-medium-expert-v0"
        ],
        "agent_name" => "SAC",
        "return_mean" => 2927.728155987557,
        "return_std =>" => 1218.962159178784
    ),
    Dict(
        "policy_path" => "ant/ant_online_5.pkl",
        "task.task_names" => [
            "ant-medium-v0",
            "ant-random-v0",
            "ant-expert-v0",
            "ant-medium-replay-v0",
            "ant-medium-expert-v0"
        ],
        "agent_name" => "SAC",
        "return_mean" => -271.0455967662947,
        "return_std =>" => 181.7343490946006
    ),
    Dict(
        "policy_path" => "ant/ant_online_6.pkl",
        "task.task_names" => [
            "ant-medium-v0",
            "ant-random-v0",
            "ant-expert-v0",
            "ant-medium-replay-v0",
            "ant-medium-expert-v0"
        ],
        "agent_name" => "SAC",
        "return_mean" => 3923.0820284011284,
        "return_std =>" => 1384.459574872169
    ),
    Dict(
        "policy_path" => "ant/ant_online_7.pkl",
        "task.task_names" => [
            "ant-medium-v0",
            "ant-random-v0",
            "ant-expert-v0",
            "ant-medium-replay-v0",
            "ant-medium-expert-v0"
        ],
        "agent_name" => "SAC",
        "return_mean" => 4564.024787293475,
        "return_std =>" => 1207.181426135141
    ),
    Dict(
        "policy_path" => "ant/ant_online_8.pkl",
        "task.task_names" => [
            "ant-medium-v0",
            "ant-random-v0",
            "ant-expert-v0",
            "ant-medium-replay-v0",
            "ant-medium-expert-v0"
        ],
        "agent_name" => "SAC",
        "return_mean" => 5116.58562094113,
        "return_std =>" => 962.8694737383373
    ),
    Dict(
        "policy_path" => "ant/ant_online_9.pkl",
        "task.task_names" => [
            "ant-medium-v0",
            "ant-random-v0",
            "ant-expert-v0",
            "ant-medium-replay-v0",
            "ant-medium-expert-v0"
        ],
        "agent_name" => "SAC",
        "return_mean" => 5176.548960934259,
        "return_std =>" => 1000.122269767824
    ),
    Dict(
        "policy_path" => "door/door_dapg_0.pkl",
        "task.task_names" => [
            "door-cloned-v0",
            "door-expert-v0",
            "door-human-v0"
        ],
        "agent_name" => "DAPG",
        "return_mean" => -53.63337645679012,
        "return_std =>" => 2.0058239428094895
    ),
    Dict(
        "policy_path" => "door/door_dapg_10.pkl",
        "task.task_names" => [
            "door-cloned-v0",
            "door-expert-v0",
            "door-human-v0"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 2974.9306587121887,
        "return_std =>" => 52.48250668645121
    ),
    Dict(
        "policy_path" => "door/door_dapg_1.pkl",
        "task.task_names" => [
            "door-cloned-v0",
            "door-expert-v0",
            "door-human-v0"
        ],
        "agent_name" => "DAPG",
        "return_mean" => -51.41658735064874,
        "return_std =>" => 0.6978335854285623
    ),
    Dict(
        "policy_path" => "door/door_dapg_2.pkl",
        "task.task_names" => [
            "door-cloned-v0",
            "door-expert-v0",
            "door-human-v0"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 86.28632719532406,
        "return_std =>" => 256.30747202806475
    ),
    Dict(
        "policy_path" => "door/door_dapg_3.pkl",
        "task.task_names" => [
            "door-cloned-v0",
            "door-expert-v0",
            "door-human-v0"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 1282.0275007615646,
        "return_std =>" => 633.9669441391286
    ),
    Dict(
        "policy_path" => "door/door_dapg_4.pkl",
        "task.task_names" => [
            "door-cloned-v0",
            "door-expert-v0",
            "door-human-v0"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 1607.4255566289276,
        "return_std =>" => 499.58651630841575
    ),
    Dict(
        "policy_path" => "door/door_dapg_5.pkl",
        "task.task_names" => [
            "door-cloned-v0",
            "door-expert-v0",
            "door-human-v0"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 2142.36638691816,
        "return_std =>" => 442.0537003890031
    ),
    Dict(
        "policy_path" => "door/door_dapg_6.pkl",
        "task.task_names" => [
            "door-cloned-v0",
            "door-expert-v0",
            "door-human-v0"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 2525.495218483574,
        "return_std =>" => 160.8683834534215
    ),
    Dict(
        "policy_path" => "door/door_dapg_7.pkl",
        "task.task_names" => [
            "door-cloned-v0",
            "door-expert-v0",
            "door-human-v0"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 2794.653907232321,
        "return_std =>" => 62.78226619278986
    ),
    Dict(
        "policy_path" => "door/door_dapg_8.pkl",
        "task.task_names" => [
            "door-cloned-v0",
            "door-expert-v0",
            "door-human-v0"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 2870.85173247603,
        "return_std =>" => 37.96052715176604
    ),
    Dict(
        "policy_path" => "door/door_dapg_9.pkl",
        "task.task_names" => [
            "door-cloned-v0",
            "door-expert-v0",
            "door-human-v0"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 2959.4718836123457,
        "return_std =>" => 53.31391818495784
    ),
    Dict(
        "policy_path" => "halfcheetah/halfcheetah_online_0.pkl",
        "task.task_names" => [
            "halfcheetah-medium-v0",
            "halfcheetah-random-v0",
            "halfcheetah-expert-v0",
            "halfcheetah-medium-replay-v0",
            "halfcheetah-medium-expert-v0"
        ],
        "agent_name" => "SAC",
        "return_mean" => -309.2417932614121,
        "return_std =>" => 91.3640277992432
    ),
    Dict(
        "policy_path" => "halfcheetah/halfcheetah_online_10.pkl",
        "task.task_names" => [
            "halfcheetah-medium-v0",
            "halfcheetah-random-v0",
            "halfcheetah-expert-v0",
            "halfcheetah-medium-replay-v0",
            "halfcheetah-medium-expert-v0"
        ],
        "agent_name" => "SAC",
        "return_mean" => 12695.696030461002,
        "return_std =>" => 209.98612023443096
    ),
    Dict(
        "policy_path" => "halfcheetah/halfcheetah_online_1.pkl",
        "task.task_names" => [
            "halfcheetah-medium-v0",
            "halfcheetah-random-v0",
            "halfcheetah-expert-v0",
            "halfcheetah-medium-replay-v0",
            "halfcheetah-medium-expert-v0"
        ],
        "agent_name" => "SAC",
        "return_mean" => 5686.148033603298,
        "return_std =>" => 77.60317050580818
    ),
    Dict(
        "policy_path" => "halfcheetah/halfcheetah_online_2.pkl",
        "task.task_names" => [
            "halfcheetah-medium-v0",
            "halfcheetah-random-v0",
            "halfcheetah-expert-v0",
            "halfcheetah-medium-replay-v0",
            "halfcheetah-medium-expert-v0"
        ],
        "agent_name" => "SAC",
        "return_mean" => 6898.252473142946,
        "return_std =>" => 131.2808199171071
    ),
    Dict(
        "policy_path" => "halfcheetah/halfcheetah_online_3.pkl",
        "task.task_names" => [
            "halfcheetah-medium-v0",
            "halfcheetah-random-v0",
            "halfcheetah-expert-v0",
            "halfcheetah-medium-replay-v0",
            "halfcheetah-medium-expert-v0"
        ],
        "agent_name" => "SAC",
        "return_mean" => 7843.345957832609,
        "return_std =>" => 119.82879594969056
    ),
    Dict(
        "policy_path" => "halfcheetah/halfcheetah_online_4.pkl",
        "task.task_names" => [
            "halfcheetah-medium-v0",
            "halfcheetah-random-v0",
            "halfcheetah-expert-v0",
            "halfcheetah-medium-replay-v0",
            "halfcheetah-medium-expert-v0"
        ],
        "agent_name" => "SAC",
        "return_mean" => 8661.367146815282,
        "return_std =>" => 142.1433195543218
    ),
    Dict(
        "policy_path" => "halfcheetah/halfcheetah_online_5.pkl",
        "task.task_names" => [
            "halfcheetah-medium-v0",
            "halfcheetah-random-v0",
            "halfcheetah-expert-v0",
            "halfcheetah-medium-replay-v0",
            "halfcheetah-medium-expert-v0"
        ],
        "agent_name" => "SAC",
        "return_mean" => 9197.889639800613,
        "return_std =>" => 125.40543058761767
    ),
    Dict(
        "policy_path" => "halfcheetah/halfcheetah_online_6.pkl",
        "task.task_names" => [
            "halfcheetah-medium-v0",
            "halfcheetah-random-v0",
            "halfcheetah-expert-v0",
            "halfcheetah-medium-replay-v0",
            "halfcheetah-medium-expert-v0"
        ],
        "agent_name" => "SAC",
        "return_mean" => 9623.789519132608,
        "return_std =>" => 130.91946985245835
    ),
    Dict(
        "policy_path" => "halfcheetah/halfcheetah_online_7.pkl",
        "task.task_names" => [
            "halfcheetah-medium-v0",
            "halfcheetah-random-v0",
            "halfcheetah-expert-v0",
            "halfcheetah-medium-replay-v0",
            "halfcheetah-medium-expert-v0"
        ],
        "agent_name" => "SAC",
        "return_mean" => 10255.26711299773,
        "return_std =>" => 173.52116806555978
    ),
    Dict(
        "policy_path" => "halfcheetah/halfcheetah_online_8.pkl",
        "task.task_names" => [
            "halfcheetah-medium-v0",
            "halfcheetah-random-v0",
            "halfcheetah-expert-v0",
            "halfcheetah-medium-replay-v0",
            "halfcheetah-medium-expert-v0"
        ],
        "agent_name" => "SAC",
        "return_mean" => 10899.460856799158,
        "return_std =>" => 324.2557642475202
    ),
    Dict(
        "policy_path" => "halfcheetah/halfcheetah_online_9.pkl",
        "task.task_names" => [
            "halfcheetah-medium-v0",
            "halfcheetah-random-v0",
            "halfcheetah-expert-v0",
            "halfcheetah-medium-replay-v0",
            "halfcheetah-medium-expert-v0"
        ],
        "agent_name" => "SAC",
        "return_mean" => 11829.054827593913,
        "return_std =>" => 240.63510160394745
    ),
    Dict(
        "policy_path" => "hammer/hammer_dapg_0.pkl",
        "task.task_names" => [
            "hammer-cloned-v0",
            "hammer-expert-v0",
            "hammer-human-v0"
        ],
        "agent_name" => "DAPG",
        "return_mean" => -236.37114898868305,
        "return_std =>" => 5.2941436284324075
    ),
    Dict(
        "policy_path" => "hammer/hammer_dapg_10.pkl",
        "task.task_names" => [
            "hammer-cloned-v0",
            "hammer-expert-v0",
            "hammer-human-v0"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 17585.58837262877,
        "return_std =>" => 96.53489547795978
    ),
    Dict(
        "policy_path" => "hammer/hammer_dapg_1.pkl",
        "task.task_names" => [
            "hammer-cloned-v0",
            "hammer-expert-v0",
            "hammer-human-v0"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 128.60395654435058,
        "return_std =>" => 30.68441678661929
    ),
    Dict(
        "policy_path" => "hammer/hammer_dapg_2.pkl",
        "task.task_names" => [
            "hammer-cloned-v0",
            "hammer-expert-v0",
            "hammer-human-v0"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 7408.354956936379,
        "return_std =>" => 7294.096332941535
    ),
    Dict(
        "policy_path" => "hammer/hammer_dapg_3.pkl",
        "task.task_names" => [
            "hammer-cloned-v0",
            "hammer-expert-v0",
            "hammer-human-v0"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 15594.112899701715,
        "return_std =>" => 197.28904701529942
    ),
    Dict(
        "policy_path" => "hammer/hammer_dapg_4.pkl",
        "task.task_names" => [
            "hammer-cloned-v0",
            "hammer-expert-v0",
            "hammer-human-v0"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 16245.548923178216,
        "return_std =>" => 262.7060238728634
    ),
    Dict(
        "policy_path" => "hammer/hammer_dapg_5.pkl",
        "task.task_names" => [
            "hammer-cloned-v0",
            "hammer-expert-v0",
            "hammer-human-v0"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 16595.136728219404,
        "return_std =>" => 124.5270089215883
    ),
    Dict(
        "policy_path" => "hammer/hammer_dapg_6.pkl",
        "task.task_names" => [
            "hammer-cloned-v0",
            "hammer-expert-v0",
            "hammer-human-v0"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 17065.590900836418,
        "return_std =>" => 55.85140116556182
    ),
    Dict(
        "policy_path" => "hammer/hammer_dapg_7.pkl",
        "task.task_names" => [
            "hammer-cloned-v0",
            "hammer-expert-v0",
            "hammer-human-v0"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 17209.380445590097,
        "return_std =>" => 35.922080086069116
    ),
    Dict(
        "policy_path" => "hammer/hammer_dapg_8.pkl",
        "task.task_names" => [
            "hammer-cloned-v0",
            "hammer-expert-v0",
            "hammer-human-v0"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 17388.10343669515,
        "return_std =>" => 71.04818789434533
    ),
    Dict(
        "policy_path" => "hammer/hammer_dapg_9.pkl",
        "task.task_names" => [
            "hammer-cloned-v0",
            "hammer-expert-v0",
            "hammer-human-v0"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 17565.807571496796,
        "return_std =>" => 83.22119300427666
    ),
    Dict(
        "policy_path" => "hopper/hopper_online_0.pkl",
        "task.task_names" => [
            "hopper-medium-v0",
            "hopper-random-v0",
            "hopper-expert-v0",
            "hopper-medium-replay-v0",
            "hopper-medium-expert-v0"
        ],
        "agent_name" => "SAC",
        "return_mean" => 89.08207455972816,
        "return_std =>" => 45.69740377810402
    ),
    Dict(
        "policy_path" => "hopper/hopper_online_10.pkl",
        "task.task_names" => [
            "hopper-medium-v0",
            "hopper-random-v0",
            "hopper-expert-v0",
            "hopper-medium-replay-v0",
            "hopper-medium-expert-v0"
        ],
        "agent_name" => "SAC",
        "return_mean" => 1290.7677147248753,
        "return_std =>" => 86.34701290680572
    ),
    Dict(
        "policy_path" => "hopper/hopper_online_1.pkl",
        "task.task_names" => [
            "hopper-medium-v0",
            "hopper-random-v0",
            "hopper-expert-v0",
            "hopper-medium-replay-v0",
            "hopper-medium-expert-v0"
        ],
        "agent_name" => "SAC",
        "return_mean" => 1134.244611055915,
        "return_std =>" => 407.6547443287992
    ),
    Dict(
        "policy_path" => "hopper/hopper_online_2.pkl",
        "task.task_names" => [
            "hopper-medium-v0",
            "hopper-random-v0",
            "hopper-expert-v0",
            "hopper-medium-replay-v0",
            "hopper-medium-expert-v0"
        ],
        "agent_name" => "SAC",
        "return_mean" => 727.0768143435397,
        "return_std =>" => 92.94955320157855
    ),
    Dict(
        "policy_path" => "hopper/hopper_online_3.pkl",
        "task.task_names" => [
            "hopper-medium-v0",
            "hopper-random-v0",
            "hopper-expert-v0",
            "hopper-medium-replay-v0",
            "hopper-medium-expert-v0"
        ],
        "agent_name" => "SAC",
        "return_mean" => 1571.2810005160163,
        "return_std =>" => 447.3216244940128
    ),
    Dict(
        "policy_path" => "hopper/hopper_online_4.pkl",
        "task.task_names" => [
            "hopper-medium-v0",
            "hopper-random-v0",
            "hopper-expert-v0",
            "hopper-medium-replay-v0",
            "hopper-medium-expert-v0"
        ],
        "agent_name" => "SAC",
        "return_mean" => 1140.2394986005213,
        "return_std =>" => 671.1379607505328
    ),
    Dict(
        "policy_path" => "hopper/hopper_online_5.pkl",
        "task.task_names" => [
            "hopper-medium-v0",
            "hopper-random-v0",
            "hopper-expert-v0",
            "hopper-medium-replay-v0",
            "hopper-medium-expert-v0"
        ],
        "agent_name" => "SAC",
        "return_mean" => 1872.571834592923,
        "return_std =>" => 793.8865779126361
    ),
    Dict(
        "policy_path" => "hopper/hopper_online_6.pkl",
        "task.task_names" => [
            "hopper-medium-v0",
            "hopper-random-v0",
            "hopper-expert-v0",
            "hopper-medium-replay-v0",
            "hopper-medium-expert-v0"
        ],
        "agent_name" => "SAC",
        "return_mean" => 3088.2017624993064,
        "return_std =>" => 356.52713477862386
    ),
    Dict(
        "policy_path" => "hopper/hopper_online_7.pkl",
        "task.task_names" => [
            "hopper-medium-v0",
            "hopper-random-v0",
            "hopper-expert-v0",
            "hopper-medium-replay-v0",
            "hopper-medium-expert-v0"
        ],
        "agent_name" => "SAC",
        "return_mean" => 1726.0060438089222,
        "return_std =>" => 761.6326666292086
    ),
    Dict(
        "policy_path" => "hopper/hopper_online_8.pkl",
        "task.task_names" => [
            "hopper-medium-v0",
            "hopper-random-v0",
            "hopper-expert-v0",
            "hopper-medium-replay-v0",
            "hopper-medium-expert-v0"
        ],
        "agent_name" => "SAC",
        "return_mean" => 2952.957468938808,
        "return_std =>" => 682.5831907733249
    ),
    Dict(
        "policy_path" => "hopper/hopper_online_9.pkl",
        "task.task_names" => [
            "hopper-medium-v0",
            "hopper-random-v0",
            "hopper-expert-v0",
            "hopper-medium-replay-v0",
            "hopper-medium-expert-v0"
        ],
        "agent_name" => "SAC",
        "return_mean" => 2369.7998719150673,
        "return_std =>" => 1119.4914225331481
    ),
    Dict(
        "policy_path" => "maze2d_large/maze2d_large_dapg_0.pkl",
        "task.task_names" => [
            "maze2d-large-v1"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 2.21,
        "return_std =>" => 8.873888662812938
    ),
    Dict(
        "policy_path" => "maze2d_large/maze2d_large_dapg_10.pkl",
        "task.task_names" => [
            "maze2d-large-v1"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 627.86,
        "return_std =>" => 161.0254650668645
    ),
    Dict(
        "policy_path" => "maze2d_large/maze2d_large_dapg_1.pkl",
        "task.task_names" => [
            "maze2d-large-v1"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 41.74,
        "return_std =>" => 72.2068722491149
    ),
    Dict(
        "policy_path" => "maze2d_large/maze2d_large_dapg_2.pkl",
        "task.task_names" => [
            "maze2d-large-v1"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 124.9,
        "return_std =>" => 131.5638628195448
    ),
    Dict(
        "policy_path" => "maze2d_large/maze2d_large_dapg_3.pkl",
        "task.task_names" => [
            "maze2d-large-v1"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 107.78,
        "return_std =>" => 109.32251186283638
    ),
    Dict(
        "policy_path" => "maze2d_large/maze2d_large_dapg_4.pkl",
        "task.task_names" => [
            "maze2d-large-v1"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 289.46,
        "return_std =>" => 262.69070862898826
    ),
    Dict(
        "policy_path" => "maze2d_large/maze2d_large_dapg_5.pkl",
        "task.task_names" => [
            "maze2d-large-v1"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 356.17,
        "return_std =>" => 276.9112151936068
    ),
    Dict(
        "policy_path" => "maze2d_large/maze2d_large_dapg_6.pkl",
        "task.task_names" => [
            "maze2d-large-v1"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 393.87,
        "return_std =>" => 309.08651394067647
    ),
    Dict(
        "policy_path" => "maze2d_large/maze2d_large_dapg_7.pkl",
        "task.task_names" => [
            "maze2d-large-v1"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 517.4,
        "return_std =>" => 274.58688970888613
    ),
    Dict(
        "policy_path" => "maze2d_large/maze2d_large_dapg_8.pkl",
        "task.task_names" => [
            "maze2d-large-v1"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 565.42,
        "return_std =>" => 210.94450360225082
    ),
    Dict(
        "policy_path" => "maze2d_large/maze2d_large_dapg_9.pkl",
        "task.task_names" => [
            "maze2d-large-v1"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 629.22,
        "return_std =>" => 123.23023817229276
    ),
    Dict(
        "policy_path" => "maze2d_medium/maze2d_medium_dapg_0.pkl",
        "task.task_names" => [
            "maze2d-medium-v1"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 83.15,
        "return_std =>" => 177.59827561099797
    ),
    Dict(
        "policy_path" => "maze2d_medium/maze2d_medium_dapg_10.pkl",
        "task.task_names" => [
            "maze2d-medium-v1"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 442.35,
        "return_std =>" => 161.2205554512203
    ),
    Dict(
        "policy_path" => "maze2d_medium/maze2d_medium_dapg_1.pkl",
        "task.task_names" => [
            "maze2d-medium-v1"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 177.8,
        "return_std =>" => 218.1089635938881
    ),
    Dict(
        "policy_path" => "maze2d_medium/maze2d_medium_dapg_2.pkl",
        "task.task_names" => [
            "maze2d-medium-v1"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 249.33,
        "return_std =>" => 237.2338110388146
    ),
    Dict(
        "policy_path" => "maze2d_medium/maze2d_medium_dapg_3.pkl",
        "task.task_names" => [
            "maze2d-medium-v1"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 214.81,
        "return_std =>" => 246.09809812349224
    ),
    Dict(
        "policy_path" => "maze2d_medium/maze2d_medium_dapg_4.pkl",
        "task.task_names" => [
            "maze2d-medium-v1"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 254.63,
        "return_std =>" => 262.0181541420365
    ),
    Dict(
        "policy_path" => "maze2d_medium/maze2d_medium_dapg_5.pkl",
        "task.task_names" => [
            "maze2d-medium-v1"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 238.76,
        "return_std =>" => 260.3596404975241
    ),
    Dict(
        "policy_path" => "maze2d_medium/maze2d_medium_dapg_6.pkl",
        "task.task_names" => [
            "maze2d-medium-v1"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 374.9,
        "return_std =>" => 222.18107480161314
    ),
    Dict(
        "policy_path" => "maze2d_medium/maze2d_medium_dapg_7.pkl",
        "task.task_names" => [
            "maze2d-medium-v1"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 379.68,
        "return_std =>" => 228.59111443798514
    ),
    Dict(
        "policy_path" => "maze2d_medium/maze2d_medium_dapg_8.pkl",
        "task.task_names" => [
            "maze2d-medium-v1"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 392.9,
        "return_std =>" => 217.99805044999832
    ),
    Dict(
        "policy_path" => "maze2d_medium/maze2d_medium_dapg_9.pkl",
        "task.task_names" => [
            "maze2d-medium-v1"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 432.03,
        "return_std =>" => 173.93714123211294
    ),
    Dict(
        "policy_path" => "maze2d_umaze/maze2d_umaze_dapg_0.pkl",
        "task.task_names" => [
            "maze2d-umaze-v1"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 22.19,
        "return_std =>" => 25.18320670605711
    ),
    Dict(
        "policy_path" => "maze2d_umaze/maze2d_umaze_dapg_10.pkl",
        "task.task_names" => [
            "maze2d-umaze-v1"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 250.64,
        "return_std =>" => 36.357810715168206
    ),
    Dict(
        "policy_path" => "maze2d_umaze/maze2d_umaze_dapg_1.pkl",
        "task.task_names" => [
            "maze2d-umaze-v1"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 43.33,
        "return_std =>" => 66.01621846182951
    ),
    Dict(
        "policy_path" => "maze2d_umaze/maze2d_umaze_dapg_2.pkl",
        "task.task_names" => [
            "maze2d-umaze-v1"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 100.97,
        "return_std =>" => 95.598060126762
    ),
    Dict(
        "policy_path" => "maze2d_umaze/maze2d_umaze_dapg_3.pkl",
        "task.task_names" => [
            "maze2d-umaze-v1"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 115.26,
        "return_std =>" => 120.07919220247945
    ),
    Dict(
        "policy_path" => "maze2d_umaze/maze2d_umaze_dapg_4.pkl",
        "task.task_names" => [
            "maze2d-umaze-v1"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 106.56,
        "return_std =>" => 123.82562901112192
    ),
    Dict(
        "policy_path" => "maze2d_umaze/maze2d_umaze_dapg_5.pkl",
        "task.task_names" => [
            "maze2d-umaze-v1"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 142.5,
        "return_std =>" => 111.55568116416124
    ),
    Dict(
        "policy_path" => "maze2d_umaze/maze2d_umaze_dapg_6.pkl",
        "task.task_names" => [
            "maze2d-umaze-v1"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 172.13,
        "return_std =>" => 118.24048841238772
    ),
    Dict(
        "policy_path" => "maze2d_umaze/maze2d_umaze_dapg_7.pkl",
        "task.task_names" => [
            "maze2d-umaze-v1"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 190.98,
        "return_std =>" => 73.81706848690214
    ),
    Dict(
        "policy_path" => "maze2d_umaze/maze2d_umaze_dapg_8.pkl",
        "task.task_names" => [
            "maze2d-umaze-v1"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 228.17,
        "return_std =>" => 39.635856241539685
    ),
    Dict(
        "policy_path" => "maze2d_umaze/maze2d_umaze_dapg_9.pkl",
        "task.task_names" => [
            "maze2d-umaze-v1"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 239.34,
        "return_std =>" => 37.597664821102924
    ),
    Dict(
        "policy_path" => "pen/pen_dapg_0.pkl",
        "task.task_names" => [
            "pen-cloned-v0",
            "pen-expert-v0",
            "pen-human-v0"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 1984.096763504694,
        "return_std =>" => 1929.6110474391166
    ),
    Dict(
        "policy_path" => "pen/pen_dapg_10.pkl",
        "task.task_names" => [
            "pen-cloned-v0",
            "pen-expert-v0",
            "pen-human-v0"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 3808.794849593491,
        "return_std =>" => 1932.9965631785215
    ),
    Dict(
        "policy_path" => "pen/pen_dapg_1.pkl",
        "task.task_names" => [
            "pen-cloned-v0",
            "pen-expert-v0",
            "pen-human-v0"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 2480.1224231814135,
        "return_std =>" => 2125.5773427152635
    ),
    Dict(
        "policy_path" => "pen/pen_dapg_2.pkl",
        "task.task_names" => [
            "pen-cloned-v0",
            "pen-expert-v0",
            "pen-human-v0"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 2494.1335875747145,
        "return_std =>" => 2118.0014860996175
    ),
    Dict(
        "policy_path" => "pen/pen_dapg_3.pkl",
        "task.task_names" => [
            "pen-cloned-v0",
            "pen-expert-v0",
            "pen-human-v0"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 2802.87073294418,
        "return_std =>" => 2120.3981104287323
    ),
    Dict(
        "policy_path" => "pen/pen_dapg_4.pkl",
        "task.task_names" => [
            "pen-cloned-v0",
            "pen-expert-v0",
            "pen-human-v0"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 3136.18545171068,
        "return_std =>" => 2112.923714191993
    ),
    Dict(
        "policy_path" => "pen/pen_dapg_5.pkl",
        "task.task_names" => [
            "pen-cloned-v0",
            "pen-expert-v0",
            "pen-human-v0"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 3110.619191864754,
        "return_std =>" => 2012.2585161410343
    ),
    Dict(
        "policy_path" => "pen/pen_dapg_6.pkl",
        "task.task_names" => [
            "pen-cloned-v0",
            "pen-expert-v0",
            "pen-human-v0"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 3410.4384362331157,
        "return_std =>" => 2029.187357465904
    ),
    Dict(
        "policy_path" => "pen/pen_dapg_7.pkl",
        "task.task_names" => [
            "pen-cloned-v0",
            "pen-expert-v0",
            "pen-human-v0"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 3489.353704450997,
        "return_std =>" => 2035.2279026017748
    ),
    Dict(
        "policy_path" => "pen/pen_dapg_8.pkl",
        "task.task_names" => [
            "pen-cloned-v0",
            "pen-expert-v0",
            "pen-human-v0"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 3673.9622983303598,
        "return_std =>" => 2052.8837762657795
    ),
    Dict(
        "policy_path" => "pen/pen_dapg_9.pkl",
        "task.task_names" => [
            "pen-cloned-v0",
            "pen-expert-v0",
            "pen-human-v0"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 3683.932983177092,
        "return_std =>" => 2028.9543873822265
    ),
    Dict(
        "policy_path" => "relocate/relocate_dapg_0.pkl",
        "task.task_names" => [
            "relocate-cloned-v0",
            "relocate-expert-v0",
            "relocate-human-v0"
        ],
        "agent_name" => "DAPG",
        "return_mean" => -4.4718813284277195,
        "return_std =>" => 0.9021515021945451
    ),
    Dict(
        "policy_path" => "relocate/relocate_dapg_10.pkl",
        "task.task_names" => [
            "relocate-cloned-v0",
            "relocate-expert-v0",
            "relocate-human-v0"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 3481.7834354311035,
        "return_std =>" => 813.1857720257618
    ),
    Dict(
        "policy_path" => "relocate/relocate_dapg_1.pkl",
        "task.task_names" => [
            "relocate-cloned-v0",
            "relocate-expert-v0",
            "relocate-human-v0"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 5.070946470816939,
        "return_std =>" => 31.708695854456067
    ),
    Dict(
        "policy_path" => "relocate/relocate_dapg_2.pkl",
        "task.task_names" => [
            "relocate-cloned-v0",
            "relocate-expert-v0",
            "relocate-human-v0"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 54.976670129729555,
        "return_std =>" => 140.09635704443158
    ),
    Dict(
        "policy_path" => "relocate/relocate_dapg_3.pkl",
        "task.task_names" => [
            "relocate-cloned-v0",
            "relocate-expert-v0",
            "relocate-human-v0"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 54.11338525066304,
        "return_std =>" => 146.87277676706216
    ),
    Dict(
        "policy_path" => "relocate/relocate_dapg_4.pkl",
        "task.task_names" => [
            "relocate-cloned-v0",
            "relocate-expert-v0",
            "relocate-human-v0"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 97.16474411169358,
        "return_std =>" => 164.81156449057102
    ),
    Dict(
        "policy_path" => "relocate/relocate_dapg_5.pkl",
        "task.task_names" => [
            "relocate-cloned-v0",
            "relocate-expert-v0",
            "relocate-human-v0"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 366.3185681324701,
        "return_std =>" => 581.577837554543
    ),
    Dict(
        "policy_path" => "relocate/relocate_dapg_6.pkl",
        "task.task_names" => [
            "relocate-cloned-v0",
            "relocate-expert-v0",
            "relocate-human-v0"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 1254.0676523894747,
        "return_std =>" => 929.5248207929493
    ),
    Dict(
        "policy_path" => "relocate/relocate_dapg_7.pkl",
        "task.task_names" => [
            "relocate-cloned-v0",
            "relocate-expert-v0",
            "relocate-human-v0"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 2700.2361856493385,
        "return_std =>" => 1089.9871332809942
    ),
    Dict(
        "policy_path" => "relocate/relocate_dapg_8.pkl",
        "task.task_names" => [
            "relocate-cloned-v0",
            "relocate-expert-v0",
            "relocate-human-v0"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 2570.351217370911,
        "return_std =>" => 1266.9305994339466
    ),
    Dict(
        "policy_path" => "relocate/relocate_dapg_9.pkl",
        "task.task_names" => [
            "relocate-cloned-v0",
            "relocate-expert-v0",
            "relocate-human-v0"
        ],
        "agent_name" => "DAPG",
        "return_mean" => 3379.424369497742,
        "return_std =>" => 948.6183219418235
    ),
    Dict(
        "policy_path" => "walker/walker_online_0.pkl",
        "task.task_names" => [
            "walker2d-medium-v0",
            "walker2d-random-v0",
            "walker2d-expert-v0",
            "walker2d-medium-replay-v0",
            "walker2d-medium-expert-v0"
        ],
        "agent_name" => "SAC",
        "return_mean" => 17.57372020467802,
        "return_std =>" => 51.686802739349666
    ),
    Dict(
        "policy_path" => "walker/walker_online_10.pkl",
        "task.task_names" => [
            "walker2d-medium-v0",
            "walker2d-random-v0",
            "walker2d-expert-v0",
            "walker2d-medium-replay-v0",
            "walker2d-medium-expert-v0"
        ],
        "agent_name" => "SAC",
        "return_mean" => 4120.947079569632,
        "return_std =>" => 468.1515654051671
    ),
    Dict(
        "policy_path" => "walker/walker_online_1.pkl",
        "task.task_names" => [
            "walker2d-medium-v0",
            "walker2d-random-v0",
            "walker2d-expert-v0",
            "walker2d-medium-replay-v0",
            "walker2d-medium-expert-v0"
        ],
        "agent_name" => "SAC",
        "return_mean" => 193.84631742541606,
        "return_std =>" => 185.16785303932383
    ),
    Dict(
        "policy_path" => "walker/walker_online_2.pkl",
        "task.task_names" => [
            "walker2d-medium-v0",
            "walker2d-random-v0",
            "walker2d-expert-v0",
            "walker2d-medium-replay-v0",
            "walker2d-medium-expert-v0"
        ],
        "agent_name" => "SAC",
        "return_mean" => 942.6191179097373,
        "return_std =>" => 532.9834162811841
    ),
    Dict(
        "policy_path" => "walker/walker_online_3.pkl",
        "task.task_names" => [
            "walker2d-medium-v0",
            "walker2d-random-v0",
            "walker2d-expert-v0",
            "walker2d-medium-replay-v0",
            "walker2d-medium-expert-v0"
        ],
        "agent_name" => "SAC",
        "return_mean" => 2786.7497792224794,
        "return_std =>" => 477.5450988462439
    ),
    Dict(
        "policy_path" => "walker/walker_online_4.pkl",
        "task.task_names" => [
            "walker2d-medium-v0",
            "walker2d-random-v0",
            "walker2d-expert-v0",
            "walker2d-medium-replay-v0",
            "walker2d-medium-expert-v0"
        ],
        "agent_name" => "SAC",
        "return_mean" => 914.4680927038296,
        "return_std =>" => 559.5155757967623
    ),
    Dict(
        "policy_path" => "walker/walker_online_5.pkl",
        "task.task_names" => [
            "walker2d-medium-v0",
            "walker2d-random-v0",
            "walker2d-expert-v0",
            "walker2d-medium-replay-v0",
            "walker2d-medium-expert-v0"
        ],
        "agent_name" => "SAC",
        "return_mean" => 3481.491012709211,
        "return_std =>" => 87.12729823320758
    ),
    Dict(
        "policy_path" => "walker/walker_online_6.pkl",
        "task.task_names" => [
            "walker2d-medium-v0",
            "walker2d-random-v0",
            "walker2d-expert-v0",
            "walker2d-medium-replay-v0",
            "walker2d-medium-expert-v0"
        ],
        "agent_name" => "SAC",
        "return_mean" => 2720.2509272083826,
        "return_std =>" => 746.9753406110725
    ),
    Dict(
        "policy_path" => "walker/walker_online_7.pkl",
        "task.task_names" => [
            "walker2d-medium-v0",
            "walker2d-random-v0",
            "walker2d-expert-v0",
            "walker2d-medium-replay-v0",
            "walker2d-medium-expert-v0"
        ],
        "agent_name" => "SAC",
        "return_mean" => 3926.346852318098,
        "return_std =>" => 365.4230491920236
    ),
    Dict(
        "policy_path" => "walker/walker_online_8.pkl",
        "task.task_names" => [
            "walker2d-medium-v0",
            "walker2d-random-v0",
            "walker2d-expert-v0",
            "walker2d-medium-replay-v0",
            "walker2d-medium-expert-v0"
        ],
        "agent_name" => "SAC",
        "return_mean" => 3695.4887678612936,
        "return_std =>" => 262.0350155576298
    ),
    Dict(
        "policy_path" => "walker/walker_online_9.pkl",
        "task.task_names" => [
            "walker2d-medium-v0",
            "walker2d-random-v0",
            "walker2d-expert-v0",
            "walker2d-medium-replay-v0",
            "walker2d-medium-expert-v0"
        ],
        "agent_name" => "SAC",
        "return_mean" => 4122.358396232011,
        "return_std =>" => 107.76279305206488
    )
]