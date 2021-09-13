import Printf.@sprintf
export dm_params

function dm_params()
    rodent = keys(DM_LOCOMOTION_RODENT)
    humanoid = keys(DM_LOCOMOTION_HUMANOID)
    dm_control = keys(DM_CONTROL_SUITE_SIZE)

    game = Dict("rodent" => rodent, "humanoid" => humanoid, "dm_control" => dm_control)
    shards = 0:4

    @info game shards
end

const DM_LOCOMOTION_RODENT = Dict{String, String}(
    "rodent_gaps" => "dm_locomotion/rodent_gaps/seq2",
    "rodent_escape" => "dm_locomotion/rodent_bowl_escape/seq2",
    "rodent_two_touch" => "dm_locomotion/rodent_two_touch/seq40",
    "rodent_mazes" => "dm_locomotion/rodent_mazes/seq40"
)

const DM_LOCOMOTION_RODENT_SIZE = Dict{String, Tuple}(
    "observation/walker/actuator_activation" => (38,),
    # "observation/walker/sensors_torque" => (),
    # "observation/walker/sensors_force" => (),
    "observation/walker/body_height" => (1,),
    "observation/walker/end_effectors_pos" => (12,),
    "observation/walker/joints_pos"=> (30,),
    "observation/walker/joints_vel"=> (30,),
    "observation/walker/tendons_pos"=> (8,),
    "observation/walker/tendons_vel"=> (8,),
    "observation/walker/appendages_pos"=> (15,),
    "observation/walker/world_zaxis"=> (3,),
    "observation/walker/sensors_accelerometer"=> (3,),
    "observation/walker/sensors_velocimeter"=> (3,),
    "observation/walker/sensors_gyro"  => (3,),
    "observation/walker/sensors_touch"=> (4,),
    "observation/walker/egocentric_camera"=> (64, 64, 3),
    "action"=> (38,),
    "discount"=> (),
    "reward"=> (),
    "step_type"=> ()
)

const DM_LOCOMOTION_HUMANOID = Dict{String, String}(
    "humanoid_corridor" => "dm_locomotion/humanoid_corridor/seq2",
    "humanoid_gaps" => "dm_locomotion/humanoid_gaps/seq2",
    "humanoid_walls" => "dm_locomotion/humanoid_walls/seq40"
)

const DM_LOCOMOTION_HUMANOID_SIZE = Dict{String, Tuple}(
    # "observation/walker/actuator_activation" => (0,),
    "observation/walker/sensors_torque" => (6,),
    # "observation/walker/sensors_force" => (),
    "observation/walker/joints_vel"=> (56,),
    "observation/walker/sensors_velocimeter"=> (3,),
    "observation/walker/sensors_gyro"=> (3,),
    "observation/walker/joints_pos"=> (56,),
    "observation/walker/appendages_pos" => (15,),
    "observation/walker/world_zaxis"=> (3,),
    "observation/walker/body_height"=> (1,),
    "observation/walker/sensors_accelerometer"=> (3,),
    "observation/walker/end_effectors_pos"=> (12,),
    "observation/walker/egocentric_camera"=> (
        64,
        64,
        3,
    ),
    "action"=> (56,),
    "discount"=> (),
    "reward"=> (),
    # "episodic_reward"=> (),
    "step_type"=> ()
)

const DM_CONTROL_SUITE_SIZE = Dict{String, Dict{String, Tuple}}(
    "cartpole_swingup" => Dict{String, Tuple}(
        "observation/position"=> (3,),
        "observation/velocity"=> (2,),
        "action"=> (1,),
        "discount"=> (),
        "reward"=> (),
        "episodic_reward"=> (),
        "step_type"=> ()
    ),
    "cheetah_run" => Dict{String, Tuple}(
        "observation/position"=> (8,),
        "observation/velocity"=> (9,),
        "action"=> (6,),
        "discount"=> (),
        "reward"=> (),
        "episodic_reward"=> (),
        "step_type"=> ()
    ),
    "finger_turn_hard" => Dict{String, Tuple}(
        "observation/position"=> (4,),
        "observation/velocity"=> (3,),
        "observation/touch"=> (2,),
        "observation/target_position"=> (2,),
        "observation/dist_to_target"=> (1,),
        "action"=> (2,),
        "discount"=> (),
        "reward"=> (),
        "episodic_reward"=> (),
        "step_type"=> ()
    ),
    "fish_swim" => Dict{String, Tuple}(
        "observation/target"=> (3,),
        "observation/velocity"=> (13,),
        "observation/upright"=> (1,),
        "observation/joint_angles"=> (7,),
        "action"=> (5,),
        "discount"=> (),
        "reward"=> (),
        "episodic_reward"=> (),
        "step_type"=> ()
    ),
    "humanoid_run" => Dict{String, Tuple}(
        "observation/velocity"=> (27,),
        "observation/com_velocity"=> (3,),
        "observation/torso_vertical"=> (3,),
        "observation/extremities"=> (12,),
        "observation/head_height"=> (1,),
        "observation/joint_angles"=> (21,),
        "action"=> (21,),
        "discount"=> (),
        "reward"=> (),
        "episodic_reward"=> (),
        "step_type"=> ()
    ),
    "manipulator_insert_ball" => Dict{String, Tuple}(
        "observation/arm_pos"=> (16,),
        "observation/arm_vel"=> (8,),
        "observation/touch"=> (5,),
        "observation/hand_pos"=> (4,),
        "observation/object_pos"=> (4,),
        "observation/object_vel"=> (3,),
        "observation/target_pos"=> (4,),
        "action"=> (5,),
        "discount"=> (),
        "reward"=> (),
        "episodic_reward"=> (),
        "step_type"=> ()
    ),
    "manipulator_insert_peg" => Dict{String, Tuple}(
        "observation/arm_pos"=> (16,),
        "observation/arm_vel"=> (8,),
        "observation/touch"=> (5,),
        "observation/hand_pos"=> (4,),
        "observation/object_pos"=> (4,),
        "observation/object_vel"=> (3,),
        "observation/target_pos"=> (4,),
        "episodic_reward"=> (),
        "action"=> (5,),
        "discount"=> (),
        "reward"=> (),
        "step_type"=> ()
    ),
    "walker_stand" => Dict{String, Tuple}(
        "observation/orientations"=> (14,),
        "observation/velocity"=> (9,),
        "observation/height"=> (1,),
        "action"=> (6,),
        "discount"=> (),
        "reward"=> (),
        "episodic_reward"=> (),
        "step_type"=> ()
    ),
    "walker_walk" => Dict{String, Tuple}(
        "observation/orientations"=> (14,),
        "observation/velocity"=> (9,),
        "observation/height"=> (1,),
        "action"=> (6,),
        "discount"=> (),
        "reward"=> (),
        "episodic_reward"=> (),
        "step_type"=> ()
    )
)

const DM_LOCOMOTION = merge(DM_LOCOMOTION_HUMANOID, DM_LOCOMOTION_RODENT)

num_shards = 100

function dm_init()
    repo = "rl-unplugged-dm"
    for task in keys(DM_CONTROL_SUITE_SIZE)
        for index in 0:num_shards-1
            register(
                DataDep(
                    "$repo-$task-$index",
                    """
                    Dataset: RL Unplugged dm-control
                    Credits: https://arxiv.org/abs/2006.13888
                    Url: https://github.com/deepmind/deepmind-research/tree/master/rl_unplugged
                    Authors: Caglar Gulcehre, Ziyu Wang, Alexander Novikov, Tom Le Paine,
                    Sergio Gómez Colmenarejo, Konrad Zolna, Rishabh Agarwal,
                    Josh Merel, Daniel Mankowitz, Cosmin Paduraru, Gabriel
                    Dulac-Arnold, Jerry Li, Mohammad Norouzi, Matt Hoffman,
                    Ofir Nachum, George Tucker, Nicolas Heess, Nando deFreitas
                    Year: 2020

                    DeepMind Control Suite Tassa et al., 2018 is a set of control tasks implemented in MuJoCo Todorov et al., 2012. 
                    We consider a subset of the tasks provided in the suite that cover a wide range of difficulties.
                    Most of the datasets in this domain are generated using D4PG. For the environments Manipulator insert 
                    ball and Manipulator insert peg we use V-MPO Song et al., 2020 to generate the data as D4PG is unable to 
                    solve these tasks. Datasets are available for 9 dm-control-suite tasks. For details on how the dataset was generated, 
                    please refer to the paper. DeepMind Control Suite is a traditional continuous action RL benchmark. In particular, it is recommended 
                    that you test your approach in DeepMind Control Suite if you are interested in comparing against other state of the art offline RL methods.
                    """,
                    "gs://rl_unplugged/dm_control_suite/$task/"*@sprintf("train-%05i-of-%05i", index, num_shards);
                    fetch_method = fetch_gc_file,
                )
            )
        end
    end
    for task in keys(DM_LOCOMOTION)
        for index in 0:num_shards-1
            register(
                DataDep(
                    "$repo-$task-$index",
                    """
                    Dataset: RL Unplugged dm-locomotion
                    Credits: https://arxiv.org/abs/2006.13888
                    Url: https://github.com/deepmind/deepmind-research/tree/master/rl_unplugged
                    Authors: Caglar Gulcehre, Ziyu Wang, Alexander Novikov, Tom Le Paine,
                    Sergio Gómez Colmenarejo, Konrad Zolna, Rishabh Agarwal,
                    Josh Merel, Daniel Mankowitz, Cosmin Paduraru, Gabriel
                    Dulac-Arnold, Jerry Li, Mohammad Norouzi, Matt Hoffman,
                    Ofir Nachum, George Tucker, Nicolas Heess, Nando deFreitas
                    Year: 2020

                    These tasks are made up of the corridor locomotion tasks involving the CMU Humanoid, for which prior efforts 
                    have either used motion capture data Merel et al., 2019a, Merel et al., 2019b or training from scratch Song et 
                    al., 2020. In addition, the DM Locomotion repository contains a set of tasks adapted to be suited to a virtual 
                    rodent Merel et al., 2020. We emphasize that the DM Locomotion tasks feature the combination of challenging high-DoF 
                    continuous control along with perception from rich egocentric observations. For details on how the dataset was 
                    generated, please refer to the paper.

                    It is recommended that you to try offline RL methods on DeepMind Locomotion dataset, if you are interested in very challenging 
                    offline RL dataset with continuous action space.
                    """,
                    "gs://rl_unplugged/$(DM_LOCOMOTION[task])/"*@sprintf("train-%05i-of-%05i", index, num_shards);
                    fetch_method = fetch_gc_file
                )
            )
        end
    end
end