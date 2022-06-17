gcs_prefix = "gs://gresearch/deep-ope/d4rl"
folder_prefix = "deep-ope-d4rl" 
policies = D4RL_POLICIES

function deep_ope_d4rl_init()
    for policy in policies
        gcs_policy_folder = policy["policy_path"]
        local_policy_folder = chop(split(gcs_policy_folder, "/")[end], head=0, tail=4)
        register(
            DataDep(
                "$(folder_prefix)-$(local_policy_folder)",
                """
                Policy: Benchmarks for Deep Off-Policy Evaluation 
                Credits: https://openreview.net/forum?id=kWSeGEeHvF8
                Url: https://github.com/google-research/deep_ope
                Authors: Justin Fu, Mohammad Norouzi, Ofir Nachum, George Tucker, ziyu wang, Alexander Novikov, Mengjiao Yang, Michael R Zhang, 
                Yutian Chen, Aviral Kumar, Cosmin Paduraru, Sergey Levine, Thomas Paine
                Year: 2021
        
                Deep OPE contains:
                Policies for the tasks in the D4RL, DeepMind Locomotion and Control Suite datasets.
                Policies trained with the following algorithms (D4PG, ABM, CRR, SAC, DAPG and BC) and snapshots along the training trajectory. This facilitates 
                benchmarking offline model selection.

                D4RL:
                A subset of the tasks within the D4RL (Fu et. al. 2020) for offline reinforcement learning is included. These tasks include maze navigation
                with different robot morphologies, hand manipulation tasks (Rajeswaran et. al. 2017), and tasks from the OpenAI Gym bechmark (Brockman et. al. 2016).
                Each task includes a variety of datasets in order to study the interaction between dataset distributions and policies. For further information on 
                what datasets are available, please refer to D4RL: Datasets for Deep Data-Driven Reinforcement Learning.
                """,
                "$(gcs_prefix)/$(gcs_policy_folder)";
                fetch_method=fetch_gc_file
            )
        )
    end
end