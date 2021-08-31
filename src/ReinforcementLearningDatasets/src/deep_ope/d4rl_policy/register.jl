using JSON

policy_file_path = "d4rl_policies.json"
gcs_prefix = "gs://gresearch/deep-ope/d4rl"
policy_file = JSON.parsefile(policy_file_path; dicttype=Dict, inttype=Int64, use_mmap=true)

function d4rl_policies_init()
    for dict in policy_file
        policy_path = dict["policy_path"]
        file_name = split(policy_path, "/")[2]
        dep_name = chop(file_name, head=0, tail=4)
        register(
        DataDep(
            "deep-ope-d4rl-$(dep_name)",
            """
            Credits: https://openreview.net/forum?id=kWSeGEeHvF8
            Policy: Benchmarks for Deep Off-Policy Evaluation
            Url: https://github.com/google-research/deep_ope
            Authors: Justin Fu, Mohammad Norouzi, Ofir Nachum, George Tucker, ziyu wang, Alexander Novikov, 
            Mengjiao Yang, Michael R Zhang, Yutian Chen, Aviral Kumar, Cosmin Paduraru, Sergey Levine, Thomas Paine
            Year: 2021
            """,
            "$(gcs_prefix)/$(policy_path)";
            fetch_method = fetch_gc_file,
        )
        )
    end
end