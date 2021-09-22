@def title = "Enriching Offline Reinforcement Learning Algorithms in ReinforcementLearning.jl"
@def description = """
    This is the phase 2 technical report of the summer OSPP project [Enriching Offline Reinforcement Learning Algorithms in ReinforcementLearning.jl](https://summer.iscas.ac.cn/#/org/prodetail/210370539?lang=en). This report will be continuously updated during the project. Currently, this report contains specific weekly plans and completed work. After all the work is completed, this report will be organized into a complete version.
    """
@def is_enable_toc = true
@def has_code = true
@def has_math = true

@def front_matter = """
    {
        "authors": [
            {
                "author":"Guoyu Yang",
                "authorURL":"https://github.com/pilgrimygy",
                "affiliation":"Nanjing University, LAMDA Group",
                "affiliationURL":"https://www.lamda.nju.edu.cn"
            }
        ],
        "publishedDate":"2021-09-21",
        "citationText":"Guoyu Yang, 2021"
    }"""

@def appendix = """
    ### Corrections
    If you see mistakes or want to suggest changes, please [create an issue](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/issues) in the source repository.
    """

@def bibliography = "bibliography.bib"

# Technical Report
This technical report is the second phase technical report of Project "Enriching Offline Reinforcement Learning Algorithms in ReinforcementLearning.jl" in OSPP. It summarizes all the results of the second phase.

## Weekly Plans
#### Week 7
This week, we will prepare to implement FisherBRC \dcite{DBLP:conf/icml/KostrikovFTN21} algorithm. FisherBRC is an improved version of SAC\dcite{DBLP:journals/corr/abs-1812-05905} and BRAC\dcite{DBLP:journals/corr/abs-1911-11361}.
- In first step, it trains a behavior policy with entropy (Implementation of official code). We need to implement a simple component because existing [`BehaviorCloningPolicy`](https://juliareinforcementlearning.org/docs/rlzoo/#ReinforcementLearningZoo.BehaviorCloningPolicy-Union{Tuple{},%20Tuple{A}}%20where%20A)  does not contain entropy terms and support continuous action space.
- Design how to add a gradient penalty regularizer in Julia.

Besides, we need some useful components in the task of continuous action space. For example, we only implement [`GaussianNetwork`](https://juliareinforcementlearning.org/docs/rlcore/#ReinforcementLearningCore.GaussianNetwork), but we need the Gaussian Mixture Model to handle many complex tasks.

#### Week 8
Last week we finished the FisherBRC algorithm.

This week we will read the papers of BEAR\dcite{DBLP:conf/nips/KumarFSTL19} and UWAC\dcite{DBLP:conf/icml/0001ZSSZSG21} (the improvement of BEAR), and their python code implementation.

And we continue to design and implement GMM (Gaussian Mixture Model).

## Completed Work
#### Offline RL algorithms
##### FisherBRC
The pseudocode of FisherBRC:

\dfig{body;FisherBRC.png}

Firstly, it needs to pre-train a behavior policy $\mu$ by Behavior Cloning. In official python implementation, it adds an entropy term in negative log-likelihood of actions in a given state. Mathematical formulation:

$$\mathcal{L}(\mu) = \mathbb{E}[-\log \mu(s|a) + \alpha \mathcal{H}(\mu)]$$

Besides, it automatically adjusts entropy term like SAC:

$$J(\alpha) = -\alpha \mathbb{E}_{a_t\sim \mu_t}[\log\mu(a_t|s_t) + \bar{\mathcal{H}}]$$

Where $\bar{\mathcal{H}}$ is target entropy. But in [ReinforcementLearningZoo.jl](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/tree/master/src/ReinforcementLearningZoo), [`BehaviorCloningPolicy`](https://juliareinforcementlearning.org/docs/rlzoo/#ReinforcementLearningZoo.BehaviorCloningPolicy-Union{Tuple{},%20Tuple{A}}%20where%20A) does not contain entropy terms and does not support continuous action space. So, we define `EntropyBC`:
```julia
mutable struct EntropyBC{A<:NeuralNetworkApproximator}
    policy::A
    α::Float32
    lr_alpha::Float32
    target_entropy::Float32
    # Logging
    policy_loss::Float32
end
```
Users only need to set parameter `policy` and `lr_alpha`. `policy` usually uses a [`GaussianNetwork`](https://juliareinforcementlearning.org/docs/rlcore/#ReinforcementLearningCore.GaussianNetwork). `lr_alpha` is the learning rate of `α`, which is an entropy term. `target_entropy` is set to $-\dim(\mathcal{A})$, and $\mathcal{A}$ is action space.

Afterwards, the FisherBRC learner is updated. When updating Actor, it adds an entropy term in Q-value loss and automatically adjusts entropy. It updates Critic by this equation:

$$\min_\theta J(O_\theta + \log\mu(a|s)) + \lambda \mathbb{E}_{s\sim D, a\sim \pi_\phi(\cdot|s)}[\|\nabla_a O_\theta(s,a)\|^2]$$

There are a few key concepts that need to be introduced. $J$ is the standard Q-value loss function. $O_\theta(s,a)$ is offset network:

$$Q_\theta(s,a) = O_\theta(s,a) + \log\mu(a|s)$$

Instead of $Q_\theta(s,a)$, $O_\theta(s,a)$ will provide a richer representation of Q-values. However, this parameterization can potentially put us back in the fully-parameterized $Q_\theta$ regime of vanilla actor critic. So it uses a gradient penalty regularizer of the form $\|\nabla_a O_\theta(s,a)\|$. The implementation is as follows: 

```julia
a_policy = l.policy(l.rng, s; is_sampling=true)
q_grad_1 = gradient(Flux.params(l.qnetwork1)) do
    q1 = l.qnetwork1(q_input) |> vec
    q1_grad_norm = gradient(Flux.params([a_policy])) do 
        q1_reg = mean(l.qnetwork1(vcat(s, a_policy)))
    end
    reg = mean(q1_grad_norm[a_policy] .^ 2)
    loss = mse(q1 .+ log_μ, y) + l.f_reg * reg  # y is target value
end
```

Please refer to this link for specific code ([link](https://juliareinforcementlearning.org/docs/rlzoo/#ReinforcementLearningZoo.FisherBRCLearner-Tuple{})). The brief function parameters are as follows:

```julia
mutable struct FisherBRCLearner{BA1, BC1, BC2, R} <: AbstractLearner
    ### Omit other parameters
    policy::BA1
    behavior_policy::EntropyBC
    qnetwork1::BC1
    qnetwork2::BC2
    target_qnetwork1::BC1
    target_qnetwork2::BC2
    α::Float32
    f_reg::Float32
    reward_bonus::Float32
    pretrain_step::Int
    lr_alpha::Float32
    target_entropy::Float32
end
```
`f_reg` is the regularization parameter of $\|\nabla_a O_\theta(s,a)\|$. `reward_bonus` is generally set to 5, which is added in the reward to improve performance. `pretrain_step` is used for pre-training `behavior_policy`. `α`, `lr_alpha` and `target_entropy` are parameters used to add an entropy term and automatically adjust the entropy.

Performance curve of FisherBRC algorithm in Pendulum (`pertrain_step=100`):

\dfig{body;JuliaRL_FisherBRC_Pendulum.png}

FisherBRC's performance is better than online SAC.