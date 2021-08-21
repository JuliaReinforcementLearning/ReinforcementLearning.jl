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
        "publishedDate":"2021-08-20",
        "citationText":"Guoyu Yang, 2021"
    }"""

@def appendix = """
    ### Corrections
    If you see mistakes or want to suggest changes, please [create an issue](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/issues) in the source repository.
    """

@def bibliography = "bibliography.bib"

# Technical Report
This technical report is the second phase technical report of Project "Enriching Offline Reinforcement Learning Algorithms in ReinforcementLearning.jl" in OSPP. Now, it includes two components: weekly plans and completed work.

## Weekly Plans
### Week 7
This week, we will prepare to implement FisherBRC \dcite{DBLP:conf/icml/KostrikovFTN21} algorithm. FisherBRC is an improved version of SAC\dcite{DBLP:journals/corr/abs-1812-059051} and BRAC\dcite{DBLP:journals/corr/abs-1911-11361}.
\dfig{body;FisherBRC.png}
- In first step, it trains a behavior policy with entropy (Implementation of official code). We need to implement a simple component because existing [`BehaviorCloningPolicy`](https://juliareinforcementlearning.org/docs/rlzoo/#ReinforcementLearningZoo.BehaviorCloningPolicy-Union{Tuple{},%20Tuple{A}}%20where%20A)  does not contain entropy terms and support continuous action space.
- Design how to add a gradient penalty regularizer in Julia.

Besides, we need some useful components in the task of continuous action space. For example, we only implement [`GaussianNetwork`](https://juliareinforcementlearning.org/docs/rlcore/#ReinforcementLearningCore.GaussianNetwork), but we need the Gaussian Mixture Model to handle many complex tasks.

## Completed Work
Waiting for update.