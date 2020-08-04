export AbstractAgent,
    get_role,
    PreExperimentStage,
    PostExperimentStage,
    PreEpisodeStage,
    PostEpisodeStage,
    PreActStage,
    PostActStage,
    PRE_EXPERIMENT_STAGE,
    POST_EXPERIMENT_STAGE,
    PRE_EPISODE_STAGE,
    POST_EPISODE_STAGE,
    PRE_ACT_STAGE,
    POST_ACT_STAGE,
    Training,
    Testing

"""
    (agent::AbstractAgent)(env) = agent(PRE_ACT_STAGE, env) -> action
    (agent::AbstractAgent)(stage::AbstractStage, env)

Similar to [`AbstractPolicy`](@ref), an agent is also a functional object which takes in an observation and returns an action.
The main difference is that, we divide an experiment into the following stages:

- `PRE_EXPERIMENT_STAGE`
- `PRE_EPISODE_STAGE`
- `PRE_ACT_STAGE`
- `POST_ACT_STAGE`
- `POST_EPISODE_STAGE`
- `POST_EXPERIMENT_STAGE`

In each stage, different types of agents may have different behaviors, like updating experience buffer, environment model or policy.
"""
abstract type AbstractAgent end

function get_role(::AbstractAgent) end

"""
                      +-----------------------------------------------------------+                      
                      |Episode                                                    |                      
                      |                                                           |                      
PRE_EXPERIMENT_STAGE  |            PRE_ACT_STAGE    POST_ACT_STAGE                | POST_EXPERIMENT_STAGE
         |            |                  |                |                       |          |           
         v            |        +-----+   v   +-------+    v   +-----+             |          v           
         --------------------->+ env +------>+ agent +------->+ env +---> ... ------->......             
                      |  ^     +-----+       +-------+ action +-----+          ^  |                      
                      |  |                                                     |  |                      
                      |  +--PRE_EPISODE_STAGE            POST_EPISODE_STAGE----+  |                      
                      |                                                           |                      
                      |                                                           |                      
                      +-----------------------------------------------------------+     
"""
abstract type AbstractStage end

struct PreExperimentStage <: AbstractStage end
struct PostExperimentStage <: AbstractStage end
struct PreEpisodeStage <: AbstractStage end
struct PostEpisodeStage <: AbstractStage end
struct PreActStage <: AbstractStage end
struct PostActStage <: AbstractStage end

const PRE_EXPERIMENT_STAGE = PreExperimentStage()
const POST_EXPERIMENT_STAGE = PostExperimentStage()
const PRE_EPISODE_STAGE = PreEpisodeStage()
const POST_EPISODE_STAGE = PostEpisodeStage()
const PRE_ACT_STAGE = PreActStage()
const POST_ACT_STAGE = PostActStage()

(agent::AbstractAgent)(env) = agent(PRE_ACT_STAGE, env)
function (agent::AbstractAgent)(stage::AbstractStage, env) end

struct Training{T<:AbstractStage} end
Training(s::T) where {T<:AbstractStage} = Training{T}()
struct Testing{T<:AbstractStage} end
Testing(s::T) where {T<:AbstractStage} = Testing{T}()

Base.show(io::IO, agent::AbstractAgent) =
    AbstractTrees.print_tree(io, StructTree(agent), get(io, :max_depth, 10))
