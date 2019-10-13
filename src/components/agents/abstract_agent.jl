export AbstractAgent

"""
An agent is a functional object, which takes in an observation and returns an action. Agents must also implement the `update!(agent::AbstractAgent, obs_action::Pair)` method to indicate how to update the internal state of the agent.
"""
abstract type AbstractAgent end