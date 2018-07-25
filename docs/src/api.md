# API

New learners, policies, callbacks, environments, evaluation metrics or stopping
criteria need to implement the following functions.

## Learners
```
update!(learner, buffer)
```
Returns nothing.

```
selectaction(learner, policy, state)
```
Returns an action.

```
defaultbuffer(learner, environment, preprocessor)
```
Returns nothing.

## Policies
```
selectaction(policy, values)
```
Returns an action.

```
getactionprobabilities(policy, state)
```
Returns a normalized (1-norm) vector with non-negative entries.

## Callbacks
```
callback!(callback, rlsetup, state, action, reward, done)
```
Returns nothing.

## [Environments](@id api_environments)
```
interact!(action, environment)
```
Returns state, reward, done.

```
getstate(environment)
```
Returns state, done.

```
reset!(environment)
```
Returns nothing.

## [Evaluation Metrics](@id getvalue)

```
getvalue(metric)
```
Any return value allowed.

## Stopping Criteria
```
isbreak!(stoppingcriterion, state, action, reward, done)
```
Returns true or false.
