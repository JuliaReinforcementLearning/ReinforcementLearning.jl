export SARTS
export SART
export SA
export RLDataSet

abstract type RLDataSet end

"""
(:state, :action, :reward, :terminal, :next_state)
type of the returned batches.
"""
const SARTS = (:state, :action, :reward, :terminal, :next_state)

"""
(:state, :action, :reward, :terminal)
type of the returned batches.
"""
const SART = (:state, :action, :reward, :terminal)

"""
(:state, :action)
type of the returned batches.
"""
const SA = (:state, :action)