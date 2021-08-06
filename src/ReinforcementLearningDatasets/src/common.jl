export SARTS
export SART
export RLDataSet

abstract type RLDataSet end

const SARTS = (:state, :action, :reward, :terminal, :next_state)
const SART = (:state, :action, :reward, :terminal)