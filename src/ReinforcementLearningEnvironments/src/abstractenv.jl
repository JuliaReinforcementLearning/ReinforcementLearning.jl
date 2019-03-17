export AbstractEnv, observe, reset!, interact!, action_space, observation_space, render

abstract type AbstractEnv end

function observe end
function reset! end
function interact! end
function action_space end
function observation_space end
function render end