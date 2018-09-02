export AbstractEnv
abstract type AbstractEnv end

"Get current state of an environment"
function getstate end

"Reset an environment to the initial state"
function reset! end

"Take an action in an environment and return the current state"
function interact! end

"Get the action space of an environment"
function actionspace end

"Plot the current state of an environment"
function plotenv end