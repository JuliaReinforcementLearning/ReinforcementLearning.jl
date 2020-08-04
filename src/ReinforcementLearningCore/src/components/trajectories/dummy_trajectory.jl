export DummyTrajectory

struct DummyTrajectory <: AbstractTrajectory{(), Tuple{}} end

Base.length(t::DummyTrajectory) = 0