abstract type Memory{A <: Union{Float32, Int}} end


struct ReplayBuffer{A <: Union{Float32, Int}} <: Memory{A}
    observations_t::Matrix{Float32}
    actions::Matrix{A}
    rewards::Vector{Float32}
    dones::Vector{Bool}
    observations_tp1::Matrix{Float32}
    capacity::Int
    _count::Base.RefValue{Int}
end

"""
    ReplayBuffer(
        continuous::Bool, observation_dim::Int, action_dim::Int, capacity::Int
    )

Simple replay buffer used in off-policy reinforcement learning.

Args:
  * `continuous`: Indicates whether we have continuous or discrete actions.
  * `observation_dim`: Dimension of the observation space.
  * `action_dim`: Dimension of the action space.
  * `capacity`: Total number of stored transitions.

Fields:
  * `observations_t`: A history of observations at time t.
  * `actions`: A history of actions.
  * `rewards`: A history of rewards.
  * `dones`: A history of terminations.
  * `observations_tp1`: A history of observations at time t + 1.
  * `capacity`: Total number of stored transitions.
"""
function ReplayBuffer(
    continuous::Bool, observation_dim::Int, action_dim::Int, capacity::Int
)
    observations_t = zeros(Float32, observation_dim, capacity)
    if continuous
        actions = zeros(Float32, action_dim, capacity)
    else
        actions = zeros(Int, 1, capacity)
    end
    rewards = zeros(Float32, capacity)
    dones = zeros(Bool, capacity)
    return ReplayBuffer(
        observations_t,
        actions,
        rewards,
        dones,
        copy(observations_t),
        capacity,
        Base.RefValue(0),
    )
end

get_index(b::ReplayBuffer) = mod1(b._count[], b.capacity)
upto_index(b::ReplayBuffer) = min(b._count[], b.capacity)

function update_and_get_index!(b::ReplayBuffer)
    b._count[] += 1
    return mod1(b._count[], b.capacity)
end

function sample_buffer(
    buffer::ReplayBuffer, mini_batch_size::Int, rng::AbstractRNG = default_rng()
)
    upto = upto_index(buffer)
    indices = sample(rng, Base.OneTo(upto), mini_batch_size; replace=false)
    return (
        buffer.observations_t[:, indices],
        buffer.actions[:, indices],
        buffer.rewards[indices],
        buffer.dones[indices],
        buffer.observations_tp1[:, indices],
    )
end
