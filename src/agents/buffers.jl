# """Contains memory buffers to use for reinforcement learning agents."""

# abstract type Memory{A <: Union{Float32, Int}} end


# """Circular Replay Buffer for on-policy learning.

# Creates a struct with representing a simple circular buffer used in on-policy
# reinforcement learning.

# Args:
#   * continuous: Indicates whether we have continuous or discrete actions.
#   * dₒ: Dimension of the observation space.
#   * dₐ: Dimension of the action space.
#   * capacity: Number of stored transitions.
#   * recurrence: Whether to add recurrence stats to the memory (default: false).
#   * dₕ: If using recurrence, the dimensionality (default: nothing).

# Fields:
#   * observations: A history of observations.
#   * actions: A history of actions.
#   * rewards: A history of rewards.
#   * dones: A history of terminations.
#   * capacity: Number of stored transitions.
# """
# struct CircularReplayBuffer{
#     A <: Union{Float32, Int},
#     C <: Union{Nothing, Matrix{Float32}},
# } <: Memory{A}
#     observations::Matrix{Float32}
#     actions::Matrix{A}
#     rewards::Vector{Float32}
#     dones::Vector{Bool}
#     capacity::Int
#     _cell_states::C
# end

# function CircularReplayBuffer(
#     continuous::Bool,
#     dₒ::Int,
#     dₐ::Int,
#     capacity::Int,
#     recurrence::Bool = false,
#     dₕ::Union{Nothing, Int} = nothing,
# )
#     if recurrence && isnothing(dₕ)
#         throw(
#             ArgumentError(
#                 "Cannot have recurrence with empty hidden dimensionality."
#             )
#         )
#     end

#     observations = zeros(Float32, dₒ, capacity + 1)
#     if continuous
#         actions = zeros(Float32, dₐ, capacity)
#     else
#         actions = zeros(Int, 1, capacity)
#     end
#     rewards = zeros(Float32, capacity)
#     dones = zeros(Bool, capacity)
#     if recurrence
#         _cell_states = zeros(Float32, dₕ, capacity + 1)
#     else
#         _cell_states = nothing
#     end
#     return CircularReplayBuffer(
#         observations, actions, rewards, dones, capacity, _cell_states
#     )
# end

# function bootstrap_reward!(c::CircularReplayBuffer, value::Float32, γ::Float32)
#     c.rewards[end] += γ * (1 - c.dones[end]) * value
#     return nothing
# end

# function calculate_returns(c::CircularReplayBuffer, γ::Float32)
#     returns = zero(c.rewards)
#     returns[end] = c.rewards[end]
#     for i in c.capacity - 1 : -1 : 1
#         returns[i] = c.rewards[i] + γ * (1 - c.dones[i]) * returns[i + 1]
#     end
#     return returns
# end

# function calculate_advantages(
#     c::CircularReplayBuffer, values::Vector{Float32}, γ::Float32, λ::Float32
# )
#     δ₁ = @. (
#         c.rewards + γ * (1 - c.dones) * values[2:end] - values[1 : end - 1]
#     )
#     advantages = zero(c.rewards)
#     advantages[end] = δ₁[end]
#     for i in c.capacity - 1 : -1 : 1
#         advantages[i] = δ₁[i] + γ * λ * (1 - c.dones[i]) * advantages[i + 1]
#     end
#     return advantages
# end

# function calculate_targets(
#     c::CircularReplayBuffer, values::Vector{Float32}, γ::Float32, λ::Float32
# )
#     δ₁ = @. (
#         c.rewards + γ * (1 - c.dones) * values[2:end] - values[1 : end - 1]
#     )
#     advantages = zero(c.rewards)
#     advantages[end] = δ₁[end]
#     for i in c.capacity - 1 : -1 : 1
#         advantages[i] = δ₁[i] + γ * λ * (1 - c.dones[i]) * advantages[i + 1]
#     end
#     return advantages .+ values[1 : end - 1]
# end


# """Replay buffer for off-policy learning.

# Creates a struct with representing a simple buffer used in off-policy
# reinforcement learning.

# Args:
#   * continuous: Indicates whether we have continuous or discrete actions.
#   * dₒ: Dimension of the observation space.
#   * dₐ: Dimension of the action space.
#   * capacity: Total number of stored transitions.

# Fields:
#   * observations: A history of observations at time t.
#   * actions: A history of actions.
#   * rewards: A history of rewards.
#   * dones: A history of terminations.
#   * observations′: A history of observations at time t + 1.
#   * capacity: Total number of stored transitions.
# """
# struct PrioritizedReplayBuffer{A <: Union{Float32, Int}} <: Memory{A}
#     observations::Matrix{Float32}
#     actions::Matrix{A}
#     rewards::Vector{Float32}
#     dones::Vector{Bool}
#     observations′::Matrix{Float32}
#     capacity::Int
#     _count::RefValue{Int}
# end

# function PrioritizedReplayBuffer(
#     continuous::Bool, dₒ::Int, dₐ::Int, capacity::Int
# )
#     observations = zeros(Float32, dₒ, capacity)
#     if continuous
#         actions = zeros(Float32, dₐ, capacity)
#     else
#         actions = zeros(Int, 1, capacity)
#     end
#     rewards = zeros(Float32, capacity)
#     dones = zeros(Bool, capacity)
#     return PrioritizedReplayBuffer(
#         observations,
#         actions,
#         rewards,
#         dones,
#         copy(observations),
#         capacity,
#         RefValue(0),
#     )
# end

# function upto_index(buffer::PrioritizedReplayBuffer)
#     return min(buffer._count[], buffer.capacity)
# end

# function update_and_get_index!(buffer::PrioritizedReplayBuffer)
#     buffer._count[] += 1
#     return mod1(buffer._count[], buffer.capacity)
# end

# function sample_buffer(
#     buffer::PrioritizedReplayBuffer, rng::AbstractRNG, minibatch_size::Int
# )
#     upto = upto_index(buffer)
#     indices = sample(rng, OneTo(upto), minibatch_size; replace=false)
#     return (
#         buffer.observations[:, indices],
#         buffer.actions[:, indices],
#         buffer.rewards[indices],
#         buffer.dones[indices],
#         buffer.observations′[:, indices],
#     )
# end
