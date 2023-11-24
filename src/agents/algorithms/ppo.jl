# """Proximal policy optimisation with dual value networks."""

# """Parameters for PPO agent with DNA.

# Fields:
#   * action_scale: The scale to match the action space of the continuous
#         environment.
#   * action_bias: The bias to match the action space of the continous
#         environment.
#   * capacity: Number of transitions stored in memory (default: 512).
#   * γ: Discount parameter (default: 0.999).
#   * λ: Parameter for GAEs (default: [0.97, 0.97]).
#   * hiddens: Dimensions of hidden layers (default: [128, 128]).
#   * logσ²ₘᵢₙ: Minium log standard deviation for stable training (default: -10).
#   * logσ²ₘₐₓ: Maximum log standard deviation for stable training (default: 3).
#   * minibatch_sizes : Minibatch size for policy, value and distillation phases
#         (default: [128, 64, 64]).
#   * training_steps: Number of training steps per epoch for the network
#         parameters (default: [2, 2, 2]).
#   * clips: Global gradient clippings for phases (default: [5.0, 5.0, 5.0]).
#   * decay: Weight decays for phases(default: [1e-4, 1e-4, 1e-4]).
#   * η: Learning rates for phases (default: [3e-4, 3e-4, 3e-4]).
#   * 𝜀: PPO clip parameter (default: 0.2).
#   * α: Entropy bonus weight (default: 0.0).
#   * β: KL divergence weight (default: 1.0).
#   * warmup_normalisation_episodes: Number of initial episodes for observation
#         normalisation (default: 50).
#   * epochs: Number of training epochs (default: 10000).
# """
# @kwdef struct PPOParameters <: AgentParameters
#     action_scale::Vector{Float32}
#     action_bias::Vector{Float32}
#     capacity::Int = 512
#     γ::Float32 = 0.99
#     λ::Vector{Float32} = [0.97, 0.97]
#     hiddens::Vector{Int} = [128, 128]
#     logσ²ₘᵢₙ::Float32 = -10
#     logσ²ₘₐₓ::Float32 = 3
#     minibatch_sizes::Vector{Int} = [128, 64, 64]
#     training_steps::Vector{Int} = [2, 2, 2]
#     decays::Vector{Float32} = [0.0, 0.0, 0.0]
#     clips::Vector{Float32} = [5.0, 5.0, 5.0]
#     η::Vector{Float32} = [3e-4, 3e-4, 3e-4]
#     𝜀::Float32 = 0.2
#     α::Float32 = 0.0
#     β::Float32 = 1.0
#     warmup_normalisation_episodes::Int = 50
#     epochs::Int = 1000
# end

# """PPO Agent.

# Struct of a PPO agent with options for PPO loss (arXiv: 1707.06347) with either
# the standard actor-critic shared network architecture or a dual network
# architecture (DNA) (arXiv: 2206.10027).

# Args:
#   * env: The environment that the agent learns.
#   * kwargs: Keyword arguments for agent parameters, activation and
#         initialisation functions, and rngs.

# Fields:
#   * params: Hyper parameters for the agent.
#   * rng: Agent RNG.
#   * memory: Replay buffer with a history of transitions.
#   * logπᵒˡᵈ: Old agent policy / policy parameters.
#   * networks: Neural networks.
#   * opt_states: Neural networks optimiser states.
# """
# struct PPOAgent{
#     R <: AbstractRNG,
#     M <: CircularReplayBuffer,
#     # 𝒟 <: Union{FluxCPUDevice, FluxCUDADevice},
#     𝒩 <: DualNetworksArchitecture,
#     𝒪 <: AbstractVector,
# } <: Agent{R, M, 𝒩, 𝒪}
#     params::PPOParameters
#     rng::R
#     memory::M
#     logπᵒˡᵈ::Matrix{Float32}
#     # device::𝒟
#     networks::𝒩
#     opt_states::𝒪
# end

# function PPOAgent(
#     env::QuantumControlEnvironment;
#     # device::Union{FluxCPUDevice, FluxCUDADevice} = get_device(),
#     activation::Function = relu,
#     init::Function = glorot_uniform,
#     rng::AbstractRNG = default_rng(),
#     kwargs...,
# )
#     dₒ = length(env.observation_space)
#     dₐ = length(env.action_space)
#     continuous = env.action_space isa Vector{ClosedInterval{eltype(env)}}
#     recurrence = dₒ == dₐ + 1
#     if continuous
#         action_scale = (
#             (rightendpoint.(env.action_space) - leftendpoint.(env.action_space))
#             / 2
#         )
#         action_bias = (
#             (rightendpoint.(env.action_space) + leftendpoint.(env.action_space))
#             / 2
#         )
#     else
#         action_scale = zeros(dₐ)
#         action_bias = zeros(dₐ)
#     end

#     params = PPOParameters(
#         ; action_scale=action_scale, action_bias=action_bias, kwargs...
#     )
#     if recurrence
#         if !iszero(mod(params.capacity, env.model_function.tₙ))
#             throw(
#                 ArgumentError(
#                     "If using recurrence, the capacity must be a multiple of"
#                     * " the number of time steps."
#                 )
#             )
#         end
#     end
#     networks = DualNetworksArchitecture(
#         continuous,
#         dₒ,
#         dₐ,
#         params.hiddens,
#         recurrence;
#         activation=activation,
#         init=init,
#         rng=rng,
#     )
#     opt_states = [
#         setup(
#             OptimiserChain(
#                 ClipNorm(params.clips[i]),
#                 AdamW(params.η[i], (0.9f0, 0.999f0), params.decays[i]),
#             ),
#             network,
#         )
#         for (i, network) in [
#             (1, networks.actor_critic),
#             (2, networks.value_layers),
#             (3, networks.actor_critic),
#         ]
#     ]
#     memory = CircularReplayBuffer(
#         continuous,
#         dₒ,
#         dₐ,
#         params.capacity,
#         recurrence,
#         params.hiddens[1],
#     )
#     logπᵒˡᵈ = ones(Float32, dₐ + continuous * dₐ, params.capacity)
#     return PPOAgent(params, rng, memory, logπᵒˡᵈ, networks, opt_states)
# end

# ###################
# # Getting actions #
# ###################
# function get_action(
#     agent::PPOAgent{R, CircularReplayBuffer{Float32, Nothing}},
#     observation::Vector{<:AbstractFloat},
# ) where {R}
#     μ, logσ²ᵤ = get_π̃(agent.networks, f32(observation))
#     action, untransformed_action = _get_and_transform_action(agent, μ, logσ²ᵤ)
#     return convert(typeof(observation), action), untransformed_action
# end

# function get_action(
#     agent::PPOAgent{R, CircularReplayBuffer{Float32, Matrix{Float32}}},
#     cell_state::Matrix{Float32},
#     observation::Vector{<:AbstractFloat},
# ) where {R}
#     cell_state, (μ, logσ²ᵤ) = get_π̃(
#         agent.networks, cell_state, f32(observation)
#     )
#     action, untransformed_action = _get_and_transform_action(agent, μ, logσ²ᵤ)
#     return (
#         cell_state, convert(typeof(observation), action), untransformed_action
#     )
# end

# function get_random_action(
#     agent::PPOAgent,
#     action_space::Vector{<:ClosedInterval{<:AbstractFloat}},
# )
#     return rand.(agent.rng, action_space)
# end

# function _get_and_transform_action(
#     agent::PPOAgent, μ::Vector{Float32}, logσ²ᵤ::Vector{Float32}
# )
#     logσ² = @. (
#         agent.params.logσ²ₘᵢₙ
#         + (tanh(logσ²ᵤ) + 1)
#         * (agent.params.logσ²ₘₐₓ - agent.params.logσ²ₘᵢₙ)
#         / 2
#     )
#     𝒩₀₁ = randn(agent.rng, eltype(μ), length(μ))
#     untransformed_action = @. μ + exp(logσ² / 2) * 𝒩₀₁
#     action = @. (
#         tanh(untransformed_action) * agent.params.action_scale
#         + agent.params.action_bias
#     )
#     return action, untransformed_action
# end

# ######################
# # Evaluation methods #
# ######################
# function evaluation_steps!(
#     agent::PPOAgent{R, CircularReplayBuffer{Float32, Nothing}},
#     env::QuantumControlEnvironment,
# ) where {R}
#     ∑ᵣ = zero(eltype(env))

#     observation = reset!(env)
#     for i in 1:agent.memory.capacity
#         agent.memory.observations[:, i] = observation

#         action, untransformed_action = get_action(agent, observation)
#         agent.memory.actions[:, i] = untransformed_action
#         # agent.memory.actions[:, i] = action

#         observation, done, reward = step!(env, action)
#         agent.memory.rewards[i] = reward[end]
#         # agent.memory.rewards[i] = reward[1]
#         agent.memory.dones[i] = done
#         ∑ᵣ += reward[1]
#         if done
#             observation = reset!(env)
#         end
#     end
#     agent.memory.observations[:, end] = observation
#     return ∑ᵣ
# end

# function evaluation_steps!(
#     agent::PPOAgent{R, CircularReplayBuffer{Float32, Matrix{Float32}}},
#     env::QuantumControlEnvironment,
# ) where{R}
#     ∑ᵣ = zero(eltype(env))

#     observation = reset!(env)
#     cell_state = zeros(Float32, agent.params.hiddens[1], 1)
#     for i in 1:agent.memory.capacity
#         agent.memory.observations[:, i] = observation
#         agent.memory._cell_states[:, i] = cell_state

#         cell_state, action, untransformed_action = get_action(
#             agent, cell_state, observation
#         )
#         agent.memory.actions[:, i] = untransformed_action
#         # agent.memory.actions[:, i] = action

#         observation, done, reward = step!(env, action)
#         agent.memory.rewards[i] = reward[end]
#         # agent.memory.rewards[i] = reward[1]
#         agent.memory.dones[i] = done
#         ∑ᵣ += reward[1]
#         if done
#             observation = reset!(env)
#             cell_state = zeros(Float32, agent.params.hiddens[1], 1)
#         end
#     end
#     agent.memory.observations[:, end] = observation
#     agent.memory._cell_states[:, end] = cell_state
#     return ∑ᵣ
# end

# function _initial_steps(agent::PPOAgent, env::QuantumControlEnvironment)
#     for _ in 1:agent.params.warmup_normalisation_episodes
#         _ = reset!(env)
#         done = false
#         while !done
#             _, done, _ = step!(env, get_random_action(agent, env.action_space))
#         end
#     end
#     return nothing
# end

# ###################
# # Trainer methods #
# ###################
# function trainer_steps!(
#     agent::PPOAgent{R, CircularReplayBuffer{Float32, Nothing}}
# ) where {R}
#     π̃, _, Vᵥ = agent.networks(agent.memory.observations)
#     bootstrap_reward!(agent.memory, Vᵥ[end], agent.params.γ)
#     _update_logπᵒˡᵈ!(agent, π̃)

#     advantages = calculate_advantages(
#         agent.memory, vec(Vᵥ), agent.params.γ, agent.params.λ[1]
#     )
#     targets = calculate_targets(
#         agent.memory, vec(Vᵥ), agent.params.γ, agent.params.λ[2]
#     )
#     # targets = calculate_returns(agent.memory, agent.params.γ)

#     losses = zeros(Float32, 6)
#     losses[1:3] = _update_agent_policy!(agent, advantages)
#     losses[4] = _update_agent_value!(agent, targets)
#     losses[5:6] = _update_agent_distillation!(agent)
#     return losses
# end

# function trainer_steps!(
#     agent::PPOAgent{R, CircularReplayBuffer{Float32, Matrix{Float32}}}
# ) where {R}
#     π̃, _, Vᵥ = agent.networks(
#         agent.memory._cell_states, agent.memory.observations
#     )
#     bootstrap_reward!(agent.memory, Vᵥ[end], agent.params.γ)
#     _update_logπᵒˡᵈ!(agent, π̃)

#     advantages = calculate_advantages(
#         agent.memory, vec(Vᵥ), agent.params.γ, agent.params.λ[1]
#     )
#     targets = calculate_targets(
#         agent.memory, vec(Vᵥ), agent.params.γ, agent.params.λ[2]
#     )
#     # targets = calculate_returns(agent.memory, agent.params.γ)

#     losses = zeros(Float32, 6)
#     losses[1:3] = _update_agent_policy!(agent, advantages)
#     losses[4] = _update_agent_value!(agent, targets)
#     losses[5:6] = _update_agent_distillation!(agent)
#     return losses
# end

# function _update_logπᵒˡᵈ!(
#     agent::PPOAgent{R, CircularReplayBuffer{Float32, C}},
#     π̃::Tuple{Matrix{Float32}, Matrix{Float32}},
# ) where {R, C}
#     half = length(agent.params.action_scale)
#     agent.logπᵒˡᵈ[1:half, :] = π̃[1][:, 1 : end - 1]
#     agent.logπᵒˡᵈ[half + 1 : end, :] = π̃[2][:, 1 : end - 1]
#     return nothing
# end

# ###########################################
# # Continuous non-recurrent update methods #
# ###########################################
# function _update_agent_policy!(
#     agent::PPOAgent{R, CircularReplayBuffer{Float32, Nothing}},
#     advantages::Vector{Float32},
# ) where {R}
#     losses = zeros(Float32, 3, agent.params.training_steps[1])

#     half = length(agent.params.action_scale)
#     for i in 1:agent.params.training_steps[1]
#         for (batchᵒ, batchᵒˡᵈ, batchᵃ, batchᴬ) in DataLoader(
#             (
#                 agent.memory.observations[:, 1 : end - 1],
#                 agent.logπᵒˡᵈ,
#                 agent.memory.actions,
#                 advantages,
#             );
#             batchsize=agent.params.minibatch_sizes[1],
#         )
#             μᵒˡᵈ = batchᵒˡᵈ[1:half, :]
#             logσ²ᵒˡᵈ = @. (
#                 agent.params.logσ²ₘᵢₙ
#                 + (tanh(batchᵒˡᵈ[half + 1 : end, :]) + 1)
#                 * (agent.params.logσ²ₘₐₓ - agent.params.logσ²ₘᵢₙ)
#                 / 2
#             )
#             logπᵒˡᵈₐ = vec(
#                 sum(
#                     @. -(logσ²ᵒˡᵈ + (batchᵃ - μᵒˡᵈ) ^ 2 / exp(logσ²ᵒˡᵈ)) / 2;
#                     dims=1,
#                 )
#             )
#             ∇ = gradient(agent.networks.actor_critic) do m
#                 μ, logσ²ᵤ = get_π̃(m, batchᵒ)
#                 logσ² = @. (
#                     agent.params.logσ²ₘᵢₙ
#                     + (tanh(logσ²ᵤ) + 1)
#                     * (agent.params.logσ²ₘₐₓ - agent.params.logσ²ₘᵢₙ)
#                     / 2
#                 )
#                 logπᶿₐ = vec(
#                     sum(@. -(logσ² + (batchᵃ - μ) ^ 2 / exp(logσ²)) / 2; dims=1)
#                 )
#                 ratio = @. exp(logπᶿₐ - logπᵒˡᵈₐ)
#                 L = batchᴬ .* ratio
#                 g = @. (
#                     batchᴬ
#                     * clamp(ratio, 1 - agent.params.𝜀, 1 + agent.params.𝜀)
#                 )
#                 policy_loss = -mean(min.(L, g))
#                 entropy = mean(vec(sum(logσ² ./ 2; dims=1)))
#                 entropy_bonus = -agent.params.α * entropy
#                 ignore() do
#                     losses[1, i] += policy_loss
#                     losses[2, i] += entropy
#                     losses[3, i] += entropy_bonus
#                 end
#                 return policy_loss + entropy_bonus
#             end
#             update!(agent.opt_states[1], agent.networks.actor_critic, ∇[1])
#         end
#         losses[:, i] ./= agent.params.capacity ÷ agent.params.minibatch_sizes[1]
#     end
#     return vec(mean(losses; dims=2))
# end

# function _update_agent_value!(
#     agent::PPOAgent{R, CircularReplayBuffer{Float32, Nothing}},
#     targets::Vector{Float32},
# ) where {R}
#     loss = zero(Float32)

#     for _ in 1:agent.params.training_steps[2]
#         for (batchᵒ, batchᵀ) in DataLoader(
#             (agent.memory.observations[:, 1 : end - 1], targets);
#             batchsize=agent.params.minibatch_sizes[2],
#         )
#             l, ∇ = withgradient(agent.networks.value_layers) do m
#                 Vᵠ = vec(m(batchᵒ))
#                 return mse(Vᵠ, batchᵀ)
#             end
#             loss += l
#             update!(agent.opt_states[2], agent.networks.value_layers, ∇[1])
#         end
#         loss /= agent.params.capacity ÷ agent.params.minibatch_sizes[2]
#     end
#     return loss / agent.params.training_steps[2]
# end

# function _update_agent_distillation!(
#     agent::PPOAgent{R, CircularReplayBuffer{Float32, Nothing}}
# ) where {R}
#     losses = zeros(Float32, 2, agent.params.training_steps[3])

#     π̃ = get_π̃(agent.networks.actor_critic, agent.memory.observations)
#     _update_logπᵒˡᵈ!(agent, π̃)

#     half = length(agent.params.action_scale)
#     for i in 1:agent.params.training_steps[3]
#         for (batchᵒ, μᵒˡᵈ, logσᵤ²ᵒˡᵈ) in DataLoader(
#             (
#                 agent.memory.observations[:, 1 : end - 1],
#                 agent.logπᵒˡᵈ[1:half, :],
#                 agent.logπᵒˡᵈ[half + 1 : end, :],
#             );
#             batchsize=agent.params.minibatch_sizes[3],
#         )
#             Vᵠ = agent.networks.value_layers(batchᵒ)
#             logσ²ᵒˡᵈ = @. (
#                 agent.params.logσ²ₘᵢₙ
#                 + (tanh(logσᵤ²ᵒˡᵈ) + 1)
#                 * (agent.params.logσ²ₘₐₓ - agent.params.logσ²ₘᵢₙ)
#                 / 2
#             )
#             ∇ = gradient(agent.networks.actor_critic) do m
#                 (μ, logσᵤ²), Vᶿ = m(batchᵒ)
#                 logσ² = @. (
#                     agent.params.logσ²ₘᵢₙ
#                     + (tanh(logσᵤ²) + 1)
#                     * (agent.params.logσ²ₘₐₓ - agent.params.logσ²ₘᵢₙ)
#                     / 2
#                 )
#                 t₁ = @. (μᵒˡᵈ - μ) ^ 2 / exp(logσ²)
#                 t₂ = @. exp(logσ²ᵒˡᵈ - logσ²)
#                 t₃ = @. (logσ² - logσ²ᵒˡᵈ)
#                 kl_divergence = agent.params.β * mean(t₁ + t₂ + t₃) / 2
#                 # kl_divergence = agent.params.β * mse(μ, μᵒˡᵈ)
#                 value_distillation = mse(Vᶿ, Vᵠ)
#                 ignore() do
#                     losses[1, i] += value_distillation
#                     losses[2, i] += kl_divergence
#                 end
#                 return value_distillation + kl_divergence
#             end
#             update!(agent.opt_states[3], agent.networks.actor_critic, ∇[1])
#         end
#         losses[:, i] ./= agent.params.capacity ÷ agent.params.minibatch_sizes[3]
#     end
#     return vec(mean(losses; dims=2))
# end

# #######################################
# # Continuous recurrent update methods #
# #######################################
# function _update_agent_policy!(
#     agent::PPOAgent{R, CircularReplayBuffer{Float32, Matrix{Float32}}},
#     advantages::Vector{Float32},
# ) where {R}
#     losses = zeros(Float32, 3, agent.params.training_steps[1])

#     half = length(agent.params.action_scale)
#     for i in 1:agent.params.training_steps[1]
#         for (batchᵒ, batchᶜ, batchᵒˡᵈ, batchᵃ, batchᴬ) in DataLoader(
#             (
#                 agent.memory.observations[:, 1 : end - 1],
#                 agent.memory._cell_states[:, 1 : end - 1],
#                 agent.logπᵒˡᵈ,
#                 agent.memory.actions,
#                 advantages,
#             );
#             batchsize=agent.params.minibatch_sizes[1],
#         )
#             μᵒˡᵈ = batchᵒˡᵈ[1:half, :]
#             logσ²ᵒˡᵈ = @. (
#                 agent.params.logσ²ₘᵢₙ
#                 + (tanh(batchᵒˡᵈ[half + 1 : end, :]) + 1)
#                 * (agent.params.logσ²ₘₐₓ - agent.params.logσ²ₘᵢₙ)
#                 / 2
#             )
#             logπᵒˡᵈₐ = vec(
#                 sum(
#                     @. -(logσ²ᵒˡᵈ + (batchᵃ - μᵒˡᵈ) ^ 2 / exp(logσ²ᵒˡᵈ)) / 2;
#                     dims=1,
#                 )
#             )
#             # 𝐨, 𝐡 = (
#             #     reshape(batchᵒ, size(batchᵒ, 1), 38, 5),
#             #     reshape(batchᶜ, size(batchᶜ, 1), 38, 5),
#             # )
#             𝐨 = reshape(batchᵒ, size(batchᵒ, 1), 38, 5)
#             ∇ = gradient(agent.networks.actor_critic) do m
#                 _, (μ, logσ²ᵤ) = get_π̃(m, batchᶜ, batchᵒ)
#                 # _, (μ, logσ²ᵤ) = get_π̃(m, 𝐡, 𝐨)
#                 logσ² = @. (
#                     agent.params.logσ²ₘᵢₙ
#                     + (tanh(logσ²ᵤ) + 1)
#                     * (agent.params.logσ²ₘₐₓ - agent.params.logσ²ₘᵢₙ)
#                     / 2
#                 )
#                 logπᶿₐ = vec(
#                     sum(@. -(logσ² + (batchᵃ - μ) ^ 2 / exp(logσ²)) / 2; dims=1)
#                 )
#                 ratio = @. exp(logπᶿₐ - logπᵒˡᵈₐ)
#                 L = batchᴬ .* ratio
#                 g = @. (
#                     batchᴬ
#                     * clamp(ratio, 1 - agent.params.𝜀, 1 + agent.params.𝜀)
#                 )
#                 policy_loss = -mean(min.(L, g))
#                 entropy = mean(vec(sum(logσ² ./ 2; dims=1)))
#                 entropy_bonus = -agent.params.α * entropy
#                 ignore() do
#                     losses[1, i] += policy_loss
#                     losses[2, i] += entropy
#                     losses[3, i] += entropy_bonus
#                 end
#                 return policy_loss + entropy_bonus
#             end
#             update!(agent.opt_states[1], agent.networks.actor_critic, ∇[1])
#         end
#         losses[:, i] ./= agent.params.capacity ÷ agent.params.minibatch_sizes[1]
#     end
#     return vec(mean(losses; dims=2))
# end

# function _update_agent_value!(
#     agent::PPOAgent{R, CircularReplayBuffer{Float32, Matrix{Float32}}},
#     targets::Vector{Float32},
# ) where {R}
#     loss = zero(Float32)

#     for _ in 1:agent.params.training_steps[2]
#         for (batchᵒ, batchᶜ, batchᵀ) in DataLoader(
#             (
#                 agent.memory.observations[:, 1 : end - 1],
#                 agent.memory._cell_states[:, 1 : end - 1],
#                 targets,
#             );
#             batchsize=agent.params.minibatch_sizes[2],
#         )
#             l, ∇ = withgradient(agent.networks.value_layers) do m
#                 Vᵠ = vec(get_Vᵥ(m, batchᶜ, batchᵒ))
#                 return mse(Vᵠ, batchᵀ)
#             end
#             loss += l
#             update!(agent.opt_states[2], agent.networks.value_layers, ∇[1])
#         end
#         loss /= agent.params.capacity ÷ agent.params.minibatch_sizes[2]
#     end
#     return loss / agent.params.training_steps[2]
# end

# function _update_agent_distillation!(
#     agent::PPOAgent{R, CircularReplayBuffer{Float32, Matrix{Float32}}}
# ) where {R}
#     losses = zeros(Float32, 2, agent.params.training_steps[3])

#     _, π̃ = get_π̃(
#         agent.networks.actor_critic,
#         agent.memory._cell_states,
#         agent.memory.observations,
#     )
#     _update_logπᵒˡᵈ!(agent, π̃)

#     half = length(agent.params.action_scale)
#     for i in 1:agent.params.training_steps[3]
#         for (batchᵒ, batchᶜ, μᵒˡᵈ, logσᵤ²ᵒˡᵈ) in DataLoader(
#             (
#                 agent.memory.observations[:, 1 : end - 1],
#                 agent.memory._cell_states[:, 1 : end - 1],
#                 agent.logπᵒˡᵈ[1:half, :],
#                 agent.logπᵒˡᵈ[half + 1 : end, :],
#             );
#             batchsize=agent.params.minibatch_sizes[3],
#         )
#             Vᵠ = get_Vᵥ(agent.networks.value_layers, batchᶜ, batchᵒ)
#             logσ²ᵒˡᵈ = @. (
#                 agent.params.logσ²ₘᵢₙ
#                 + (tanh(logσᵤ²ᵒˡᵈ) + 1)
#                 * (agent.params.logσ²ₘₐₓ - agent.params.logσ²ₘᵢₙ)
#                 / 2
#             )
#             ∇ = gradient(agent.networks.actor_critic) do m
#                 (μ, logσᵤ²), Vᶿ = m(batchᶜ, batchᵒ)
#                 logσ² = @. (
#                     agent.params.logσ²ₘᵢₙ
#                     + (tanh(logσᵤ²) + 1)
#                     * (agent.params.logσ²ₘₐₓ - agent.params.logσ²ₘᵢₙ)
#                     / 2
#                 )
#                 t₁ = @. (μᵒˡᵈ - μ) ^ 2 / exp(logσ²)
#                 t₂ = @. exp(logσ²ᵒˡᵈ - logσ²)
#                 t₃ = @. (logσ² - logσ²ᵒˡᵈ)
#                 kl_divergence = agent.params.β * mean(t₁ + t₂ + t₃) / 2
#                 # kl_divergence = agent.params.β * mse(μ, μᵒˡᵈ)
#                 value_distillation = mse(Vᶿ, Vᵠ)
#                 ignore() do
#                     losses[1, i] += value_distillation
#                     losses[2, i] += kl_divergence
#                 end
#                 return value_distillation + kl_divergence
#             end
#             update!(agent.opt_states[3], agent.networks.actor_critic, ∇[1])
#         end
#         losses[:, i] ./= agent.params.capacity ÷ agent.params.minibatch_sizes[3]
#     end
#     return vec(mean(losses; dims=2))
# end

# #####################
# # Discrete  methods #
# #####################
# # function get_action(
# #     agent::PPOAgent{R, CircularReplayBuffer{Int, Nothing}},
# #     observation::Vector{<:AbstractFloat},
# #     time_step::Int,
# # ) where {R}
# #     logits = get_π̃(agent.networks, f32(observation))
# #     action = wsample(agent.rng, softmax(logits))
# #     agent.memory.actions[:, time_step] = action
# #     return action
# # end

# # function get_action(
# #     agent::PPOAgent{R, CircularReplayBuffer{Int, Nothing}},
# #     cell_state::Matrix{Float32},
# #     observation::Vector{Float32},
# # ) where {R}
# #     logits = get_π̃(agent.networks, observation)
# #     action = wsample(agent.rng, softmax(logits))
# #     return action
# # end

# # function get_random_action(
# #     agent::PPOAgent{R, CircularReplayBuffer{Int}}, action_space::OneTo{Int}
# # ) where {R}
# #     return rand(agent.rng, action_space)
# # end

# # function _update_logπᵒˡᵈ!(
# #     agent::PPOAgent{T, R, PT, CircularReplayBuffer{T, Int}}, π̃::Matrix{T}
# # ) where {T, R, PT}
# #     agent.logπᵒˡᵈ .= logsoftmax(π̃[:, 1 : end - 1]; dims=1)
# #     return nothing
# # end

# # function _update_agent_policy!(
# #     agent::PPOAgent{T, PT, CircularReplayBuffer{T, Int}}, advantages::Matrix{T}
# # ) where {T, PT}
# #     losses = zeros(T, 3, agent.params.training_steps[1])

# #     for i in 1:agent.params.training_steps[1]
# #         for (batchᵒ, batchᵒˡᵈ, batchᵃ, batchᴬ) in DataLoader(
# #             (
# #                 agent.memory.observations[:, 1 : end - 1],
# #                 agent.logπᵒˡᵈ,
# #                 agent.memory.actions,
# #                 advantages,
# #             );
# #             batchsize=agent.params.minibatch_sizes[1],
# #         )
# #             logπᵒˡᵈₐ = [
# #                 batchᵒˡᵈ[batchᵃ[i], i]
# #                 for i in 1:agent.params.minibatch_sizes[1]
# #             ]
# #             ∇ = gradient(agent.networks.actor_critic) do m
# #                 logits = get_π̃(m, batchᵒ)
# #                 πᶿ = softmax(logits; dims=1)
# #                 logπᶿ = logsoftmax(logits; dims=1)
# #                 logπᶿₐ = [
# #                     logπᶿ[batchᵃ[i], i]
# #                     for i in 1:agent.params.minibatch_sizes[1]
# #                 ]
# #                 ratio = @. exp(logπᶿₐ - logπᵒˡᵈₐ)
# #                 L = vec(batchᴬ) .* ratio
# #                 g = @. (
# #                     batchᴬ * clamp(ratio, 1 - agent.params.𝜀, 1 + agent.params.𝜀)
# #                 )
# #                 policy_loss = -mean(min.(L, g))
# #                 entropy = -mean(sum(πᶿ .* logπᶿ; dims=1))
# #                 entropy_bonus = -agent.params.α * entropy
# #                 ignore() do
# #                     losses[1, i] += policy_loss
# #                     losses[2, i] += entropy
# #                     losses[3, i] += entropy_bonus
# #                 end
# #                 return policy_loss + entropy_bonus
# #             end
# #             update!(agent.opt_states[1], agent.networks.actor_critic, ∇[1])
# #         end
# #         losses[:, i] ./= agent.params.capacity ÷ agent.params.minibatch_sizes[1]
# #     end
# #     return vec(mean(losses; dims=2))
# # end

# # function _update_agent_distillation!(
# #     agent::PPOAgent{T, PT, R, CircularReplayBuffer{T, Int}}
# # ) where {T, R, PT}
# #     losses = zeros(T, 2, agent.params.training_steps[3])

# #     π̃ = get_π̃(agent.networks, agent.memory.observations)
# #     _update_logπᵒˡᵈ!(agent, π̃)
# #     for i in 1:agent.params.training_steps[3]
# #         for (batchᵒ, batchᵒˡᵈ) in DataLoader(
# #             (agent.memory.observations[:, 1 : end - 1], agent.logπᵒˡᵈ);
# #             batchsize=agent.params.minibatch_sizes[3],
# #         )
# #             valuesᵥ = agent.networks.value_layers(batchᵒ)
# #             ∇ = gradient(agent.networks.actor_critic) do m
# #                 logits, valuesₚ = m(batchᵒ)
# #                 logπᶿ = logsoftmax(logits; dims=1)
# #                 kl_divergence = agent.params.β * mean(
# #                     sum(@. exp(batchᵒˡᵈ) * (batchᵒˡᵈ - logπᶿ); dims=1)
# #                 )
# #                 value_distillation = mse(valuesₚ, valuesᵥ)
# #                 ignore() do
# #                     losses[1, i] += value_distillation
# #                     losses[2, i] += kl_divergence
# #                 end
# #                 return value_distillation + kl_divergence
# #             end
# #             update!(agent.opt_states[3], agent.networks.actor_critic, ∇[1])
# #         end
# #         losses[:, i] ./= agent.params.capacity ÷ agent.params.minibatch_sizes[3]
# #     end
# #     return vec(mean(losses; dims=2))
# # end

# ##################
# # Agent learning #
# ##################
# function learn!(agent::PPOAgent, env::QuantumControlEnvironment)
#     rewards = zeros(eltype(env), agent.params.epochs)
#     dones = zeros(Int, agent.params.epochs)
#     losses = zeros(Float32, 6, agent.params.epochs)

#     _initial_steps(agent, env)

#     for epoch in 1:agent.params.epochs
#         epochᵣ = evaluation_steps!(agent, env)
#         epochₗ = trainer_steps!(agent)

#         rewards[epoch] = epochᵣ
#         dones[epoch] = sum(agent.memory.dones)
#         losses[:, epoch] = epochₗ

#         println(
#             "Epoch: ",
#             epoch,
#             "| Terminations: ",
#             dones[epoch],
#             "| Average Episodic Reward: ",
#             rewards[epoch] / dones[epoch],
#         )
#     end
#     return rewards ./ dones, losses
# end
