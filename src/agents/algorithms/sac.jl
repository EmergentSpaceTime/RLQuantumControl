# """Soft actor-critic with options for using dropout Q-networks and truncated
# quantile critics.
# """

# """Parameters for the SAC algorithm with options of using DroQ or TQC.

# Fields:
#   * action_scale: The scale to match the action space of the continuous
#         environment.
#   * action_bias: The bias to match the action space of the continous
#         environment.
#   * H̄: Target entropy, usually equals to -dim𝒜.
#   * capacity: Number of transitions stored in memory (default: 100000).
#   * hiddens: Dimensions of hidden layers (default: [256, 256]).
#   * logσ²ₘᵢₙ: Minium log standard deviation for stable training (default: -10).
#   * logσ²ₘₐₓ: Maximum log standard deviation for stable training (default: 3).
#   * use_tqc: Whether using a distribution of Q-networks (default: true).
#   * qₙ: Number of output q-values for TQC (default: 25).
#   * qₖ: Bottom number of quintiles to keep for TQC (default: 21).
#   * dropout: Dropout probability for dropout Q-networks (default: 0.01).
#   * layer_norm: Whether to use layer normalisation after each layer for
#         Q-networks (default: true).
#   * γ: Discount parameter for extrinsic/intrinsic rewards (default: 0.99).
#   * minibatch_size: Minibatch size for policy and Q-networks (default: 256).
#   * training_steps: Number of training steps per episode (default: 20).
#   * clips: Global gradient clippings (default: [5.0, 5.0, 5.0, 5.0]).
#   * decays: Weight decays (default: [0.0, 0.0, 0.0, 0.0]).
#   * η: Learning rates (default: [3e-4, 3e-4, 3e-4, 3e-4]).
#   * ρ: Polyak update parameter (default: 0.005).
#   * warmup_normalisation_episodes: Number of initial episodes for observation
#         normalisation (default: 50).
#   * warmup_evaluation_episodes: Number of initial episodes to populate the
#         buffer (default: 50).
#   * episodes: Number of total episodes (default: 5000).
# """
# @kwdef struct SACParameters
#     action_scale::Vector{Float32}
#     action_bias::Vector{Float32}
#     H̄::Float32
#     capacity::Int = 100000
#     hiddens::Vector{Int} = [256, 256]
#     logσ²ₘᵢₙ::Float32 = -10
#     logσ²ₘₐₓ::Float32 = 3
#     use_tqc::Bool = true
#     qₙ::Int = 25
#     qₖ::Int = 46
#     dropout::Float32 = 0.01
#     layer_norm::Bool = true
#     γ::Float32 = 0.99
#     minibatch_size::Int = 256
#     training_steps::Int = 20
#     decays::Vector{Float32} = [0.0, 0.0, 0.0, 0.0]
#     clips::Vector{Float32} = [5.0, 5.0, 5.0, 5.0]
#     η::Vector{Float32} = [3e-4, 3e-4, 3e-4, 3e-4]
#     ρ::Float32 = 0.005
#     warmup_normalisation_episodes::Int = 50
#     warmup_evaluation_episodes::Int = 50
#     episodes::Int = 1000
# end


# """SAC Agent.

# Struct of a agent that uses the Soft Actor Critic algorithm (arXiv: 1801.01290)
# with optional dropout Q-networks (arXiv: 2101.05982) and TQC (arXiv: 2005.04269)
# to improve sample efficiency.

# Args:
#   * env: The environment that the agent learns.
#   * kwargs: Keyword arguments for agent parameters, activation and
#         initialisation functions, and rngs.

# Fields:
#   * params: Hyper parameters for the agent.
#   * rng: Agent RNG.
#   * memory: Replay buffer with a history of transitions.
#   * networks: Neural networks.
#   * device: Device for neural networks.
#   * opt_states: Neural networks opt_states.
# """
# struct SACAgent{
#     R <: AbstractRNG,
#     M <: PrioritizedReplayBuffer,
#     𝒩 <: SACNetworks,
#     𝒪 <: AbstractVector,
# } <: Agent{R, M, 𝒩, 𝒪}
#     params::SACParameters
#     rng::R
#     memory::M
#     networks::𝒩
#     opt_states::𝒪
# end

# function SACAgent(
#     env::QuantumControlEnvironment;
#     activation::Function = relu,
#     init::Function = glorot_uniform,
#     rng::AbstractRNG = default_rng(),
#     kwargs...,
# )
#     dₒ = length(env.observation_space)
#     dₐ = length(env.action_space)
#     continuous = env.action_space isa Vector{ClosedInterval{Float64}}
#     if continuous
#         action_scale = (
#             (rightendpoint.(env.action_space) - leftendpoint.(env.action_space))
#             / 2
#         )
#         action_bias = (
#             (rightendpoint.(env.action_space) + leftendpoint.(env.action_space))
#             / 2
#         )
#         H̄ = -dₐ
#     else
#         action_scale = zeros(dₐ)
#         action_bias = zeros(dₐ)
#         H̄ = 0.98 * log(dₐ)
#     end

#     params = SACParameters(
#         ; action_scale=action_scale, action_bias=action_bias, H̄=H̄, kwargs...
#     )
#     memory = PrioritizedReplayBuffer(continuous, dₒ, dₐ, params.capacity)
#     networks = SACNetworks(
#         continuous,
#         dₒ,
#         dₐ,
#         params.use_tqc ? params.qₙ : 1,
#         params.hiddens,
#         params.dropout,
#         params.layer_norm;
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
#             (1, networks.policy_layers),
#             (2, networks.Q₁_layers),
#             (3, networks.Q₂_layers),
#             (4, networks.logα),
#         ]
#     ]
#     return SACAgent(params, rng, memory, networks, opt_states)
# end

# ###################
# # Getting actions #
# ###################
# function get_action(
#     agent::SACAgent{R, PrioritizedReplayBuffer{Float32}},
#     observation::Vector{Float64},
# ) where {R}
#     μ, logσ²ᵤ = agent.networks.policy_layers(f32(observation))
#     logσ² = @. (
#         agent.params.logσ²ₘᵢₙ
#         + (tanh(logσ²ᵤ) + 1)
#         * (agent.params.logσ²ₘₐₓ - agent.params.logσ²ₘᵢₙ)
#         / 2
#     )
#     𝒩₀₁ = randn(agent.rng, Float32, length(μ))
#     action = @. (
#        tanh(μ + exp(logσ² / 2) * 𝒩₀₁) * agent.params.action_scale
#        + agent.params.action_bias
#     )
#     return convert(Vector{Float64}, action)
# end

# function get_random_action(
#     agent::SACAgent{R, PrioritizedReplayBuffer{Float32}},
#     action_space::Vector{<:ClosedInterval{Float64}},
# ) where {R}
#     return rand.(agent.rng, action_space)
# end

# #############################
# # Shared evaluation methods #
# #############################
# function evaluation_steps!(agent::SACAgent, env::QuantumControlEnvironment)
#     ∑ᵣ = zero(Float64)

#     observation = reset!(env)
#     done = false
#     while !done
#         index = update_and_get_index!(agent.memory)
#         agent.memory.observations[:, index] = observation

#         action = get_action(agent, observation)
#         agent.memory.actions[:, index] = action

#         observation, done, reward = step!(env, action)
#         agent.memory.observations′[:, index] = observation
#         # agent.memory.rewards[index] = reward[end]
#         agent.memory.rewards[index] = reward[1]
#         agent.memory.dones[index] = done
#         ∑ᵣ += reward[1]
#     end
#     return ∑ᵣ
# end

# function _initial_steps!(agent::SACAgent, env::QuantumControlEnvironment)
#     if (
#         (env.observation_function isa NormalisedObservation)
#         | (env.reward_function isa NormalisedReward)
#     )
#         for _ in 1:agent.params.warmup_normalisation_episodes
#             _ = reset!(env)
#             done = false
#             while !done
#                 _, done, _ = step!(
#                     env, get_random_action(agent, env.action_space)
#                 )
#             end
#         end
#     end
#     for _ in 1:agent.params.warmup_evaluation_episodes
#         _ = evaluation_steps!(agent, env)
#     end
#     return nothing
# end

# ##########################
# # Shared trainer methods #
# ##########################
# function trainer_steps!(agent::SACAgent)
#     metrics = zeros(Float32, 7, agent.params.training_steps)
#     for i in 1:agent.params.training_steps
#         if agent.params.use_tqc
#             metrics[:, i] = _update_agent_networks_tqc!(agent)
#         else
#             metrics[:, i] = _update_agent_networks_base!(agent)
#         end
#         _polyak_update!(agent)
#     end
#     return vec(mean(metrics; dims=2))
# end

# function _polyak_update!(agent::SACAgent)
#     for (target, source) in zip(
#         params(agent.networks.Q₁_target_layers),
#         params(agent.networks.Q₁_layers),
#     )
#         @. target = (1 - agent.params.ρ) * target + agent.params.ρ * source
#     end
#     for (target, source) in zip(
#         params(agent.networks.Q₂_target_layers),
#         params(agent.networks.Q₂_layers),
#     )
#         @. target = (1 - agent.params.ρ) * target + agent.params.ρ * source
#     end
#     return nothing
# end

# #########################
# # Unique update methods #
# #########################
# function _update_agent_networks_tqc!(
#     agent::SACAgent{R, PrioritizedReplayBuffer{Float32}}
# ) where {R}
#     losses = zeros(Float32, 7)

#     𝐬, 𝐚, 𝐫ᵥ, 𝐝ᵥ, 𝐬′ = sample_buffer(
#         agent.memory, agent.rng, agent.params.minibatch_size
#     )
#     𝐝 = unsqueeze(𝐝ᵥ; dims=1)
#     𝐫 = unsqueeze(𝐫ᵥ; dims=1)

#     μ′, logσ²′ᵤ = agent.networks.policy_layers(𝐬′)
#     logσ²′ = @. (
#         agent.params.logσ²ₘᵢₙ
#         + (tanh(logσ²′ᵤ) + 1)
#         * (agent.params.logσ²ₘₐₓ - agent.params.logσ²ₘᵢₙ)
#         / 2
#     )
#     u′ = @. μ′ + exp(logσ²′ / 2) * $randn(agent.rng, $eltype(μ′), $size(μ′))
#     v′ = tanh.(u′)
#     𝐚′ = @. v′ * agent.params.action_scale + agent.params.action_bias
#     logπₐ′ = sum(
#         @. (
#             -0.5f0 * (logσ²′ + (u′ - μ′) ^ 2 / exp(logσ²′) + log(2f0π))
#             - log(agent.params.action_scale * (1 - v′ ^ 2) + eps(Float32))
#         );
#         dims=1,
#     )
#     𝐬′𝐚′ = vcat(𝐬′, 𝐚′)
#     𝐬𝐚 = vcat(𝐬, 𝐚)

#     Q₁ᵀ = agent.networks.Q₁_target_layers(𝐬′𝐚′)
#     Q₂ᵀ = agent.networks.Q₂_target_layers(𝐬′𝐚′)
#     Qᵀ = sort(vcat(Q₁ᵀ, Q₂ᵀ); dims=1)[1 : agent.params.qₖ, :]
#     y = unsqueeze(
#         @. (
#             𝐫
#             + agent.params.γ
#             * (1 - 𝐝)
#             * (Qᵀ - exp(agent.networks.logα) * logπₐ′)
#         );
#         dims=2,
#     )
#     cumulative = unsqueeze(
#         unsqueeze(
#             @. ($collect(1:agent.params.qₙ) - 0.5f0) / agent.params.qₙ; dims=2
#         );
#         dims=1,
#     )
#     losses[4], ∇ = withgradient(agent.networks.Q₁_layers) do m
#         Q₁ = unsqueeze(m(𝐬𝐚); dims=1)

#         δ = y .- Q₁
#         δ₊ = abs.(δ)

#         huber_term = @. ifelse(δ₊ > 1, δ₊ - 0.5f0, 0.5f0 * δ ^ 2)
#         loss = @. abs(cumulative - (δ < 0)) * huber_term
#         return mean(loss)
#     end
#     update!(agent.opt_states[2], agent.networks.Q₁_layers, ∇[1])
#     losses[5], ∇ = withgradient(agent.networks.Q₂_layers) do m
#         Q₂ = unsqueeze(m(𝐬𝐚); dims=1)

#         δ = y .- Q₂
#         δ₊ = abs.(δ)

#         huber_term = @. ifelse(δ₊ > 1, δ₊ - 0.5f0, 0.5f0 * δ ^ 2)
#         loss = @. abs(cumulative - (δ < 0)) * huber_term
#         return mean(loss)
#     end
#     update!(agent.opt_states[3], agent.networks.Q₂_layers, ∇[1])

#     losses[1], ∇ = withgradient(agent.networks.policy_layers) do m
#         μ, logσ²ᵤ = m(𝐬)
#         logσ² = @. (
#             agent.params.logσ²ₘᵢₙ
#             + (tanh(logσ²ᵤ) + 1)
#             * (agent.params.logσ²ₘₐₓ - agent.params.logσ²ₘᵢₙ)
#             / 2
#         )
#         u = @. μ + exp(logσ² / 2) * $randn(agent.rng, $eltype(μ), $size(μ))
#         v = tanh.(u)
#         𝐚̃ = @. v * agent.params.action_scale + agent.params.action_bias
#         logπᶿₐ̃ = vec(
#             sum(
#                 @. (
#                     -0.5f0 * (logσ² + (u - μ) ^ 2 / exp(logσ²) + log(2f0π))
#                     - log(
#                         agent.params.action_scale * (1 - v ^ 2) + eps(Float32)
#                     )
#                 );
#                 dims=1,
#             )
#         )
#         𝐬𝐚̃ = vcat(𝐬, 𝐚̃)

#         Q̃₁ = unsqueeze(agent.networks.Q₁_layers(𝐬𝐚̃); dims=1)
#         Q̃₂ = unsqueeze(agent.networks.Q₂_layers(𝐬𝐚̃); dims=1)
#         Q̃ = vcat(Q̃₁, Q̃₂)
#         Q̃ₘₑₐₙ = vec(mean(Q̃; dims=(1, 2)))

#         ignore() do
#             losses[2] = -mean(logπᶿₐ̃)
#             losses[6] = mean(Q̃₁)
#             losses[7] = mean(Q̃₂)
#         end
#         return mean(@. exp(agent.networks.logα) * logπᶿₐ̃ - Q̃ₘₑₐₙ)
#     end
#     update!(agent.opt_states[1], agent.networks.policy_layers, ∇[1])

#     losses[3], ∇ = withgradient(agent.networks.logα) do m
#         return mean(@. m * (losses[2] - agent.params.H̄))
#     end
#     update!(agent.opt_states[4], agent.networks.logα, ∇[1])
#     return losses
# end

# function _update_agent_networks_base!(
#     agent::SACAgent{R, PrioritizedReplayBuffer{Float32}}
# ) where {R}
#     losses = zeros(Float32, 7)

#     𝐬, 𝐚, 𝐫, 𝐝, 𝐬′ = sample_buffer(
#         agent.memory, agent.rng, agent.params.minibatch_size
#     )

#     μ′, logσ²′ᵤ = agent.networks.policy_layers(𝐬′)
#     logσ²′ = @. (
#         agent.params.logσ²ₘᵢₙ
#         + (tanh(logσ²′ᵤ) + 1)
#         * (agent.params.logσ²ₘₐₓ - agent.params.logσ²ₘᵢₙ)
#         / 2
#     )
#     u′ = @. μ′ + exp(logσ²′ / 2) * $randn(agent.rng, $eltype(μ′), $size(μ′))
#     v′ = tanh.(u′)
#     𝐚′ = @. v′ * agent.params.action_scale + agent.params.action_bias
#     logπₐ′ = vec(
#         sum(
#             @. (
#                 -0.5f0 * (logσ²′ + (u′ - μ′) ^ 2 / exp(logσ²′) + log(2f0π))
#                 - log(agent.params.action_scale * (1 - v′ ^ 2) + eps(Float32))
#             );
#             dims=1,
#         )
#     )
#     𝐬′𝐚′ = [𝐬′; 𝐚′]
#     𝐬𝐚 = [𝐬; 𝐚]

#     Q₁ᵀ = vec(agent.networks.Q₁_target_layers(𝐬′𝐚′))
#     Q₂ᵀ = vec(agent.networks.Q₂_target_layers(𝐬′𝐚′))
#     Qᵀ = min.(Q₁ᵀ, Q₂ᵀ)
#     y = @. (
#         𝐫 + agent.params.γ * (1 - 𝐝) * (Qᵀ - exp(agent.networks.logα) * logπₐ′)
#     )
#     losses[4], ∇ = withgradient(agent.networks.Q₁_layers) do m
#         Q₁ = vec(m(𝐬𝐚))
#         return mse(Q₁, y)
#     end
#     update!(agent.opt_states[2], agent.networks.Q₁_layers, ∇[1])
#     losses[5], ∇ = withgradient(agent.networks.Q₂_layers) do m
#         Q₂ = vec(m(𝐬𝐚))
#         return mse(Q₂, y)
#     end
#     update!(agent.opt_states[3], agent.networks.Q₂_layers, ∇[1])

#     losses[1], ∇ = withgradient(agent.networks.policy_layers) do m
#         μ, logσ²ᵤ = m(𝐬)
#         logσ² = @. (
#             agent.params.logσ²ₘᵢₙ
#             + (tanh(logσ²ᵤ) + 1)
#             * (agent.params.logσ²ₘₐₓ - agent.params.logσ²ₘᵢₙ)
#             / 2
#         )
#         u = @. μ + exp(logσ² / 2) * $randn(agent.rng, $eltype(μ), $size(μ))
#         v = tanh.(u)
#         𝐚̃ = @. v * agent.params.action_scale + agent.params.action_bias
#         logπᶿₐ̃ = vec(
#             sum(
#                 @. (
#                     -0.5f0 * (logσ² + (u - μ) ^ 2 / exp(logσ²) + log(2f0π))
#                     - log(
#                         agent.params.action_scale * (1 - v ^ 2) + eps(Float32)
#                     )
#                 );
#                 dims=1,
#             )
#         )
#         𝐬𝐚̃ = [𝐬; 𝐚̃]
#         Q₁ = vec(agent.networks.Q₁_layers(𝐬𝐚̃))
#         Q₂ = vec(agent.networks.Q₂_layers(𝐬𝐚̃))
#         Qₘᵢₙ = min.(Q₁, Q₂)
#         ignore() do
#             losses[2] = -mean(logπᶿₐ̃)
#             losses[6] = mean(Q₁)
#             losses[7] = mean(Q₂)
#         end
#         return mean(@. exp(agent.networks.logα) * logπᶿₐ̃ - Qₘᵢₙ)
#     end
#     update!(agent.opt_states[1], agent.networks.policy_layers, ∇[1])
#     losses[3], ∇ = withgradient(agent.networks.logα) do m
#         return mean(@. m * (losses[2] - agent.params.H̄))
#     end
#     update!(agent.opt_states[4], agent.networks.logα, ∇[1])
#     return losses
# end

# ##################
# # Agent learning #
# ##################
# function learn!(agent::SACAgent, env::QuantumControlEnvironment)
#     rewards = zeros(agent.params.episodes)
#     losses = zeros(Float32, 7, agent.params.episodes)

#     _initial_steps!(agent, env)

#     for episode in 1:agent.params.episodes
#         episodeᵣ = evaluation_steps!(agent, env)
#         episodeₗ = trainer_steps!(agent)

#         rewards[episode] = episodeᵣ
#         losses[:, episode] = episodeₗ
#         println("Episode: ", episode, "| Rewards: ", rewards[episode])
#     end
#     return rewards, losses
# end
