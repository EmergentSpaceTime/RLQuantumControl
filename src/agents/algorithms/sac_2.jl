# """
#     SACParameters(
#         action_scale::Vector{Float32},
#         action_bias::Vector{Float32},
#         H_bar::Float32;
#         kwargs...,
#     )

# Parameter struct for SAC algorithm.

# Fields:
#   * `action_scale`: The scale to match the action space of the continuous
#         environment.
#   * `action_bias`: The bias to match the action space of the continous
#         environment.
#   * `H_bar`: Target entropy, usually equals to -dim(`action_space`).
#   * `capacity`: Number of transitions stored in memory (default: `100000`).
#   * `hiddens`: Dimensions of hidden layers (default: `[256, 256]`).
#   * `log_var_min`: Minium log standard deviation for stable training (default:
#         `-10`).
#   * `log_var_max`: Maximum log standard deviation for stable training (default:
#         `3`).
#   * `use_tqc`: Whether using a distribution of Q-networks (default: `true`).
#   * `n_q`: Number of output q-values for TQC (default: `25`).
#   * `k_q`: Bottom number of quintiles to keep for TQC (default: `21`).
#   * `dropout`: Dropout probability for dropout Q-networks (default: `0.01`).
#   * `layer_norm`: Whether to use layer normalisation after each layer for
#         Q-networks (default: `true`).
#   * `gamma`: Discount parameter for extrinsic/intrinsic rewards (default:
#         `0.99`).
#   * `minibatch_size`: Minibatch size for policy and Q-networks (default: `256`).
#   * `training_steps`: Number of training steps per episode (default: `20`).
#   * `clips`: Global gradient clippings (default: `[5.0, 5.0, 5.0, 5.0]`).
#   * `decays`: Weight decays (default: `[0.0, 0.0, 0.0, 0.0]`).
#   * `eta`: Learning rates (default: `[3e-4, 3e-4, 3e-4, 3e-4]`).
#   * `rho`: Polyak update parameter (default: `0.005`).
#   * `warmup_normalisation_episodes`: Number of initial episodes for observation
#         normalisation (default: `50`).
#   * `warmup_evaluation_episodes`: Number of initial episodes to populate the
#         buffer (default: `50`).
#   * `episodes`: Number of total episodes (default: `5000`).
# """
# @kwdef struct SACParameters
#     action_scale::Vector{Float32}
#     action_bias::Vector{Float32}
#     H_bar::Float32
#     capacity::Int = 100000
#     hiddens::Vector{Int} = [256, 256]
#     log_var_min::Float32 = -10
#     log_var_max::Float32 = 3
#     use_tqc::Bool = true
#     n_q::Int = 25
#     k_q::Int = 46
#     dropout::Float32 = 0.01
#     layer_norm::Bool = true
#     gamma::Float32 = 0.99
#     minibatch_size::Int = 256
#     training_steps::Int = 20
#     decays::Vector{Float32} = [0.0, 0.0, 0.0, 0.0]
#     clips::Vector{Float32} = [5.0, 5.0, 5.0, 5.0]
#     eta::Vector{Float32} = [3e-4, 3e-4, 3e-4, 3e-4]
#     rho::Float32 = 0.005
#     warmup_normalisation_episodes::Int = 50
#     warmup_evaluation_episodes::Int = 50
#     episodes::Int = 1000
# end


# struct SACAgent{
#     M <: ReplayBuffer,
#     N <: SACNetworks,
#     O <: AbstractVector,
#     D <: AbstractDevice,
# } <: Agent{M, N, O, D}
#     params::SACParameters
#     memory::M
#     networks::N
#     opt_states::O
#     device::D
# end

# """
#     SACAgent(
#         env::QuantumControlEnvironment,
#         device::AbstractDevice = FluxCPUDevice(),
#         rng::AbstractRNG = default_rng();
#         kwargs...
#     )

# Struct of a agent that uses the Soft Actor Critic algorithm
# [haarnoja2018soft](@cite). with optional dropout Q-networks
# [chen2021randomized](@cite) and TQC [kuznetsov2020controlling](@cite) to improve
# sample efficiency.

# Args:
#   * `env`: The environment that the agent learns. This just extracts relavant
#         information from the environment such as the observation and action
#         spaces.
#   * `device`: Device for neural networks (default: [`Flux.FluxCPUDevice()`]()).
#   * `rng`: Random number generator (default: [`Random.default_rng()`]()).

# Kwargs:
#   * `activation`: Activation function for neural networks (default: relu).
#   * `init`: Initialisation function for neural networks (default:
#         glorot_normal).
#   * `params_kwargs`: Keyword arguments for SACParameters.

# Fields:
#   * `params`: Hyper parameters for the agent.
#   * `memory`: Replay buffer with a history of transitions.
#   * `networks`: Neural networks.
#   * `opt_states`: Neural networks optimiser states.
#   * `device`: Device for neural networks.
# """
# function SACAgent(
#     env::QuantumControlEnvironment,
#     device::AbstractDevice = FluxCPUDevice(),
#     rng::AbstractRNG = default_rng();
#     activation::Function = relu,
#     init::Function = glorot_normal,
#     kwargs...,
# )
#     observation_dim = length(env.observation_space)
#     action_dim = length(env.action_space)
#     action_scale = @. (
#         (rightendpoint(env.action_space) - leftendpoint(env.action_space)) / 2
#     )
#     action_bias = @. (
#         (rightendpoint(env.action_space) + leftendpoint(env.action_space)) / 2
#     )
#     H_bar = -action_dim

#     params = SACParameters(
#         ;
#         action_scale=action_scale,
#         action_bias=action_bias,
#         H_bar=H_bar,
#         kwargs...,
#     )
#     memory = ReplayBuffer(true, observation_dim, action_dim, params.capacity)
#     networks = device(
#         SACNetworks(
#             true,
#             observation_dim,
#             action_dim,
#             params.use_tqc ? params.n_q : 1,
#             params.hiddens,
#             params.dropout,
#             params.layer_norm;
#             activation=activation,
#             init=init,
#             rng=rng,
#         )
#     )
#     opt_states = [
#         setup(
#             OptimiserChain(
#                 ClipNorm(params.clips[i]),
#                 AdamW(params.eta[i], (0.9f0, 0.999f0), params.decays[i]),
#             ),
#             network,
#         )
#         for (i, network) in [
#             (1, networks.policy_layers),
#             (2, networks.Q_1_layers),
#             (3, networks.Q_2_layers),
#             (4, networks.logα),
#         ]
#     ]
#     return SACAgent(params, memory, networks, opt_states, device)
# end

# """
#     get_action(
#         agent::SACAgent,
#         observation::Vector{Float64},
#         rng::AbstractRNG = default_rng(),
#     )

# Retrieve an action from a policy informed by the current observation.

# Args:
#   * `agent`: The SAC agent.
#   * `observation`: The current observation.
#   * `rng`: Random number generator (default: [`Random.default_rng()`]()).

# Returns:
#   * `Vector{Float64}`: The action to take.
# """
# function get_action(
#     agent::SACAgent,
#     observation::Vector{Float64},
#     rng::AbstractRNG = default_rng(),
# )
#     μ, log_var_u = cpu(
#         agent.networks.policy_layers(agent.device(f32(observation)))
#     )
#     log_var = @. (
#         agent.params.log_var_min
#         + (tanh(log_var_u) + 1)
#         * (agent.params.log_var_max - agent.params.log_var_min)
#         / 2
#     )
#     N_01 = randn(rng, Float32, length(μ))
#     action = @. (
#        tanh(μ + exp(log_var / 2) * N_01) * agent.params.action_scale
#        + agent.params.action_bias
#     )
#     return convert(Vector{Float64}, action)
# end

# function get_random_action(agent::SACAgent, rng::AbstractRNG = default_rng())
#     return convert(
#         Vector{Float64},
#         @. (
#             tanh($randn(rng, Float32, $length(agent.params.action_scale)))
#             * agent.params.action_scale
#             + agent.params.action_bias
#         )
#     )
# end

# function evaluation_steps!(
#     agent::SACAgent,
#     env::QuantumControlEnvironment,
#     rng::AbstractRNG = default_rng(),
# )
#     sum_r = zero(Float64)

#     observation = reset!(env, rng)
#     done = false
#     while !done
#         index = update_and_get_index!(agent.memory)
#         agent.memory.observations_t[:, index] = observation

#         action = get_action(agent, observation, rng)
#         agent.memory.actions[:, index] = action

#         observation, done, reward = step!(env, action)
#         agent.memory.observations_tp1[:, index] = observation
#         # agent.memory.rewards[index] = reward[end]
#         agent.memory.rewards[index] = reward[1]
#         agent.memory.dones[index] = done
#         sum_r += reward[1]
#     end
#     return sum_r
# end

# function _initial_steps!(
#     agent::SACAgent,
#     env::QuantumControlEnvironment,
#     rng::AbstractRNG = default_rng(),
# )
#     if (
#         (env.observation_function isa NormalisedObservation)
#         | (env.reward_function isa NormalisedReward)
#     )
#         for _ in 1:agent.params.warmup_normalisation_episodes
#             _ = reset!(env, rng)
#             done = false
#             while !done
#                 _, done, _ = step!(env, get_random_action(agent, rng))
#             end
#         end
#     end
#     for _ in 1:agent.params.warmup_evaluation_episodes
#         _ = evaluation_steps!(agent, env, rng)
#     end
#     return nothing
# end

# function trainer_steps!(agent::SACAgent, rng::AbstractRNG = default_rng())
#     metrics = zeros(Float32, 7, agent.params.training_steps)
#     for i in 1:agent.params.training_steps
#         if agent.params.use_tqc
#             metrics[:, i] = _update_agent_networks_tqc!(agent, rng)
#         else
#             metrics[:, i] = _update_agent_networks_base!(agent, rng)
#         end
#         _polyak_update!(agent)
#     end
#     return vec(mean(metrics; dims=2))
# end

# function _polyak_update!(agent::SACAgent)
#     for (target, source) in zip(
#         params(agent.networks.Q_1_target_layers),
#         params(agent.networks.Q_1_layers),
#     )
#         @. target = (1 - agent.params.rho) * target + agent.params.rho * source
#     end
#     for (target, source) in zip(
#         params(agent.networks.Q_2_target_layers),
#         params(agent.networks.Q_2_layers),
#     )
#         @. target = (1 - agent.params.rho) * target + agent.params.rho * source
#     end
#     return nothing
# end

# function _update_agent_networks_tqc!(
#     agent::SACAgent, rng::AbstractRNG = default_rng()
# )
#     losses = zeros(Float32, 7)

#     𝐬, 𝐚, 𝐫ᵥ, 𝐝ᵥ, 𝐬′ = sample_buffer(
#         agent.memory, agent.params.minibatch_size, rng
#     )
#     𝐝 = unsqueeze(𝐝ᵥ; dims=1)
#     𝐫 = unsqueeze(𝐫ᵥ; dims=1)

#     μ′, log_var′ᵤ = agent.networks.policy_layers(𝐬′)
#     log_var′ = @. (
#         agent.params.log_var_min
#         + (tanh(log_var′ᵤ) + 1)
#         * (agent.params.log_var_max - agent.params.log_var_min)
#         / 2
#     )
#     u′ = @. μ′ + exp(log_var′ / 2) * $randn(rng, $eltype(μ′), $size(μ′))
#     v′ = tanh.(u′)
#     𝐚′ = @. v′ * agent.params.action_scale + agent.params.action_bias
#     logπₐ′ = sum(
#         @. (
#             -0.5f0 * (log_var′ + (u′ - μ′) ^ 2 / exp(log_var′) + log(2f0π))
#             - log(agent.params.action_scale * (1 - v′ ^ 2) + eps(Float32))
#         );
#         dims=1,
#     )
#     𝐬′𝐚′ = vcat(𝐬′, 𝐚′)
#     𝐬𝐚 = vcat(𝐬, 𝐚)

#     Q₁ᵀ = agent.networks.Q_1_target_layers(𝐬′𝐚′)
#     Q₂ᵀ = agent.networks.Q_2_target_layers(𝐬′𝐚′)
#     Qᵀ = sort(vcat(Q₁ᵀ, Q₂ᵀ); dims=1)[1 : agent.params.k_q, :]
#     y = unsqueeze(
#         @. (
#             𝐫
#             + agent.params.gamma
#             * (1 - 𝐝)
#             * (Qᵀ - exp(agent.networks.logα) * logπₐ′)
#         );
#         dims=2,
#     )
#     cumulative = unsqueeze(
#         unsqueeze(
#             @. ($collect(1:agent.params.n_q) - 0.5f0) / agent.params.n_q; dims=2
#         );
#         dims=1,
#     )
#     losses[4], ∇ = withgradient(agent.networks.Q_1_layers) do m
#         Q₁ = unsqueeze(m(𝐬𝐚); dims=1)

#         δ = y .- Q₁
#         δ₊ = abs.(δ)

#         huber_term = @. ifelse(δ₊ > 1, δ₊ - 0.5f0, 0.5f0 * δ ^ 2)
#         loss = @. abs(cumulative - (δ < 0)) * huber_term
#         return mean(loss)
#     end
#     update!(agent.opt_states[2], agent.networks.Q_1_layers, ∇[1])
#     losses[5], ∇ = withgradient(agent.networks.Q_2_layers) do m
#         Q₂ = unsqueeze(m(𝐬𝐚); dims=1)

#         δ = y .- Q₂
#         δ₊ = abs.(δ)

#         huber_term = @. ifelse(δ₊ > 1, δ₊ - 0.5f0, 0.5f0 * δ ^ 2)
#         loss = @. abs(cumulative - (δ < 0)) * huber_term
#         return mean(loss)
#     end
#     update!(agent.opt_states[3], agent.networks.Q_2_layers, ∇[1])

#     losses[1], ∇ = withgradient(agent.networks.policy_layers) do m
#         μ, log_var_u = m(𝐬)
#         log_var = @. (
#             agent.params.log_var_min
#             + (tanh(log_var_u) + 1)
#             * (agent.params.log_var_max - agent.params.log_var_min)
#             / 2
#         )
#         u = @. μ + exp(log_var / 2) * $randn(rng, $eltype(μ), $size(μ))
#         v = tanh.(u)
#         𝐚̃ = @. v * agent.params.action_scale + agent.params.action_bias
#         logπᶿₐ̃ = vec(
#             sum(
#                 @. (
#                     -0.5f0 * (log_var + (u - μ) ^ 2 / exp(log_var) + log(2f0π))
#                     - log(
#                         agent.params.action_scale * (1 - v ^ 2) + eps(Float32)
#                     )
#                 );
#                 dims=1,
#             )
#         )
#         𝐬𝐚̃ = vcat(𝐬, 𝐚̃)

#         Q̃₁ = unsqueeze(agent.networks.Q_1_layers(𝐬𝐚̃); dims=1)
#         Q̃₂ = unsqueeze(agent.networks.Q_2_layers(𝐬𝐚̃); dims=1)
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
#         return mean(@. m * (losses[2] - agent.params.H_bar))
#     end
#     update!(agent.opt_states[4], agent.networks.logα, ∇[1])
#     return losses
# end

# function _update_agent_networks_base!(
#     agent::SACAgent, rng::AbstractRNG = default_rng()
# )
#     losses = zeros(Float32, 7)

#     𝐬, 𝐚, 𝐫, 𝐝, 𝐬′ = sample_buffer(
#         agent.memory, rng, agent.params.minibatch_size
#     )

#     μ′, log_var′ᵤ = agent.networks.policy_layers(𝐬′)
#     log_var′ = @. (
#         agent.params.log_var_min
#         + (tanh(log_var′ᵤ) + 1)
#         * (agent.params.log_var_max - agent.params.log_var_min)
#         / 2
#     )
#     u′ = @. μ′ + exp(log_var′ / 2) * $randn(rng, $eltype(μ′), $size(μ′))
#     v′ = tanh.(u′)
#     𝐚′ = @. v′ * agent.params.action_scale + agent.params.action_bias
#     logπₐ′ = vec(
#         sum(
#             @. (
#                 -0.5f0 * (log_var′ + (u′ - μ′) ^ 2 / exp(log_var′) + log(2f0π))
#                 - log(agent.params.action_scale * (1 - v′ ^ 2) + eps(Float32))
#             );
#             dims=1,
#         )
#     )
#     𝐬′𝐚′ = [𝐬′; 𝐚′]
#     𝐬𝐚 = [𝐬; 𝐚]

#     Q₁ᵀ = vec(agent.networks.Q_1_target_layers(𝐬′𝐚′))
#     Q₂ᵀ = vec(agent.networks.Q_2_target_layers(𝐬′𝐚′))
#     Qᵀ = min.(Q₁ᵀ, Q₂ᵀ)
#     y = @. (
#         𝐫
#         + agent.params.gamma
#         * (1 - 𝐝)
#         * (Qᵀ - exp(agent.networks.logα) * logπₐ′)
#     )
#     losses[4], ∇ = withgradient(agent.networks.Q_1_layers) do m
#         Q₁ = vec(m(𝐬𝐚))
#         return mse(Q₁, y)
#     end
#     update!(agent.opt_states[2], agent.networks.Q_1_layers, ∇[1])
#     losses[5], ∇ = withgradient(agent.networks.Q_2_layers) do m
#         Q₂ = vec(m(𝐬𝐚))
#         return mse(Q₂, y)
#     end
#     update!(agent.opt_states[3], agent.networks.Q_2_layers, ∇[1])

#     losses[1], ∇ = withgradient(agent.networks.policy_layers) do m
#         μ, log_var_u = m(𝐬)
#         log_var = @. (
#             agent.params.log_var_min
#             + (tanh(log_var_u) + 1)
#             * (agent.params.log_var_max - agent.params.log_var_min)
#             / 2
#         )
#         u = @. μ + exp(log_var / 2) * $randn(rng, $eltype(μ), $size(μ))
#         v = tanh.(u)
#         𝐚̃ = @. v * agent.params.action_scale + agent.params.action_bias
#         logπᶿₐ̃ = vec(
#             sum(
#                 @. (
#                     -0.5f0 * (log_var + (u - μ) ^ 2 / exp(log_var) + log(2f0π))
#                     - log(
#                         agent.params.action_scale * (1 - v ^ 2) + eps(Float32)
#                     )
#                 );
#                 dims=1,
#             )
#         )
#         𝐬𝐚̃ = [𝐬; 𝐚̃]
#         Q₁ = vec(agent.networks.Q_1_layers(𝐬𝐚̃))
#         Q₂ = vec(agent.networks.Q_2_layers(𝐬𝐚̃))
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
#         return mean(@. m * (losses[2] - agent.params.H_bar))
#     end
#     update!(agent.opt_states[4], agent.networks.logα, ∇[1])
#     return losses
# end

# function learn!(
#     agent::SACAgent,
#     env::QuantumControlEnvironment,
#     save_dir::String,
#     rng::AbstractRNG = default_rng(),
# )
#     # Initialisation.
#     if "data.h5" in readdir(save_dir)
#         h5open(save_dir * "data.h5") do file
#             current_step .= read(file, "step")
#         end
#         rewards = zeros(agent.params.episodes)
#         losses = zeros(Float32, 7, agent.params.episodes)
#         h5open(save_dir * "data.h5") do file
#             rewards .= read(file, "r")
#             losses .= read(file, "l")
#         end
#     else
#         _initial_steps!(agent, env, rng)

#         @save save_dir * "agent.bson" agent
#         @save save_dir * "env.bson" env

#         rewards = zeros(agent.params.episodes)
#         losses = zeros(Float32, 7, agent.params.episodes)
#         h5open(save_dir * "data.h5", "cw") do file
#             create_dataset(file, "step", Int,  (1, ))
#             create_dataset(file, "r", eltype(rewards), size(losses))
#             create_dataset(file, "l", eltype(rewards), size(losses))
#             write(file["step"], current_step)
#             write(file["r"], rewards)
#             write(file["l"], losses)
#         end
#     else

#     end
#     # Training.
#     for episode in current_step:agent.params.episodes
#         r_episode = evaluation_steps!(agent, env, rng)
#         l_episode = trainer_steps!(agent, rng)

#         rewards[episode] = r_episode
#         losses[:, episode] = l_episode
#         println("Episode: ", episode, "| Rewards: ", rewards[episode])

#         if iszero(mod(episode, 500))
#             @save save_dir * "agent.bson" agent
#             @save save_dir * "env.bson" env

#             h5open(save_dir * "data.h5", "w") do file
#                 create_dataset(file, "r", eltype(rewards), size(losses))
#                 create_dataset(file, "l", eltype(rewards), size(losses))
#                 write(file["r"], rewards)
#                 write(file["l"], losses)
#             end
#         end
#     end
#     return rewards, losses
# end
