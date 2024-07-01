"""
    SACParameters(
        action_scale::AbstractVecOrMat{Float32},
        action_bias::AbstractVecOrMat{Float32},
        kwargs...,
    )

Parameter struct for SAC algorithm.

Fields:
  * `action_scale`: The scale to match the action space of the continuous
        environment (on device memory).
  * `action_bias`: The bias to match the action space of the continous
        environment (on device memory).
  * `H_bar`: Target entropy, usually equals to -dim(`action_space`).
  * `capacity`: Number of transitions stored in memory (default: `100000`).
  * `hiddens`: Dimensions of hidden layers (default: `[256, 256]`).
  * `log_var_min`: Minium log standard deviation for stable training (default:
        `-10`).
  * `log_var_max`: Maximum log standard deviation for stable training (default:
        `3`).
  * `use_tqc`: Whether using a distribution of Q-networks (default: `true`).
  * `n_q`: Number of output q-values for TQC (default: `25`).
  * `k_q`: Bottom number of quintiles to keep for TQC (default: `21`).
  * `dropout`: Dropout probability for dropout Q-networks (default: `0.01`).
  * `layer_norm`: Whether to use layer normalisation after each layer for
        Q-networks (default: `true`).
  * `gamma`: Discount parameter for extrinsic/intrinsic rewards (default:
        `0.99`).
  * `minibatch_size`: Minibatch size for policy and Q-networks (default: `256`).
  * `training_steps`: Number of training steps per episode (default: `20`).
  * `clips`: Global gradient clippings (default: `[5.0, 5.0, 5.0, 5.0]`).
  * `decays`: Weight decays (default: `[0.0, 0.0, 0.0, 0.0]`).
  * `eta`: Learning rates (default: `[3e-4, 3e-4, 3e-4, 3e-4]`).
  * `rho`: Polyak update parameter (default: `0.005`).
  * `warmup_normalisation_episodes`: Number of initial episodes for observation
        normalisation (default: `50`).
  * `warmup_evaluation_episodes`: Number of initial episodes to populate the
        buffer (default: `50`).
  * `episodes`: Number of total episodes (default: `5000`).
"""
@kwdef struct SACParameters{A <: AbstractVecOrMat{Float32}} <: AgentParameters
    action_scale::A
    action_bias::A
    H_bar::Float32 = length(action_scale)
    capacity::Int = 100000
    hiddens::Vector{Int} = [256, 256]
    log_var_min::Float32 = -10
    log_var_max::Float32 = 3
    use_tqc::Bool = true
    n_q::Int = 25
    k_q::Int = 46
    dropout::Float32 = 0.01
    layer_norm::Bool = true
    gamma::Float32 = 0.99
    minibatch_size::Int = 256
    training_steps::Int = 20
    decays::Vector{Float32} = [0.0, 0.0, 0.0, 0.0]
    clips::Vector{Float32} = [5.0, 5.0, 5.0, 5.0]
    eta::Vector{Float32} = [3e-4, 3e-4, 3e-4, 3e-4]
    rho::Float32 = 0.005
    warmup_normalisation_episodes::Int = 50
    warmup_evaluation_episodes::Int = 50
    episodes::Int = 1000
end


struct SACNetworks{
    B <: Union{Nothing, GPT},
    P <: Chain,
    Q <: Chain,
    A <: AbstractArray{Float32, 0},
    T <: AbstractVecOrMat{Float32},
} <: AgentNetworks
    base_layer::B
    policy_layers::P
    Q_1_layers::Q
    Q_1_target_layers::Q
    Q_2_layers::Q
    Q_2_target_layers::Q
    log_alpha::A
    log_var_min::Float32
    log_var_max::Float32
    action_scale::T
    action_bias::T
end

@layer SACNetworks

function trainable(n::SACNetworks)
    if isnothing(n.base_layer)
        return (
            n.policy_layers,
            n.Q_1_layers,
            n.Q_1_target_layers,
            n.Q_2_layers,
            n.Q_2_target_layers,
            n.log_alpha,
        )
    end
    return (
        n.base_layer,
        n.policy_layers,
        n.Q_1_layers,
        n.Q_1_target_layers,
        n.Q_2_layers,
        n.Q_2_target_layers,
        n.log_alpha,
    )
end

function SACNetworks(
    observation_dim::Int,
    action_dim::Int;
    action_scale::AbstractVecOrMat{Float32},
    action_bias::AbstractVecOrMat{Float32},
    log_var_min::Float32 = -10f0,
    log_var_max::Float32 = 3f0,
    n_q::Int = 1,
    hidden_dims::Vector{Int} = [128, 128],
    q_dropout::Float32 = 0.0f0,
    q_ln::Bool = false,
    ffn_activation::Function = relu,
    ffn_bias::Bool = true,
    init::Function = glorot_normal,
    base_layer::Union{Nothing, GPT} = nothing,
)
    policy_layers = _create_ffn(
        isnothing(base_layer) ? observation_dim : _embedding_dim(base_layer),
        action_dim,
        true;
        hidden_dims=hidden_dims,
        activation=ffn_activation,
        bias=ffn_bias,
        init=init,
    )
    Q_1_layers = _create_ffn(
        observation_dim + action_dim,
        n_q;
        hidden_dims=hidden_dims,
        dropout=iszero(q_dropout) ? nothing : q_dropout,
        layer_norm=q_ln,
        activation=ffn_activation,
        bias=ffn_bias,
        init=init,
    )
    Q_2_layers = _create_ffn(
        observation_dim + action_dim,
        n_q;
        hidden_dims=hidden_dims,
        dropout=iszero(q_dropout) ? nothing : q_dropout,
        layer_norm=q_ln,
        activation=ffn_activation,
        bias=ffn_bias,
        init=init,
    )
    log_alpha = zeros(Float32, ())
    return SACNetworks(
        base_layer,
        policy_layers,
        Q_1_layers,
        deepcopy(Q_1_layers),
        Q_2_layers,
        deepcopy(Q_2_layers),
        log_alpha,
        action_scale,
        action_bias,
        log_var_min,
        log_var_max,
    )
end

function Base.show(io::IO, m::SACNetworks)
    print(io, "SACNetworks(")
    if isnothing(m.base_layer)
        print(io, size(m.policy_layers[1].weight, 2), " => (")
    else
        if isa(m.base_layer.token_embedding, Embedding)
            print(io, size(m.base_layer.token_embedding.weight, 2), " => (")
        else
            print(io, size(m.base_layer.token_embedding[1].weight, 2), " => (")
        end
    end
    print(
        io,
        size(m.policy_layers[end].layer_1.weight, 1),
        ", ",
        size(m.Q_1_layers[end].weight, 1),
        ")"
    )
    isnothing(m.base_layer) || print(io, ", ", m.base_layer)
    if !iszero(isa.(m.Q_1_layers, Dropout))
        print(io, "; ", "q_dropout=", m.Q_1_layers[2].p)
    end
    iszero(isa.(m.Q_1_layers, CustomLayerNorm)) || print(io, "; ", "q_ln=true")
    print(io, ")")
    return nothing
end

function get_action(
    n::SACNetworks{GPT},
    o::Union{AbstractMatrix{Float32}, AbstractVector{Int}},
    rng::AbstractRNG,
)
    mu, log_var_u = n.base_layer(unsqueeze(o; dims=ndims(o) + 1))[:, end, 1]
    log_var = @. (
        n.log_var_min
        + (tanh(log_var_u) + 1) * (n.log_var_max - n.log_var_min) / 2
    )
    action = @. (
       tanh(mu + exp(log_var / 2) * randn(rng, Float32, $size(mu)))
       * n.action_scale
       + n.action_bias
    )
    return convert.(Float64, cpu(action))
end








struct SACAgent{
    P <: SACParameters,
    N <: SACNetworks,
    M <: ReplayBuffer,
    O <: AbstractVector,
    D <: AbstractDevice,
    R <: AbstractRNG,
} <: Agent{P, N, M, O, D, R}
    params::P
    networks::N
    memory::M
    opt_states::O
    device::D
    rng::R
end

"""
    SACAgent(
        observation_dim::Int,
        action_space::Vector{ClosedInterval{Float64}},
        device::AbstractDevice = FluxCPUDevice(),
        rng::AbstractRNG = default_rng(),
        kwargs...
    )

Struct of a agent that uses the Soft Actor Critic algorithm
[haarnoja2018soft](@cite). with optional dropout Q-networks
[chen2021randomized](@cite) and TQC [kuznetsov2020controlling](@cite) to improve
sample efficiency.

Args:
  * `observation_dim`: Environment observation dimension.
  * `action_space`: Legal environment action space.
  * `device`: Device for neural networks (default: [`Flux.FluxCPUDevice()`]()).
  * `rng`: Device RNG for agent that should match the device (e.g.
        [`CUDA.default_rng()`]()) (default: [`Random.default_rng()`]()).

Kwargs:
  * `activation`: Activation function for neural networks (default:
        [`NNlib.relu`]()).
  * `init`: Initialisation function for neural networks (default:
        [`Flux.glorot_normal`]()).
  * `params_kwargs`: Keyword arguments for `SACParameters`.

Fields:
  * `params`: Hyper parameters for the agent.
  * `memory`: Replay buffer with a history of transitions.
  * `networks`: Neural networks.
  * `opt_states`: Neural networks optimiser states.
  * `device`: Device for neural networks.
  * `rng`: Device rng for agent.
"""
function SACAgent(
    observation_dim::Int,
    action_space::Vector{ClosedInterval{Float64}},
    device::AbstractDevice = FluxCPUDevice(),
    rng::Union{AbstractRNG, Vector{<:AbstractRNG}} = default_rng();
    activation::Function = relu,
    init::Function = glorot_normal,
    kwargs...,
)
    action_dim = length(action_space)
    action_scale = @. (
        (rightendpoint(action_space) - leftendpoint(action_space)) / 2
    )
    action_bias = @. (
        (rightendpoint(action_space) + leftendpoint(action_space)) / 2
    )

    params = SACParameters(
        ;
        action_scale=action_scale,
        action_bias=action_bias,
        H_bar=-action_dim,
        kwargs...,
    )
    memory = ReplayBuffer(true, observation_dim, action_dim, params.capacity)
    networks = device(
        SACNetworks(
            observation_dim,
            action_dim;
            n_q=params.use_tqc ? params.n_q : 1,
            hiddens=params.hiddens,
            dropout=params.dropout,
            layer_norm=params.layer_norm,
            activation=activation,
            init=init,
            rng,
        )
    )
    opt_states = [
        setup(
            OptimiserChain(
                ClipNorm(params.clips[i]),
                AdamW(params.eta[i], (0.9f0, 0.999f0), params.decays[i]),
            ),
            network,
        )
        for (i, network) in [
            (1, networks.policy_layers),
            (2, networks.Q_1_layers),
            (3, networks.Q_2_layers),
            (4, networks.log_alpha),
        ]
    ]
    return SACAgent(params, memory, networks, opt_states, device, rng)
end

"""
    get_action(agent::SACAgent, observation::VecOrMat{Float64})

Retrieve an action from a policy informed by an environment observation.

Args:
  * `agent`: The SAC agent.
  * `observation`: Environment observation.

Returns:
  * `VecOrMat{Float64}`: The actions to take.
"""
function get_action(agent::SACAgent, observation::VecOrMat{Float64})
    μ, log_var_u = agent.networks.policy_layers(agent.device(f32(observation)))
    log_var = @. (
        agent.params.log_var_min
        + (tanh(log_var_u) + 1)
        * (agent.params.log_var_max - agent.params.log_var_min)
        / 2
    )
    action = @. (
       tanh(μ + exp(log_var / 2) * $randn(agent.rng, Float32, $size(μ)))
       * agent.params.action_scale
       + agent.params.action_bias
    )
    return convert.(Float64, cpu(action))
end

function get_random_action(agent::SACAgent)
    action = @. (
        tanh($randn(agent.rng, Float32, $size(agent.params.action_scale)))
        * agent.params.action_scale
        + agent.params.action_bias
    )
    return convert.(Float64, cpu(action))
end

function evaluation_steps!(agent::SACAgent, env::QuantumControlEnvironment)
    sum_r = zero(Float64)

    observation = reset!(env, agent.rng)
    done = false
    while !done
        index = update_and_get_index!(agent.memory)
        agent.memory.observations_t[:, index] = observation

        action = get_action(agent, observation)
        agent.memory.actions[:, index] = action

        observation, done, reward = step!(env, action)
        agent.memory.observations_tp1[:, index] = observation
        agent.memory.rewards[index] = reward
        agent.memory.dones[index] = done

        sum_r += reward
    end
    return sum_r
end

function trainer_steps!(agent::SACAgent, rng::AbstractRNG = default_rng())
    metrics = zeros(Float32, 7, agent.params.training_steps)
    for i in 1:agent.params.training_steps
        if agent.params.use_tqc
            metrics[:, i] = _update_agent_networks_tqc!(agent)
        else
            metrics[:, i] = _update_agent_networks_base!(agent, rng)
        end
        _polyak_update!(agent)
    end
    return vec(mean(metrics; dims=2))
end

function learn!(
    agent::SACAgent,
    env::QuantumControlEnvironment,
    init::Bool = true,
    rng::AbstractRNG = default_rng(),
)
    rewards = zeros(agent.params.episodes)
    losses = zeros(Float32, 7, agent.params.episodes)

    init && _initial_steps!(agent, env)

    for episode in 1:agent.params.episodes
        r_episode = evaluation_steps!(agent, env)
        l_episode = trainer_steps!(agent, rng)

        rewards[episode] = r_episode
        losses[:, episode] = l_episode
        println("Episode: ", episode, "| Reward: ", rewards[episode])
    end
    return rewards, losses
end

function _initial_steps!(agent::SACAgent, env::QuantumControlEnvironment)
    if (
        isa(env.observation_function, NormalisedObservation)
        | isa(env.reward_function, NormalisedReward)
    )
        for _ in 1:agent.params.warmup_normalisation_episodes
            _ = reset!(env, agent.rng)
            done = false
            while !done
                _, done, _ = step!(env, get_random_action(agent))
            end
        end
    end
    for _ in 1:agent.params.warmup_evaluation_episodes
        _ = evaluation_steps!(agent, env, agent.rng)
    end
    return nothing
end

function _polyak_update!(agent::SACAgent)
    for (target, source) in zip(
        params(agent.networks.Q_1_target_layers),
        params(agent.networks.Q_1_layers),
    )
        @. target = (1 - agent.params.rho) * target + agent.params.rho * source
    end
    for (target, source) in zip(
        params(agent.networks.Q_2_target_layers),
        params(agent.networks.Q_2_layers),
    )
        @. target = (1 - agent.params.rho) * target + agent.params.rho * source
    end
    return nothing
end

function _update_agent_networks_tqc!(agent::SACAgent)
    losses = zeros(Float32, 7)

    states, actions, rewards, dones, states_p = sample_buffer(
        agent.memory, agent.device, agent.params.minibatch_size, agent.rng
    )

    mean_p, log_var_p_u = agent.networks.policy_layers(states_p)
    log_var_p = @. (
        agent.params.log_var_min
        + (tanh(log_var_p_u) + 1)
        * (agent.params.log_var_max - agent.params.log_var_min)
        / 2
    )
    n_01 = agent.device(randn(agent.rng, eltype(mean_p), size(mean_p)))
    u_p = @. mean_p + exp(log_var_p / 2) * n_01
    v_p = tanh.(u_p)

    actions_p = @. v_p * agent.params.action_scale + agent.params.action_bias
    logπₐ′ = sum(
        @. (
            -0.5f0 * (log_var_p + (u_p - mean_p) ^ 2 / exp(log_var_p) + log(2f0π))
            - log(agent.params.action_scale * (1 - v_p ^ 2) + eps(Float32))
        );
        dims=1,
    )
    𝐬′𝐚′ = vcat(states_p, actions_p)
    𝐬𝐚 = vcat(states, actions)

    Q₁ᵀ = agent.networks.Q_1_target_layers(𝐬′𝐚′)
    Q₂ᵀ = agent.networks.Q_2_target_layers(𝐬′𝐚′)
    Qᵀ = sort(vcat(Q₁ᵀ, Q₂ᵀ); dims=1)[1 : agent.params.k_q, :]
    y = unsqueeze(
        @. (
            rewards
            + agent.params.gamma
            * (1 - dones)
            * (Qᵀ - exp(agent.networks.log_alpha) * logπₐ′)
        );
        dims=2,
    )
    cumulative = unsqueeze(
        unsqueeze(
            @. ($collect(1:agent.params.n_q) - 0.5f0) / agent.params.n_q; dims=2
        );
        dims=1,
    )
    losses[4], ∇ = withgradient(agent.networks.Q_1_layers) do m
        Q₁ = unsqueeze(m(𝐬𝐚); dims=1)

        δ = y .- Q₁
        δ₊ = abs.(δ)

        huber_term = @. ifelse(δ₊ > 1, δ₊ - 0.5f0, 0.5f0 * δ ^ 2)
        loss = @. abs(cumulative - (δ < 0)) * huber_term
        return mean(loss)
    end
    update!(agent.opt_states[2], agent.networks.Q_1_layers, ∇[1])
    losses[5], ∇ = withgradient(agent.networks.Q_2_layers) do m
        Q₂ = unsqueeze(m(𝐬𝐚); dims=1)

        δ = y .- Q₂
        δ₊ = abs.(δ)

        huber_term = @. ifelse(δ₊ > 1, δ₊ - 0.5f0, 0.5f0 * δ ^ 2)
        loss = @. abs(cumulative - (δ < 0)) * huber_term
        return mean(loss)
    end
    update!(agent.opt_states[3], agent.networks.Q_2_layers, ∇[1])

    losses[1], ∇ = withgradient(agent.networks.policy_layers) do m
        μ, log_var_u = m(states)
        log_var = @. (
            agent.params.log_var_min
            + (tanh(log_var_u) + 1)
            * (agent.params.log_var_max - agent.params.log_var_min)
            / 2
        )
        u = @. μ + exp(log_var / 2) * $randn(agent.rng, $eltype(μ), $size(μ))
        v = tanh.(u)
        𝐚̃ = @. v * agent.params.action_scale + agent.params.action_bias
        logπᶿₐ̃ = vec(
            sum(
                @. (
                    -0.5f0 * (log_var + (u - μ) ^ 2 / exp(log_var) + log(2f0π))
                    - log(
                        agent.params.action_scale * (1 - v ^ 2) + eps(Float32)
                    )
                );
                dims=1,
            )
        )
        𝐬𝐚̃ = vcat(states, 𝐚̃)

        Q̃₁ = unsqueeze(agent.networks.Q_1_layers(𝐬𝐚̃); dims=1)
        Q̃₂ = unsqueeze(agent.networks.Q_2_layers(𝐬𝐚̃); dims=1)
        Q̃ = vcat(Q̃₁, Q̃₂)
        Q̃ₘₑₐₙ = vec(mean(Q̃; dims=(1, 2)))

        ignore() do
            losses[2] = -mean(logπᶿₐ̃)
            losses[6] = mean(Q̃₁)
            losses[7] = mean(Q̃₂)
        end
        return mean(@. exp(agent.networks.log_alpha) * logπᶿₐ̃ - Q̃ₘₑₐₙ)
    end
    update!(agent.opt_states[1], agent.networks.policy_layers, ∇[1])

    losses[3], ∇ = withgradient(agent.networks.log_alpha) do m
        return mean(@. m * (losses[2] - agent.params.H_bar))
    end
    update!(agent.opt_states[4], agent.networks.log_alpha, ∇[1])
    return losses
end

function _update_agent_networks_base!(
    agent::SACAgent, rng::AbstractRNG = default_rng()
)
    losses = zeros(Float32, 7)

    𝐬, 𝐚, 𝐫, 𝐝, 𝐬′ = sample_buffer(
        agent.memory, rng, agent.params.minibatch_size
    )

    μ′, log_var′ᵤ = agent.networks.policy_layers(𝐬′)
    log_var′ = @. (
        agent.params.log_var_min
        + (tanh(log_var′ᵤ) + 1)
        * (agent.params.log_var_max - agent.params.log_var_min)
        / 2
    )
    u′ = @. μ′ + exp(log_var′ / 2) * $randn(rng, $eltype(μ′), $size(μ′))
    v′ = tanh.(u′)
    𝐚′ = @. v′ * agent.params.action_scale + agent.params.action_bias
    logπₐ′ = vec(
        sum(
            @. (
                -0.5f0 * (log_var′ + (u′ - μ′) ^ 2 / exp(log_var′) + log(2f0π))
                - log(agent.params.action_scale * (1 - v′ ^ 2) + eps(Float32))
            );
            dims=1,
        )
    )
    𝐬′𝐚′ = [𝐬′; 𝐚′]
    𝐬𝐚 = [𝐬; 𝐚]

    Q₁ᵀ = vec(agent.networks.Q_1_target_layers(𝐬′𝐚′))
    Q₂ᵀ = vec(agent.networks.Q_2_target_layers(𝐬′𝐚′))
    Qᵀ = min.(Q₁ᵀ, Q₂ᵀ)
    y = @. (
        𝐫
        + agent.params.gamma
        * (1 - 𝐝)
        * (Qᵀ - exp(agent.networks.log_alpha) * logπₐ′)
    )
    losses[4], ∇ = withgradient(agent.networks.Q_1_layers) do m
        Q₁ = vec(m(𝐬𝐚))
        return mse(Q₁, y)
    end
    update!(agent.opt_states[2], agent.networks.Q_1_layers, ∇[1])
    losses[5], ∇ = withgradient(agent.networks.Q_2_layers) do m
        Q₂ = vec(m(𝐬𝐚))
        return mse(Q₂, y)
    end
    update!(agent.opt_states[3], agent.networks.Q_2_layers, ∇[1])

    losses[1], ∇ = withgradient(agent.networks.policy_layers) do m
        μ, log_var_u = m(𝐬)
        log_var = @. (
            agent.params.log_var_min
            + (tanh(log_var_u) + 1)
            * (agent.params.log_var_max - agent.params.log_var_min)
            / 2
        )
        u = @. μ + exp(log_var / 2) * $randn(rng, $eltype(μ), $size(μ))
        v = tanh.(u)
        𝐚̃ = @. v * agent.params.action_scale + agent.params.action_bias
        logπᶿₐ̃ = vec(
            sum(
                @. (
                    -0.5f0 * (log_var + (u - μ) ^ 2 / exp(log_var) + log(2f0π))
                    - log(
                        agent.params.action_scale * (1 - v ^ 2) + eps(Float32)
                    )
                );
                dims=1,
            )
        )
        𝐬𝐚̃ = [𝐬; 𝐚̃]
        Q₁ = vec(agent.networks.Q_1_layers(𝐬𝐚̃))
        Q₂ = vec(agent.networks.Q_2_layers(𝐬𝐚̃))
        Qₘᵢₙ = min.(Q₁, Q₂)
        ignore() do
            losses[2] = -mean(logπᶿₐ̃)
            losses[6] = mean(Q₁)
            losses[7] = mean(Q₂)
        end
        return mean(@. exp(agent.networks.log_alpha) * logπᶿₐ̃ - Qₘᵢₙ)
    end
    update!(agent.opt_states[1], agent.networks.policy_layers, ∇[1])
    losses[3], ∇ = withgradient(agent.networks.log_alpha) do m
        return mean(@. m * (losses[2] - agent.params.H_bar))
    end
    update!(agent.opt_states[4], agent.networks.log_alpha, ∇[1])
    return losses
end
