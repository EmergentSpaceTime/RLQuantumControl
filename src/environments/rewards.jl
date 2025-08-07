"""Abstract callable struct to generate different types of rewards from the
quantum simulation. Custom rewards should define a [`reward_space`]() method.

These callables have the argument signature (excluding an optional RNG):
```math
    \\mathscr{R}(s_{t}, \\text{done}_{t})\\rightarrow r_{t}
```
"""
abstract type RewardFunction <: Function end


struct DenseGateFidelity{
    I <: Union{Nothing, AbstractVector{Int}}
} <: RewardFunction
    target::Matrix{ComplexF64}
    computational_indices::I
    map_to_closest_unitary::Bool
    _r_tm1::Base.RefValue{Float64}
end

"""
    DenseGateFidelity(
        target::Matrix,
        computational_indices::Union{Nothing, AbstractVector{Int}} = nothing,
        map_to_closest_unitary::Bool = false,
    )

Callable to generate a gate fidelity reward function that is defined as:
```math
    \\mathscr{R}(U_{t}) = r_{t} - r_{t - 1}
```
Where:
```math
    r_{t} = -\\log_{10}(1 - \\mathcal{F}(U_{t}, U_{\\text{target}}))
```

Args:
  * `target`: Target unitary matrix.
  * `computational_indices`: Computational subspace to calculate the fidelity.
        If `nothing`, the full matrix is used (default: `nothing`).
  * `map_to_closest_unitary`: If `true`, the input matrix is mapped to the
        closest unitary matrix (default: `false`).

Fields:
  * `target`: Target unitary matrix.
  * `computational_indices`: Computational subspace.
  * `map_to_closest_unitary`: Whether to map final matrix to the closest
        unitary.
"""
function DenseGateFidelity(
    target::Matrix,
    computational_indices::Union{Nothing, AbstractVector{Int}} = nothing,
    map_to_closest_unitary::Bool = false,
)
    is_unitary(target) || throw(ArgumentError("Target matrix must be unitary!"))
    return DenseGateFidelity(
        target,
        computational_indices,
        map_to_closest_unitary,
        Base.RefValue(zero(Float64)),
    )
end

function (r::DenseGateFidelity)(
    u::AbstractMatrix{ComplexF64}, done::Bool, ::AbstractRNG = default_rng()
)
    if isnothing(r.computational_indices)
        nlif = -log10(
            gate_infidelity(
                r.map_to_closest_unitary ? closest_unitary(u) : u, r.target
            )
            + 1e-6
        )
    else
        nlif = -log10(
            gate_infidelity(
                (
                    r.map_to_closest_unitary
                    ? closest_unitary(
                        u[r.computational_indices, r.computational_indices]
                    )
                    : u[r.computational_indices, r.computational_indices]
                ),
                r.target,
            )
            + 1e-6
        )
    end
    r = nlif - r._r_tm1[]
    r._r_tm1[] = nlif
    if done
        r._r_tm1[] = 0
    end
    return r
end

reward_space(::DenseGateFidelity) = ClosedInterval(0.0, 6.0)


struct SparseGateFidelity{
    I <: Union{Nothing, AbstractVector{Int}}
} <: RewardFunction
    target::Matrix{ComplexF64}
    computational_indices::I
    map_to_closest_unitary::Bool
end

"""
    SparseGateFidelity(
        target::Matrix,
        computational_indices::Union{Nothing, AbstractVector{Int}} = nothing,
        map_to_closest_unitary::Bool = false,
    )

Constructs a fidelity reward function that is defined as:

```math
    \\mathscr{R}(U_{t}) = \\begin{cases}
        0\\ \\forall\\ t \\neq T\\\\
        r_{t}\\ \\ \\ t = T
    \\end{cases}
```
Where:
```math
    r_{T} = -\\log_{10}(1 - \\mathcal{F}(U_{T}, U_{\\text{target}}))
```

Args:
  * `target`: Target unitary matrix.
  * `computational_indices`: Computational subspace to calculate the fidelity.
        If `nothing`, the full matrix is used (default: `nothing`).
  * `map_to_closest_unitary`: If `true`, the input matrix is mapped to the
        closest unitary matrix (default: `false`).

Fields:
  * `target`: Target unitary matrix.
  * `computational_indices`: Computational subspace.
  * `map_to_closest_unitary`: Whether to map final matrix to the closest
        unitary.
"""
function SparseGateFidelity(
    target::Matrix,
    computational_indices::Union{Nothing, AbstractVector{Int}} = nothing,
    map_to_closest_unitary::Bool = false,
)
    is_unitary(target) || throw(ArgumentError("Target matrix must be unitary!"))
    return SparseGateFidelity{typeof(computational_indices)}(
        target, computational_indices, map_to_closest_unitary
    )
end

function (r::SparseGateFidelity)(
    u::AbstractMatrix{ComplexF64}, done::Bool, ::AbstractRNG = default_rng()
)
    if done
        if isnothing(r.computational_indices)
            return -log10(
                gate_infidelity(
                    r.map_to_closest_unitary ? closest_unitary(u) : u, r.target
                )
                + 1e-6
            )
        end
        return -log10(
            gate_infidelity(
                (
                    r.map_to_closest_unitary
                    ? closest_unitary(
                        u[r.computational_indices, r.computational_indices]
                    )
                    : u[r.computational_indices, r.computational_indices]
                ),
                r.target,
            )
            + 1e-6
        )
    end
    return zero(Float64)
end

reward_space(::SparseGateFidelity) = ClosedInterval(0.0, 6.0)


struct RobustGateReward{
    M <: ModelFunction,
    P <: Union{PulseFunction, Chain{<:Tuple{Vararg{PulseFunction}}}},
    O <: Union{Nothing, ObservationFunction},
    L <: Function,
    S <: Function,
    I <: Union{Nothing, AbstractVector{Int}},
} <: RewardFunction
    target::Matrix{ComplexF64}
    model_function::M
    pulse_function::P
    observation_function::O
    loss_function::L
    stat_function::S
    computational_indices::I
    map_to_closest_unitary::Bool
    n_runs::Int
    _pulse_history::SubArray{
        Float64,
        2,
        Matrix{Float64},
        Tuple{Base.Slice{Base.OneTo{Int64}}, Base.Slice{Base.OneTo{Int64}}},
        true,
    }
end

"""
    RobustGateReward(
        target::Matrix,
        model_function::ModelFunction,
        pulse_history::Matrix{Float64},
        pulse_function::Union{
            PulseFunction, Chain{<:Tuple{Vararg{PulseFunction}}}
        },
        observation_function::Union{Nothing, ObservationFunction} = nothing,
        loss_function::Function = gate_infidelity,
        stat_function::Function = mean,
        computational_indices::Union{Nothing, AbstractVector{Int}} = nothing,
        map_to_closest_unitary::Bool = false,
        n_runs::Int = 1000,
    )

Callable generating a robust gate reward where a protocol choice is simulated
with a number of different noise realisations and the reward is the average
reward function over the runs. There should be noise in either the model or
pulse protocol.

Args:
  * `target`: Target unitary matrix.
  * `model_function`: Model function (optionally with noise).
  * `pulse_history`: Pulse history matrix (a view is made to avoid copying). If
        there is pulse shaping, use the shaped pulse history.
  * `pulse_function`: Pulse function (optionally with noise).
  * `observation_function`: Observation function if using statistics to get
        final gate (default: `nothing`).
  * `loss_function`: Loss function to calculate the reward (default:
        [`gate_infidelity`](@ref)).
  * `stat_function`: Statistical function to calculate the colleciton of loss
        functions (default: [`Statistics.mean`]()).
  * `computational_indices`: Computational subspace to calculate the fidelity.
        If `nothing`, the full matrix is used (default: `nothing`).
  * `map_to_closest_unitary`: If `true`, the input matrix is mapped to the
        closest unitary matrix (default: `false`).

Kwargs:
  * `n_runs`: Number of runs (default: `1000`).

Fields:
  * `target`: Target unitary matrix.
  * `model_function`: Model function.
  * `pulse_function`: Pulse function.
  * `observation_function`: Observation function.
  * `loss_function`: Loss function.
  * `stat_function`: Loss function.
  * `computational_indices`: Computational subspace.
  * `map_to_closest_unitary`: Whether to map final matrix to the closest
        unitary.
  * `n_runs`: Number of runs.
"""
function RobustGateReward(
    target::Matrix,
    model_function::ModelFunction,
    pulse_history::Matrix{Float64},
    pulse_function::Union{PulseFunction, Chain{<:Tuple{Vararg{PulseFunction}}}},
    observation_function::Union{Nothing, ObservationFunction} = nothing,
    loss_function::Function = gate_infidelity,
    stat_function::Function = mean,
    computational_indices::Union{Nothing, AbstractVector{Int}} = nothing,
    map_to_closest_unitary::Bool = false;
    n_runs::Int = 1000,
)
    is_unitary(target) || throw(ArgumentError("Target matrix must be unitary!"))
    if isnothing(computational_indices)
        if map_to_closest_unitary
            throw(
                ArgumentError(
                    "Computational indices cannot be `nothing` whilst also"
                    * " mapping to the closest unitary."
                )
            )
        end
    end
    return RobustGateReward{
        typeof(model_function),
        typeof(pulse_function),
        typeof(observation_function),
        typeof(loss_function),
        typeof(stat_function),
        typeof(computational_indices),
    }(
        target,
        model_function,
        pulse_function,
        observation_function,
        loss_function,
        stat_function,
        computational_indices,
        map_to_closest_unitary,
        n_runs,
        view(pulse_history, :, :),
    )
end

function (r::RobustGateReward)(
    ::AbstractMatrix{ComplexF64}, done::Bool, rng::AbstractRNG = default_rng()
)
    if done
        rewards = zeros(Float64, r.n_runs)
        for i in 1:r.n_runs
            reset!(r.model_function, rng)
            reset!(r.pulse_function, rng)
            u = Matrix{ComplexF64}(I, _m_size(r.model_function))
            for j in axes(r._pulse_history, 2)
                u .= (
                    r.model_function(
                        r.pulse_function(j, r._pulse_history[:, j])
                    )
                    * u
                )
            end
            if isa(r.observation_function, UnitaryTomography)
                u .= r.observation_function(
                    vcat(
                        zeros(size(r._pulse_history, 1) + 1),
                        vec(reinterpret(Float64, u)),
                    ),
                    rng,
                )
            end
            if isnothing(r.computational_indices)
                rewards[i] = (
                    r.loss_function(
                        r.map_to_closest_unitary ? closest_unitary(u) : u,
                        r.target,
                    )
                    + 1e-6
                )
            else
                rewards[i] = (
                    r.loss_function(
                        (
                            r.map_to_closest_unitary
                            ? closest_unitary(
                                u[
                                    r.computational_indices,
                                    r.computational_indices,
                                ]
                            )
                            : u[
                                r.computational_indices, r.computational_indices
                            ]
                        ),
                        r.target,
                    )
                    + 1e-6
                )
            end
        end
        return r.stat_function(rewards)
    end
    return zero(Float64)
end

function (
    r::RobustGateReward{
        <:ModelFunction,
        <:Union{PulseFunction, Chain{<:Tuple{Vararg{PulseFunction}}}},
        SingleShotTomography,
    }
)(::AbstractMatrix{ComplexF64}, done::Bool, rng::AbstractRNG = default_rng())
    if done
        p = zeros(
            Float64,
            r.observation_function._unitary_dim,
            2 * r.observation_function._unitary_dim,
        )
        for i in 1:r.observation_function._unitary_dim
            for _ in 1:r.n_runs
                reset!(r.model_function, rng)
                reset!(r.pulse_function, rng)
                u = Matrix{ComplexF64}(
                    I,
                    r.observation_function._unitary_dim,
                    r.observation_function._unitary_dim,
                )
                for j in axes(r._pulse_history, 2)
                    u .= (
                        r.model_function(
                            r.pulse_function(j, r._pulse_history[:, j])
                        )
                        * u
                    )
                end
                outcome = r.observation_function(
                    vcat(
                        zeros(size(r._pulse_history, 1) + 1),
                        vec(reinterpret(Float64, u)),
                    ),
                    rng,
                    i,
                )
                p[i, outcome[2]] += 1
            end
        end
        u_reconstructed = _get_u_from_probabilities(
            p ./ r.n_runs,
            r.observation_function._unitary_dim,
            r.observation_function.a,
            r.observation_function.b,
        )
        if isnothing(r.computational_indices)
            return -log10(
                r.loss_function(
                    (
                        r.map_to_closest_unitary
                        ? closest_unitary(u_reconstructed)
                        : u_reconstructed
                    ),
                    r.target,
                )
                + 1e-6
            )
        end
        return -log10(
            r.loss_function(
                (
                    r.map_to_closest_unitary
                    ? closest_unitary(
                        u_reconstructed[
                            r.computational_indices, r.computational_indices
                        ]
                    )
                    : u_reconstructed[
                        r.computational_indices, r.computational_indices
                    ]
                ),
                r.target,
            )
            + 1e-6
        )
    end
    return zero(Float64)
end

reward_space(::RobustGateReward) = ClosedInterval(0.0, 6.0)


struct NormalisedReward{R <: RewardFunction} <: RewardFunction
    base_function::R
    gamma::Float64
    return_e::Base.RefValue{Float64}
    returns_mean::Base.RefValue{Float64}
    returns_var::Base.RefValue{Float64}
    count::Base.RefValue{Int}
end

"""
    NormalisedReward(base_function::RewardFunction, gamma::Float64)

Reward function with output values normalised to unit normal of the returns.

Args:
  * `base_function`: Base reward function.
  * `gamma`: Discount factor.

Fields:
  * `base_function`: Base reward function.
  * `gamma`: Discount factor.
  * `return_e`: Return of episode.
  * `returns_mean`: Return mean.
  * `returns_var`: Return variance.
  * `count`: Number of observed returns.
"""
function NormalisedReward(base_function::RewardFunction, gamma::Float64)
    if base_function isa NormalisedReward
        throw(
            ArgumentError(
                "Base reward function can't already be the normalised reward"
                * "function!"
            )
        )
    end
    return NormalisedReward(
        base_function,
        gamma,
        Base.RefValue(zero(Float64)),
        Base.RefValue(zero(Float64)),
        Base.RefValue(one(Float64)),
        Base.RefValue(0),
    )
end

function (r::NormalisedReward)(
    u::AbstractMatrix{ComplexF64}, done::Bool, rng::AbstractRNG = default_rng()
)
    reward = r.base_function(u, done, rng)
    r.return_e[] = r.return_e[] * r.gamma + reward
    delta_r = r.return_e[] - r.returns_mean[]  # Update mean.
    r.returns_mean[] += delta_r / (r.count[] + 1)
    delta_r_new = r.return_e[] - r.returns_mean[]
    r.returns_var[] = (  # Update variance.
        r.count[] * r.returns_var[] / (r.count[] + 1)
        + delta_r * delta_r_new / (r.count[] + 1)
    )
    r.count[] += 1  # Update count.
    if done
        r.return_e[] = 0
    end
    return reward, reward / sqrt(r.returns_var[] + 1e-6)
end

reward_space(r::NormalisedReward) = reward_space(r.base_function)


struct RobustGateRewardG{
    M <: ModelFunction,
    P <: Union{PulseFunction, Chain{<:Tuple{Vararg{PulseFunction}}}},
    O <: Union{Nothing, ObservationFunction},
    L <: Function,
    S <: Function,
    I <: Union{Nothing, AbstractVector{Int}},
} <: RewardFunction
    target::Matrix{ComplexF64}
    model_function::M
    pulse_function::P
    observation_function::O
    loss_function::L
    stat_function::S
    computational_indices::I
    map_to_closest_unitary::Bool
    noise_level::Float64
    n_runs::Int
    _pulse_history::SubArray{
        Float64,
        2,
        Matrix{Float64},
        Tuple{Base.Slice{Base.OneTo{Int64}}, Base.Slice{Base.OneTo{Int64}}},
        true,
    }
end

function RobustGateRewardG(
    target::Matrix,
    model_function::ModelFunction,
    pulse_history::Matrix{Float64},
    pulse_function::Union{PulseFunction, Chain{<:Tuple{Vararg{PulseFunction}}}},
    observation_function::Union{Nothing, ObservationFunction} = nothing,
    loss_function::Function = gate_infidelity,
    stat_function::Function = mean,
    computational_indices::Union{Nothing, AbstractVector{Int}} = nothing,
    map_to_closest_unitary::Bool = false,
    noise_level::Float64 = 10000.0;
    n_runs::Int = 1000,
)
    is_unitary(target) || throw(ArgumentError("Target matrix must be unitary!"))
    if isnothing(computational_indices)
        if map_to_closest_unitary
            throw(
                ArgumentError(
                    "Computational indices cannot be `nothing` whilst also"
                    * " mapping to the closest unitary."
                )
            )
        end
    end
    return RobustGateRewardG{
        typeof(model_function),
        typeof(pulse_function),
        typeof(observation_function),
        typeof(loss_function),
        typeof(stat_function),
        typeof(computational_indices),
    }(
        target,
        model_function,
        pulse_function,
        observation_function,
        loss_function,
        stat_function,
        computational_indices,
        map_to_closest_unitary,
        noise_level,
        n_runs,
        view(pulse_history, :, :),
    )
end

function (r::RobustGateRewardG)(
    ::AbstractMatrix{ComplexF64}, done::Bool, rng::AbstractRNG = default_rng()
)
    if done
        rewards = zeros(Float64, r.n_runs)
        for i in 1:r.n_runs
            reset!(r.model_function, rng)
            reset!(r.pulse_function, rng)
            u = Matrix{ComplexF64}(I, _m_size(r.model_function))
            for j in axes(r._pulse_history, 2)
                u .= (
                    r.model_function(
                        r.pulse_function(j, r._pulse_history[:, j])
                    )
                    * u
                )
            end
            if isa(r.observation_function, UnitaryTomography)
                u .= r.observation_function(
                    vcat(
                        zeros(size(r._pulse_history, 1) + 1),
                        vec(reinterpret(Float64, u)),
                    ),
                    rng,
                )
            end
            u = closest_unitary(u .+ randn(rng, ComplexF64, size(u)) ./ r.noise_level)
            if isnothing(r.computational_indices)
                rewards[i] = (
                    r.loss_function(
                        r.map_to_closest_unitary ? closest_unitary(u) : u,
                        r.target,
                    )
                    + 1e-6
                )
            else
                rewards[i] = (
                    r.loss_function(
                        (
                            r.map_to_closest_unitary
                            ? closest_unitary(
                                u[
                                    r.computational_indices,
                                    r.computational_indices,
                                ]
                            )
                            : u[
                                r.computational_indices, r.computational_indices
                            ]
                        ),
                        r.target,
                    )
                    + 1e-6
                )
            end
        end
        return r.stat_function(rewards)
    end
    return zero(Float64)
end

reward_space(::RobustGateRewardG) = ClosedInterval(0.0, 6.0)
