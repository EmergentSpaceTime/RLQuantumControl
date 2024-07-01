"""Abstract callable struct to generate different types of rewards from the
quantum simulation. Custom rewards should define a [`reward_space`]() method.

These callables have the argument signature:
```math
    \\mathscr{R}(s_{t}, done_{t})\\rightarrow r_{t}
```
"""
abstract type RewardFunction <: Function end


struct DenseGateFidelity{
    I <: Union{Nothing, AbstractVector{Int}}
} <: RewardFunction
    target::Matrix{ComplexF64}
    computational_indices::I
    _r_tm1::Base.RefValue{Float64}
end

"""
    DenseGateFidelity(
        target::Matrix,
        computational_indices::Union{Nothing, AbstractVector{Int}} = nothing,
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

Fields:
  * `target`: Target unitary matrix.
  * `computational_indices`: Computational subspace.
"""
function DenseGateFidelity(
    target::Matrix,
    computational_indices::Union{Nothing, AbstractVector{Int}} = nothing,
)
    is_unitary(target) || throw(ArgumentError("Target matrix must be unitary!"))
    return DenseGateFidelity(
        target, computational_indices, Base.RefValue(zero(Float64))
    )
end

function (r::DenseGateFidelity)(u::AbstractMatrix{ComplexF64}, done::Bool)
    if isnothing(r.computational_indices)
        nlif = -log10(1 - gate_fidelity(u, r.target) + 1e-6)
    else
        nlif = -log10(
            1
            - gate_fidelity(
                u[r.computational_indices, r.computational_indices], r.target
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
end

"""
    function SparseGateFidelity(
        target::Matrix,
        computational_indices::Union{Nothing, AbstractVector{Int}} = nothing,
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

Fields:
  * `target`: Target unitary matrix.
  * `computational_indices`: Computational subspace.
"""
function SparseGateFidelity(
    target::Matrix,
    computational_indices::Union{Nothing, AbstractVector{Int}} = nothing,
)
    is_unitary(target) || throw(ArgumentError("Target matrix must be unitary!"))
    return SparseGateFidelity{typeof(computational_indices)}(
        target, computational_indices
    )
end

function (r::SparseGateFidelity)(u::AbstractMatrix{ComplexF64}, done::Bool)
    if done
        if isnothing(r.computational_indices)
            return -log10(1 - gate_fidelity(u, r.target) + 1e-6)
        end
        return -log10(
            1
            - gate_fidelity(
                u[r.computational_indices, r.computational_indices], r.target
            )
            + 1e-6
        )
    end
    return zero(Float64)
end

reward_space(::SparseGateFidelity) = ClosedInterval(0.0, 6.0)


struct RobustGateFidelity{
    M <: ModelFunction,
    P <: Union{PulseFunction, Chain{<:Tuple{Vararg{PulseFunction}}}},
} <: RewardFunction
    target::Matrix{ComplexF64}
    model_function::M
    pulse_function::P
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
    RobustGateFidelity(
        target::Matrix,
        model_function::ModelFunction,
        pulse_history::Matrix{Float64},
        pulse_function::Union{
            PulseFunction, Chain{<:Tuple{Vararg{PulseFunction}}}
        };
        n_runs::Int = 50,
    )

Callable generating a robust gate fidelity reward where a protocol choice is
simulated with a number of different noise realisations and the reward is the
average fidelity over the runs.

Args:
  * `target`: Target unitary matrix.
  * `model_function`: Model function (optionally with noise).
  * `pulse_history`: Pulse history matrix (a view is made to avoid copying).
  * `pulse_function`: Pulse function (optionally with noise).

Kwargs:
  * `n_runs`: Number of runs (default: `50`).

Fields:
  * `target`: Target unitary matrix.
  * `model_function`: Model function.
  * `pulse_function`: Pulse function.
  * `n_runs`: Number of runs.
"""
function RobustGateFidelity(
    target::Matrix,
    model_function::ModelFunction,
    pulse_history::Matrix{Float64},
    pulse_function::Union{PulseFunction, Chain{<:Tuple{Vararg{PulseFunction}}}};
    n_runs::Int = 50,
)
    is_unitary(target) || throw(ArgumentError("Target matrix must be unitary!"))
    if !has_noise(model_function) & !has_noise(pulse_function)
        throw(
            ArgumentError(
                "Do not use this reward if the process does not contain noise"
                * "processes."
            )
        )
    end
    return RobustGateFidelity(
        target,
        model_function,
        pulse_function,
        n_runs,
        view(pulse_history, :, :),
    )
end

function (r::RobustGateFidelity)(
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
                    u
                    * r.model_function(
                        r.pulse_function(j, r._pulse_history[:, j])
                    )
                )
            end
            rewards[i] = -log10(
                1
                - gate_fidelity(
                    u[
                        computational_indices(r.model_function),
                        computational_indices(r.model_function),
                    ],
                    r.target,
                )
                + 1e-6
            )
        end
        return mean(rewards)
    end
    return zero(Float64)
end

reward_space(::RobustGateFidelity) = ClosedInterval(0.0, 6.0)


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

function (r::NormalisedReward)(u::AbstractMatrix{ComplexF64}, done::Bool)
    reward = r.base_function(u, done)
    r.return_e[] = r.return_e[] * r.gamma + reward
    # Update mean
    delta_r = r.return_e[] - r.returns_mean[]
    r.returns_mean[] += delta_r / (r.count[] + 1)
    # Update variance
    delta_r_new = r.return_e[] - r.returns_mean[]
    r.returns_var[] = (
        r.count[] * r.returns_var[] / (r.count[] + 1)
        + delta_r * delta_r_new / (r.count[] + 1)
    )
    # Update count
    r.count[] += 1
    if done
        r.return_e[] = 0
    end
    return reward / sqrt(r.returns_var[] + 1e-6)
end

reward_space(r::NormalisedReward) = reward_space(r.base_function)
