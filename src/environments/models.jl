"""Quantum environment simulation models and parameters."""

abstract type ModelFunction end


##############################
# Basis Generating Functions #
##############################
# function _get_H_control_terms₂()
#     σˣ = [0 1; 1 0]
#     σʸ = [0 -im; im 0]

#     σˣᴵ = kron(σˣ, I(2))
#     σᴵˣ = kron(I(2), σˣ)
#     σʸᴵ = kron(σʸ, I(2))
#     σᴵʸ = kron(I(2), σʸ)
#     return [σˣᴵ, σᴵˣ, σʸᴵ, σᴵʸ]
# end


function _get_H_drift_terms₂ꜛꜜ()
    σᶻ = [1 0; 0 -1]
    σᶻ¹ = kron(σᶻ, I(2), I(2), I(2))
    σᶻ² = kron(I(2), σᶻ, I(2), I(2))
    σᶻ³ = kron(I(2), I(2), σᶻ, I(2))
    σᶻ⁴ = kron(I(2), I(2), I(2), σᶻ)
    return [
        ((-3 * σᶻ¹ + σᶻ² + σᶻ³ + σᶻ⁴) / 8),
        ((-σᶻ¹ - σᶻ² + σᶻ³ + σᶻ⁴) / 4),
        ((-σᶻ¹ - σᶻ² - σᶻ³ + 3 * σᶻ⁴) / 8),
    ]
end


function _get_H_control_terms₂ꜛꜜ()
    σᶻ = [1 0; 0 -1]
    σˣ = [0 1; 1 0]
    σʸ = [0 -im; im 0]

    σ¹² = (
        kron(σˣ, σˣ, I(2), I(2))
        .+ real(kron(σʸ, σʸ, I(2), I(2)))
        .+ kron(σᶻ, σᶻ, I(2), I(2))
    )
    σ²³ = (
        kron(I(2), σˣ, σˣ, I(2))
        .+ real(kron(I(2), σʸ, σʸ, I(2)))
        .+ kron(I(2), σᶻ, σᶻ, I(2))
    )
    σ³⁴ = (
        kron(I(2), I(2), σˣ, σˣ)
        .+ real(kron(I(2), I(2), σʸ, σʸ))
        .+ kron(I(2), I(2), σᶻ, σᶻ)
    )
    return [σ¹², σ²³, σ³⁴] ./ 4
end


function _get_H_coupling_term₂ꜛꜜ(J₀::Real, ϵ₀::Real, Eᶜ::Real)
    Hᶜ = zeros(16, 16)
    Hᶜ[[6, 7, 10, 11], [6, 7, 10, 11]] .= (
        reshape([1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1], 4, 4)
        .* (6.58119e-7 / 16)
        .* (Eᶜ * (J₀ / ϵ₀) ^ 2)
    )
    return Hᶜ
end


# function _get_H_drift_term₂ᴶ(
#     δⱼ::Vector{T}, αⱼ::Vector{T}, J::T
# ) where {T <: AbstractFloat}
#     Hᵈ = diagm(
#         [
#             zero(T),
#             δⱼ[2],
#             δⱼ[1],
#             δⱼ[1] + δⱼ[2],
#             2 * δⱼ[2] + αⱼ[2],
#             2 * δⱼ[1] + αⱼ[1],
#             δⱼ[1] + 2 * δⱼ[2] + αⱼ[2],
#             δⱼ[2] + 2 * δⱼ[1] + αⱼ[1],
#             2 * (δⱼ[1] + δⱼ[2]) + αⱼ[1] + αⱼ[2],
#         ]
#     )
#     Hᵈ[2, 3] = Hᵈ[3, 2] = J
#     Hᵈ[7, 8] = Hᵈ[8, 7] = 2 * J
#     return Hᵈ
# end


# function _get_H_control_terms₂ᴶ(Ωⱼ::Vector{<:AbstractFloat})
#     H₁ = (
#         (Ωⱼ[1] / 2)
#         .* diagm(
#             2 => [1, 1, 0, 0, 1, 0, √2], 3 => [0, 0, √2], 4 => [0, 0, 0, √2]
#         )
#     )
#     H₂ = (
#         (Ωⱼ[2] / 2)
#         .* diagm(
#             1 => [1, 0, 1, 0, 0, 0, 0, √2],
#             2 => [0, 0, 0, 0, 0, 1],
#             3  => [0, √2, 0, √2],
#         )
#     )
#     return [H₁, collect(H₁'), H₂, collect(H₂')]
# end


##########
# Models #
##########
# """Simple Hamiltonian model.

# Model struct for a generic Hamiltonian model with a drift term, a vector of
# controls, a vector of coupings, and a vector of episodic drift (Gaussian) noise.
# The unitary evolution is given by the Schrödinger equation.

# ```math
# H = Hᵈ + ∑ᵢϵᵢHᵢᵋ + ∑ᵢf(ϵᵢ)Hᵢᶜ ∑ᵢsᵢHᵢⁿ
# U(t) = 𝒯[exp(-i∫H(t)dt)]
# ```

# Note that the struct is created with the broadest data type, so you should use
# the simplest common data type that is compatible with the model.

# Args:
#   * H_drift: Drift component of Hamiltonian.
#   * H_controls: Control components of Hamiltonian.

# Kwargs:
#   * H_drift_noises: Slow (episodic) noise on the drift components of the
#         Hamiltonian (default: []). This noise at the start of the episode.

# Fields:
#   * H_drift: Drift component of Hamiltonian.
#   * H_controls: Control components of Hamiltonian.
#   * H_drift_noises: Slow (episodic) noise of drift components of Hamiltonian.
# """
# struct SimpleModel{H <: Matrix} <: ModelFunction
#     H_drift::H
#     H_controls::Vector{H}
#     H_drift_noises::Vector{H}
#     _H_drift_episode::H
# end

# function SimpleModel(
#     H_drift::AbstractMatrix,
#     H_controls::AbstractVector;
#     H_drift_noises::AbstractVector = [],
# )
#     all_H = [H_drift, H_controls..., H_drift_noises...]
#     for H in all_H
#         if !isapprox(H, H')
#             throw(ArgumentError("Hamiltonian must be Hermitian."))
#         end
#     end
#     if !allequal(size.(all_H))
#         throw(
#             ArgumentError("Hamiltonian components should be of the same size.")
#         )
#     end
#     return SimpleModel{Matrix{promote_type(eltype.(all_H)...)}}(
#         H_drift, H_controls, zeros(size(H_drift))
#     )
# end

# Base.eltype(::SimpleModel{Matrix{T}}) where {T <: Number} = T

# function reset!(m::SimpleModel, rng::AbstractRNG = default_rng())
#     if isempty(m.H_drift_noises)
#         m._H_drift_episode .= m.H_drift
#     else
#         @. m._H_drift_episode = (
#             m.H_drift
#             + $sum(
#                 $rand($eltype(m), rng, $length(m.H_drift_noises))
#                 * m.H_drift_noises
#             )
#         )
#     end
#     return nothing
# end

# function get_unitary(
#     m::SimpleModel, Δt::T, ϵₜ::AbstractVector{T}
# ) where {T <: AbstractFloat}
#     return cis(
#         -Hermitian(@. Δt * (m._H_drift_episode + $sum(ϵₜ * m.H_controls)))
#     )
# end


"""Two-qubit quantum dot quantum gate control parameters.

Quantum dot callable struct for two qubit double quantum dot environment with
three controls [1] and a Schrödinger evolution. Units in ns⁻¹ / GHz unless
specified otherwise.

```math
ℳ(ϵₜ) = U(t) = exp(-iΔtH(J(ϵₜ)))
```

where:

```math
H(J(ϵₜ)) = ⅛b₁₂(-3σᶻ⁽¹⁾ + σᶻ⁽²⁾ + σᶻ⁽³⁾ + σᶻ⁽⁴⁾)
    + ¼b₂₃(-σᶻ⁽¹⁾ - σᶻ⁽²⁾ + σᶻ⁽³⁾ + σᶻ⁽⁴⁾)
    + ⅛b₃₄(-σᶻ⁽¹⁾ - σᶻ⁽²⁾ - σᶻ⁽³⁾ + 3σᶻ⁽⁴⁾)
    + ∑³ᵢⱼ₌ᵢ₊₁[¼Jᵢⱼ(ϵᵢⱼₜ)𝛔̄⁽ⁱ⁾⋅𝛔̄⁽ʲ⁾]
```

In addition, there is a coupling term, and a quasi-static (slow) noise that
peturbs the magnetic field gradient.

Args:
  * tₙ: Number of time steps per episode (equal to number of inputted actions).
  * Δt: Length of each time step. Note that if pulse shaping this should be
        the length of each sub-time step if using a filter.

Kwargs:
  * bᵢⱼ: Magnetic field gradients (default: [1.0, 7.0, -1.0]).
  * J₀: Unit parameter for tilt-control (default: 1.0).
  * ϵ₀: Unit field in meV (default: 0.272).
  * Eᶜ: Coupling parameter for leakage in μeV (default: 0.0).
  * σᵇ: Standard deviation of the slow noise on the magnetic field gradients
        (default: 0.0264).

Fields:
  * tₙ: Number of time steps per episode (equal to number of inputted actions).
  * Δt: Time step size (may include oversampling).
  * H_drift: Drift component of Hamiltonian.
  * H_controls: Control components of Hamiltonian.
  * H_coupling: Coupling component of Hamiltonian.
  * H_drift_noises: Slow (episodic) noise of drift components of Hamiltonian.

[1] Cerfontaine, P. et al. *High-fidelity gate set for exchange-coupled
singlet-triplet qubits*. Physical Review B 101, 155311 (2020).
https://doi.org/10.1103/PhysRevB.101.155311
"""
struct QuantumDot₂ <: ModelFunction
    tₙ::Int
    Δt::Float64
    H_drift::Matrix{Float64}
    H_controls::Vector{Matrix{Float64}}
    H_coupling::Matrix{Float64}
    H_drift_noises::Vector{Matrix{Float64}}
    _H_drift_episode::Matrix{Float64}
    _unitary_indices::Vector{Int}
    _computational_indices::UnitRange{Int}
end

function QuantumDot₂(
    tₙ::Int,
    Δt::Real;
    bᵢⱼ::Vector{<:Real} = [1.0, 7.0, -1.0],
    J₀::Real = 1.0,
    ϵ₀::Real = 0.272,
    Eᶜ::Real = 0.0,
    σᵇ::Real = 0.0264,
)
    if tₙ <= 0
        throw(ArgumentError("Number of time steps must be greater than 0!"))
    end
    if Δt <= zero(Δt)
        throw(ArgumentError("Length of time step must be greater than 0!"))
    end
    if σᵇ == zero(σᵇ)
        H_drift_noises = Vector{Matrix{Float64}}[]
    else
        H_drift_noises = σᵇ .* _get_H_drift_terms₂ꜛꜜ()
    end
    return QuantumDot₂(
        tₙ,
        Δt,
        sum(bᵢⱼ .* _get_H_drift_terms₂ꜛꜜ()),
        J₀ .* _get_H_control_terms₂ꜛꜜ(),
        _get_H_coupling_term₂ꜛꜜ(J₀, ϵ₀, Eᶜ),
        H_drift_noises,
        zeros(16, 16),
        [4, 6, 7, 10, 11, 13],
        2:5,
    )
end

function reset!(m::QuantumDot₂, rng::AbstractRNG = default_rng())
    if isempty(m.H_drift_noises)
        m._H_drift_episode .= m.H_drift
    else
        m._H_drift_episode .= (
            m.H_drift .+ sum(rand(rng, 3) .* m.H_drift_noises)
        )
    end
    return nothing
end

function (m::QuantumDot₂)(ϵₜ::Vector{Float64})
    U = cis(
        -Hermitian(
            @. m.Δt * (
                m._H_drift_episode
                + ϵₜ[1] * m.H_controls[1]
                + ϵₜ[2] * m.H_controls[2]
                + ϵₜ[3] * m.H_controls[3]
                + ϵₜ[1] * ϵₜ[3] * m.H_coupling
            )
        )
    )
    return @view U[m._unitary_indices, m._unitary_indices]
end


# """One qubit quantum gate control model.

# Model struct for an idealised one logical qubit quantum gate control
# environment. Arbitrary units.

# ```math
# H = Bᶻσᶻ + ϵ(t)σˣ
# ```

# Fields:
#   * target: Target unitary gate.
#   * tₙ: Number of time steps (default: 28).
#   * Δt: Length of each time step (default: 1.0 / 28).
#   * Bᶻ: The drift component in the z-direction (default: 1.0).
# """
# @kwdef struct QuantumGateControl₁{T <: AbstractFloat} <: QuantumModel{T}
#     target::Matrix{Complex{T}}
#     tₙ::Int = 28
#     Δt::T = 1.0 / 28
#     Bᶻ::T = 1.0
#     _H_drift::Matrix{Int} = Bᶻ .* [1 0; 0 -1]
#     _H_controls::Vector{Matrix{Int}} = [[0 1; 1 0]]
#     _computational_indices::UnitRange{Int} = 1:2
# end

# function get_unitary(m::QuantumGateControl₁{T}, ϵₜ::AbstractVector{T}) where {T}
#     k = √(ϵₜ[]^2 + m.Bᶻ^2)
#     return @. (
#         cos(k * m.Δt) * $I(2)
#         - im * (sin(k * m.Δt) / k) * (m._H_drift + ϵₜ[] * m._H_controls[])
#     )
# end


# """One qubit quantum dot quantum gate control model.

# Model struct for an idealised one logical qubit quantum gate control
# environment. Units in ns⁻¹ / GHz unless specified otherwise.

# ```math
# H = ½(ΔBᶻσᶻ + J(ϵₜ)σˣ)
# ```

# Fields:
#   * target: Target unitary gate.
#   * tₙ: Number of time steps (default: 28).
#   * Δt: Length of each time step (default: 1.0 / 28).
#   * Bᶻ: The drift component in the z-direction (default: 1.0).
# """
# @kwdef struct QuantumGateControl₁ꜛꜜ{T <: AbstractFloat} <: QuantumModel{T}
#     target::Matrix{Complex{T}}
#     tₙ::Int = 20
#     Δt::T = 1.0 / 20
#     ΔBᶻ::T = 1.0
#     _H_drift::Matrix{Int} = (ΔBᶻ / 2) .* kron([1 0; 0 -1], [1 0; 0 -1])
#     _H_controls::Vector{Matrix{Int}} = [kron([0 1; 1 0], [0 1; 1 0])]
#     _computational_indices::UnitRange{Int} = 1:2
# end

# function get_unitary(
#     m::QuantumGateControl₁ꜛꜜ{T}, ϵₜ::AbstractVector{T}
# ) where {T}
#     k = √(ϵₜ[]^2 + m.ΔBᶻ^2)
#     return @. (
#         cos(k * m.Δt) * $I(2)
#         - im * (sin(k * m.Δt) / k) * (m._H_drift + ϵₜ[] * m._H_controls[])
#     )
# end


# """Two qubit quantum gate control model.

# Model struct for an idealised two logical qubits quantum gate control
# environment. Arbitrary units.

# ```math
# H = Bᶻᶻσᶻ⊗σᶻ + ϵ₁(t)σˣ⊗𝟙 + ϵ₂(t)𝟙⊗σˣ + ϵ₃(t)σʸ⊗𝟙 + ϵ₄(t)𝟙⊗σʸ
# ```

# Fields:
#   * target: Target unitary gate.
#   * tₙ: Number of time steps (default: 38).
#   * Δt: Length of each time step (default: 1.0 / 38).
#   * Bᶻᶻ: The drift component in the z⊗z-direction (default: 1.0).
# """
# @kwdef struct QuantumGateControl₂{T <: AbstractFloat} <: QuantumModel{T}
#     target::Matrix{Complex{T}}
#     tₙ::Int = 38
#     Δt::T = 1.0 / 38
#     Bᶻᶻ::T = 1.0
#     _H_drift::Matrix{T} = Bᶻᶻ .* kron([1 0; 0 -1], [1 0; 0 -1])
#     _H_controls::Vector{Matrix{Complex{Int}}} = _get_H_control_terms₂()
#     _computational_indices::UnitRange{Int} = 1:4
# end

# function get_unitary(m::QuantumGateControl₂{T}, ϵₜ::AbstractVector{T}) where {T}
#     H = Hermitian(
#         @. (
#             m.Δt
#             * (
#                 m._H_drift
#                 + ϵₜ[1] * m._H_controls[1]
#                 + ϵₜ[2] * m._H_controls[2]
#                 + ϵₜ[3] * m._H_controls[3]
#                 + ϵₜ[4] * m._H_controls[4]
#             )
#         )
#     )
#     return exp(-im * H)
# end





# """Quantum dot quantum gate control parameters.

# Model struct for a two qubit transmon quantum environment (Mageson and Gambetta,
# 2020). Units are in ns⁻¹ / GHz unless specified otherwise. We set two controls
# to be zero (d₁, u₂₁).

# ```math
# H = ∑²ⱼ₌₁[δⱼbᵗⱼbⱼ + ½αbᵗⱼbᵗⱼbⱼbⱼ] + J(bᵗ₁b₂ + b₁bᵗ₂)
#     + ½Ω₁[exp(iδ₁t)d₁(t) + exp(iδ₂t)u₁₂(t)]b₁ + h.c.
#     + ½Ω₂[exp(iδ₂t)d₂(t) + exp(iδ₁t)u₂₁(t)]b₂ + h.c.
# ```

# Fields:
#   * target: Target unitary gate.
#   * tₙ: Number of time steps (default: 50).
#   * Δt: Length of each time step in ns (default: 4.0).
#   * Ωⱼ: Transmon drive parameter (default: [0.2047, 0.1585]).
#   * δⱼ: Transmon detuning parameter (default: [-0.499, 0.0]).
#   * αⱼ: Transmon anharmonicity parameter (default: [-0.35, -0.35]).
#   * J: Transmon coupling parameter (default: -0.009).
# """
# @kwdef struct QuantumGateControl₂ᴶ{T <: AbstractFloat} <: QuantumModel{T}
#     target::Matrix{Complex{T}}
#     tₙ::Int = 20
#     Δt::T = 250 / tₙ
#     Ωⱼ::Vector{T} = [0.2047, 0.1585]
#     δⱼ::Vector{T} = [-0.0866, 0.0]
#     αⱼ::Vector{T} = [-0.3105, -0.3139]
#     J::T = 0.0022
#     _H_drift::Matrix{T} = _get_H_drift_term₂ᴶ(δⱼ, αⱼ, J)
#     _H_controls::Vector{Matrix{T}} = _get_H_control_terms₂ᴶ(Ωⱼ)
#     _computational_indices::UnitRange{Int} = 1:4
# end

# function get_unitary(
#     m::QuantumGateControl₂ᴶ{T}, ϵₜ::AbstractVector{T}
# ) where {T}
#     H = Hermitian(
#         @. (
#             m.Δt / 10
#             * (
#                 m._H_drift
#                 + (ϵₜ[1] + im * ϵₜ[2]) * m._H_controls[1]
#                 + (ϵₜ[1] - im * ϵₜ[2]) * m._H_controls[2]
#                 + (ϵₜ[3] + im * ϵₜ[4]) * m._H_controls[3]
#                 + (ϵₜ[3] - im * ϵₜ[4]) * m._H_controls[4]
#                 + Hₙ
#             )
#         )
#     )
#     return cis(-H)
# end
