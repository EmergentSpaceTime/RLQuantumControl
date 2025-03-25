module RLQuantumControl
    using BSON: @save
    using ChainRulesCore: @ignore_derivatives
    using Dierckx: Spline1D
    using Distributions: Multinomial
    using FFTW: irfft, rfft, rfftfreq
    using Flux: AdamW, Chain, ClipNorm, Dense, Dropout, Embedding, GRUv3Cell,
        LayerNorm, MultiHeadAttention, OptimiserChain, Scale, cpu, f32, gelu,
        glorot_normal, ignore, normalise, mse, params, relu, setup, unsqueeze,
        update!, withgradient, @layer, _size_check
    using HDF5: close, create_dataset, h5open, write
    using IntervalSets: ClosedInterval, leftendpoint, rightendpoint
    using LinearAlgebra: Hermitian, I, diag, diagm, dot, eigvals, qr, svd
    using NNlib: make_causal_mask
    using Random: AbstractRNG, default_rng
    using Statistics: mean
    using StatsBase: sample


    # Environment files.
    include("environments/utils.jl")
    include("environments/inputs.jl")
    include("environments/shapings.jl")
    include("environments/pulses.jl")
    include("environments/models.jl")
    include("environments/observations.jl")
    include("environments/rewards.jl")
    include("environments/environment.jl")

    # Agent files.
    include("agents/networks/ln.jl")
    include("agents/networks/gpt.jl")
    include("agents/networks/base.jl")

    include("agents/buffers.jl")

    include("agents/algorithms/agent.jl")
    include("agents/algorithms/sac.jl")

    # Re-export some used functions.
    export I, Spline1D
    export Chain

    # Environment exports.
    export closest_unitary, gate_fidelity, is_unitary, power_noise, rand_unitary
    export IdentityInput, InputFunction, StepInput, is_valid_input
    export ExponentialShaping, FilterShaping, IdentityShaping, ShapingFunction
    export ColouredNoiseInjection, ExponentialPulse, IdentityPulse,
        LogarithmPulse, PulseFunction, StaticNoiseInjection, WhiteNoiseInjection
    export ModelFunction, QuantumDot2, Simple1DSystem
    export ExactTomography, FullObservation, NormalisedObservation,
        MinimalObservation, ObservationFunction, UnitaryTomography
    export DenseGateFidelity, NormalisedReward, RewardFunction,
        RobustGateFidelity, SparseGateFidelity
    export QuantumControlEnvironment, step!
    export reset!

    # Agent exports.
    # export CustomLayerNorm, GPT
    export Memory, ReplayBuffer
    export Agent, SACAgent, SACNetworks, SACParameters, evaluation_steps!,
        get_action, get_random_action, learn!, trainer_steps!
end
