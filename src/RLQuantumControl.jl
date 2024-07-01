module RLQuantumControl
    using ChainRulesCore: @ignore_derivatives
    using Dierckx: Spline1D
    using Distributions: Multinomial
    using FFTW: ifft
    using Flux: AbstractDevice, AdamW, Chain, ClipNorm, Dense, Dropout,
        Embedding, FluxCPUDevice, GRUv3Cell, MultiHeadAttention, OptimiserChain,
        Scale, cpu, f32, gelu, glorot_normal, ignore, normalise, mse, params,
        relu, setup, unsqueeze, update!, withgradient, @layer,
        _greek_ascii_depwarn, _size_check
    using IntervalSets: ClosedInterval, leftendpoint, rightendpoint
    using LinearAlgebra: Hermitian, I, diag, diagm, dot, qr, svd
    using NNlib: make_causal_mask
    using Random: AbstractRNG, default_rng
    using StatsBase: mean, sample

    import Flux: hasaffine, trainable


    # Environment files
    include("environments/utils.jl")
    include("environments/inputs.jl")
    include("environments/shapings.jl")
    include("environments/pulses.jl")
    include("environments/models.jl")
    include("environments/observations.jl")
    include("environments/rewards.jl")
    include("environments/environment.jl")

    # Agent files
    include("agents/networks/ln.jl")
    include("agents/networks/gpt.jl")
    include("agents/networks/base.jl")

    include("agents/buffers.jl")

    include("agents/algorithms/agent.jl")
    include("agents/algorithms/sac.jl")

    # Re-export some used functions
    export I, Spline1D
    export Chain

    # Environment exports
    export gate_fidelity, is_unitary, power_noise, rand_unitary
    export IdentityInput, InputFunction, StepInput, is_valid_input
    export FilterShaping, IdentityShaping, ShapingFunction
    export ColouredNoiseInjection, ExponentialPulse, IdentityPulse,
        PulseFunction, StaticNoiseInjection, WhiteNoiseInjection
    export ModelFunction, QuantumDot2
    export ExactTomography, FullObservation, NormalisedObservation,
        MinimalObservation, ObservationFunction, UnitaryTomography
    export DenseGateFidelity, NormalisedReward, RewardFunction,
        RobustGateFidelity, SparseGateFidelity
    export QuantumControlEnvironment, step!
    export reset!

    # Agent exports
    export GPT
    export Memory, ReplayBuffer
    export Agent, SACAgent, SACNetworks, SACParameters, evaluation_steps!,
        get_action, get_random_action, learn!, trainer_steps!
end
