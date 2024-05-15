struct DoubleHead{D <: Dense}
    layer_1::D
    layer_2::D
end

@layer DoubleHead

function DoubleHead(
    in::Int,
    out::Int;
    init::Function = glorot_normal,
    rng::AbstractRNG = default_rng(),
)
    layer_1 = Dense(in, out; init=init(rng))
    layer_2 = Dense(in, out; init=init(rng))
    return DoubleHead(layer_1, layer_2)
end

function (m::DoubleHead)(x::AbstractVecOrMat{Float32})
    return m.layer_1(x), m.layer_2(x)
end


struct SACNetworks{P <: Chain, Q <: Chain, A <: AbstractArray{Float32}}
    policy_layers::P
    Q_1_layers::Q
    Q_1_target_layers::Q
    Q_2_layers::Q
    Q_2_target_layers::Q
    logα::A
end

@layer SACNetworks

function SACNetworks(
    continuous::Bool,
    observation_dim::Int,
    action_dim::Int,
    n_q::Int,
    hiddens::Vector{Int},
    dropout::Float32,
    layer_norm::Bool;
    activation::Function = relu,
    init::Function = glorot_normal,
    rng::AbstractRNG = default_rng(),
)
    policy_layers = _create_ffn(
        observation_dim,
        hiddens;
        out=action_dim,
        double_out=continuous,
        activation=activation,
        init=init,
        rng=rng,
    )
    Q_1_layers = _create_ffn(
        observation_dim + action_dim,
        hiddens;
        out=n_q,
        dropout=iszero(dropout) ? nothing : dropout,
        layer_norm=layer_norm,
        activation=activation,
        init=init,
        rng=rng,
    )
    Q_2_layers = _create_ffn(
        observation_dim + action_dim,
        hiddens;
        out=n_q,
        dropout=iszero(dropout) ? nothing : dropout,
        layer_norm=layer_norm,
        activation=activation,
        init=init,
        rng=rng,
    )
    logα = zeros(Float32)
    return SACNetworks(
        policy_layers,
        Q_1_layers,
        deepcopy(Q_1_layers),
        Q_2_layers,
        deepcopy(Q_2_layers),
        logα,
    )
end

function _create_ffn(
    in::Int,
    hiddens::Vector{Int};
    out::Union{Nothing, Int} = nothing,
    double_out::Bool = false,
    dropout::Union{Nothing, Float32} = nothing,
    layer_norm::Bool = false,
    recurrence::Bool = false,
    activation::Function = relu,
    init::Function = glorot_normal,
    rng::AbstractRNG = default_rng(),
)
    if isnothing(out)
        final_layer = []
    else
        if double_out
            final_layer = [DoubleHead(hiddens[end], out; init=init, rng=rng)]
        else
            final_layer = [Dense(hiddens[end], out; init=init(rng))]
        end
    end
    if recurrence
        intermediate_layers = vcat(
            isnothing(dropout) ? [] : Dropout(dropout; rng=rng),
            !layer_norm ? [] : CustomLayerNorm(hiddens[1]),
            isnothing(dropout) & !layer_norm ? [] : activation,
            GRUv3Cell(hiddens[1], hiddens[1]; init=init(rng)),
        )
    else
        intermediate_layers = vcat(
            isnothing(dropout) ? [] : Dropout(dropout; rng=rng),
            !layer_norm ? [] : CustomLayerNorm(hiddens[1]),
            isnothing(dropout) & !layer_norm ? [] : activation,
        )
    end
    return Chain(
        Dense(
            in,
            hiddens[1],
            isnothing(dropout) & !layer_norm ? activation : identity;
            init=init(rng),
        ),
        intermediate_layers...,
        [
            layer
            for i in 2 : length(hiddens)
            for layer in vcat(
                Dense(
                    hiddens[i - 1],
                    hiddens[i],
                    isnothing(dropout) & !layer_norm ? activation : identity;
                    init=init(rng),
                ),
                isnothing(dropout) ? [] : Dropout(dropout; rng=rng),
                !layer_norm ? [] : CustomLayerNorm(hiddens[i]),
                isnothing(dropout) & !layer_norm ? [] : activation,
            )
        ]...,
        final_layer...,
    )
end
