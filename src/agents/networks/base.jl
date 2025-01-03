struct DoubleHead{D <: Dense}
    layer_1::D
    layer_2::D
end

@layer :noexpand DoubleHead

function DoubleHead(
    in::Int,
    out::Int;
    bias::Bool = true,
    init::Function = glorot_normal,
    rng::AbstractRNG = default_rng(),
)
    layer_1 = Dense(in, out; bias=bias, init=init(rng))
    layer_2 = Dense(in, out; bias=bias, init=init(rng))
    return DoubleHead(layer_1, layer_2)
end

function (m::DoubleHead)(x::AbstractVecOrMat{Float32})
    return m.layer_1(x), m.layer_2(x)
end

function Base.show(io::IO, m::DoubleHead)
    print(io, "DoubleHead(")
    print(io, join(m.layer_1.weight.size, ", "))
    m.layer_1.bias == false && print(io, "; bias=false")
    print(io, ")")
    return nothing
end


function _create_ffn(
    in::Int,
    hiddens::Vector{Int};
    out::Union{Nothing, Int} = nothing,
    double_out::Bool = false,
    dropout::Union{Nothing, Float32} = nothing,
    layer_norm::Bool = false,
    activation::Function = relu,
    dense_bias::Bool = true,
    ln_bias::Bool = true,
    init::Function = glorot_normal,
    rng::AbstractRNG = default_rng(),
)
    if isnothing(out)
        final_layer = []
    else
        if double_out
            final_layer = [
                DoubleHead(
                    hiddens[end],
                    out;
                    bias=dense_bias,
                    init=init,
                    rng=rng,
                )
            ]
        else
            final_layer = [
                Dense(hiddens[end], out; bias=dense_bias, init=init(rng))
            ]
        end
    end
    intermediate_layers = vcat(
        isnothing(dropout) ? [] : Dropout(dropout; rng=rng),
        !layer_norm ? [] : CustomLayerNorm(hiddens[1]; bias=ln_bias),
        isnothing(dropout) & !layer_norm ? [] : activation,
    )
    return Chain(
        Dense(
            in,
            hiddens[1],
            isnothing(dropout) & !layer_norm ? activation : identity;
            bias=dense_bias,
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
                    bias=dense_bias,
                    init=init(rng),
                ),
                isnothing(dropout) ? [] : Dropout(dropout; rng=rng),
                !layer_norm ? [] : CustomLayerNorm(hiddens[i]; bias=ln_bias),
                isnothing(dropout) & !layer_norm ? [] : activation,
            )
        ]...,
        final_layer...,
    )
end
