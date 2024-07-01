struct DoubleHead{D <: Dense}
    layer_1::D
    layer_2::D
end

@layer DoubleHead

function DoubleHead(
    in_dim::Int,
    out_dim::Int;
    bias::Bool = true,
    init::Function = glorot_normal,
)
    layer_1 = Dense(in_dim, out_dim; bias=bias, init=init)
    layer_2 = Dense(in_dim, out_dim; bias=bias, init=init)
    return DoubleHead(layer_1, layer_2)
end

(l::DoubleHead)(x::AbstractVecOrMat{Float32}) = l.layer_1(x), l.layer_2(x)

function Base.show(io::IO, l::DoubleHead)
    print(io, "DoubleHead(")
    print(io,  size(l.layer_1.weight, 2), " => ", size(l.layer_1.weight, 1))
    l.layer_1.bias == false && print(io, "; bias=false")
    print(io, ")")
    return nothing
end


function _create_ffn(
    in_dim::Int,
    out_dim::Union{Nothing, Int} = nothing,
    double_out::Bool = false;
    hidden_dims::Vector{Int} = [128, 128],
    dropout::Union{Nothing, Float32} = nothing,
    layer_norm::Bool = false,
    activation::Function = relu,
    bias::Bool = true,
    init::Function = glorot_normal,
)
    if isnothing(out_dim)
        final_layer = []
    else
        if double_out
            final_layer = [
                DoubleHead(hidden_dims[end], out_dim; bias=bias, init=init)
            ]
        else
            final_layer = [
                Dense(hidden_dims[end], out_dim; bias=bias, init=init)
            ]
        end
    end
    return Chain(
        [
            layer
            for i in 2 : length(vcat(in_dim, hidden_dims))
            for layer in vcat(
                Dense(
                    vcat(in_dim, hidden_dims)[i - 1],
                    vcat(in_dim, hidden_dims)[i];
                    bias=bias,
                    init=init,
                ),
                isnothing(dropout) ? [] : Dropout(dropout),
                !layer_norm ? [] : CustomLayerNorm(
                    vcat(in_dim, hidden_dims)[i]; bias=bias
                ),
                activation,
            )
        ]...,
        final_layer...,
    )
end
