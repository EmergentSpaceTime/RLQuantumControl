# """Agent module containing neural networks, algorithms, and memory sub-modules.
# """
module Agents
    using Flux: Dense, Chain, glorot_uniform, relu, Dropout, LayerNorm,
    GRUv3Cell, MultiHeadAttention, gelu, Embedding
    using Functors: @functor
    using NNlib: make_causal_mask

    include("networks/base.jl")
    include("networks/gpt.jl")

    export GPT, TransformerBlock
end
