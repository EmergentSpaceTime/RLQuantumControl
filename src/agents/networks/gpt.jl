# struct _TransformerBlock{
#     CLN <: CustomLayerNorm,
#     MHA <: MultiHeadAttention,
#     D <: Dropout,
#     FFN <: Chain,
# }
#     layer_norm_pre::CLN
#     multi_head_attention::MHA
#     residual_dropout::D
#     layer_norm_post::CLN
#     feed_forward_network::FFN
# end

# @layer _TransformerBlock

# function _TransformerBlock(
#     ;
#     embedding_dim::Int = 128,
#     n_heads::Int = 4,
#     attention_bias::Bool = false,
#     attention_dropout::T = 0.0f0,
#     residual_dropout::T = 0.0f0,
#     intermediate_dim::Int = 4 * embedding_dim,
#     ffn_activation::Function = gelu,
#     ffn_bias::Bool = true,
#     ln_bias::Bool = true,
#     init::Function = glorot_normal,
# ) where {T <: AbstractFloat}
#     return _TransformerBlock(
#         CustomLayerNorm(embedding_dim; bias=ln_bias, eps=convert(T, 1f-5)),
#         MultiHeadAttention(
#             embedding_dim;
#             nheads=n_heads,
#             bias=attention_bias,
#             init=init,
#             dropout_prob=attention_dropout,
#         ),
#         Dropout(residual_dropout),
#         CustomLayerNorm(embedding_dim; bias=ln_bias, eps=convert(T, 1f-5)),
#         Chain(
#             Dense(embedding_dim, intermediate_dim; init=init, bias=ffn_bias),
#             ffn_activation,
#             Dense(intermediate_dim, embedding_dim; init=init, bias=ffn_bias),
#             Dropout(residual_dropout),
#         ),
#     )
# end

# function (m::_TransformerBlock)(
#     q::A, kv::A = q; mask::AbstractMatrix{Bool}
# ) where {A <: AbstractArray{<:AbstractFloat, 3}}
#     a, _ = m.multi_head_attention(
#         m.layer_norm_pre(q), m.layer_norm_pre(kv); mask=mask
#     )
#     a = m.residual_dropout(a)
#     a += q
#     a += m.feed_forward_network(m.layer_norm_post(a))
#     return a
# end


# struct GPT{
#     TE <: Union{Embedding, Chain},
#     PE <: Embedding,
#     D <: Dropout,
#     TB <: Chain{<:Tuple{Vararg{<:_TransformerBlock}}},
#     CLN <: CustomLayerNorm
# }
#     token_embedding::TE
#     positional_embedding::PE
#     embedding_dropout::D
#     transformer_blocks::TB
#     final_layer_norm::CLN
# end

# @layer GPT

# """
#     GPT(
#         observation_dim::Int,
#         continuous_observation::Bool,
#         sequence_length::Int;
#         kwargs...
#     )

# GPT model for partially observable quantum environments.

# Transformer model based on GPT-2 [radford2019language](@cite) which allows
# including a memory of past actions and observations when an agent has limited
# information of the model. For state control, measurement outcomes are single
# outcomes of a operator, whereas for gate control, measurement corresponds to a
# partial or full tomography of the system.

# Args:
#   * observation_dim: Dimension of the observation space.
#   * continuous_observation: Whether the observation space is discrete (typically
#         for state control (single-shot experiments)) or continuous
#         (corresponding to a partial / noisy tomography).
#   * sequence_length: Length of the sequence of observations (number of discrete
#         time steps in protocol).

# Kwargs:
#   * embedding_dim: Dimension of the embedding space (default: `128`).
#   * embedding_dropout: Dropout probability for the embedding (default: `0.0f0`).
#   * n_blocks: Number of transformer blocks (default: `2`).
#   * n_heads: Number of attention heads (default: `8`).
#   * attention_bias: Whether to include a bias in the attention layer (default:
#         `false`).
#   * attention_dropout: Dropout probability for the attention layer (default:
#         `0.0f0`).
#   * residual_dropout: Dropout probability for the residual connections (default:
#         `0.0f0`).
#   * intermediate_dim: Dimension of the intermediate layer in the feed-forward
#         network (default: `4 × embedding_dim`).
#   * ffn_activation: Activation function for the feed-forward network (default:
#         [`NNlib.gelu`]()).
#   * ffn_bias: Whether to include a bias in the feed-forward network (default:
#         `true`).
#   * ln_bias: Whether to include a learnable bias in the layer norm layer (
#         default: `true`).
#   * init: Initialisation function for the weights (default:
#         [`NNlib.glorot_normal`]()).

# Fields:
#   * token_embedding: Embedding layer for the tokens.
#   * positional_embedding: Embedding layer for the positions.
#   * embedding_dropout: Dropout layer for the embeddings.
#   * transformer_blocks: Transformer blocks.
#   * final_layer_norm: Layer norm for the final output.
# """
# function GPT(
#     observation_dim::Int,
#     continuous_observation::Bool,
#     sequence_length::Int;
#     embedding_dim::Int = 128,
#     embedding_dropout::T = 0.0f0,
#     n_blocks::Int = 2,
#     n_heads::Int = 8,
#     attention_bias::Bool = false,
#     attention_dropout::T = 0.0f0,
#     residual_dropout::T = 0.0f0,
#     intermediate_dim::Int = 4 * embedding_dim,
#     ffn_activation::Function = gelu,
#     ffn_bias::Bool = true,
#     ln_bias::Bool = true,
#     init::Function = glorot_normal,
# ) where {T <: AbstractFloat}
#     if continuous_observation
#         token_embedding = Chain(
#             Dense(observation_dim, embedding_dim; init=init, bias=ffn_bias),
#             ffn_activation,
#         )
#     else
#         token_embedding = Embedding(observation_dim, embedding_dim; init=init)
#     end
#     transformer_blocks = Chain(
#         [
#             _TransformerBlock(
#                 ;
#                 embedding_dim=embedding_dim,
#                 n_heads=n_heads,
#                 attention_bias=attention_bias,
#                 attention_dropout=attention_dropout,
#                 residual_dropout=residual_dropout,
#                 intermediate_dim=intermediate_dim,
#                 ffn_activation=ffn_activation,
#                 ffn_bias=ffn_bias,
#                 ln_bias=ln_bias,
#                 init=init,
#             )
#             for _ in 1:n_blocks
#         ]...
#     )
#     for block in transformer_blocks  # GPT-2 scaling
#         block.multi_head_attention.out_proj.weight .= (
#             init(embedding_dim, embedding_dim) / sqrt(2 * n_blocks)
#         )
#     end
#     return GPT(
#         token_embedding,
#         Embedding(sequence_length, embedding_dim; init=init),
#         Dropout(embedding_dropout),
#         transformer_blocks,
#         CustomLayerNorm(embedding_dim; bias=ln_bias, eps=convert(T, 1f-5)),
#     )
# end

# """
#     (::GPT)(q, kv = q; mask=make_causal_mask(q))

# Forward pass of the GPT model.

# Args:
#   * q: Observation at time step or set of observations of shape (observation_dim
#     , sequence_length or 1, batch_size).
#   * kv: History of past observations of shape (observation_dim, sequence_length,
#         batch_size) (default: `q`).

# Kwargs:
#   * mask: Mask for the attention layer of shape (sequence_length or 1,
#         sequence_length) (default: [`NNlib.make_causal_mask(q)`]()).

# Returns:
#   * Output of the transformer of shape (embedding_dim, sequence_length or 1,
#         batch_size).
# """
# function (m::GPT)(
#     q::AbstractArray{<:AbstractFloat, 3};
#     mask::AbstractMatrix{Bool} = make_causal_mask(q),
# )
#     x = m.embedding_dropout(
#         m.token_embedding(q) .+ m.positional_embedding(1:size(q, 2))
#     )
#     for block in m.transformer_blocks
#         x = block(x, x; mask=mask)
#     end
#     return m.final_layer_norm(x)
# end

# function decode!(m::GPT, q::A, kv::A = q; mask::AbstractMatrix{Bool} = make_causal_mask(q)) where {A <: AbstractArray{<:AbstractFloat, 3}}
#     # token embedding
#     query = m.embedding_dropout(m.token_embedding(q))
#     key_value = m.embedding_dropout(m.token_embedding(kv))

#     p = m.positional_embedding(1:size(kv, 2))  # positional_embedding

#     println(size(query), size(p), size(key_value))
#     history = key_value .+ p

#     # for block in m.transformer_blocks
#         # t = block(t, history; mask=mask)
#     # end

#     # t = m.final_layer_norm(t)
#     # return #t
#     return history
# end


# m = GPT(76, true, 50)
# q = randn(Float32, 76, 50, 2)
# m.positional_embedding(1:50)
# x = m.embedding_dropout(m.token_embedding(q) .+ m.positional_embedding(1:50))
# mas = make_causal_mask(q)
# m.transformer_blocks[1](x, x; mask=mas)

# kv = randn(Float32, 76, 22, 2)

# q_o = m.token_embedding(q)
# kv_o = m.positional_embedding(1:22) .+ m.token_embedding(kv)

# o, _ = m.transformer_blocks[1].multi_head_attention(q_o, k_o; mask=make_causal_mask(k_o))
# o_2, _ = m.transformer_blocks[1].multi_head_attention(q_o, kv_o)
# size(o_2)

# o == o_2
# o_2
# size(o_2)
