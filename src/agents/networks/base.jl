# """Contains various network architectures."""


# struct DoubleHead{D <: Dense}
#     layer₁::D
#     layer₂::D
# end

# @functor DoubleHead

# function DoubleHead(
#     in::Int,
#     out::Int;
#     init::Function = glorot_uniform,
#     rng::AbstractRNG = default_rng(),
# )
#     layer₁ = Dense(in, out; init=init(rng))
#     layer₂ = Dense(in, out; init=init(rng))
#     return DoubleHead(layer₁, layer₂)
# end

# function (m::DoubleHead)(x::AbstractVecOrMat{Float32})
#     return m.layer₁(x), m.layer₂(x)
# end


# # struct ActorCriticNetwork{
# #     C <: Chain, P <: Union{DoubleHead, Dense}, V <: Dense}
# #     common_layers::C
# #     policy_layer::P
# #     value_layer::V
# # end

# # @functor ActorCriticNetwork

# # function ActorCriticNetwork(
# #     continuous::Bool,
# #     dₒ::Int,
# #     dₐ::Int,
# #     hiddens::Vector{Int},
# #     recurrence::Bool;
# #     activation::Function = relu,
# #     init::Function = glorot_uniform,
# #     rng::AbstractRNG = default_rng(),
# # )
# #     common_layers = _create_ffn(
# #         dₒ,
# #         hiddens;
# #         recurrence=recurrence,
# #         activation=activation,
# #         init=init,
# #         rng=rng,
# #     )
# #     if continuous
# #         policy_layer = DoubleHead(hiddens[end], dₐ; init=init, rng=rng)
# #     else
# #         policy_layer = Dense(hiddens[end], dₐ; init=init(rng))
# #     end
# #     value_layer = Dense(hiddens[end], 1; init=init(rng))
# #     return ActorCriticNetwork(common_layers, policy_layer, value_layer)
# # end

# # function (
# #     m::ActorCriticNetwork{<:Chain{<:Tuple{<:Dense, <:Dense, Vararg{Any}}}}
# # )(observations::VecOrMat)
# #     commons = m.common_layers(observations)
# #     return m.policy_layer(commons), m.value_layer(commons)
# # end

# # function get_π̃(
# #     m::ActorCriticNetwork{<:Chain{<:Tuple{<:Dense, <:Dense, Vararg{Any}}}},
# #     observations::VecOrMat,
# # )
# #     commons = m.common_layers(observations)
# #     return m.policy_layer(commons)
# # end

# # function (
# #     m::ActorCriticNetwork{<:Chain{<:Tuple{<:Dense, <:GRUv3Cell, Vararg{Any}}}}
# # )(cell_states::Matrix, observations::VecOrMat)
# #     hidden = m.common_layers.layers[1](observations)
# #     _, hidden = m.common_layers.layers[2](cell_states, hidden)
# #     for layer in m.common_layers.layers[3 : end]
# #         hidden = layer(hidden)
# #     end
# #     return m.policy_layer(hidden), m.value_layer(hidden)
# # end

# # function get_π̃(
# #     m::ActorCriticNetwork{<:Chain{<:Tuple{<:Dense, GRUv3Cell, Vararg{Any}}}},
# #     cell_states::Matrix,
# #     observations::VecOrMat,
# # )
# #     hidden = m.common_layers.layers[1](observations)
# #     cell_states, hidden = m.common_layers.layers[2](cell_states, hidden)
# #     for layer in m.common_layers.layers[3 : end]
# #         hidden = layer(hidden)
# #     end
# #     return cell_states, m.policy_layer(hidden)
# # end


# # struct DualNetworksArchitecture{AC <: ActorCriticNetwork, V <: Chain}
# #     actor_critic::AC
# #     value_layers::V
# # end

# # @functor DualNetworksArchitecture

# # function DualNetworksArchitecture(
# #     continuous::Bool,
# #     dₒ::Int,
# #     dₐ::Int,
# #     hiddens::Vector{Int},
# #     recurrence::Bool;
# #     activation::Function = relu,
# #     init::Function = glorot_uniform,
# #     rng::AbstractRNG = default_rng(),
# # )
# #     actor_critic = ActorCriticNetwork(
# #         continuous,
# #         dₒ,
# #         dₐ,
# #         hiddens,
# #         recurrence;
# #         activation=activation,
# #         init=init,
# #         rng=rng,
# #     )
# #     value_layers = _create_ffn(
# #         dₒ,
# #         hiddens;
# #         out=1,
# #         recurrence=recurrence,
# #         activation=activation,
# #         init=init,
# #         rng=rng,
# #     )
# #     return DualNetworksArchitecture(actor_critic, value_layers)
# # end

# # function (
# #     m::DualNetworksArchitecture{
# #         <:ActorCriticNetwork{<:Chain{<:Tuple{<:Dense, <:Dense, Vararg{Any}}}}
# #     }
# # )(observations::VecOrMat)
# #     π̃, Vₚ = m.actor_critic(observations)
# #     Vᵥ = m.value_layers(observations)
# #     return π̃, Vₚ, Vᵥ
# # end

# # function get_π̃(
# #     m::DualNetworksArchitecture{
# #         <:ActorCriticNetwork{<:Chain{<:Tuple{<:Dense, <:Dense, Vararg{Any}}}}
# #     },
# #     observations::VecOrMat,
# # )
# #     return get_π̃(m.actor_critic, observations)
# # end

# # function (
# #     m::DualNetworksArchitecture{
# #         <:ActorCriticNetwork{
# #             <:Chain{<:Tuple{<:Dense, <:GRUv3Cell, Vararg{Any}}}
# #         }
# #     }
# # )(cell_states::Matrix, observations::VecOrMat)
# #     π̃, Vₚ = m.actor_critic(cell_states, observations)
# #     hidden = m.value_layers.layers[1](observations)
# #     _, hidden = m.value_layers.layers[2](cell_states, hidden)
# #     for layer in m.value_layers.layers[3 : end - 1]
# #         hidden = layer(hidden)
# #     end
# #     Vᵥ = m.value_layers.layers[end](hidden)
# #     return π̃, Vₚ, Vᵥ
# # end

# # function get_π̃(
# #     m::DualNetworksArchitecture{
# #         <:ActorCriticNetwork{
# #             <:Chain{<:Tuple{<:Dense, <:GRUv3Cell, Vararg{Any}}}
# #         }
# #     },
# #     cell_states::Matrix,
# #     observations::VecOrMat,
# # )
# #     return get_π̃(m.actor_critic, cell_states, observations)
# # end


# # function get_Vᵥ(
# #     m::Chain{<:Tuple{<:Dense, <:GRUv3Cell, Vararg{Any}}},
# #     cell_states::Matrix,
# #     observations::VecOrMat,
# # )
# #     hidden = m.layers[1](observations)
# #     _, hidden = m.layers[2](cell_states, hidden)
# #     for layer in m.layers[3 : end - 1]
# #         hidden = layer(hidden)
# #     end
# #     Vᵥ = m.layers[end](hidden)
# #     return Vᵥ
# # end


# struct SACNetworks{P <: Chain, Q <: Chain, A <: AbstractArray{Float32}}
#     policy_layers::P
#     Q₁_layers::Q
#     Q₁_target_layers::Q
#     Q₂_layers::Q
#     Q₂_target_layers::Q
#     logα::A
# end

# @functor SACNetworks

# function SACNetworks(
#     continuous::Bool,
#     dₒ::Int,
#     dₐ::Int,
#     qₙ::Int,
#     hiddens::Vector{Int},
#     dropout::Float32,
#     layer_norm::Bool;
#     activation::Function = relu,
#     init::Function = glorot_uniform,
#     rng::AbstractRNG = default_rng(),
# )
#     policy_layers = _create_ffn(
#         dₒ,
#         hiddens;
#         out=dₐ,
#         double_out=continuous,
#         activation=activation,
#         init=init,
#         rng=rng,
#     )
#     Q₁_layers = _create_ffn(
#         dₒ + dₐ,
#         hiddens;
#         out=qₙ,
#         dropout=iszero(dropout) ? nothing : dropout,
#         layer_norm=layer_norm,
#         activation=activation,
#         init=init,
#         rng=rng,
#     )
#     Q₂_layers = _create_ffn(
#         dₒ + dₐ,
#         hiddens;
#         out=qₙ,
#         dropout=iszero(dropout) ? nothing : dropout,
#         layer_norm=layer_norm,
#         activation=activation,
#         init=init,
#         rng=rng,
#     )
#     logα = zeros(Float32)
#     return SACNetworks(
#         policy_layers,
#         Q₁_layers,
#         deepcopy(Q₁_layers),
#         Q₂_layers,
#         deepcopy(Q₂_layers),
#         logα,
#     )
# end

# function _create_ffn(
#     in::Int,
#     hiddens::Vector{Int};
#     out::Union{Nothing, Int} = nothing,
#     double_out::Bool = false,
#     dropout::Union{Nothing, Float32} = nothing,
#     layer_norm::Bool = false,
#     recurrence::Bool = false,
#     activation::Function = relu,
#     init::Function = glorot_uniform,
#     rng::AbstractRNG = default_rng(),
# )
#     if isnothing(out)
#         final_layer = []
#     else
#         if double_out
#             final_layer = [DoubleHead(hiddens[end], out; init=init, rng=rng)]
#         else
#             final_layer = [Dense(hiddens[end], out; init=init(rng))]
#         end
#     end
#     if recurrence
#         intermediate_layers = vcat(
#             isnothing(dropout) ? [] : Dropout(dropout; rng=rng),
#             !layer_norm ? [] : LayerNorm(hiddens[1]),
#             isnothing(dropout) & !layer_norm ? [] : activation,
#             GRUv3Cell(hiddens[1], hiddens[1]; init=init(rng)),
#         )
#     else
#         intermediate_layers = vcat(
#             isnothing(dropout) ? [] : Dropout(dropout; rng=rng),
#             !layer_norm ? [] : LayerNorm(hiddens[1]),
#             isnothing(dropout) & !layer_norm ? [] : activation,
#         )
#     end
#     return Chain(
#         Dense(
#             in,
#             hiddens[1],
#             isnothing(dropout) & !layer_norm ? activation : identity;
#             init=init(rng),
#         ),
#         intermediate_layers...,
#         [
#             layer
#             for i in 2 : length(hiddens)
#             for layer in vcat(
#                 Dense(
#                     hiddens[i - 1],
#                     hiddens[i],
#                     isnothing(dropout) & !layer_norm ? activation : identity;
#                     init=init(rng),
#                 ),
#                 isnothing(dropout) ? [] : Dropout(dropout; rng=rng),
#                 !layer_norm ? [] : LayerNorm(hiddens[i]),
#                 isnothing(dropout) & !layer_norm ? [] : activation,
#             )
#         ]...,
#         final_layer...,
#     )
# end
