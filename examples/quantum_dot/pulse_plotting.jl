using Plots
using BSON: load
using NPZ
using HDF5

include("../../src/RLQuantumControl.jl")
using .RLQuantumControl


plotly()
theme(:dao)



# function pulse_plotter(base_file_name::String) end


########
# Main #
########
# Loading agent and environment
# base_file_name = (
#     "data/experiments/protocol_length_and_inputs/qdot_"
#     * "episodes-10000_seed=551_inputs=45_shaping=none_pulse_type=none"
# )
# agent = load(base_file_name * "_agent.bson")[:agent]
# env = load(base_file_name * "_environment.bson")[:env]

agent = load(
    "examples/quantum_dot/data/plength_shaping_srate_noises/episodes=30000_seed=5771_shaping=gauss_plength=30_srate=10_agent.bson"
)[:agent]
env = load(
    "examples/quantum_dot/data/plength_shaping_srate_noises/episodes=30000_seed=5771_shaping=gauss_plength=30_srate=10_environment.bson"
)[:env]
env.model_function.H_drift

maximum(agent.memory.rewards)
max_r_index = argmax(agent.memory.rewards)
best_actions = agent.memory.actions[
    :, max_r_index - env.model_function.tₙ + 1 : max_r_index
]


f = h5open("pulse.h5", "w")
fset = create_dataset(f, "pulse", eltype(best_actions), size(best_actions))
write(f["pulse"], best_actions)
HDF5.close(f)

best_actions
r = 0
reset!(env)
for i in 1:env.model_function.tₙ
    _, _, r = step!(env, Float64.(best_actions[:, i]))
end
env.shaping_function.shaped_pulse_history
r
env._stateₘ
plot(env.shaping_function.kernel(0:0.1:10))

######
# Get Unitary #
#####
U = Matrix{ComplexF64}(I, 6, 6)
epsil = exp.(env.shaping_function.shaped_pulse_history)
U_list = zeros(ComplexF64, size(epsil, 2) + 1, 6, 6)

U_list[1, :, :] .= U
for i in axes(epsil, 2)
    U .= env.model_function(epsil[:, i]) * U
    U_list[i + 1, :, :] .= U
end
npzwrite(
    "data/experiments/protocol_length_and_inputs/paths/unitaries_15_1.npy",
    U_list,
)


plot(
    reverse(
        [
            plot(
                LinRange(0, 50, 45),
                [
                    exp.(env.shaping_function.pulse_history[i, :]),
                ];
                legend=:none,
                # labels=["1" "2"],
                # legend=i == 1 ? :top : nothing,
                grid=false,
                linetype=:steppost,
                c=[RGB(55 / 255, 113 / 255, 185 / 255) RGB(219 / 255, 91 / 255, 49 / 255)],
            )
            for i in 1:3
        ]
    )...;
    layout=(3, 1),
    ylabel=["J₃₄[ns⁻¹]" "J₂₃[ns⁻¹]" "J₁₂[10⁻²ns⁻¹]"],
    ytickfont=font(36, "Helvetica"),
    yticks=[[0, 5, 10] [0, 5, 10] [0, 1e-2, 2e-2]],
    yformatter=[y -> y y -> y y -> string(Int(y / 1e-2))],
    yguidefont=font(36, "Helvetica"),
    xticks=[false false true],
    xlabel=["" "" "t [ns]"],
    xtickfont=font(36, "Helvetica"),
    xguidefont=font(36, "Helvetica"),
    size=(1800, 800),
    lw=4,
    # labels=[["1" "2"] nothing nothing],
    # legend=[:right nothing nothing],
    # framewidth=10
    left_margin = 10 * Plots.mm,
    framestyle=:box,
    # thickness_scaling=1,
)


plot(
    reverse(
        [
            plot(
                LinRange(0, 50, 50 * env.shaping_function.sampling_rate),
                [
                    exp.(env.shaping_function.pulse_history[i, 51 : end]),
                    3 .* exp.(env.shaping_function.shaped_pulse_history[i, :]),
                ];
                legend=:none,
                # labels=["1" "2"],
                # legend=i == 1 ? :top : nothing,
                grid=false,
                c=[RGB(55 / 255, 113 / 255, 185 / 255) RGB(219 / 255, 91 / 255, 49 / 255)],
            )
            for i in 1:3
        ]
    )...;
    layout=(3, 1),
    ylabel=["J₃₄[ns⁻¹]" "J₂₃[ns⁻¹]" "J₁₂[ns⁻¹]"],
    ytickfont=font(36, "Helvetica"),
    yticks=[[0, 5, 10] [0, 1, 2] [0, 5, 10]],
    yguidefont=font(36, "Helvetica"),
    xticks=[false false true],
    xlabel=["" "" "t [ns]"],
    xtickfont=font(36, "Helvetica"),
    xguidefont=font(36, "Helvetica"),
    size=(1800, 800),
    lw=4,
    # labels=[["1" "2"] nothing nothing],
    # legend=[:right nothing nothing],
    # framewidth=10
    left_margin = 10 * Plots.mm,
    framestyle=:box,
    # thickness_scaling=1,
)

plot(
    [
        plot(
            eachindex(Jᵢⱼ[i, :]) .- 1,
            Jᵢⱼ[i, :];
            linetype=:steppre,
            legend=false,
            color=RGB(25 / 255, 116 / 255, 191 / 255),
            lw=2.0,
        )
    for i in axes(Jᵢⱼ, 1)
    ]...;
    layout=(4, 1),
    xlabel=["" "" "" "𝑡 (1 ns)"],
    xtickfont=font(10, "Helvetica"),
    xguidefont=font(15, "Helvetica"),
    ylabel=["X₂" "X₂" "Y₁" "Y₂"],
    yguidefont=font(11, "Helvetica"),
    # yticks=[true true true],
    ytickfont=font(8, "Helvetica"),
    grid=false,
    xticks=[false false false true],
    framestyle=:box,
    xlims=(0, env.model.tₙ),
    size=(600, 400),
)

# Seeing best action and trying it on environment many times
































maximum(agent.memory.rewards)
max_r_index = argmax(agent.memory.rewards)
best_actions = agent.memory.actions[
    :, max_r_index - env.model_function.tₙ + 1 : max_r_index
]
rewards = zeros(100)

reset!(env)
for i in 1:env.model_function.tₙ
    _, _, r = step!(env, Float64.(best_actions[:, i]))
end
x = LinRange(0, 50, 20 * 10)
plot(x, repeat(env.shaping_function.pulse_history[1, :]; inner=10); linetype=:steppre)
plot!(x, env.shaping_function.shaped_pulse_history[1, :])


for i in eachindex(rewards)
    reset!(env)

    r = (0.0, 0.0)
    for i in 1:env.model_function.tₙ
        _, _, r = step!(env, Float64.(best_actions[:, i]))
    end
    rewards[i] = r[1]
end

plot(rewards, ylabel="Reward", xlabel="Experiment Number", legend=:none)
hline!([Float64(agent.memory.rewards[max_r_index])])

# Comparing iterations of environement (e.g. with different noises)
actions = zeros(2, size(best_actions)...)
shaped_actions = zeros(
    2,
    size(best_actions, 1),
    size(best_actions, 2) * env.shaping_function.sampling_rate,
)

actions = zeros(2,  size(best_actions, 1),
(size(best_actions, 2) + 10) * env.shaping_function.sampling_rate,
)
shaped_actions = zeros(
    2,
    size(best_actions, 1),
    (size(best_actions, 2) + 5) * env.shaping_function.sampling_rate,
)
for i in 1:2
    # evaluation steps!
    observation = reset!(env)
    reset!(env.model_function)
    done = false
    ∑ᵣ = 0.0
    while !done
        action = get_action(agent, observation)
        observation, done, reward = step!(env, action)
        ∑ᵣ += reward[1]
    end
    println(∑ᵣ)


    # iteration = env.shaping_function.shaped_pulse_history
    # for j in axes(iteration, 2)
        # @. iteration[:, j] += env.pulse_function[1]._noise_episode
        # @. iteration[:, j] += env.pulse_function[2]._noises_episode[:, j]
        # @. iteration[:, j] = exp(iteration[:, j])
    # end

    actions[i, :, :] = env.shaping_function.pulse_history
    shaped_actions[i, :, :] = env.shaping_function.shaped_pulse_history
end
env.model_function.tₙ
plotts = [
    plot(
        LinRange(0, 50, size(shaped_actions, 3)),
        [
            # repeat(exp.(actions[1, 3, :]); inner=env.shaping_function.sampling_rate),
            exp.(shaped_actions[1, i, :]),
            exp.(shaped_actions[2, i, :]),
        ];
        legend=:none,
        size=(1000, 500),
        c=[RGB(219 / 255, 91 / 255, 49 / 255) :steelblue],
        ylabel="J₂₃[ns⁻¹]",
        ytickfont=font(36, "Helvetica"),
        # yticks=[[0, 5, 10] [0, 1, 2] [0, 5, 10]],
        yguidefont=font(36, "Helvetica"),
        # xticks=[false false true],
        xlabel="t [ns]",
        xlims=(0, 52),
        lw=4,
        grid=false,
        xtickfont=font(36, "Helvetica"),
        xguidefont=font(36, "Helvetica"),
        # lw=4,
    ) for i in 2:2
]
plot(plotts...)