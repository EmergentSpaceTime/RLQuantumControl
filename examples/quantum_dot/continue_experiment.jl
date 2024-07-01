using LinearAlgebra.BLAS: get_num_threads, set_num_threads

set_num_threads(Base.Threads.nthreads())
println(get_num_threads(), Base.Threads.nthreads())

using Random: seed!

using TOML: parsefile
using BSON: @save, @load
using HDF5: h5open, create_dataset, write, close

include("../../src/RLQuantumControl.jl")
using .RLQuantumControl

################
# Setup config #
################
CONFIG = parsefile(ARGS[1])
EPISODES = ARGS[2]
SEED = CONFIG["seed"]
LABEL = ["", "", "", ""]

# Setup seed and custom labels
seed!(SEED + parse(Int, ARGS[3]))
LABEL[1] = "_seed=" * string(SEED)
LABEL[2] = "_ndrift=" * string(CONFIG["noises_drift"])
LABEL[3] = "_nepss=" * string(CONFIG["noises_epsilon_s"])
LABEL[4] = "_nepsf=" * string(CONFIG["noises_epsilon_f"])

#####################
# Setup environment #
#####################
if CONFIG["shaping"] == "none"
    # LABEL[2] = "_shaping=none"
elseif CONFIG["shaping"] == "fir"
    # LABEL[2] = "_shaping=fir"
end

if CONFIG["pulse"] == "none"
elseif CONFIG["pulse"] == "both"
    if CONFIG["reward"] != "robust"
    else
    end
end

if CONFIG["observation"] == "full"
    # LABEL[3] = "_observation=full"
end

if CONFIG["reward"] == "sparse"
    # LABEL[5] = "_reward=sparse"
elseif CONFIG["reward"] == "robust"
    # LABEL[5] = "_reward=robust"
end

prefix = (
    "data/robust_reward_noises/qdot"
    * "_episodes="
    * EPISODES
    * LABEL[1]
    * LABEL[2]
    * LABEL[3]
    * LABEL[4]
)

@load prefix * "_environment.bson" env
@load prefix * "_agent.bson" agent

r, l = learn!(agent, env, false)

@save prefix * "_environment.bson" env
@save prefix * "_agent.bson" agent

d_file = h5open(prefix * "_data_" * ARGS[3] * ".h5", "cw")
fset = create_dataset(d_file, "r", eltype(r), size(r))
fset = create_dataset(d_file, "l", eltype(l), size(l))
write(d_file["r"], r)
write(d_file["l"], l)
close(d_file)
