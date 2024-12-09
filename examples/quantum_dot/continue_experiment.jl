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
LABEL = ["", "", "", "", ""]

# Setup seed and custom labels
seed!(SEED + parse(Int, ARGS[3]))
LABEL[1] = "_seed=" * string(SEED)
LABEL[2] = "_inputs=" * string(CONFIG["inputs"])
LABEL[3] = "_plength=" * string(CONFIG["plength"])
LABEL[4] = "_observation=" * CONFIG["observation"]
# LABEL[5] = "_reward=" * CONFIG["reward"]

prefix = (
    "data/"
    * ARGS[4]
    * "/episodes="
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
