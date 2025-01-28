using LinearAlgebra.BLAS: get_num_threads, set_num_threads

set_num_threads(Base.Threads.nthreads())
println(get_num_threads(), Base.Threads.nthreads())

using Random: seed!

using TOML: parsefile
using BSON: @save, @load
using HDF5: h5open, create_dataset, write, close

include("../../src/RLQuantumControl.jl")
using .RLQuantumControl

#################
# Setup config. #
#################
CONFIG = parsefile(ARGS[1])
SEED = CONFIG["seed"]

# Setup seed.
seed!(SEED + parse(Int, ARGS[2]))

@load CONFIG["save_directory"] * "environment.bson" env
@load CONFIG["save_directory"] * "agent.bson" agent

r, l = learn!(agent, env, false)

@save CONFIG["save_directory"] * "environment.bson" env
@save CONFIG["save_directory"] * "agent.bson" agent

d_file = h5open(CONFIG["save_directory"] * "data=" * ARGS[2] * ".h5", "cw")
fset = create_dataset(d_file, "r", eltype(r), size(r))
fset = create_dataset(d_file, "l", eltype(l), size(l))
write(d_file["r"], r)
write(d_file["l"], l)
close(d_file)
