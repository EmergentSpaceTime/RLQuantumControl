using HDF5: h5open, h5read
using Plots
using StatsBase: mean, std


gr()
# plotly()
theme(:dao)

# Load data
folder_name = "examples/quantum_dot/data/inputs_plength_srate/"
data_files = filter(x -> contains(x, "_data"), readdir(folder_name))
choices = vcat(unique.(
    collect(eachcol(mapreduce(permutedims, vcat, split.(data_files, "_"))))
))
choices[6]

r = zeros(
    length.(choices)[2:end]...,
    parse(Int, split.(choices[1], "=")[][2]),
)
for (i_1, p_1) in enumerate(choices[2])
    for (i_2, p_2) in enumerate(choices[3])
        for (i_3, p_3) in enumerate(choices[4])
            for (i_4, p_4) in enumerate(choices[5])
                for (i_5, p_5) in enumerate(choices[6])
                    try
                        file_name = (
                            folder_name
                            * choices[1][1]
                            * "_"
                            * p_1
                            * "_"
                            * p_2
                            * "_"
                            * p_3
                            * "_"
                            * p_4
                            * "_"
                            * p_5
                        )
                        h5open(file_name) do file
                            r[i_1, i_2, i_3, i_4, i_5, :] = file["r"][]
                        end
                    catch ErrorException
                        println(ErrorException)
                        continue
                    end
                end
            end
        end
    end
end

mean_r = dropdims(
    mean(
        cat(
            (r[:, :, :, :, i, :] for i in axes(r, ndims(r) - 1))...;
            dims=ndims(r) - 1,
        );
        dims=1
    );
    dims=1,
)
std_r = dropdims(
    std(
        cat(
            (r[:, :, :, :, i, :] for i in axes(r, ndims(r) - 1))...;
            dims=ndims(r) - 1,
        );
        dims=1
    );
    dims=1,
)
max_r = dropdims(
    maximum(
        cat(
            (r[:, :, :, :, i, :] for i in axes(r, ndims(r) - 1))...;
            dims=ndims(r) - 1,
        );
        dims=1
    );
    dims=1,
)
min_r = dropdims(
    minimum(
        cat(
            (r[:, :, :, :, i, :] for i in axes(r, ndims(r) - 1))...;
            dims=ndims(r) - 1,
        );
        dims=1
    );
    dims=1,
)
choices
# Plot reward for experiment
r_plots = [
    plot(
        axes(mean_r, ndims(mean_r)),
        transpose(mean_r[:, i, j, :]);
        labels=reshape(choices[3], 1, :),
    )
    for i in 1:4
    for j in 1:2
]
# r_plots = [
#     plot(
#         axes(mean_r, ndims(mean_r)),
#         transpose(
#             mean_r[
#                 vcat(
#                     [
#                         i == j ? (1:size(mean_r, i)) : 1
#                         for i in 1 : ndims(mean_r) - 1
#                     ],
#                     [1:size(mean_r, ndims(mean_r))],
#                 )...
#             ]
#         );
#         labels=reshape(choices[2 + j], 1, :),
#         legend=:topleft,
#     )
#     for j in 1 : ndims(mean_r) - 1
# ]
plot(
    r_plots...;
    size=(1800, 600),
    layout=(4, 2),
    ylabel=["" "" "" ""],
    xlabel=["" "" "" ""],
    legend=reshape([i == 1 ? true : false for i in 1:8], 1, :),
)


# Plot final rewards for each experiment
last_r = mean(mean_r[:, :, :, end - 100 : end]; dims=ndims(mean_r))
