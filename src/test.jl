using JLD2



load_path = "Results\\SimData\\MBB_sym_2026-01-06_s300r3mHa1b1cMd1l1.jld2"

@load load_path sim_results

sim_results.sim_times_conv_iter

sim_results.simulation_times