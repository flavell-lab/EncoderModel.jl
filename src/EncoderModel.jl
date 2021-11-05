module EncoderModel

using NLopt, ForwardDiff, StatsBase
include("model.jl")

export 
    # model.jl
    lesser,
    ewma,
    generate_model_nl3_vhp,
    fit_model_bound_nlopt,
    cost_rss,
    cost_abs,
    cost_mse,
    cost_cor,
    reg_var_L1,
    reg_var_L2
end # module
