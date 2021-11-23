module EncoderModel

using NLopt, ForwardDiff, StatsBase, Statistics, AnalysisBase
include("model.jl")

export 
    # model.jl
    ewma,
    generate_model_nl5,
    generate_model_nl5_partial,

    # cost.jl
    cost_rss,
    cost_abs,
    cost_mse,
    cost_cor,
    reg_var_L1,
    reg_var_L2,
    reg_L1,
    reg_L2,
    
    # fit.jl
    fit_model_glopt_bound,
    fit_model_glopt_bound_reg,
    fit_model_nlopt_bound,
    fit_model_nlopt_bound_reg
end
