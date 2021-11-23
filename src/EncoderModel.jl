module EncoderModel

using FlavellBase, GalacticOptim, NLopt, ForwardDiff, StatsBase, Statistics
include("model.jl")
include("fit.jl")

export 
    # model.jl
    ewma,
    generate_model_nl5,
    generate_model_nl5_partial,
    init_ps_model5,

    # fit.jl
    fit_model_glopt_bound,
    fit_model_glopt_bound_reg,
    fit_model_nlopt_bound,
    fit_model_nlopt_bound_reg
end
