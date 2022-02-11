module EncoderModel

using FlavellBase, GalacticOptim, NLopt, ForwardDiff, StatsBase, Statistics
include("model.jl")
include("fit.jl")

export 
    # fit.jl
    fit_model_glopt_bound,
    fit_model_glopt_bound_reg,
    fit_model_nlopt_bound,
    fit_model_nlopt_bound_reg,

    # model.jl
    ewma,
    ModelEncoder,
    n_ps,
    generate_model_f!,
    init_model_ps!,
    lesser,

    # nl5
    generate_model_nl5,
    generate_model_nl5_partial,
    init_ps_model_nl5,
    
    # nl6
    generate_model_nl6,
    generate_model_nl6_partial,
    init_ps_model_nl6,
    generate_model_nl6a,
    init_ps_model_nl6a,
    generate_model_nl6b,
    init_ps_model_nl6b,
    generate_model_nl6c,
    init_ps_model_nl6c,
    generate_model_nl6d,
    init_ps_model_nl6d,
    ModelEncoderNL6,
    ModelEncoderNL6a,
    ModelEncoderNL6b,
    ModelEncoderNL6c,
    ModelEncoderNL6d

end
