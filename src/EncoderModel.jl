module EncoderModel

using NLopt, ForwardDiff, StatsBase, Statistics, AnalysisBase
include("model.jl")

export 
    # model.jl
    lesser,
    ewma,
    generate_model_nl3_vhp,
    fit_model_bound_nlopt
end # module
