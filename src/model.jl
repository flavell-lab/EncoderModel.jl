zstd = FlavellBase.standardize
logistic(x,x0,k) = 1 / (1 + exp(-k * (x - x0)))
leaky_logistic(x,x0,k,m) = logistic(x,x0,k) + m * (x - x0)
lesser(x,x0) = leaky_logistic(x,x0,50,1e-3)

function ewma(λ::T, x, idx_splits) where {T}
    return_array = zeros(T, idx_splits[end][end])
    for split = idx_splits
        max_t = length(split)
        x_avg = zeros(T, max_t)
        s = sum([exp(-(max_t-t)*λ) for t=1:max_t])
        x_avg[1] = x[split[1]] / s
        for t=2:max_t
            t_x = split[1] + t -1
            x_avg[t] = T(x_avg[t-1] * (s-1) / s + x[t_x] / s)
        end
        return_array[split] = x_avg
    end

    return return_array
end

include("model/model_nl5.jl")
include("model/model_nl6.jl")