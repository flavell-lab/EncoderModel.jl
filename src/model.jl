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
        x_avg[1] = T(0.)
        for t=2:max_t
            t_x = split[1] + t -1
            x_avg[t] = T(x_avg[t-1] * (s-1) / s + x[t_x] / s)
        end
        return_array[split] = x_avg
    end
    
    return return_array
end

function generate_model_nl5(xs, idx_splits)
    x1 = xs[1,:] # velocity
    x2 = xs[2,:] # θh
    x3 = xs[3,:] # pumping
    x4 = xs[4,:] # ang vel
    x5 = xs[5,:] #curvature
    
    return function (ps)
        ps[18] .+ ps[17] .* ewma(ps[16],
            (sin(ps[1]) .* x1 .+ cos(ps[1])) .*
            (sin(ps[2]) .* (1 .- 2 .* lesser.(x1, ps[3])) .+ cos(ps[2])) .*

            (sin(ps[4]) .* x2 .+ cos(ps[4])) .*
            (sin(ps[5]) .* (1 .- 2 .* lesser.(x2, ps[6])) .+ cos(ps[5])) .*

            (sin(ps[7]) .* x3 .+ cos(ps[7])) .*
            (sin(ps[8]) .* (1 .- 2 .* lesser.(x3, ps[9])) .+ cos(ps[8])) .*

            (sin(ps[10]) .* x4 .+ cos(ps[10])) .*
            (sin(ps[11]) .* (1 .- 2 .* lesser.(x4, ps[12])) .+ cos(ps[11])) .*

            (sin(ps[13]) .* x5 .+ cos(ps[13])) .*
            (sin(ps[14]) .* (1 .- 2 .* lesser.(x5, ps[15])) .+ cos(ps[14])), idx_splits)
    end
end

function generate_model_nl5_partial(xs, idx_splits, idx_valid)
    x1 = xs[1,:] # velocity
    x2 = xs[2,:] # θh
    x3 = xs[3,:] # pumping
    x4 = xs[4,:] # ang vel
    x5 = xs[5,:] #curvature
    
    return function (ps_::AbstractVector{T}) where T
        ps = zeros(T, 18)
        ps[idx_valid] .= ps_
        ps[18] .+ ps[17] .* ewma(ps[16],
            (sin(ps[1]) .* x1 .+ cos(ps[1])) .*
            (sin(ps[2]) .* (1 .- 2 .* lesser.(x1, ps[3])) .+ cos(ps[2])) .*

            (sin(ps[4]) .* x2 .+ cos(ps[4])) .*
            (sin(ps[5]) .* (1 .- 2 .* lesser.(x2, ps[6])) .+ cos(ps[5])) .*

            (sin(ps[7]) .* x3 .+ cos(ps[7])) .*
            (sin(ps[8]) .* (1 .- 2 .* lesser.(x3, ps[9])) .+ cos(ps[8])) .*

            (sin(ps[10]) .* x4 .+ cos(ps[10])) .*
            (sin(ps[11]) .* (1 .- 2 .* lesser.(x4, ps[12])) .+ cos(ps[11])) .*

            (sin(ps[13]) .* x5 .+ cos(ps[13])) .*
            (sin(ps[14]) .* (1 .- 2 .* lesser.(x5, ps[15])) .+ cos(ps[14])), idx_splits)
    end
end