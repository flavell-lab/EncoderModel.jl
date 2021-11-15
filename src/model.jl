logistic(x,x0,k) = 1 / (1 + exp(-k * (x - x0)))
leaky_logistic(x,x0,k,m) = logistic(x,x0,k) + m * (x-x0)
lesser(x,x0) = leaky_logistic(x,x0,50,1e-3)


function ewma(γ::T, x, idx_splits) where {T}
    return_array = zeros(T, idx_splits[end][end])
    for split = idx_splits
        max_t = length(split)
        x_avg = zeros(T, max_t)
        s = sum([exp(-(max_t-t)*γ) for t=1:max_t])
        x_avg[1] = T(x[split[1]]/s)
        for t=2:max_t
            t_x = split[1] + t -1
            x_avg[t] = T(x_avg[t-1] * (s-1) / s + x[t_x] / s)
        end
        return_array[split] = x_avg
    end
    
    return return_array
end


function generate_model_nl3_1var(behaviors, idx_splits)
    var1 = behaviors[1]
    
    return function (ps)
        ps[5] .+ (exp.((ps[1] .* EncoderModel.ewma(ps[4], var1, idx_splits) .+
                ps[2] .* EncoderModel.ewma(ps[4], 1 .- 2 .* lesser.(ps[3], var1), idx_splits))
            )
        )
    end
end

function generate_model_nl3_2var(behaviors, idx_splits)
    var1 = behaviors[1]
    var2 = behaviors[2]
    
    return function(ps)
        ps[8] .+ (exp.((ps[1] .* EncoderModel.ewma(ps[7], var1, idx_splits) .+
                ps[2] .* EncoderModel.ewma(ps[7], 1 .- 2 .* lesser.(ps[3], var1), idx_splits) .+
                ps[4] .* EncoderModel.ewma(ps[7], var2, idx_splits) .+
                ps[5] .* EncoderModel.ewma(ps[7], 1 .- 2 .* lesser.(ps[6], var2), idx_splits))
            )
        )
    end
end

function generate_model_nl3_3var(behaviors, idx_splits)
    var1 = behaviors[1]
    var2 = behaviors[2]
    var3 = behaviors[3]
    
    return function (ps)
        ps[11] .+ (exp.((ps[1] .* ewma(ps[10], var1, idx_splits) .+
                ps[2] .* ewma(ps[10], 1 .- 2 .* lesser.(ps[3], var1), idx_splits) .+
                ps[4] .* ewma(ps[10], var2, idx_splits) .+
                ps[5] .* ewma(ps[10], 1 .- 2 .* lesser.(ps[6], var2), idx_splits) .+
                ps[7] .* ewma(ps[10], var3, idx_splits) .+
                ps[8] .* ewma(ps[10], 1 .- 2 .* lesser.(ps[9], var3), idx_splits))
            )
        )
    end
end




function fit_model_bound_nlopt(traces, model_generator, f_cost, ps_0, ps_min, ps_max,
        idx_trace, behaviors, optimizer_g, optimizer_l, idx_splits;
        max_time=60, max_evals=20000, xtol=1e-4, ftol=1e-4, train=1:1600, λ=0)

    list_σ2 = [var(behaviors[(i+1)÷2]) for i in 1:2*length(behaviors)]
    ps_idx = [i + (i-1)÷2 for i in 1:2*length(behaviors)]

    model_fn = model_generator(behaviors, idx_splits)

    f(ps) = f_cost(model_fn(ps)[train], traces[idx_trace,train]) + λ * reg_var_L2(ps[ps_idx], list_σ2)
    g(ps) = ForwardDiff.gradient(f, ps)
    function cost(ps::Vector, grad::Vector)
        if length(grad) > 0
            grad .= g(ps)
        end
        return f(ps)
    end

    n_ps = length(ps_0)
    opt = Opt(optimizer_g, n_ps)
    opt.min_objective = cost
    opt.maxtime = max_time
    opt.maxeval = max_evals
    opt.lower_bounds = ps_min
    opt.upper_bounds = ps_max
    opt.min_objective = cost
    opt.xtol_rel = xtol
    opt.ftol_rel = ftol

    if !isnothing(optimizer_l)
        local_optimizer = Opt(optimizer_l, n_ps)
        local_optimizer.xtol_rel = xtol
        local_optimizer.ftol_rel = ftol
        local_optimizer!(opt, local_optimizer)
    end

    NLopt.optimize(opt, ps_0), opt.numevals # ((final cost, u_opt, exit code), num f eval)
end
