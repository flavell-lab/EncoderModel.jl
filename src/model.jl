logistic(x,x0,k) = 1 / (1 + exp(-k * (x - x0)))
leaky_logistic(x,x0,k,m) = logistic(x,x0,k) + m * (x-x0)
lesser(x,x0) = leaky_logistic(x,x0,100,1e-3)


function cost_rss(y, y_pred)
    sum((y .- y_pred) .^ 2)
end

function cost_abs(y, y_pred)
    sum(abs.(y .- y_pred))
end

function cost_mse(y, y_pred)
    mean((y .- y_pred) .^ 2)
end

function cost_cor(y, y_pred)
    - cor(y, y_pred)
end

function reg_var_L1(list_θ, list_σ2)
    sum(abs.(list_θ ./ (list_σ2 .+ 1)))
end

function reg_var_L2(list_θ, list_σ2)
    sum((list_θ ./ (list_σ2 .+ 1)) .^ 2)
end


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



function generate_model_nl3_vhp(behaviors, idx_splits)
    velocity = behaviors[1]
    θh = behaviors[2]
    pumping = behaviors[3]
    
    return function (ps)
        ps[11] .+ (exp.((ps[1] .* ewma(ps[10], velocity, idx_splits) .+
                ps[2] .* ewma(ps[10], 1 .- 2 .* lesser.(ps[3], velocity), idx_splits) .+
                ps[4] .* ewma(ps[10], θh, idx_splits) .+
                ps[5] .* ewma(ps[10], 1 .- 2 .* lesser.(ps[6], θh), idx_splits) .+
                ps[7] .* ewma(ps[10], pumping, idx_splits) .+
                ps[8] .* ewma(ps[10], 1 .- 2 .* lesser.(ps[9], pumping), idx_splits))
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
