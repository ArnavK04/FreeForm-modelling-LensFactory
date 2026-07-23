# importing modules and files
using LensFactory
using LensFactory.Constants
using LensFactory.LensModel.LensModelIO
using Optim
using JLD2
using LineSearches
using LinearAlgebra
using ProgressMeter
using FiniteDiff
using Statistics
using ArgParse

include("FreeFormLens.jl")
include("utility_functions.jl")

function neg_logpost_MEM(θ_vec::M) where M <: ROA
    """
    Returns the negative log-posterior for given log(convergence) matrix.
    It minimises wrt log(κ) to ensure positivity of κ. Since log is monotonic, 
    this is equivalent to minimising wrt κ.
    """
    global model, param_ref, gridx, gridy, prior_kappa, reg_factor, full_kernel
    #println("starting logpost calc...")
    κ_vec = exp.(θ_vec)
    println("κ range: ", extrema(κ_vec), "  θ range: ", extrema(θ_vec))
    κ = reshape(κ_vec, size(gridx))  # Reshape κ_vec to match the shape of prior_kappa
    t0 = time()
    lens = FreeFormLens.init_FreeFormLens(κ, gridx, gridy, true)  # kernel_flag = true to compute kernel for the lens
    t1 = time()
    #print("lens init took: ", t1-t0, " s, ")
    lp = LikelihoodFunctions.neg_logprior_MEM(κ_vec, prior_kappa, reg_factor)
    t2 = time()
    #print("log-prior calc took: ", t2-t1, "  s, ")
    ll = LikelihoodFunctions.neg_loglikelihood_MEM(model, lens, param_ref, full_kernel)
    t3 = time()
    #println("log-likelihood calc took: ", t3-t2, "  s, ")
    #println("logpost calc done.")

    return lp + ll
end

function neg_logpost_MEM_κ(κ_vec::M) where M <: ROA
    """
    Returns the negative log-posterior for given convergence matrix.
    It takes in κ_vec directly, without the log transformation. This function is useful for
    calculating hessian in the κ space directly.
    """
    global model, param_ref, gridx, gridy, prior_kappa, reg_factor, full_kernel
        
    κ = reshape(κ_vec, size(gridx))  # Reshape κ_vec to match the shape of prior_kappa

    lens = FreeFormLens.init_FreeFormLens(κ, gridx, gridy, true)

    lp = LikelihoodFunctions.neg_logprior_MEM(κ_vec, prior_kappa, reg_factor)
    ll = LikelihoodFunctions.neg_loglikelihood_MEM(model, lens, param_ref, full_kernel)

    return lp + ll
end

function logpost_grad!(grad_vec_θ::M, θ_vec::M) where M <: ROA
    """
    Compute the gradient of log-posterior wrt θ in place.
    This function combines the gradients of the log-prior and log-likelihood.
    """
    global prior_kappa, reg_factor, gridx, gridy, model, param_ref, full_kernel
    #println("starting logpost grad calc...")
    κ_vec = exp.(θ_vec)  # Convert θ_vec back to κ_vec

    # Compute gradients of log-prior and log-likelihood
    t0 = time()
    lp_grad = LikelihoodFunctions.logprior_grad!(κ_vec, prior_kappa, reg_factor)
    t1 = time()
    print("log-prior grad calc took: ", t1-t0, "  s, ")
    ll_grad = LikelihoodFunctions.loglikelihood_grad!(κ_vec, prior_kappa, gridx, gridy, model, param_ref, full_kernel)
    t2 = time()
    println("log-likelihood grad calc took: ", t2-t1, "  s, ")

    # Combine the gradients
    grad_vec_θ .= (lp_grad .+ ll_grad) .* κ_vec  # Chain rule: d/dθ = d/dκ * κ
    #println("logpost grad calc done.")
end

function give_inversehessian(κ::M, prior_kappa::M, gridx::M, gridy::M, model::ModelConfig, param_ref::Dict{Tuple{Symbol, Symbol},Float64}) where {M <: ROA}
    """
    Returns the covariance matrix (inverse Hessian) of the convergence map κ. This assumes a 
    gaussian distribution near the minimum of the target function.
    """
    κ_vec = vec(κ)
    hessian = zeros(length(κ_vec), length(κ_vec))

    f0 = neg_logpost_MEM_κ(κ_vec)
    buf1 = copy(κ_vec)
    buf2 = copy(κ_vec)
    buf3 = copy(κ_vec)      # making three copies here 

    @showprogress "Computing Hessian..." for i in eachindex(κ_vec)

        hi = max(1e-8, abs(κ_vec[i]) * 1e-5)  # step size based on the magnitude of κ_vec[i]
        buf1[i] += hi
        f1 = neg_logpost_MEM_κ(buf1)

        for j in eachindex(κ_vec)

            if i != j
                hj = max(1e-8, abs(κ_vec[j]) * 1e-5)  # step size based on the magnitude of κ_vec[j]
                buf2[i] += hi
                buf2[j] += hj
                f2 = neg_logpost_MEM_κ(buf2)

                buf3[j] += hj
                f3 = neg_logpost_MEM_κ(buf3)

                hessian[i, j] = (f2 - f1 - f3 + f0) / (hi * hj)

                # Reset the buffers for the next iteration
                buf2[j] = κ_vec[j]
                buf3[j] = κ_vec[j]
                buf2[i] = κ_vec[i]
            else
                buf2[i] -= hi
                f2 = neg_logpost_MEM_κ(buf2)
                hessian[i, i] = (f1 - 2 * f0 + f2) / (hi^2)
                buf2[i] = κ_vec[i]
            end
        end
        buf1[i] = κ_vec[i]
    end    

    return inv(hessian)

end

function give_inversehessian_fast(κ::M, prior_kappa::M, gridx::M, gridy::M, model::ModelConfig, param_ref::Dict{Tuple{Symbol, Symbol},Float64}) where {M <: ROA}
    κ_vec = vec(κ)
    n = length(κ_vec)
    
    hessian = zeros(n, n)
    
    cache = FiniteDiff.HessianCache(κ_vec)
    
    FiniteDiff.finite_difference_hessian!(
        hessian, 
        x -> neg_logpost_MEM_κ(x), 
        κ_vec, 
        cache
    )
    
    return inv(hessian)
end

function give_errormap(hessian::M) where M <: ROA
    """
    Computes the error map from the inverse Hessian matrix.
    """
    global gridx
    diag_elements = diag(hessian)
    if any(diag_elements .< 0)
        println("Warning: Negative diagonal elements in Hessian.")
        println("There are $(sum(diag_elements .< 0)) negative diagonal elements out of $(length(diag_elements)).")
        return nothing
    end
    errormap = reshape(sqrt.(diag_elements), size(gridx))
    return errormap
end

function main()

    t0 = time()

    global param_ref, reg_factor, prior_kappa, model, gridx, gridy, new_guess, full_kernel

    settings = ArgParseSettings()

    @add_arg_table settings begin
        "--input_file"
            help = "Path to the input YAML file containing lens model parameters."
            arg_type = String
            default = "../LensFactory-Examples/LensModel/GalaxyLens/MockLens/galaxy_parameter.yaml"
        "--seed"
            help = "Random seed for reproducibility."
            arg_type = Int
            default = nothing
        "--pix"
            help = "size of smoothening kernel for random guess."
            arg_type = Int
            default = nothing
        "--sigma"
            help = "Sigma value for gaussian guess."
            arg_type = Float64
            default = nothing
        "--g_flag"
            help = "Prior flag indicating the type of initial guess to use."
            arg_type = Int
            default = 1
        "--prior_from_prev"
            help = "Boolean flag to indicate if prior should be loaded from previous run."
            arg_type = Bool
            default = false
        "--reg_factor"
            help = "Regularization factor for the prior."
            arg_type = Float64
            default = 1.0
        "--guess_value"
            help = "Initial guess value for the convergence map."
            arg_type = Float64
            default = 0.5
        "--runnumber"
            help = "Run number for this optimization run."
            arg_type = Int
            default = 1
        "--p_value"
            help = "Prior value used in constructing the prior map."
            arg_type = Float64
            default = 0.5
    end

    args = parse_args(settings)

    seed = args["seed"]
    input_file = args["input_file"]
    pix = args["pix"]
    sigma = args["sigma"]
    g_flag = args["g_flag"]
    prior_from_prev = args["prior_from_prev"]
    reg_factor = args["reg_factor"]
    guess_value = args["guess_value"]
    runnumber = args["runnumber"]
    p_value = args["p_value"]

    model = LensModel.read_input(input_file)
    param_ref = Dict(p.key => p.refer for p in model.parameters)



    prior_kappa, gridx, gridy = LikelihoodFunctions.construct_prior(model; prior_flag=1, prior_value=p_value)
    # get full_kernel for the run
    full_kernel = FreeFormLens.compute_fullkernel(model, gridx, gridy)

    println("size of gridx: ", size(gridx))
    new_guess, _, _ = LikelihoodFunctions.construct_prior(model; prior_flag=g_flag, seed=seed,pix=pix,sigma=sigma, prior_value=guess_value)

    filename = "MEM_fit_result4res_125FOV_reg$(reg_factor)_gflag$(g_flag)_pvalue$(p_value)_$(guess_value)_$(seed)_$(pix)_$(sigma)_$(runnumber)"
    filename_fromprev = "MEM_fit_result4res_125FOV_reg$(reg_factor)_gflag$(g_flag)_pvalue$(p_value)_$(guess_value)_$(seed)_$(pix)_$(sigma)_$(runnumber+1)"

    # load prior from previous run converged map and refine to a finer grid
    if prior_from_prev
        data = load("../Diagnostics/files/$(filename).jld2")
        prior_kappa_ = data["κ_map"]
        new_guess_ = data["prior_kappa"]
        gridx_ = data["gridx"]
        gridy_ = data["gridy"]

        res_ = gridx_[2,1] - gridx_[1,1]
        print("min kappa: ", minimum(prior_kappa_), " max kappa: ", maximum(prior_kappa_), " mean kappa: ", mean(prior_kappa_))

        #fin_res = res_/2.0
        fin_res = res_
        res_factor = round(Int,fin_res/res_)

        if res_factor != 1
            println("Refining the prior from previous run by a factor of: ", res_factor)
            new_guess, gridx, gridy = UtilityFunctions.refine_map(new_guess_, gridx_, gridy_, gridx_[end,1], gridy_[1,end], fin_res, 1)  # refine to a required grid
            prior_kappa, _, _ = UtilityFunctions.refine_map(prior_kappa_, gridx_, gridy_, gridx_[end,1], gridy_[1,end], fin_res, 1)
        else
            println("No refinement needed for the prior from previous run.")
            new_guess = new_guess_
            prior_kappa = prior_kappa_
            gridx = gridx_
            gridy = gridy_
        end
        println("loaded prior from previous run and refined to a finer grid with resolution: ", fin_res, " arcsec/pixel")
    end

    κ0 = vec(new_guess)
    θ0 = log.(κ0)  # Initial guess in θ space

    # testing grad provided by finitediff module
    function g!(grad_vec_θ, θ_vec)
        FiniteDiff.finite_difference_gradient!(grad_vec_θ, neg_logpost_MEM, θ_vec)
        return grad_vec_θ
    end

    result = optimize(
        neg_logpost_MEM,
        logpost_grad!,
        #g!,
        θ0,
        Optim.LBFGS(linesearch = LineSearches.HagerZhang()),
        Optim.Options(
            show_trace  = true,
            show_every  = 5,
            iterations  = 2000,
            time_limit  = 43200,  # 12 hours
            g_tol       = 1e-2,
        ),
        #autodiff  = AutoFiniteDiff(),
    )

    θ_map = reshape(Optim.minimizer(result), size(gridx))
    κ_map = exp.(θ_map)  # Convert back to κ space

    println("Stopped by:  ", result.stopped_by)
    println("Final value: ", Optim.minimum(result))

    t1 = time()
    #hessian = give_inversehessian(κ_map, prior_kappa, gridx, gridy, model, param_ref)
    t2 = time()
    #hessian_fast = give_inversehessian_fast(κ_map, prior_kappa, gridx, gridy, model, param_ref)
    t_fast = time()
    #errors = give_errormap(hessian_fast)

    # WILL MOVE HESSIAN CALC TO A DIFFEENT FILE ALTOGETHER. IT NEEDS PARALLELIZATION.

    println("Converged:     ", Optim.converged(result))
    println("Iterations:    ", Optim.iterations(result))
    println("Final value:   ", Optim.minimum(result))
    println("Stopped by:    ", result.stopped_by)
    println("Gradient norm: ", result.g_residual)

    # final chi2
    final_lens = FreeFormLens.init_FreeFormLens(κ_map, gridx, gridy, true)            # general diagnostics lens, so no computing kernel here
    final_chi2 = 2 * LikelihoodFunctions.neg_loglikelihood_MEM(model, final_lens, param_ref, full_kernel)

    println("\nFinal -ve log likelihood (approx): ", final_chi2)
    println("\nFinal -ve log posterior (approx): ", neg_logpost_MEM(vec(θ_map))) 

    if !prior_from_prev
        jldsave("../Diagnostics/files/$(filename).jld2";
            model_config = model,
            κ_map        = κ_map,
            #hessian      = hessian,
            #hessian_fast = hessian_fast,
            #errors       = errors,
            gridx        = gridx,
            gridy        = gridy,
            chi2         = final_chi2,
            neg_logpost  = neg_logpost_MEM(vec(θ_map)),
            θ0           = θ0,
            prior_kappa  = prior_kappa,
            init_guess   = new_guess,
            seed         = seed,
            pix          = pix,
            sigma        = sigma,
            prior_flag   = g_flag,
            reg_factor   = reg_factor,
            minimum_value= Optim.minimum(result),
            iterations   = Optim.iterations(result),
            time_run     = Optim.time_run(result),
            stopped_by   = result.stopped_by,
            converged    = Optim.converged(result)
        )

    else 
        jldsave("../Diagnostics/files/$(filename_fromprev).jld2";
            model_config = model,
            κ_map        = κ_map,
            #hessian      = hessian,
            #hessian_fast = hessian_fast,
            #errors       = errors,
            gridx        = gridx,
            gridy        = gridy,
            chi2         = final_chi2,
            neg_logpost  = neg_logpost_MEM(vec(θ_map)),
            θ0           = θ0,
            prior_kappa  = prior_kappa,
            init_guess   = new_guess,
            seed         = seed,
            pix          = pix,
            sigma        = sigma,
            prior_flag   = g_flag,
            reg_factor   = reg_factor,
            minimum_value= Optim.minimum(result),
            iterations   = Optim.iterations(result),
            time_run     = Optim.time_run(result),
            stopped_by   = result.stopped_by,
            converged    = Optim.converged(result)
        )
    end

    t3 = time()
    println("Time taken for optimization: ", t1 - t0, " seconds")
    println("Time taken for Hessian computation: ", t2 - t1, " seconds")
    println("Time taken for fast Hessian computation: ", t_fast - t2, " seconds")
    #println("max difference between hessians: ", maximum(abs.(hessian - hessian_fast)))
    #println("mean difference between hessians: ", mean(abs.(hessian - hessian_fast)))
    println("Total time taken: ", t3 - t0, " seconds")


    """println("printing stats for fast hessian matrix....")

    println("Condition number:  ", cond(hessian_fast))
    println("Min eigenvalue:    ", minimum(eigvals(hessian_fast)))
    println("Max eigenvalue:    ", maximum(eigvals(hessian_fast)))
    println("Min diagonal:      ", minimum(diag(hessian_fast)))
    println("Max diagonal:      ", maximum(diag(hessian_fast)))

    println("Symmetry error:    ", maximum(abs.(hessian_fast .- hessian_fast')))
    
    eigs_fast = eigvals(hessian_fast)
    println("Positive definite: ", all(eigs_fast .> 0))
    println("N negative eigs:   ", sum(eigs_fast .< 0))
    println("N near-zero eigs:  ", sum(abs.(eigs_fast) .< 1e-6))"""
end

main()
