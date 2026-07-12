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

include("FreeFormLens.jl")
include("utility_functions.jl")

function neg_logpost_MEM(θ_vec::M) where M <: ROA
   """
   Returns the negative log-posterior for given log(convergence) matrix.
   It minimises wrt log(κ) to ensure positivity of κ. Since log is monotonic, 
   this is equivalent to minimising wrt κ.
   """
   global model, param_ref, gridx, gridy, prior_kappa, reg_factor
   println("starting logpost calc...")
   κ_vec = exp.(θ_vec)

   κ = reshape(κ_vec, size(gridx))  # Reshape κ_vec to match the shape of prior_kappa
   t0 = time()
   lens = FreeFormLens.init_FreeFormLens(κ, gridx, gridy)
   t1 = time()
   print("lens init took: ", t1-t0, " s, ")
   lp = LikelihoodFunctions.neg_logprior_MEM(κ_vec, prior_kappa, reg_factor)
   t2 = time()
   print("log-prior calc took: ", t2-t1, "  s, ")
   ll = LikelihoodFunctions.neg_loglikelihood_MEM(model, lens, param_ref)
   t3 = time()
   println("log-likelihood calc took: ", t3-t2, "  s, ")
   println("logpost calc done.")

   return lp + ll
end

function neg_logpost_MEM_κ(κ_vec::M) where M <: ROA
   """
   Returns the negative log-posterior for given convergence matrix.
   It takes in κ_vec directly, without the log transformation. This function is useful for
   calculating hessian in the κ space directly.
   """
   global model, param_ref, gridx, gridy, prior_kappa, reg_factor

   κ = reshape(κ_vec, size(gridx))  # Reshape κ_vec to match the shape of prior_kappa

   lens = FreeFormLens.init_FreeFormLens(κ, gridx, gridy)

   lp = LikelihoodFunctions.neg_logprior_MEM(κ_vec, prior_kappa, reg_factor)
   ll = LikelihoodFunctions.neg_loglikelihood_MEM(model, lens, param_ref)

   return lp + ll
end

function logpost_grad!(grad_vec_θ::M, θ_vec::M) where M <: ROA
    """
    Compute the gradient of log-posterior wrt θ in place.
    This function combines the gradients of the log-prior and log-likelihood.
    """
    global prior_kappa, reg_factor, gridx, gridy, model, param_ref
    println("starting logpost grad calc...")
    κ_vec = exp.(θ_vec)  # Convert θ_vec back to κ_vec

    # Compute gradients of log-prior and log-likelihood
    t0 = time()
    lp_grad = LikelihoodFunctions.logprior_grad!(κ_vec, prior_kappa, reg_factor)
    t1 = time()
    print("log-prior grad calc took: ", t1-t0, "  s, ")
    ll_grad = LikelihoodFunctions.loglikelihood_grad!(κ_vec, prior_kappa, gridx, gridy, model, param_ref)
    t2 = time()
    println("log-likelihood grad calc took: ", t2-t1, "  s, ")

    # Combine the gradients
    grad_vec_θ .= (lp_grad .+ ll_grad) .* κ_vec  # Chain rule: d/dθ = d/dκ * κ
    println("logpost grad calc done.")
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
    
    # 1. Allocate the output matrix
    hessian = zeros(n, n)
    
    # 2. Create an allocating Hessian cache (Do this once!)
    # By default, it uses high-accuracy central differences (:hcentral)
    cache = FiniteDiff.HessianCache(κ_vec)
    
    # 3. Call the highly optimized, non-allocating routine
    # We pass an anonymous function wrapping your posterior calculation
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
    input_file = "../LensFactory-Examples/LensModel/GalaxyLens/MockLens/galaxy_parameter.yaml"

    global param_ref, reg_factor, prior_kappa, model, gridx, gridy

    model = LensModel.read_input(input_file)
    param_ref = Dict(p.key => p.refer for p in model.parameters)
    reg_factor = 1
    prior_kappa, gridx, gridy = LikelihoodFunctions.construct_prior(model, param_ref; prior_flag=1)
    new_guess = prior_kappa
    prior_from_prev = false

    filename = "MEM_fit_result32x32_1_125FOV_reg1_timecheck"

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
        new_guess, gridx, gridy = UtilityFunctions.refine_map(new_guess_, gridx_, gridy_, gridx_[end,1], gridy_[1,end], fin_res, 1)  # refine to a required grid
        prior_kappa, _, _ = UtilityFunctions.refine_map(prior_kappa_, gridx_, gridy_, gridx_[end,1], gridy_[1,end], fin_res, 1)
        println("loaded prior from previous run and refined to a finer grid with resolution: ", fin_res, " arcsec/pixel")
    end 

    κ0 = vec(new_guess)
    θ0 = log.(κ0)  # Initial guess in θ space

    result = optimize(
        neg_logpost_MEM,
        logpost_grad!,
        θ0,
        Optim.LBFGS(linesearch = LineSearches.HagerZhang()),
        Optim.Options(
            show_trace  = true,
            show_every  = 1,
            iterations  = 2000,
            time_limit  = 7200,  # 6 hours
            g_tol       = 1e-3,
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

    # WILL MOVE HESSIAN CALC TO A DIFFEENT FILE ALTOGETHER

    println("Converged:     ", Optim.converged(result))
    println("Iterations:    ", Optim.iterations(result))
    println("Final value:   ", Optim.minimum(result))
    println("Stopped by:    ", result.stopped_by)
    println("Gradient norm: ", result.g_residual)

    # final chi2
    final_lens = FreeFormLens.init_FreeFormLens(κ_map, gridx, gridy)
    final_chi2 = 2 * LikelihoodFunctions.neg_loglikelihood_MEM(model, final_lens, param_ref)

    println("\nFinal -ve log likelihood (approx): ", final_chi2)
    println("\nFinal -ve log posterior (approx): ", neg_logpost_MEM(vec(θ_map))) 

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
        reg_factor   = reg_factor,
        minimum_value= Optim.minimum(result),
        iterations   = Optim.iterations(result),
        time_run     = Optim.time_run(result),
        stopped_by   = result.stopped_by,
        converged    = Optim.converged(result)
    )
    t3 = time()
    println("Time taken for optimization: ", t1 - t0, " seconds")
    println("Time taken for Hessian computation: ", t2 - t1, " seconds")
    println("Time taken for fast Hessian computation: ", t_fast - t2, " seconds")
    #println("max difference between hessians: ", maximum(abs.(hessian - hessian_fast)))
    #println("mean difference between hessians: ", mean(abs.(hessian - hessian_fast)))
    println("Total time taken: ", t3 - t0, " seconds")

    """println("printing some diagnostics for the hessian matrix....")
    # 1. Basic stats
    println("Condition number:  ", cond(hessian))
    println("Min eigenvalue:    ", minimum(eigvals(hessian)))
    println("Max eigenvalue:    ", maximum(eigvals(hessian)))
    println("Min diagonal:      ", minimum(diag(hessian)))
    println("Max diagonal:      ", maximum(diag(hessian)))

    # 2. Symmetry — should be ~machine precision
    println("Symmetry error:    ", maximum(abs.(hessian .- hessian')))

    # 3. Positive definiteness — all eigenvalues > 0?
    eigs = eigvals(hessian)
    println("Positive definite: ", all(eigs .> 0))
    println("N negative eigs:   ", sum(eigs .< 0))
    println("N near-zero eigs:  ", sum(abs.(eigs) .< 1e-6))

    # 4. Diagonal dominance — rough check of numerical quality
    diag_dom = all(abs.(diag(hessian)) .>= sum(abs.(hessian), dims=2)[:])
    println("Diagonally dominant: ", diag_dom)

    println("printing stats for fast hessian matrix....")

    # 1. Basic stats
    println("Condition number:  ", cond(hessian_fast))
    println("Min eigenvalue:    ", minimum(eigvals(hessian_fast)))
    println("Max eigenvalue:    ", maximum(eigvals(hessian_fast)))
    println("Min diagonal:      ", minimum(diag(hessian_fast)))
    println("Max diagonal:      ", maximum(diag(hessian_fast)))

    # 2. Symmetry — should be ~machine precision
    println("Symmetry error:    ", maximum(abs.(hessian_fast .- hessian_fast')))
    
    # 3. Positive definiteness — all eigenvalues > 0?
    eigs_fast = eigvals(hessian_fast)
    println("Positive definite: ", all(eigs_fast .> 0))
    println("N negative eigs:   ", sum(eigs_fast .< 0))
    println("N near-zero eigs:  ", sum(abs.(eigs_fast) .< 1e-6))

    # 4. Diagonal dominance — rough check of numerical quality
    diag_dom_fast = all(abs.(diag(hessian_fast)) .>= sum(abs.(hessian_fast), dims=2)[:])
    println("Diagonally dominant: ", diag_dom_fast)"""
end

main()
