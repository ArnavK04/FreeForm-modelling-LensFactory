# importing modules and files
using LensFactory
using LensFactory.Constants
using LensFactory.LensModel.LensModelIO
using Optim
using JLD2
using LineSearches

include("FreeFormLens.jl")

function neg_logprior_MEM(κ_vec::M) where M <: ROA
   """
   Give the log prior for the convergence matrix κ. 
   The prior_flag can be used to specify different types of priors, such as "flat", "gaussian", etc.
   Currently, this function is a placeholder and needs to be implemented based on the specific prior requirements.
   """
   global prior_kappa, reg_factor

   κ = reshape(κ_vec, size(prior_kappa))  # Reshape κ_vec to match the shape of prior_kappa

   any(κ .< 0) && return 1e300
   
   return reg_factor * sum( @. prior_kappa - κ + κ * log(κ / prior_kappa) )

end

function neg_loglikelihood_MEM(model::ModelConfig, lens::Lenses.AbstractLens, param_ref::Dict{Tuple{Symbol,Symbol},Float64})
   """
   Computes the log-likelihood(-0.5 * χ²) for the convergence matrix κ given the observed data.
   data is a dictionary containing the observed data and its associated uncertainties.
   """
   adis = LensModel.LensModelUtils.adis_current(model, param_ref)

   if model.sampler.scheme == :SourcePlane
      # Calculate deflection at image positions
      _, αx_all, αy_all, A_all = LensModel.LensModelUtils.lens_quantities(model, lens)

      # Calculate position likelihood
      pos_chi2 = LensModel.Likelihood.chi2_sourceplane(model, adis, αx_all, αy_all, A_all)
   else
      error("Unsupported sampling scheme: $(model.sampler.scheme)")
   end

   return -0.5 * pos_chi2

end

function construct_prior(model::ModelConfig, param_ref::Dict{Tuple{Symbol,Symbol},Float64}; prior_flag::Int = 1)
   """
   Construct the convergence matrix κ based on the model configuration and parameter references.
   This function is a placeholder and needs to be implemented based on the specific requirements for constructing κ.
   """
   global gridx, gridy
   
   X_max, Y_max = model.observation.FOV
   pixel_scale = model.observation.pixel_scale
   prior_value = param_ref[(:lens1, :prior)]
   gridx, gridy = Lenses.get_meshgrid(X_max, Y_max, pixel_scale)

   if prior_flag == 1
      prior_kappa = fill(prior_value, size(gridx))
   else
      error("Unsupported prior type: $prior_flag")
   end

   return prior_kappa
end

function neg_logpost_MEM(κ_vec::M) where M <: ROA
   """
   Returns the negative log-posterior for given ocnvergence matrix.
   """
   global model, param_ref, gridx, gridy
   κ = reshape(κ_vec, size(gridx))  # Reshape κ_vec to match the shape of prior_kappa

   lens = FreeFormLens.init_FreeFormLens(κ, gridx, gridy)

   lp = neg_logprior_MEM(κ_vec)
   ll = neg_loglikelihood_MEM(model, lens, param_ref)

   return lp + ll
end

function logprior_grad!(κ_vec::M) where M <: ROA
    """
    Compute the gradient of log-prior wrt κ in place.
    We have ∂logP/∂κ_{mn} = - reg_factor + 1 + log(κ_{mn}/prior_kappa_{mn})
    """
   global prior_kappa, reg_factor

   κ = reshape(κ_vec, size(prior_kappa))  # Reshape κ_vec 
   lp_grad = zeros(size(prior_kappa))  # Initialize gradient array with the same shape as prior_kappa

   ax1, ax2 = axes(prior_kappa, 1), axes(prior_kappa, 2)

   @inbounds for j in ax2
      @inbounds for i in ax1
        if κ[i,j] <= 0
            lp_grad[i,j] = 1e5 * κ[i,j] # large negative gradient for non-positive κ
        else
            lp_grad[i,j] = - reg_factor + 1 + log(κ[i,j] / prior_kappa[i,j])
        end
      end
   end
   return vec(lp_grad)  # Flatten the gradient vector to match the shape of κ_vec
end

function loglikelihood_grad!(κ_vec::M) where M <: ROA
    """
    Compute the gradient of log-likelihood wrt κ in place.
    This function is a placeholder and needs to be implemented based on the specific requirements for computing the gradient of the log-likelihood.
    """
    global prior_kappa, gridx, gridy, model, param_ref

    κ = reshape(κ_vec, size(prior_kappa))  # Reshape κ_vec to match the shape of prior_kappa

    f0 = neg_loglikelihood_MEM(model, FreeFormLens.init_FreeFormLens(κ, gridx, gridy), param_ref)

    ll_grad = zeros(length(κ_vec))
    h = 1e-5


    Threads.@threads for i in eachindex(κ_vec)
        buf = copy(κ_vec)
        buf[i] += h

        ll_grad[i] = (neg_loglikelihood_MEM(
            model,
            FreeFormLens.init_FreeFormLens(reshape(buf, size(gridx)), gridx, gridy),
            param_ref
        ) - f0) / h
    end
    return vec(ll_grad)
end

function logpost_grad!(grad_vec::M, κ_vec::M) where M <: ROA
    """
    Compute the gradient of log-posterior wrt κ in place.
    This function combines the gradients of the log-prior and log-likelihood.
    """
    global prior_kappa

    # Compute gradients of log-prior and log-likelihood
    lp_grad = logprior_grad!(κ_vec)
    ll_grad = loglikelihood_grad!(κ_vec)

    # Combine the gradients
    grad_vec .= lp_grad .+ ll_grad

end

function give_inversehessian(κ::M) where M <: ROA
    """
    Returns the covariance matrix (inverse Hessian) of the convergence map κ. This assumes a 
    gaussian distribution near the minimum of the target function.
    """
    global prior_kappa, gridx, gridy, model, param_ref
    κ_vec = vec(κ)
    hessian = zeros(length(κ_vec), length(κ_vec))

    f0 = neg_logpost_MEM(κ_vec)
    h = 1e-5

    Threads.@threads for i in eachindex(κ_vec)
        buf = copy(κ_vec)
        buf[i] += h
        f1 = neg_logpost_MEM(buf)

        for j in eachindex(κ_vec)
            buf2 = copy(buf)
            buf2[j] += h
            f2 = neg_logpost_MEM(buf2)

            buf3 = copy(κ_vec)
            buf3[j] += h
            f3 = neg_logpost_MEM(buf3)

            hessian[i, j] = (f2 - f1 - f3 + f0) / (h^2)
            println("Hessian computation progress: ", i, ",", j)
        end
    end    
    return inv(hessian)
end

function give_errormap(hessian::M) where M <: ROA
    """
    Computes the error map from the inverse Hessian matrix.
    """
    global gridx
    diag_elements = diag(hessian)
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
    prior_kappa = construct_prior(model, param_ref; prior_flag=1)

    #data = load("MEM_fit_result32x32_1.jld2")
    #prior_kappa = data["κ_map"]

    x0 = vec(prior_kappa)

    result = optimize(
        neg_logpost_MEM,
        logpost_grad!,
        x0,
        Optim.LBFGS(linesearch = LineSearches.BackTracking()),
        Optim.Options(
            show_trace  = true,
            show_every  = 5,
            iterations  = 10000,
            g_tol       = 1e-3,
        ),
        #autodiff  = AutoFiniteDiff(),
    )

    κ_map = reshape(Optim.minimizer(result), size(gridx))
    println("Stopped by:  ", result.stopped_by)
    println("Final value: ", Optim.minimum(result))

    t1 = time()
    println("Time taken for optimization: ", t1 - t0, " seconds")
    hessian = give_inversehessian(κ_map)
    t2 = time()
    println("Time taken for Hessian computation: ", t2 - t1, " seconds")
    errors = give_errormap(hessian)

    println("Converged:     ", Optim.converged(result))
    println("Iterations:    ", Optim.iterations(result))
    println("Final value:   ", Optim.minimum(result))
    println("Stopped by:    ", result.stopped_by)
    println("Gradient norm: ", result.g_residual)

    # final chi2
    final_lens = FreeFormLens.init_FreeFormLens(κ_map, gridx, gridy)
    final_chi2 = 2 * neg_loglikelihood_MEM(model, final_lens, param_ref)

    println("\nFinal -ve log likelihood (approx): ", final_chi2)
    println("\nFinal -ve log posterior (approx): ", neg_logpost_MEM(vec(κ_map))) 

    jldsave("../Diagnostics/files/MEM_fit_result16x16_1_125FOV_reg1.jld2";
        model_config = model,
        κ_map        = κ_map,
        errors       = errors,
        gridx        = gridx,
        gridy        = gridy,
        chi2         = final_chi2,
        neg_logpost  = neg_logpost_MEM(vec(κ_map)),
        x0           = x0,
        prior_kappa  = prior_kappa,
        reg_factor   = reg_factor,
        minimum_value= Optim.minimum(result),
        iterations   = Optim.iterations(result),
        time_run     = Optim.time_run(result),
        stopped_by   = result.stopped_by,
        converged    = Optim.converged(result)
    )
    t3 = time()
    println("Total time taken: ", t3 - t0, " seconds")
end

main()
