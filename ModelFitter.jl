# importing modules and files
using LensFactory
using LensFactory.Constants
using LensFactory.LensModel.LensModelIO
using Statistics
using StatsPlots
using Optim
using ADTypes
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
   gridx = range(-X_max, X_max, step=pixel_scale)
   # repeat along y-axis to create a 2D grid
   gridx = repeat(gridx, 1, length(gridx))
   gridy = range(-Y_max, Y_max, step=pixel_scale)
   # repeat along x-axis to create a 2D grid
   gridy = repeat(gridy', length(gridy), 1)

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
   κ = reshape(κ_vec, size(prior_kappa))  # Reshape κ_vec to match the shape of prior_kappa

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
        copyto!(buf, κ_vec)   # reset buffer
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

input_file = "../LensFactory-Examples/LensModel/GalaxyLens/MockLens/galaxy_parameter.yaml"

global param_ref, reg_factor, prior_kappa, model, gridx, gridy

model = LensModel.read_input(input_file)
param_ref = Dict(p.key => p.refer for p in model.parameters)
reg_factor = 0.8
prior_kappa = construct_prior(model, param_ref; prior_flag=1)

x0 = vec(prior_kappa)
using LineSearches

result = optimize(
    neg_logpost_MEM,
    logpost_grad!,
    x0,
    Optim.LBFGS(linesearch = LineSearches.BackTracking()),
    Optim.Options(
        show_trace  = true,
        show_every  = 5,
        iterations  = 1000,
        g_tol       = 1e-3,
    ),
    #autodiff  = AutoFiniteDiff(),
)

κ_map = reshape(Optim.minimizer(result), size(gridx))
println("Stopped by:  ", result.stopped_by)
println("Final value: ", Optim.minimum(result))

println("Converged:     ", Optim.converged(result))
println("Iterations:    ", Optim.iterations(result))
println("Final value:   ", Optim.minimum(result))
println("Stopped by:    ", result.stopped_by)
println("Gradient norm: ", result.g_residual)

println("κ_map stats:")
println("  min:  ", minimum(κ_map))
println("  max:  ", maximum(κ_map))
println("  mean: ", mean(κ_map))


println("\nFinal -ve log posterior (approx): ", 2 * Optim.minimum(result))  # rough estimate
using Plots
h = heatmap(κ_map, title="κ MAP estimate", color=:viridis, size=(800,800))
savefig(h, "kappa_mapAres.png")  # Save the heatmap to a file
