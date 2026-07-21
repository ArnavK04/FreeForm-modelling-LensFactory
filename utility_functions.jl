module UtilityFunctions

using LensFactory
using LensFactory.Constants
using LensFactory.LensModel.LensModelIO
using FITSIO
using JLD2
using Interpolations
using LinearAlgebra
using CairoMakie
using Statistics

function refine_map(map_coarse::M, gridx::M, gridy::M, new_grid_limx::T, new_grid_limy::T, resolution::T, order::Int) where {T <:RV, M <: ROA}
    """
    Uses Bicubic (now linear) interpolation to return a higher res map of the convergence κ.
    """
    x_nodes = range(gridx[1, 1], stop=gridx[end, 1], length=size(gridx, 2))
    y_nodes = range(gridy[1, 1], stop=gridy[1, end], length=size(gridy, 1))

    if order == 1
        itp = interpolate(map_coarse, BSpline(Linear()))
    elseif order == 2
        itp = interpolate(map_coarse, BSpline(Quadratic(Line(OnGrid()))))
    elseif order == 3
        itp = interpolate(map_coarse, BSpline(Cubic(Line(OnGrid()))))
    else
        error("Unsupported interpolation order. upto cubic is allowed.")
    end
    
    itp = scale(itp, x_nodes, y_nodes)

    x_fine, y_fine = Lenses.get_meshgrid(new_grid_limx, new_grid_limy, resolution)

    map_fine = itp.(x_fine, y_fine)

    return map_fine, x_fine, y_fine

end

function coarsen_map(map_fine::M, gridx::M, gridy::M, new_grid_limx::T, new_grid_limy::T, factor::Int) where {T <:RV, M <: ROA}
    """
    Returns a lower res map of the convergence κ downsampled by factor.
    """
    original_res = gridx[2, 1] - gridx[1, 1]
    new_res = original_res * factor

    nx_full, ny_full = size(map_fine)
    nx_used = div(nx_full, factor) * factor   # largest multiple of factor <= nx_full
    ny_used = div(ny_full, factor) * factor

    if nx_used != nx_full || ny_used != ny_full
        println("Warning: cropping map_fine from ($nx_full, $ny_full) to ($nx_used, $ny_used) to fit factor=$factor evenly.")
    end

    n_coarse_x = div(nx_used, factor)
    n_coarse_y = div(ny_used, factor)
    coarse_map = zeros(n_coarse_x, n_coarse_y)
    coarse_x = zeros(size(coarse_map))
    coarse_y = zeros(size(coarse_map))

    ax1, ax2 = axes(coarse_map, 1), axes(coarse_map, 2)
    for j in ax2
        for i in ax1
            coarse_map[i,j] = mean(map_fine[(i-1)*factor+1:i*factor, (j-1)*factor+1:j*factor])
            coarse_x[i,j] = mean(gridx[(i-1)*factor+1:i*factor, (j-1)*factor+1:j*factor])
            coarse_y[i,j] = mean(gridy[(i-1)*factor+1:i*factor, (j-1)*factor+1:j*factor])
        end
    end

    coarse_map_, coarse_x_, coarse_y_ = refine_map(coarse_map, coarse_x, coarse_y, new_grid_limx, new_grid_limy, new_res, 3)

    return coarse_map_, coarse_x_, coarse_y_, new_res
end

function load_fitsfile(filename::String)
    """
    Loads a FITS file and returns coordinate grid along with kappa, gamma1, gamma2 matrices.
    """
    local gridx_fits, gridy_fits, kappa, gamma1, gamma2

    FITS("../$(filename)_data/kappa_z9_0.fits") do f
        hdr = read_header(f[1])
        kappa = read(f[1])

        NX = hdr["NAXIS1"]
        NY = hdr["NAXIS2"]

        pixel_scale = hdr["CDELT1"] * 3600.0  # Convert degrees to arcseconds
        CRVAL1 = hdr["CRVAL1"] * 3600.0
        CRVAL2 = hdr["CRVAL2"] * 3600.0
        CRPIX1 = hdr["CRPIX1"]
        CRPIX2 = hdr["CRPIX2"]

        X_max = (NX - CRPIX1) * pixel_scale + CRVAL1
        Y_max = (NY - CRPIX2) * pixel_scale + CRVAL2

        gridx_fits, gridy_fits = Lenses.get_meshgrid(X_max, Y_max, pixel_scale)
    end

    FITS("../$(filename)_data/gammax_z9_0.fits") do f
        gamma1 = read(f[1])
    end

    FITS("../$(filename)_data/gammay_z9_0.fits") do f
        gamma2 = read(f[1])
    end

    return gridx_fits[2:end, 2:end], gridy_fits[2:end, 2:end], kappa, gamma1, gamma2    # because the fits arrays are 2048, but grid_fits are 2049
end

function make_gridfrom_model(model::LensModel.ModelConfig)
    """
    Constructs a grid based on the model configuration.
    Returns the grid coordinates (gridx, gridy).
    """
    X_max, Y_max = model.observation.FOV
    pixel_scale = model.observation.pixel_scale
    gridx, gridy = Lenses.get_meshgrid(X_max, Y_max, pixel_scale)

    return gridx, gridy
end

function predict_image(lens::Lenses.AbstractLens, gridx::M, gridy::M, θx::N, θy::N, adis::T, sid::Int, kid::Int, images_obs, plot_flag::Bool, path::String) where {T <: RV, M <: ROA, N <: ROA}
    """
    Predicts the image positions based on the lens model and source positions.
    """
    αx, αy = Lenses.get_deflection(lens, θx, θy)

    βx = θx .- αx .* adis
    βy = θy .- αy .* adis

    μ_obs = Lenses.get_magnification_image(lens, θx, θy, adis)

    # Calculate barycenter source position
    βx_model = sum(βx .* μ_obs.^2) / sum(μ_obs.^2)
    βy_model = sum(βy .* μ_obs.^2) / sum(μ_obs.^2)

    images = Lenses.get_image(lens, gridx, gridy, adis, (βx_model, βy_model))

    if plot_flag
        fig, axes = Lenses.plot_image_plane(lens, gridx, gridy, adis, source=(βx_model, βy_model), two_panel=true)
        makie_points = [Point2f(pt) for pt in images_obs]
        scatter!(axes[2], makie_points, color=:cyan, markersize=8)
        outpath = path * "images_sid$(sid)_kid$(kid).png"
        save(outpath, fig)
    end
    return images
    
end

function give_image_rmsscatter(model::LensModel.ModelConfig, lens::Lenses.AbstractLens, param_ref::Dict{Tuple{Symbol,Symbol},Float64}, gridx::M, gridy::M, plot_flag::Bool, path::String) where M <: ROA

    adis = LensModel.LensModelUtils.adis_current(model, param_ref)

    sid = 1
    kid = 1
    sum_rms = 0.0
    count = 0
    for src in model.source_config.sources
        adis_value = adis[sid]
        kid = 1
        for knot in src.knots
            x = knot.x
            y = knot.y
            images_obs = [(xi, yi) for (xi, yi) in zip(x, y)]
            images_pred = predict_image(lens, gridx, gridy, x, y, adis_value, sid, kid, images_obs, plot_flag, path)
            println(length(images_obs), " Observed images: ", images_obs)
            println(length(images_pred), " Predicted images: ", images_pred)
            sum_rms_= give_sum_rms(images_pred, images_obs)
            sum_rms += sum_rms_
            count += length(images_obs)
            kid += 1
            println("$(sid), $(kid) has sum_rms = $(sum_rms_), count = $(length(images_obs)), rms = $(sqrt(sum_rms_/length(images_obs)))")
            println("------------------------------------------------------")
        end
        sid += 1
    end
    
    return sqrt(sum_rms / count)
    println("count: ", count, " sum_rms: ", sum_rms, " rms: ", sqrt(sum_rms / count))
end

function give_sum_rms(images_pred, images_obs)
    """
    Computes rms between predicted and observed image positions. Can handle false predictions as well
    since it matches the pairs first. The arguments are vectors of tuples of (x, y) positions.
    """

    sum_rms = 0.0

    for image in images_obs
        pred_closest = nothing
        len_closest = Inf
        for pred in images_pred
            dist = sqrt((image[1] - pred[1])^2 + (image[2] - pred[2])^2)
            if dist < len_closest
                len_closest = dist
                pred_closest = pred
            end
        end
        sum_rms += len_closest^2
    end
    return sum_rms
end

end

module LikelihoodFunctions

using LensFactory
using LensFactory.Constants
using LensFactory.LensModel.LensModelIO
using FITSIO
using JLD2
using Interpolations
using LinearAlgebra
using CairoMakie
using Random
using ImageFiltering

function neg_logprior_MEM(κ_vec::M, prior_kappa::N, reg_factor::T) where {M <: ROA, N <: ROA, T <: RV}
   """
   Give the log prior for the convergence matrix κ. 
   The prior_flag can be used to specify different types of priors, such as "flat", "gaussian", etc.
   Currently, this function is a placeholder and needs to be implemented based on the specific prior requirements.
   """

   κ = reshape(κ_vec, size(prior_kappa))  # Reshape κ_vec to match the shape of prior_kappa
   
   # κ will always be +ve by construction
   return reg_factor * sum( @. prior_kappa - κ + κ * log(κ / prior_kappa) )

end

function neg_loglikelihood_MEM(model::ModelConfig, lens::Lenses.AbstractLens, param_ref::Dict{Tuple{Symbol,Symbol},Float64}, full_kernel::Vector{Vector{NTuple{6, Matrix{Float64}}}} = nothing)
   """
   Computes the log-likelihood(-0.5 * χ²) for the convergence matrix κ given the observed data.
   data is a dictionary containing the observed data and its associated uncertainties.
   """
   adis = LensModel.LensModelUtils.adis_current(model, param_ref)

   if model.sampler.scheme == :SourcePlane
      # Calculate deflection at image positions
      t00 = time()
      _, αx_all, αy_all, A_all = LensModel.LensModelUtils.lens_quantities(model, lens, full_kernel)
      t01 = time()
      #print("lens quantities calc took: ", t01-t00, "  s, ")

      # Calculate position likelihood
      pos_chi2 = LensModel.Likelihood.chi2_sourceplane(model, adis, αx_all, αy_all, A_all)
      t02 = time()
      #println("chi2 calc took: ", t02-t01, "  s, ")
   else
      error("Unsupported sampling scheme: $(model.sampler.scheme)")
   end

   return -0.5 * pos_chi2

end

function construct_prior(model::ModelConfig; prior_flag::Int = 1, prior_value::RV =  0.5, seed::Union{Int, Nothing} = 1234, pix::Union{Int, Nothing} = 2, sigma::Union{RV, Nothing} = 10.0)
    """
    Construct the convergence matrix κ based on the model configuration and parameter references.
    This function is a placeholder and needs to be implemented based on the specific requirements for constructing κ.
    """

    X_max, Y_max = model.observation.FOV
    pixel_scale = model.observation.pixel_scale
    gridx, gridy = Lenses.get_meshgrid(X_max, Y_max, pixel_scale)

    if prior_flag == 1
        prior_kappa = fill(prior_value, size(gridx))
    elseif prior_flag == 2
        # uniform random grid, then gaussian filtered 
        rng = MersenneTwister(seed)
        random_grid = prior_value .* rand(rng, size(gridx,1), size(gridx,2))
        prior_kappa = imfilter(random_grid, Kernel.gaussian(pix))
    elseif prior_flag == 3
        # gaussian random grid, κ = magnitude at grid centre
        prior_kappa = prior_value .* exp.(-((gridx .^ 2) + (gridy .^ 2)) ./ (2 * sigma^2))
    else
        error("Unsupported prior type: $prior_flag")
    end

    prior_kappa = clamp.(prior_kappa, 0.1, Inf)

    return prior_kappa, gridx, gridy
end

function logprior_grad!(κ_vec::M, prior_kappa::N, reg_factor::T) where {M <: ROA, N <: ROA, T <: RV}
    """
    Compute the gradient of log-prior wrt κ.
    We have ∂logP/∂κ_{mn} = reg_factor * log(κ_{mn}/prior_kappa_{mn})
    """

   κ = reshape(κ_vec, size(prior_kappa))  # Reshape κ_vec 
   lp_grad_κ = zeros(size(prior_kappa))  # Initialize gradient array with the same shape as prior_kappa

   ax1, ax2 = axes(prior_kappa, 1), axes(prior_kappa, 2)

   @inbounds for j in ax2
      @inbounds for i in ax1
        lp_grad_κ[i,j] = reg_factor * log(κ[i,j] / prior_kappa[i,j])
      end
   end

   return vec(lp_grad_κ)  # Flatten the gradient vector 

end

function loglikelihood_grad!(κ_vec::M, prior_kappa::N, gridx::N, gridy::N, model::ModelConfig, param_ref::Dict{Tuple{Symbol, Symbol},Float64}, full_kernel::Vector{Vector{NTuple{6, Matrix{Float64}}}}) where {M <: ROA, N <: ROA}
    """
    Compute the gradient of log-likelihood wrt κ.
    This function uses finite differences to approximate the gradient.
    """
    κ = reshape(κ_vec, size(prior_kappa))  # Reshape κ_vec to match the shape of prior_kappa
    f0 = neg_loglikelihood_MEM(model, Main.FreeFormLens.init_FreeFormLens(κ, gridx, gridy, true), param_ref, full_kernel)
        
    ll_grad = zeros(length(κ_vec))
    buf = copy(κ_vec)
    for i in eachindex(κ_vec)

        h_i = max(1e-8, abs(κ_vec[i]) * 1e-5)       # step size based on the magnitude of κ_vec[i]

        buf[i] += h_i

        ll_grad[i] = (neg_loglikelihood_MEM(
            model,
            Main.FreeFormLens.init_FreeFormLens(reshape(buf, size(gridx)), gridx, gridy, true),
            param_ref,
            full_kernel
        ) - f0) / h_i

        buf[i] = κ_vec[i]  # Reset the buffer for the next iteration
    end
    return vec(ll_grad)
end

end
