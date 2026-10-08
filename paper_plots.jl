using LensFactory
using LensFactory.Constants
using CairoMakie
using JLD2
using LaTeXStrings

include("FreeFormLens.jl")
include("utility_functions.jl")

#=
function plot_magnification_scatter(model::LensModel.ModelConfig,
                            lens_model::Lenses.AbstractLens,
                            x_grid::M,
                            y_grid::M;
                            save_plot::Bool        = true,
                            plot_name::String      = "./magnification_scatter.png",
                            resolution::Int64      = 2,
                            point_kws::NamedTuple  = (markersize  = 5, 
                                                    marker      = :circle, 
                                                    color       = :transparent, 
                                                    strokecolor = :black, 
                                                    strokewidth = 2),
                            gridqty_tuple::NTuple{6, M} = nothing, imgqty_tuple = nothing) where {M <: ROA}

   # Get list of parameters for the lens model
   param_ref = Dict(p.key => p.refer for p in model.parameters)
 
   cosmo = Cosmology.init_cosmology()       # right now it initialises default cosmo only.   
   # Get angular-diameter distance ratios
   adis = LensFactory.LensModel.adis_current(model, param_ref)

   # Collect residuals across every knot/image
   mag_obs_all = Float64[]
   mag_pred_all = Float64[]

   sid = 1
   kid = 1
   kid_global = 1
   for src in model.source_config.sources
      # Angular-diameter distance ratio for this source
      adis_value = adis[sid]
      kid = 1
      for knot in src.knots
         # Knot positions and measurement errors
         x  = knot.x
         y  = knot.y
         σx = knot.σx
         σy = knot.σy
         σθ = knot.σθ
         ψkid = imgqty_tuple[1][kid_global]
         αxkid = imgqty_tuple[2][kid_global]
         αykid = imgqty_tuple[3][kid_global]
         ψxxkid = imgqty_tuple[4][kid_global][1]
         ψyykid = imgqty_tuple[4][kid_global][4]
         ψxykid = imgqty_tuple[4][kid_global][2]
         kidqty_tuple = (ψkid, αxkid, αykid, ψxxkid, ψyykid, ψxykid)

         # Number of images for this knot
         n = length(x)
 
         # Predicted image positions
         predicted_image = UtilityFunctions.predict_image(lens_model, x_grid, y_grid, x, y, adis_value, sid, kid, nothing, false, nothing, gridqty_tuple, kidqty_tuple)
         # Convert predicted to mutable arrays for iterative removal
         pred_x = Float64[p[1] for p in predicted_image]
         pred_y = Float64[p[2] for p in predicted_image]

         psixx = adis_value * kidqty_tuple[4]
         psixy = adis_value * kidqty_tuple[6]
         psiyy = adis_value * kidqty_tuple[5]

         mag_obs_kid = @. 1.0 / (1.0 - psixx - psiyy + psixx * psiyy - psixy^2)

         A_all_kid_pred = LensFactory.Lenses.get_jacobian(lens_model, pred_x, pred_y)
         psixx_pred = adis_value * A_all_kid_pred[1]
         psiyy_pred = adis_value * A_all_kid_pred[2]
         psixy_pred = adis_value * A_all_kid_pred[3]

         mag_pred_kid = @. 1.0 / (1.0 - psixx_pred - psiyy_pred + psixx_pred * psiyy_pred - psixy_pred^2)

         # Matching observed images to predicted images
         for i in 1:n
            if isempty(pred_x)
               push!(results, "MISSING")
               continue
            end

            # Calculate distances to all remaining candidates
            dx = @. pred_x .- x[i]
            dy = @. pred_y .- y[i]
            dist_sq = @. dx^2 + dy^2

            # Find the closest predicted image index
            best_idx = argmin(dist_sq)

            # ratio of magnifications, assumed same ordering/length as knot.x, knot.y
            push!(mag_obs_all, mag_obs_kid[i])
            push!(mag_pred_all, mag_pred_kid[best_idx])

            # Remove this candidate so it can't be matched twice
            deleteat!(pred_x, best_idx)
            deleteat!(pred_y, best_idx)
            deleteat!(mag_pred_kid, best_idx)
         end
         kid_global = kid_global + 1
         kid = kid + 1
      end
      sid = sid + 1
   end

   fig = Figure(size = (500, 400))

   ax = Axis(fig[1, 1];
      xlabel = L"\mu_{obs}",
      ylabel = L"\mu_{pred}",
      xscale = log10,
      yscale = log10,
      xlabelsize = 20,
      ylabelsize = 20
   )

   # Residuals
   scatter!(ax, abs.(mag_obs_all), abs.(mag_pred_all); point_kws...)

   xmin, xmax = extrema(abs.(mag_obs_all))
   ymin, ymax = extrema(abs.(mag_pred_all))

   lo = min(xmin, ymin)
   hi = max(xmax, ymax)

   lines!(
      ax,
      [lo, hi],
      [lo, hi];
      linestyle = :dash,
      label = L"\mu_{obs} = \mu_{pred}"
   )
   # Legend
   axislegend(ax, position = :rb, framevisible = false, labelsize = 20)

   # Save
   if save_plot
      save(plot_name, fig; px_per_unit = resolution)
   end

   return fig, ax
end
=#

foldername = "/home/arnavkumar/juliacodes/ashish_workstation_data/Heraplots/MARS_FOV_reiter0_res2.0/Hera_MEM_fit_reg1.0_gflag1_pvalue0.5_0.5_nothing_nothing_nothing_10_1.0_70.0_70.0_res_0.25_thres_4.0"
filename = "Hera_MEM_fit_reg1.0_gflag1_pvalue0.5_0.5_nothing_nothing_nothing_10_1.0_70.0_70.0_diagnostics.jld2"       
filepath = joinpath(foldername, filename)        # path to file containing maps
mkpath(joinpath(foldername, "paperplots"))
# load the file

data = load(filepath)

#=     gridx_finefits = gridx_finefits,
        gridy_finefits = gridy_finefits,
        gridx          = gridx,
        gridy          = gridy,
        mag_fine       = mag_fine,
        mag_finefits   = mag_finefits,
        κ_fine         = κ_fine_,
        kappa_finefits = kappa_finefits,
        prior_kappa_fine = prior_kappa_fine_,
        init_guess_fine  = init_guess_fine_,
        κ_diff         = κ_diff_,
        κ_reldiff      = κ_reldiff,
        κ_diff_fine    = κ_diff_fine_,
        κ_reldiff_fine = κ_reldiff_fine_,
        img_pts        = makiepts,
        X_lim_plot     = X_lim_plot,
        Y_lim_plot     = Y_lim_plot,
        X_lim          = X_lim,
        Y_lim          = Y_lim,
        cluster        = clustername,
        res            = res,
        thres          = thres,
        name           = name,
        foldername     = foldername,
        RMS            = rms,
        count          = count,
        total_img      = total_img,
        χ²             = data["chi2"]
=#

println("Available keys:")
println(keys(data))

model = data["model"]
gridx_finefits = data["gridx_finefits"]
gridy_finefits = data["gridy_finefits"]
gridx_coarse = data["gridx"]
gridy_coarse = data["gridy"]
mag_model = data["mag_fine"]
mag_truth = data["mag_finefits"]
kappa_model = data["κ_fine"]
kappa_truth = data["kappa_finefits"]        # all are at z = 9
prior_kappa_fine = data["prior_kappa_fine"]
init_guess_fine = data["init_guess_fine"]
kappa_diff = data["κ_diff"]
kappa_rel_diff = data["κ_reldiff"]
kappa_diff_fine = data["κ_diff_fine"]
kappa_rel_diff_fine = data["κ_reldiff_fine"]
img_pts = data["img_pts"]
X_lim_plot = data["X_lim_plot"]
Y_lim_plot = data["Y_lim_plot"]
X_lim = data["X_lim"]
Y_lim = data["Y_lim"]
clustername = data["cluster"]
res = data["res"]
thres = data["thres"]
name = data["name"]
RMS = data["RMS"]
count = data["count"]
total_img = data["total_img"]
src_chi2 = data["χ²"]

if clustername == "Hera"
    kappa_range = (0.0, 2.70)
elseif clustername == "Ares"
    kappa_range = (0.0, 3.75)
else
    kappa_range = (0.0, 3.75)
end
mag_range = (0.0, 100.0)  # range of magnification values for colorbar
kappa_reldev_range = (-1.0, 2.0)  # range of kappa rel deviation values for colorbar
magnif_reldev_range = (-1.0, 4.0)  # range of magnification rel deviation values for colorbar

# plots to be generated
println("Generating paper plots...")

fig, axes = Lenses.plot_sky(gridx_finefits, gridy_finefits)
hm = heatmap!(axes, gridx_finefits[:,1], gridy_finefits[1,:], kappa_model, colormap = :turbo, colorrange = kappa_range)
cb = Colorbar(fig[1,2], hm; label = L"κ", width = 20)
xlims!(axes, -X_lim_plot, X_lim_plot)
ylims!(axes, -Y_lim_plot, Y_lim_plot)
save(foldername*"/"*"paperplots/"*"kappa_map.png", fig)

fig, axes = Lenses.plot_sky(gridx_finefits, gridy_finefits)
hm = heatmap!(axes, gridx_finefits[:,1], gridy_finefits[1,:], abs.(mag_model), colormap = :turbo, colorrange = mag_range)
cb = Colorbar(fig[1,2], hm; label = L"|μ|", width = 20)
xlims!(axes, -X_lim_plot, X_lim_plot)
ylims!(axes, -Y_lim_plot, Y_lim_plot)
save(foldername*"/"*"paperplots/"*"magnification_map.png", fig)

fig, axes = Lenses.plot_sky(gridx_finefits, gridy_finefits)
hm = heatmap!(axes, gridx_finefits[:,1], gridy_finefits[1,:], (kappa_model .- kappa_truth)./kappa_truth, colormap = :afmhot, colorrange = kappa_reldev_range)
cb = Colorbar(fig[1,2], hm; label = L"(κ - κ_{truth})/κ_{truth}", width = 20)
xlims!(axes, -X_lim_plot, X_lim_plot)
ylims!(axes, -Y_lim_plot, Y_lim_plot)
save(foldername*"/"*"paperplots/"*"kappa_devmap.png", fig)

fig, axes = Lenses.plot_sky(gridx_finefits, gridy_finefits)
hm = heatmap!(axes, gridx_finefits[:,1], gridy_finefits[1,:], (mag_model .- mag_truth)./mag_truth, colormap = :BrBG, colorrange = magnif_reldev_range)
cb = Colorbar(fig[1,2], hm; label = L"(|μ| - |μ|_{truth})/|μ|_{truth}", width = 20)
xlims!(axes, -X_lim_plot, X_lim_plot)
ylims!(axes, -Y_lim_plot, Y_lim_plot)
save(foldername*"/"*"paperplots/"*"magnification_devmap.png", fig)

# need to add these plots here
# radial profiles

# kappa contours on fits is from another py file


