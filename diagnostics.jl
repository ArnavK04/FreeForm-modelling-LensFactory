using LensFactory
using LensFactory.Constants
using LensFactory.LensModel.LensModelIO
using JLD2
using Interpolations
using CairoMakie

include("FreeFormLens.jl")

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

function refine_map(map_coarse::M, gridx::M, gridy::M, resolution::T) where {T <:RV, M <: ROA}
    """
    Uses Bicubic interpolation to return a higher res map of the convergence κ.
    """
    x_nodes = range(gridx[1, 1], stop=gridx[end, 1], length=size(gridx, 2))
    y_nodes = range(gridy[1, 1], stop=gridy[1, end], length=size(gridy, 1))

    itp = interpolate(map_coarse, BSpline(Cubic(Line(OnGrid()))))
    itp = scale(itp, x_nodes, y_nodes)

    x_fine, y_fine = Lenses.get_meshgrid(maximum(x_nodes), maximum(y_nodes), resolution)

    map_fine = itp.(x_fine, y_fine)


    println("before returning")
    println(size(map_fine))
    println(x_fine)
    println(y_fine)

    return map_fine, x_fine, y_fine

end

function main()
    name = "MEM_fit_result8x8_1_125FOV"
    filename = "../Diagnostics/files/$name" * ".jld2"

    global prior_kappa, gridx, gridy, model, param_ref, reg_factor
    # loading the jld2 file
    data = load(filename)
    model = data["model_config"]
    κ_map = data["κ_map"]
    prior_kappa = data["prior_kappa"]
    reg_factor = data["reg_factor"]
    errors = data["errors"]
    param_ref = Dict(p.key => p.refer for p in model.parameters)

    # making the grid
    gridx, gridy = make_gridfrom_model(model)

    res = 1  # 1 arcsec resolution for the refined map
    κ_fine, x_fine, y_fine = refine_map(κ_map, gridx, gridy, res)
    # initialize the lens
    free_lens = FreeFormLens.init_FreeFormLens(κ_fine, x_fine, y_fine)
    println("Lens initialized.")

    zs = 9
    zd = model.observation.z_d
    cosmo = Cosmology.init_cosmology()      # default cosmo
    Dds = Cosmology.angular_diameter_distance(cosmo, zd, zs)
    Ds = Cosmology.angular_diameter_distance(cosmo, 0.0, zs)
    adis = Dds / Ds

    err_fig, err_axes = Lenses.plot_sky(gridx, gridy)
    hm = heatmap!(err_axes, gridx[:,1], gridy[1,:], errors, colormap = :turbo, colorrange = (0, maximum(errors)))
    cb = Colorbar(err_fig[1,2], hm; label = "δκ", width = 20)
    save("../Diagnostics/plots/$name" * "_error_map.png", err_fig)

    rel_errors = errors ./ κ_map
    relerr_fig, relerr_axes = Lenses.plot_sky(gridx, gridy)
    hm_rel = heatmap!(relerr_axes, gridx[:,1], gridy[1,:], rel_errors, colormap = :turbo, colorrange = (0, 2))
    cb_rel = Colorbar(relerr_fig[1,2], hm_rel; label = "δκ/κ", width = 20)
    save("../Diagnostics/plots/$name" * "_relative_error_map.png", relerr_fig)

    μ_fig, μ_axes = Lenses.plot_magnification_map(free_lens, x_fine, y_fine, adis, heatmap_kws = (colormap = :turbo, colorrange = (0,100)))
    save("../Diagnostics/plots/$name" * "_magnification_map.png", μ_fig)

    cc_fig, cc_axes = Lenses.plot_image_plane(free_lens, x_fine, y_fine, adis, two_panel = true)
    save("../Diagnostics/plots/$name" * "_critical_curves.png", cc_fig)

    κ_fig, κ_axes = Lenses.plot_surface_density(free_lens, x_fine, y_fine, adis, unit = :convergence, heatmap_kws = (colormap = :turbo, colorrange = (0,maximum(κ_fine))))
    save("../Diagnostics/plots/$name" * "_kappa_map.png", κ_fig)

    prof_fig, prof_axis = Lenses.plot_magnification_profile(free_lens, x_fine, y_fine, adis)
    save("../Diagnostics/plots/$name" * "_magnification_profile.png", prof_fig)

    println("χ² of predicted image positions: ", data["chi2"])

end

main()