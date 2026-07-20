using LensFactory
using LensFactory.Constants
using LensFactory.LensModel.LensModelIO
using JLD2
using Interpolations
using CairoMakie
using FITSIO
using LinearAlgebra
using ArgParse

include("FreeFormLens.jl")
include("utility_functions.jl")

function main()

    settings = ArgParseSettings()

    @add_arg_table settings begin
        "--name"
        help = "path to saved model file"
        arg_type = String
        default = nothing
        "--fits_flag"
	help = "whether to plot fits truth maps"
	arg_type = Bool
	default = false
	"--plot_file_diag"
	help = "whether to plot run diagnostics"
	arg_type = Bool
	default = false
	"--plot_image_flag"
	help = "whether to plot each image prediction by the model"
	arg_type = Bool
	default = false
	"--plot_image_flag_og"
	help = "whether to plot image prediction by the truth"
	arg_type = Bool
	default = false
	"--X_lim"
	help = "X_lim to plot"
	arg_type = Float64
	default = 150.0
        "--Y_lim"
        help = "Y_lim to plot"
        arg_type = Float64
        default = 150.0
        "--res"
        help = "resolutin at which to plot"
        arg_type = Float64
        default = 2.0
    end

    args = parse_args(settings)

    name = args["name"]
    fits_flag = args["fits_flag"]
    plot_file_diag = args["plot_file_diag"]
    plot_image_flag = args["plot_image_flag"]
    plot_image_flag_og = args["plot_image_flag_og"]

    X_lim = args["X_lim"]
    Y_lim = args["Y_lim"]
    res = args["res"]
    fact = round(Int, res/0.14)
    println("loading fits data..")
    gridx_fits, gridy_fits, kappa, gamma1, gamma2 = UtilityFunctions.load_fitsfile("Ares")
    kappa = Float64.(kappa)  # Ensure kappa is of type Float64
    gamma1 = Float64.(gamma1)  # Ensure gamma1 is of type Float64
    gamma2 = Float64.(gamma2)  # Ensure gamma2 is of type Float64
    println("interpolating...")
    # interpolate to a particular grid
    order = 3  # cubic interpolation
    kappa_finefits, gridx_finefits, gridy_finefits = UtilityFunctions.refine_map(kappa, gridx_fits, gridy_fits, X_lim, Y_lim, res, order)
    gamma1_finefits, _, _ = UtilityFunctions.refine_map(gamma1, gridx_fits, gridy_fits, X_lim, Y_lim, res, order)
    gamma2_finefits, _, _ = UtilityFunctions.refine_map(gamma2, gridx_fits, gridy_fits, X_lim, Y_lim, res, order)
    mag_finefits = zeros(size(kappa_finefits)) 
    @. mag_finefits = 1.0 / ((1.0 - kappa_finefits)^2 - (gamma1_finefits^2 + gamma2_finefits^2))
    
    ares_path = "../Ares_data/$(res)/"
    if fits_flag

        mkpath(ares_path)

        fig_, axes_ = Lenses.plot_sky(gridx_finefits, gridy_finefits)
        hm = heatmap!(axes_, gridx_finefits[:,1], gridy_finefits[1,:], kappa_finefits, colormap = :turbo, colorrange = (0, 3.75))
        cb = Colorbar(fig_[1,2], hm; label = "κ", width = 20)
        save(ares_path * "kappa_finefits.png", fig_)

        fig_, axes_ = Lenses.plot_sky(gridx_finefits, gridy_finefits)
        hm = heatmap!(axes_, gridx_finefits[:,1], gridy_finefits[1,:], gamma1_finefits, colormap = :turbo, colorrange = (-2.5, 2.5))
        cb = Colorbar(fig_[1,2], hm; label = "γ₁", width = 20)
        save(ares_path * "gamma1_finefits.png", fig_)

        fig_, axes_ = Lenses.plot_sky(gridx_finefits, gridy_finefits)
        hm = heatmap!(axes_, gridx_finefits[:,1], gridy_finefits[1,:], gamma2_finefits, colormap = :turbo, colorrange = (-2.5, 2.5))
        cb = Colorbar(fig_[1,2], hm; label = "γ₂", width = 20)
        save(ares_path * "gamma2_finefits.png", fig_)

        fig_, axes_ = Lenses.plot_sky(gridx_finefits, gridy_finefits)
        hm = heatmap!(axes_, gridx_finefits[:,1], gridy_finefits[1,:], abs.(mag_finefits), colormap = :turbo, colorrange = (0, 100))
        cb = Colorbar(fig_[1,2], hm; label = "|μ|", width = 20)
        save(ares_path * "mag_finefits.png", fig_)
    end

    filename = "../Diagnostics/files/$name" * ".jld2"
    if plot_file_diag || plot_image_flag
        mkpath("../Diagnostics/plots/$(name)_res_$(res)/")
    end

    global prior_kappa, gridx, gridy, model, param_ref, reg_factor
    # loading the jld2 file
    data = load(filename)
    model = data["model_config"]
    κ_map = data["κ_map"]
    prior_kappa = data["prior_kappa"]
    reg_factor = data["reg_factor"]
    init_guess = data["init_guess"]
    #errors = data["errors"]
    param_ref = Dict(p.key => p.refer for p in model.parameters)

    println("size of κ_map: ", size(κ_map))
    println("size of prior_kappa: ", size(prior_kappa))

    zs = 9
    zd = model.observation.z_d
    cosmo = Cosmology.init_cosmology()      # default cosmo
    Dds = Cosmology.angular_diameter_distance(cosmo, zd, zs)
    Ds = Cosmology.angular_diameter_distance(cosmo, 0.0, zs)
    adis = Dds / Ds

    # making the grid
    gridx, gridy = UtilityFunctions.make_gridfrom_model(model)
    println("size of gridx: ", size(gridx))
    println("size of gridy: ", size(gridy))

    κ_fine, x_fine, y_fine = UtilityFunctions.refine_map(κ_map, gridx, gridy, X_lim, Y_lim, res, order)
    prior_kappa_fine, _, _ = UtilityFunctions.refine_map(prior_kappa, gridx, gridy, X_lim, Y_lim, res, order)
    init_guess_fine, _, _ = UtilityFunctions.refine_map(init_guess, gridx, gridy, X_lim, Y_lim, res, order)
    # initialize the lens
    free_lens = FreeFormLens.init_FreeFormLens(κ_fine, x_fine, y_fine)
    ares_lens = FreeFormLens.init_FreeFormLens(kappa_finefits ./ adis, gridx_finefits, gridy_finefits)
    println("Lens initialized.")

    κ_map .*= adis
    κ_fine .*= adis
    prior_kappa_fine .*= adis
    init_guess_fine .*= adis
    #errors .*= adis            # rescaling errors to source redshift 9

    if plot_file_diag
        mag_fine = Lenses.get_magnification_image(free_lens, x_fine, y_fine, adis)

        println("plotting the results...")
        fig_, axes_ = Lenses.plot_sky(gridx_finefits, gridy_finefits)
        hm = heatmap!(axes_, gridx_finefits[:,1], gridy_finefits[1,:], (mag_fine .- mag_finefits)./mag_finefits, colormap = :BrBG, colorrange = (-1.0, 4.0))
        cb = Colorbar(fig_[1,2], hm; label = L"(μ - μ_{truth})/μ_{truth}", width = 20)
        save("../Diagnostics/plots/$(name)_res_$(res)/$(name)_mag_rel_deviation.png", fig_)

        fig_, axes_ = Lenses.plot_sky(gridx_finefits, gridy_finefits)
        hm = heatmap!(axes_, gridx_finefits[:,1], gridy_finefits[1,:], (κ_fine .- kappa_finefits)./kappa_finefits, colormap = :afmhot, colorrange = (-1.0, 2.0))
        cb = Colorbar(fig_[1,2], hm; label = L"(κ - κ_{truth})/κ_{truth}", width = 20)
        save("../Diagnostics/plots/$(name)_res_$(res)/$(name)_kappa_rel_deviation.png", fig_)

        fig_, axes_ = Lenses.plot_sky(gridx_finefits, gridy_finefits)
        hm = heatmap!(axes_, gridx_finefits[:,1], gridy_finefits[1,:], prior_kappa_fine, colormap = :turbo, colorrange = (0, 3.75))
        cb = Colorbar(fig_[1,2], hm; label = "κ_prior", width = 20)
        save("../Diagnostics/plots/$(name)_res_$(res)/$(name)_prior_kappa_map.png", fig_)

        fig_, axes_ = Lenses.plot_sky(gridx_finefits, gridy_finefits)
        hm = heatmap!(axes_, gridx_finefits[:,1], gridy_finefits[1,:], init_guess_fine, colormap = :turbo, colorrange = (0, 3.75))
        cb = Colorbar(fig_[1,2], hm; label = "κ_init_guess", width = 20)
        save("../Diagnostics/plots/$(name)_res_$(res)/$(name)_init_guess_kappa_map.png", fig_)

        """err_fig, err_axes = Lenses.plot_sky(gridx, gridy)
        hm = heatmap!(err_axes, gridx[:,1], gridy[1,:], errors, colormap = :turbo, colorrange = (0, maximum(errors)))
        cb = Colorbar(err_fig[1,2], hm; label = "δκ", width = 20)
        save("../Diagnostics/plots/$(name)_res_$(res)/$(name)_error_map.png", err_fig)

        rel_errors = errors ./ κ_map
        relerr_fig, relerr_axes = Lenses.plot_sky(gridx, gridy)
        hm_rel = heatmap!(relerr_axes, gridx[:,1], gridy[1,:], rel_errors, colormap = :turbo, colorrange = (0, 2))
        cb_rel = Colorbar(relerr_fig[1,2], hm_rel; label = "δκ/κ", width = 20)
        save("../Diagnostics/plots/$(name)_res_$(res)/$(name)_relative_error_map.png", relerr_fig)"""

        μ_fig, μ_axes = Lenses.plot_magnification_map(free_lens, x_fine, y_fine, adis, heatmap_kws = (colormap = :turbo, colorrange = (0,100)))
        save("../Diagnostics/plots/$(name)_res_$(res)/$(name)_magnification_map.png", μ_fig)

        cc_fig, cc_axes = Lenses.plot_image_plane(free_lens, x_fine, y_fine, adis, two_panel = true)
        save("../Diagnostics/plots/$(name)_res_$(res)/$(name)_critical_curves.png", cc_fig)

        κ_fig, κ_axes = Lenses.plot_surface_density(free_lens, x_fine, y_fine, adis, unit = :convergence, heatmap_kws = (colormap = :turbo, colorrange = (0, 3.75)))
        save("../Diagnostics/plots/$(name)_res_$(res)/$(name)_kappa_map.png", κ_fig)

        prof_fig, prof_axis = Lenses.plot_magnification_profile(free_lens, x_fine, y_fine, adis)
        save("../Diagnostics/plots/$(name)_res_$(res)/$(name)_magnification_profile.png", prof_fig)

        println("χ² of predicted image positions: ", data["chi2"])
    end
    
    if plot_image_flag_og
        rms = UtilityFunctions.give_image_rmsscatter(model, ares_lens, param_ref, gridx_finefits, gridy_finefits, plot_image_flag_og, ares_path)
        open(ares_path * "rms.txt", "a") do io
            println(io, "rms of image positions: " * string(rms))
        end
    elseif plot_image_flag
        rms = UtilityFunctions.give_image_rmsscatter(model, free_lens, param_ref, x_fine, y_fine, plot_image_flag, "../Diagnostics/plots/$(name)_res_$(res)/")
        # save the rms to a text file
        open("../Diagnostics/plots/$(name)_res_$(res)/rms.txt", "a") do io
            println(io, "rms of image positions: " * string(rms))
            println(io, "χ² of predicted image positions: ", data["chi2"])
        end
    end
    

end

main()
