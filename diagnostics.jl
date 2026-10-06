using LensFactory
using LensFactory.Constants
using LensFactory.LensModel.LensModelIO
using JLD2
using Interpolations
using CairoMakie
using FITSIO
using LinearAlgebra
using ArgParse
using Optim
using Printf
using Statistics
using LaTeXStrings

include("FreeFormLens.jl")
include("utility_functions.jl")


function _write_fits_header!(
    header::ImageHDU,
    model::ModelConfig,
    x_grid::AbstractMatrix,
    y_grid::AbstractMatrix)

    # x varies along dim 1, y along dim 2
    nx, ny = size(x_grid)
    @assert size(y_grid) == (nx, ny) "x_grid and y_grid must have the same size"

    RA_REF  = model.observation.reference[1]
    DEC_REF = model.observation.reference[2]
    if RA_REF == 0.0 && DEC_REF == 0.0
        @warn "Reference position is (0.0, 0.0). Are you sure?"
    end

    dx = x_grid[2, 1] - x_grid[1, 1]
    dy = y_grid[1, 2] - y_grid[1, 1]
    @assert dx != 0 && dy != 0 "degenerate grid spacing: dx=$dx dy=$dy"

    FOV_x = abs(dx) * nx
    FOV_y = abs(dy) * ny

    ix = argmin(abs.(x_grid[:, 1]))
    iy = argmin(abs.(y_grid[1, :]))

    write_key(header, "CTYPE1", "RA---TAN", "RA coordinate type")
    write_key(header, "CTYPE2", "DEC--TAN", "DEC coordinate type")
    write_key(header, "CUNIT1", "deg", "Units of axis 1")
    write_key(header, "CUNIT2", "deg", "Units of axis 2")
    write_key(header, "RADESYS", "ICRS", "Reference frame")

    write_key(header, "CRVAL1", RA_REF, "RA reference value")
    write_key(header, "CRVAL2", DEC_REF, "DEC reference value")
    write_key(header, "CRPIX1", Float64(ix), "Reference pixel in x-direction")
    write_key(header, "CRPIX2", Float64(iy), "Reference pixel in y-direction")
    write_key(header, "CDELT1", -abs(dx) / 3600.0, "Pixel scale in RA (degrees)")
    write_key(header, "CDELT2",  abs(dy) / 3600.0, "Pixel scale in DEC (degrees)")

    # NAXIS1/NAXIS2 removed: FITSIO writes them from the array
    write_key(header, "FOV_X", FOV_x, "Field of view along x (arcsec)")
    write_key(header, "FOV_Y", FOV_y, "Field of view along y (arcsec)")

    write_key(header, "MODELER", string(model.observation.modeler), "Modeler name")
    write_key(header, "LENS", string(model.observation.lens), "Lens name")
    write_key(header, "Z_D", model.observation.z_d, "Lens redshift")

    return nothing
end

function save_fits_file(model::LensModel.ModelConfig, map::M, x_grid::M, y_grid::M, name::String, foldername::String) where {M <: ROA}

    # Open new FITS file
    f = FITS("../Diagnostics/plots/$(foldername)/$(name).fits", "w")
    write(f, map)

    # Write header
    hdu = f[1]
    _write_fits_header!(hdu, model, x_grid, y_grid)

    close(f)

    return nothing
end

function main()

    time_start = time()

    settings = ArgParseSettings()

    @add_arg_table settings begin
        "--cname"
        help = "Name of the cluster being fitted."
        arg_type = String
        default = "NotSpecified"
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
        help = "X_lim for convergence map"
        arg_type = Float64
        default = 150.0
        "--Y_lim"
        help = "Y_lim for convergence map"
        arg_type = Float64
        default = 150.0
        "--X_lim_plot"
        help = "X_lim for plotting"
        arg_type = Float64
        default = 150.0
        "--Y_lim_plot"
        help = "Y_lim for plotting"
        arg_type = Float64
        default = 150.0
        "--res"
        help = "resolutin at which to plot"
        arg_type = Float64
        default = 2.0
        "--thres"
        help = "threshold distance for image matching"
        arg_type = Float64
        default = 1.0
    end

    args = parse_args(settings)

    name = args["name"]
    fits_flag = args["fits_flag"]
    plot_file_diag = args["plot_file_diag"]
    plot_image_flag = args["plot_image_flag"]
    plot_image_flag_og = args["plot_image_flag_og"]
    thres = args["thres"]
    X_lim = args["X_lim"]
    Y_lim = args["Y_lim"]
    X_lim_plot = args["X_lim_plot"]
    Y_lim_plot = args["Y_lim_plot"]
    res = args["res"]
    clustername = args["cname"]

    fact = round(Int, res/0.14)

    println("loading fits data..")
    time_loadstart = time()
    gridx_fits, gridy_fits, kappa, gamma1, gamma2 = UtilityFunctions.load_fitsfile(clustername)

    Hera_offset = 12.0      # Hera image coordinates were offset by 12 in x direction, so we need to correct for that
    if clustername == "Hera"
        gridx_fits .+= Hera_offset
    end

    kappa = Float64.(kappa)  # Ensure kappa is of type Float64
    gamma1 = Float64.(gamma1)  # Ensure gamma1 is of type Float64
    gamma2 = Float64.(gamma2)  # Ensure gamma2 is of type Float64
    println("$(clustername) data loaded in ", time() - time_loadstart, " seconds.")

    flush(stdout)
    println("interpolating...")
    # interpolate to a particular grid
    order = 3  # cubic interpolation
    refinestart = time()
    kappa_finefits, gridx_finefits, gridy_finefits = UtilityFunctions.refine_map(kappa, gridx_fits, gridy_fits, X_lim, Y_lim, res, order)
    gamma1_finefits, _, _ = UtilityFunctions.refine_map(gamma1, gridx_fits, gridy_fits, X_lim, Y_lim, res, order)
    gamma2_finefits, _, _ = UtilityFunctions.refine_map(gamma2, gridx_fits, gridy_fits, X_lim, Y_lim, res, order)
    magstrt = time()
    println("interpolation done in ", magstrt - refinestart, " seconds.")
    mag_finefits = @. 1.0 / ((1.0 - kappa_finefits)^2 - (gamma1_finefits^2 + gamma2_finefits^2))
    println("magnification calculated in ", time() - magstrt, " seconds.")
    flush(stdout)

    cluster_path = "../$(clustername)_data/res$(res)_thres$(thres)/"

    kappa_dev_range = (-1.0, 2.0)
    magnif_range = (0, 100)
    if clustername == "Hera"
        kappa_range = (0, 2.70)
    elseif clustername == "Ares"
        kappa_range = (0, 3.75)
    else
        kappa_range = (0, 3.75)
    end

    if fits_flag

        mkpath(cluster_path)

        clusterplotstart = time()
        fig_, axes_ = Lenses.plot_sky(gridx_finefits, gridy_finefits)
        hm = heatmap!(axes_, gridx_finefits[:,1], gridy_finefits[1,:], kappa_finefits, colormap = :turbo, colorrange = kappa_range)
        cb = Colorbar(fig_[1,2], hm; label = L"κ", width = 20)
        xlims!(axes_, -X_lim_plot, X_lim_plot)
        ylims!(axes_, -Y_lim_plot, Y_lim_plot)
        save(cluster_path * "kappa_finefits.png", fig_)

        fig_, axes_ = Lenses.plot_sky(gridx_finefits, gridy_finefits)
        hm = heatmap!(axes_, gridx_finefits[:,1], gridy_finefits[1,:], gamma1_finefits, colormap = :turbo, colorrange = (-2.5, 2.5))
        cb = Colorbar(fig_[1,2], hm; label = L"γ₁", width = 20)
        xlims!(axes_, -X_lim_plot, X_lim_plot)
        ylims!(axes_, -Y_lim_plot, Y_lim_plot)
        save(cluster_path * "gamma1_finefits.png", fig_)

        fig_, axes_ = Lenses.plot_sky(gridx_finefits, gridy_finefits)
        hm = heatmap!(axes_, gridx_finefits[:,1], gridy_finefits[1,:], gamma2_finefits, colormap = :turbo, colorrange = (-2.5, 2.5))
        cb = Colorbar(fig_[1,2], hm; label = L"γ₂", width = 20)
        xlims!(axes_, -X_lim_plot, X_lim_plot)
        ylims!(axes_, -Y_lim_plot, Y_lim_plot)
        save(cluster_path * "gamma2_finefits.png", fig_)

        fig_, axes_ = Lenses.plot_sky(gridx_finefits, gridy_finefits)
        hm = heatmap!(axes_, gridx_finefits[:,1], gridy_finefits[1,:], abs.(mag_finefits), colormap = :turbo, colorrange = magnif_range)
        cb = Colorbar(fig_[1,2], hm; label = L"|μ|", width = 20)
        xlims!(axes_, -X_lim_plot, X_lim_plot)
        ylims!(axes_, -Y_lim_plot, Y_lim_plot)
        save(cluster_path * "mag_finefits.png", fig_)

        println("$(clustername) plots saved in ", time() - clusterplotstart, " seconds.")

        flush(stdout)
    end

    foldername = "$(name)_res_$(res)_thres_$(thres)"
    filename = "../Diagnostics/files/$name" * ".jld2"
    if plot_file_diag || plot_image_flag
        mkpath("../Diagnostics/plots/$(foldername)/")
    end

    global prior_kappa, gridx, gridy, model, param_ref, reg_factor
    # loading the jld2 file
    data = load(filename)
    model = data["model_config"]
    κ_map = data["κ_map"]
    prior_kappa = data["prior_kappa"]
    reg_factor = data["reg_factor"]
    init_guess = data["init_guess"]
    traceofrun = data["trace"]
    gridx = data["gridx"]
    gridy = data["gridy"]
    κ_diff = data["κ_diff"]
    κ_reldiff = data["κ_reldiff"]

    param_ref = Dict(p.key => p.refer for p in model.parameters)

    println("size of κ_map: ", size(κ_map))
    println("size of prior_kappa: ", size(prior_kappa))

    zs = 9
    zd = model.observation.z_d
    cosmo = Cosmology.init_cosmology()      # default cosmo
    Dds = Cosmology.angular_diameter_distance(cosmo, zd, zs)
    Ds = Cosmology.angular_diameter_distance(cosmo, 0.0, zs)
    adis = Dds / Ds


    println("size of gridx: ", size(gridx))
    println("size of gridy: ", size(gridy))


    obsrefinestart = time()
    κ_fine, x_fine, y_fine = UtilityFunctions.refine_map(κ_map, gridx, gridy, X_lim, Y_lim, res, order)
    prior_kappa_fine, _, _ = UtilityFunctions.refine_map(prior_kappa, gridx, gridy, X_lim, Y_lim, res, order)
    init_guess_fine, _, _ = UtilityFunctions.refine_map(init_guess, gridx, gridy, X_lim, Y_lim, res, order)
    κ_diff_fine, _, _ = UtilityFunctions.refine_map(κ_diff, gridx, gridy, X_lim, Y_lim, res, order)
    κ_reldiff_fine, _, _ = UtilityFunctions.refine_map(κ_reldiff, gridx, gridy, X_lim, Y_lim, res, order)
    println("refining the optimal maps done in ", time() - obsrefinestart, " seconds.")

    flush(stdout)

    # initialize the lens
    initlenstime = time()
    free_lens_nokernel = FreeFormLens.init_FreeFormLens(κ_fine, x_fine, y_fine, false)
    free_lens = FreeFormLens.init_FreeFormLens(κ_fine, x_fine, y_fine, true)
    freefull_kernel = FreeFormLens.compute_fullkernel(model, x_fine, y_fine)
    ψ_all, αx_all, αy_all, A_all = LensModel.LensModelUtils.lens_quantities(model, free_lens, freefull_kernel)
    freeimgqty_tuple = (ψ_all, αx_all, αy_all, A_all)

    # compute the lens quantities for the difference map
    diff_lens = FreeFormLens.init_FreeFormLens(κ_diff_fine, x_fine, y_fine, true)
    ψ_diff_all, αx_diff_all, αy_diff_all, A_diff_all = LensModel.LensModelUtils.lens_quantities(model, diff_lens, freefull_kernel)
    diffimgqty_tuple = (ψ_diff_all, αx_diff_all, αy_diff_all, A_diff_all)

    # lens quantities for the truth map
    if plot_image_flag_og
        cluster_lens_nokernel = FreeFormLens.init_FreeFormLens(kappa_finefits ./ adis, gridx_finefits, gridy_finefits, false)
        cluster_lens = FreeFormLens.init_FreeFormLens(kappa_finefits ./ adis, gridx_finefits, gridy_finefits, true)
        ψ_cluster, αx_cluster, αy_cluster, A_cluster = LensModel.LensModelUtils.lens_quantities(model, cluster_lens, freefull_kernel)
        clusterimgqty_tuple = (ψ_cluster, αx_cluster, αy_cluster, A_cluster)
    end
    println("Lens initialized in ", time() - initlenstime, " seconds.")
    flush(stdout)

    # calculate the lens quantities over the whole grid
    lensqtystart = time()
    ψ_free = Lenses.get_potential(free_lens, x_fine, y_fine)
    αx_free, αy_free = Lenses.get_deflection(free_lens, x_fine, y_fine)
    ψxx_free, ψyy_free, ψxy_free = Lenses.get_jacobian(free_lens, x_fine, y_fine)
    free_qty_tuple = (ψ_free, αx_free, αy_free, ψxx_free, ψyy_free, ψxy_free)

    # lens quantities for the difference map
    ψ_diff = Lenses.get_potential(diff_lens, x_fine, y_fine)
    αx_diff, αy_diff = Lenses.get_deflection(diff_lens, x_fine, y_fine)
    ψxx_diff, ψyy_diff, ψxy_diff = Lenses.get_jacobian(diff_lens, x_fine, y_fine)
    diff_qty_tuple = (ψ_diff, αx_diff, αy_diff, ψxx_diff, ψyy_diff, ψxy_diff)

    # calculate the lens quantities for the truth map
    if plot_image_flag_og
        ψ_cluster = Lenses.get_potential(cluster_lens, x_fine, y_fine)
        αx_cluster, αy_cluster = Lenses.get_deflection(cluster_lens, x_fine, y_fine)
        ψxx_cluster, ψyy_cluster, ψxy_cluster = Lenses.get_jacobian(cluster_lens, x_fine, y_fine)
        cluster_qty_tuple = (ψ_cluster, αx_cluster, αy_cluster, ψxx_cluster, ψyy_cluster, ψxy_cluster)
    end
    println("Lensing quantities calculated in ", time() - lensqtystart, " seconds.")

    flush(stdout)
    
    κ_map_ = copy(κ_map) .* adis
    κ_fine_ = copy(κ_fine) .* adis
    prior_kappa_fine_ = copy(prior_kappa_fine) .* adis
    init_guess_fine_ = copy(init_guess_fine) .* adis
    κ_diff_ = copy(κ_diff) .* adis
    κ_diff_fine_ = copy(κ_diff_fine) .* adis
    κ_reldiff_fine_ = copy(κ_reldiff_fine) .* adis

    #errors .*= adis            # rescaling errors to source redshift 9

    if plot_file_diag

        mag_fine = @. 1.0 / (1.0 + adis * adis * ψxx_free * ψyy_free - adis * ψxx_free - adis * ψyy_free - adis * adis * ψxy_free^2)

        println("plotting the results...")
        fig_magdev, axes_magdev = Lenses.plot_sky(gridx_finefits, gridy_finefits)
        hm = heatmap!(axes_magdev, gridx_finefits[:,1], gridy_finefits[1,:], (mag_fine .- mag_finefits)./mag_finefits, colormap = :BrBG, colorrange = (-1.0, 4.0))
        cb = Colorbar(fig_magdev[1,2], hm; label = L"(μ - μ_{truth})/μ_{truth}", width = 20)
        xlims!(axes_magdev, -X_lim_plot, X_lim_plot)
        ylims!(axes_magdev, -Y_lim_plot, Y_lim_plot)
        save("../Diagnostics/plots/$(foldername)/$(name)_mag_rel_deviation.png", fig_magdev)

        fig_magdev2, axes_magdev2 = Lenses.plot_sky(gridx_finefits, gridy_finefits)
        hm = heatmap!(axes_magdev2, gridx_finefits[:,1], gridy_finefits[1,:], (mag_fine .- mag_finefits)./mag_finefits, colormap = :BrBG, colorrange = (-4.0, 4.0))
        cb = Colorbar(fig_magdev2[1,2], hm; label = L"(μ - μ_{truth})/μ_{truth}", width = 20)
        xlims!(axes_magdev2, -X_lim_plot, X_lim_plot)
        ylims!(axes_magdev2, -Y_lim_plot, Y_lim_plot)
        save("../Diagnostics/plots/$(foldername)/$(name)_mag_rel_deviationBrBG.png", fig_magdev2)

        fig_kappadev, axes_kappadev = Lenses.plot_sky(gridx_finefits, gridy_finefits)
        hm = heatmap!(axes_kappadev, gridx_finefits[:,1], gridy_finefits[1,:], (κ_fine_ .- kappa_finefits)./kappa_finefits, colormap = :afmhot, colorrange = kappa_dev_range)
        cb = Colorbar(fig_kappadev[1,2], hm; label = L"(κ - κ_{truth})/κ_{truth}", width = 20)
        xlims!(axes_kappadev, -X_lim_plot, X_lim_plot)
        ylims!(axes_kappadev, -Y_lim_plot, Y_lim_plot)
        save("../Diagnostics/plots/$(foldername)/$(name)_kappa_rel_deviation.png", fig_kappadev)

        fig_kappadev2, axes_kappadev2 = Lenses.plot_sky(gridx_finefits, gridy_finefits)
        hm = heatmap!(axes_kappadev2, gridx_finefits[:,1], gridy_finefits[1,:], (κ_fine_ .- kappa_finefits)./kappa_finefits, colormap = :BrBG, colorrange = (-2.0, 2.0))
        cb = Colorbar(fig_kappadev2[1,2], hm; label = L"(κ - κ_{truth})/κ_{truth}", width = 20)
        xlims!(axes_kappadev2, -X_lim_plot, X_lim_plot)
        ylims!(axes_kappadev2, -Y_lim_plot, Y_lim_plot)
        save("../Diagnostics/plots/$(foldername)/$(name)_kappa_rel_deviationBrBG.png", fig_kappadev2)

        fig_prior, axes_prior = Lenses.plot_sky(gridx_finefits, gridy_finefits)
        hm = heatmap!(axes_prior, gridx_finefits[:,1], gridy_finefits[1,:], prior_kappa_fine_, colormap = :turbo, colorrange = kappa_range)
        cb = Colorbar(fig_prior[1,2], hm; label = L"κ_{prior}", width = 20)
        xlims!(axes_prior, -X_lim_plot, X_lim_plot)
        ylims!(axes_prior, -Y_lim_plot, Y_lim_plot)
        save("../Diagnostics/plots/$(foldername)/$(name)_prior_kappa_map.png", fig_prior)

        fig_init, axes_init = Lenses.plot_sky(gridx_finefits, gridy_finefits)
        hm = heatmap!(axes_init, gridx_finefits[:,1], gridy_finefits[1,:], init_guess_fine_, colormap = :turbo, colorrange = kappa_range)
        cb = Colorbar(fig_init[1,2], hm; label = L"κ_{init_guess}", width = 20)
        xlims!(axes_init, -X_lim_plot, X_lim_plot)
        ylims!(axes_init, -Y_lim_plot, Y_lim_plot)
        save("../Diagnostics/plots/$(foldername)/$(name)_init_guess_kappa_map.png", fig_init)

        fig_diff, axes_diff = Lenses.plot_sky(gridx, gridy)
        hm = heatmap!(axes_diff, gridx[:,1], gridy[1,:], κ_diff_, colormap = :BrBG, colorrange = (-maximum(abs.(κ_diff_)), maximum(abs.(κ_diff_))))
        cb = Colorbar(fig_diff[1,2], hm; label = L"κ_{diff}", width = 20)
        xlims!(axes_diff, -X_lim_plot, X_lim_plot)
        ylims!(axes_diff, -Y_lim_plot, Y_lim_plot)
        Label(fig_diff[2,1], @sprintf("Abs Mean = %.3e     Max |deviation| = %.3e", mean(abs.(κ_diff_)), maximum(abs.(κ_diff_))), fontsize = 16, halign = :left)
        save("../Diagnostics/plots/$(foldername)/$(name)_kappa_diff_map.png", fig_diff)

        fig_reldiff, axes_reldiff = Lenses.plot_sky(gridx, gridy)
        hm = heatmap!(axes_reldiff, gridx[:,1], gridy[1,:], κ_reldiff, colormap = :BrBG, colorrange = (-maximum(abs.(κ_reldiff)), maximum(abs.(κ_reldiff))))
        cb = Colorbar(fig_reldiff[1,2], hm; label = L"κ_{reldiff}", width = 20)
        xlims!(axes_reldiff, -X_lim_plot, X_lim_plot)
        ylims!(axes_reldiff, -Y_lim_plot, Y_lim_plot)
        Label(fig_reldiff[2,1], @sprintf("Abs Mean = %.3e     Max |deviation| = %.3e", mean(abs.(κ_reldiff)), maximum(abs.(κ_reldiff))), fontsize = 16, halign = :left)
        save("../Diagnostics/plots/$(foldername)/$(name)_kappa_reldiff_map.png", fig_reldiff)

        fig_diff, axes_diff = Lenses.plot_sky(gridx, gridy)
        hm = heatmap!(axes_diff, gridx_finefits[:,1], gridy_finefits[1,:], κ_diff_fine_, colormap = :BrBG, colorrange = (-maximum(abs.(κ_diff_fine_)), maximum(abs.(κ_diff_fine_))))
        cb = Colorbar(fig_diff[1,2], hm; label = L"κ_{diff}", width = 20)
        xlims!(axes_diff, -X_lim_plot, X_lim_plot)
        ylims!(axes_diff, -Y_lim_plot, Y_lim_plot)
        Label(fig_diff[2,1], @sprintf("Abs Mean = %.3e     Max |deviation| = %.3e", mean(abs.(κ_diff_fine_)), maximum(abs.(κ_diff_fine_))), fontsize = 16, halign = :left)
        save("../Diagnostics/plots/$(foldername)/$(name)_kappa_diff_finemap.png", fig_diff)

        fig_reldiff, axes_reldiff = Lenses.plot_sky(gridx, gridy)
        hm = heatmap!(axes_reldiff, gridx_finefits[:,1], gridy_finefits[1,:], κ_reldiff_fine_, colormap = :BrBG, colorrange = (-maximum(abs.(κ_reldiff_fine_)), maximum(abs.(κ_reldiff_fine_))))
        cb = Colorbar(fig_reldiff[1,2], hm; label = L"κ_{reldiff}", width = 20)
        xlims!(axes_reldiff, -X_lim_plot, X_lim_plot)
        ylims!(axes_reldiff, -Y_lim_plot, Y_lim_plot)
        Label(fig_reldiff[2,1], @sprintf("Abs Mean = %.3e     Max |deviation| = %.3e", mean(abs.(κ_reldiff_fine_)), maximum(abs.(κ_reldiff_fine_))), fontsize = 16, halign = :left)
        save("../Diagnostics/plots/$(foldername)/$(name)_kappa_reldiff_finemap.png", fig_reldiff)

        println("plotting the lens magnification/kappa maps...")

        fig_mag, axes_mag = Lenses.plot_sky(gridx_finefits, gridy_finefits)
        hm = heatmap!(axes_mag, gridx_finefits[:,1], gridy_finefits[1,:], abs.(mag_fine), colormap = :turbo, colorrange = magnif_range)
        cb = Colorbar(fig_mag[1,2], hm; label = L"|\mu|", width = 20)
        xlims!(axes_mag, -X_lim_plot, X_lim_plot)
        ylims!(axes_mag, -Y_lim_plot, Y_lim_plot)
        save("../Diagnostics/plots/$(foldername)/$(name)_magnification_map.png", fig_mag)

        makiepts = UtilityFunctions.add_clusterimages(model)
        scatter!(axes_mag, makiepts, color=:yellow, markersize=3)
        save("../Diagnostics/plots/$(foldername)/$(name)_magnification_map_with_images.png", fig_mag)

        time_planemap_start = time()
        cc_fig, cc_axes = Lenses.plot_image_plane(free_lens_nokernel, free_qty_tuple, x_fine, y_fine, adis, two_panel = true)        # bottleneck
        xlims!(cc_axes[1], -X_lim_plot, X_lim_plot)
        ylims!(cc_axes[1], -Y_lim_plot, Y_lim_plot)
        xlims!(cc_axes[2], -X_lim_plot, X_lim_plot)
        ylims!(cc_axes[2], -Y_lim_plot, Y_lim_plot)
        save("../Diagnostics/plots/$(foldername)/$(name)_critical_curves.png", cc_fig)
        println("plotted the critical curves and caustics in ", time() - time_planemap_start, " seconds.")

        flush(stdout)

        κ_fig, κ_axes = Lenses.plot_sky(x_fine, y_fine)
        hm = heatmap!(κ_axes, x_fine[:,1], y_fine[1,:], κ_fine_, colormap = :turbo, colorrange = kappa_range)
        cb = Colorbar(κ_fig[1,2], hm; label = L"κ", width = 20)
        xlims!(κ_axes, -X_lim_plot, X_lim_plot)
        ylims!(κ_axes, -Y_lim_plot, Y_lim_plot)
        save("../Diagnostics/plots/$(foldername)/$(name)_kappa_map.png", κ_fig)

        scatter!(κ_axes, makiepts, color=:black, markersize=3)
        xlims!(κ_axes, -X_lim_plot, X_lim_plot)
        ylims!(κ_axes, -Y_lim_plot, Y_lim_plot)
        save("../Diagnostics/plots/$(foldername)/$(name)_kappa_map_with_images.png", κ_fig)

        println("χ² of predicted image positions: ", data["chi2"])

        flush(stdout)
    end

    if plot_image_flag_og

        time_start_image = time()
        rms, count, total_img = UtilityFunctions.give_image_rmsscatter(model, cluster_lens_nokernel, param_ref, gridx_finefits, gridy_finefits, plot_image_flag_og, cluster_path, thres, cluster_qty_tuple, clusterimgqty_tuple)
        fig, ax = UtilityFunctions.plot_image_scatter(model, cluster_lens, gridx_finefits, gridy_finefits; save_plot = true, plot_name = cluster_path * "image_scatter_og.png", gridqty_tuple = cluster_qty_tuple, imgqty_tuple = clusterimgqty_tuple)
        fig, ax = UtilityFunctions.plot_magnification_scatter(model, cluster_lens, gridx_finefits, gridy_finefits; save_plot = true, plot_name = cluster_path * "magnification_scatter_og.png", gridqty_tuple = cluster_qty_tuple, imgqty_tuple = clusterimgqty_tuple)
        open(cluster_path * "rms.txt", "a") do io
            println(io, "rms of image positions with threshold using truth map from $(clustername) = $(thres): " * string(rms))
            println(io, "count: ", count, ", total_img: ", total_img)
            println(io, "time taken for rms calc: ", time() - time_start_image, " seconds.")
        end

        println("time taken for rms calc: ", time() - time_start_image, " seconds.")
    end

    if plot_image_flag
        time_start_image = time()
        rms, count, total_img = UtilityFunctions.give_image_rmsscatter(model, free_lens_nokernel, param_ref, x_fine, y_fine, plot_image_flag, "../Diagnostics/plots/$(foldername)/", thres, free_qty_tuple, freeimgqty_tuple)
        fig, ax = UtilityFunctions.plot_image_scatter(model, free_lens, x_fine, y_fine; save_plot = true, plot_name = "../Diagnostics/plots/$(foldername)/image_scatter.png", gridqty_tuple = free_qty_tuple, imgqty_tuple = freeimgqty_tuple)
        fig, ax = UtilityFunctions.plot_magnification_scatter(model, free_lens, x_fine, y_fine; save_plot = true, plot_name = "../Diagnostics/plots/$(foldername)/magnification_scatter.png", gridqty_tuple = free_qty_tuple, imgqty_tuple = freeimgqty_tuple)
        # save the rms to a text file
        open("../Diagnostics/plots/$(foldername)/rms.txt", "a") do io
            println(io, "rms of image positions with threshold using reconstructed map for $(clustername) = $(thres): " * string(rms))
            println(io, "count: ", count, ", total_img: ", total_img)
            println(io, "χ² of predicted image positions: ", data["chi2"])
            println(io, "time taken for rms calc: ", time() - time_start_image, " seconds.")
        end
        println("time taken for rms calc: ", time() - time_start_image, " seconds.")
    end

    time_end = time()
    print("the parameters used for this run are: ")
    println("clustername: ", clustername, ", res: ", res, ", thres: ", thres, ", X_lim: ", X_lim, ", Y_lim: ", Y_lim, ", X_lim_plot: ", X_lim_plot, ", Y_lim_plot: ", Y_lim_plot)
    println("the input file used for this run is: ", name)
    println("Total time taken: ", time_end - time_start, " seconds.")

    println("plotting trace diagnostics...")
    fig, ax1, ax2, ax3, ax4, ax5 = UtilityFunctions.plot_trace_stats(traceofrun)

    save("../Diagnostics/plots/$(foldername)/$(name)_trace_diagnostics.png", fig)
    println("done")
    println("------------------------------")

    # saving fits files for potential, deflection, and kappa, gamma1, gamma2 maps.
    save_fits_file(model, κ_fine_, x_fine, y_fine, "kappa_map", foldername)
    save_fits_file(model, ψ_free, x_fine, y_fine, "potential_map", foldername)
    save_fits_file(model, αx_free, x_fine, y_fine, "deflection_x_map", foldername)
    save_fits_file(model, αy_free, x_fine, y_fine, "deflection_y_map", foldername)
    save_fits_file(model, ψxx_free, x_fine, y_fine, "jacobian_xx_map", foldername)
    save_fits_file(model, ψyy_free, x_fine, y_fine, "jacobian_yy_map", foldername)
    save_fits_file(model, ψxy_free, x_fine, y_fine, "jacobian_xy_map", foldername)

    # saving all important arrays for cross iteration comparison
    jldsave("../Diagnostics/plots/$(foldername)/$(name)_diagnostics.jld2",;
        model          = model,
        gridx_finefits = gridx_finefits,
        gridy_finefits = gridy_finefits,
        gridx          = gridx,
        gridy          = gridy,
        mag_fine       = mag_fine,
        mag_finefits   = mag_finefits,
        κ_fine         = κ_fine_,
        free_qty_tuple    = free_qty_tuple,
        freeimgqty_tuple  = freeimgqty_tuple,
        kappa_finefits = kappa_finefits,
        prior_kappa_fine = prior_kappa_fine_,
        init_guess_fine  = init_guess_fine_,
        κ_diff         = κ_diff_,
        κ_reldiff      = κ_reldiff,
        κ_diff_fine    = κ_diff_fine_,
        diff_qty_tuple    = diff_qty_tuple,
        diffimgqty_tuple  = diffimgqty_tuple,
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
    )

end

main()
