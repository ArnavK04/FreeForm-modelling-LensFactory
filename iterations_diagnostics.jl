using LensFactory
using ArgParse
using JLD2
using CairoMakie
using Statistics
using Printf
using LaTeXStrings

# functino to read command line arguments

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--names"
            help = "Comma-separated list of run names to compare, e.g. \"runA,runB,runC\""
            arg_type = String
            required = true
        "--res"
            help = "Resolution for which to make maps"
            arg_type = Float64
            default = nothing
        "--thres"
            help = "Threshold for which to make maps"
            arg_type = Float64
            default = nothing
        "--diag_dir"
            help = "Base diagnostics directory"
            arg_type = String
            default = "../Diagnostics/plots"
        "--outdir"
            help = "Where to save the comparison figures (defaults to <diag_dir>/<foldername>/comparison)"
            arg_type = String
            default = nothing
        "--clustername"
            help = "Name of the cluster (e.g. Hera, Ares, etc.)"
            arg_type = String
            default = "Hera"
    end

    return parse_args(s)
end

function load_run(diag_dir::String, foldername::String, name::String)
    path = joinpath(diag_dir, foldername, "$(name)_diagnostics.jld2")
    isfile(path) || error("diagnostics file not found: $path")
    return load(path)
end

# plot a row of all runs for a given quantity (e.g. kappa, magnification, etc.)
function plot_quantity_row(names::Vector{String}, datas::Vector{Dict{String,Any}},
                            key::String, gridx_key::String, gridy_key::String;
                            colormap = :turbo,
                            colorrange::Union{Nothing,Tuple{<:Real,<:Real}} = nothing,
                            label = key,
                            transform::Function = identity,
                            symmetric::Bool = false,
                            X_lim_plot::Union{Nothing,Float64} = nothing,
                            Y_lim_plot::Union{Nothing,Float64} = nothing,
                            show_images::Bool = false, img_color = :yellow)

    n = length(names)
    missing_idx = [i for i in 1:n if !haskey(datas[i], key)]
    if !isempty(missing_idx)
        @warn "key \"$key\" missing for runs: $(names[missing_idx]); skipping this comparison"
        return nothing
    end

    vals = [transform.(datas[i][key]) for i in 1:n]

    if colorrange === nothing
        if symmetric
            m = 5*maximum(maximum(abs.(vals[end])))
            colorrange = (-m, m)
        else
            lo = 5*minimum(minimum(vals[end]))
            hi = 5*maximum(maximum(vals[end]))
            colorrange = (lo, hi)
        end
    end

    fig = Figure(size = (420 * n + 140, 480))
    hm = nothing
    for i in 1:n
        gx = datas[i][gridx_key]
        gy = datas[i][gridy_key]
        ax = Axis(fig[1, i]; aspect = DataAspect(),
                  xlabel = "θx", ylabel = i == 1 ? "θy" : "")
        hm = heatmap!(ax, gx[:, 1], gy[1, :], vals[i]; colormap = colormap, colorrange = colorrange)

        if show_images && haskey(datas[i], "img_pts")
            pts = datas[i]["img_pts"]
            if !isempty(pts)
                scatter!(ax, pts; color = img_color, markersize = 6)
            end
        end

        xl = X_lim_plot === nothing ? get(datas[i], "X_lim_plot", nothing) : X_lim_plot
        yl = Y_lim_plot === nothing ? get(datas[i], "Y_lim_plot", nothing) : Y_lim_plot
        xl !== nothing && xlims!(ax, -xl, xl)
        yl !== nothing && ylims!(ax, -yl, yl)
    end
    Colorbar(fig[1, n + 1], hm; label = label, width = 20)
    return fig
end

# plots showing how xi^2, images counted, rms change across each iteration

function plot_summary(names::Vector{String}, datas::Vector{Dict{String,Any}})

    n = length(names)
    rms   = [get(d, "RMS", NaN) for d in datas]
    chi2  = [get(d, "χ²", NaN) for d in datas]
    count = [get(d, "count", NaN) for d in datas]
    total = [get(d, "total_img", NaN) for d in datas]       # same total image for all iterations

    fig = Figure(size = (1200, 400))

    ax1 = Axis(fig[1, 1]; title = "RMS of image positions", xlabel = "run", ylabel = "RMS [arcsec]", yscale = log10)
    scatterlines!(ax1, 1:n, rms; color = :blue, markersize = 8)

    ax2 = Axis(fig[1, 2]; title = "χ² of image positions", xlabel = "run", ylabel = "χ²", yscale = log10)
    scatterlines!(ax2, 1:n, chi2; color = :blue, markersize = 8)

    ax3 = Axis(fig[1, 3]; title = "Images counted - Total = $(total[1])", xlabel = "run", ylabel = "count")
    scatterlines!(ax3, 1:n, count; color = :blue, markersize = 8)


    println("Summary stats:")
    for i in 1:n
        println("run: $(names[i]), RMS = $(rms[i]), χ² = $(chi2[i]), count = $(count[i]), total_img = $(total[i])")
    end

    return fig
end

function main()

    args = parse_commandline()

    names      = String.(strip.(split(args["names"], ",")))
    diag_dir   = args["diag_dir"]
    res        = args["res"]
    thres      = args["thres"]
    outdir     = args["outdir"]
    clustername = args["clustername"]

    kappa_dev_range = (-1.0, 2.0)
    magnif_range = (0, 100)
    if clustername == "Hera"
        kappa_range = (0, 2.70)
    elseif clustername == "Ares"
        kappa_range = (0, 3.75)
    else
        kappa_range = (0, 3.75)
    end

    if outdir == nothing
        println("outdir not specified, using default: $(joinpath(diag_dir, "comparison"))")
        outdir = joinpath(diag_dir, "comparison_res_$(res)_thres_$(thres)")
        mkpath(outdir)
    else 
        mkpath(outdir)
    end

    println("runs to be analysed have parent name: $(names[1])")
    println("loading diagnostics for $(length(names)) runs from $(diag_dir)...")
    datas = Dict{String,Any}[]

    for name in names
        foldername = "$(name)_res_$(res)_thres_$(thres)"
        d = load_run(diag_dir, foldername, name)
        push!(datas, d)
    end
    println("loaded: ", join(names, ", "))

    # reconstructed kappa at z_s = 9
    fig = plot_quantity_row(names, datas, "κ_fine", "gridx_finefits", "gridy_finefits";
                             colormap = :turbo, colorrange = kappa_range, label = "κ")
    save(joinpath(outdir, "compare_kappa_reconst.png"), fig)

    fig = plot_quantity_row(names, datas, "κ_fine", "gridx_finefits", "gridy_finefits";
                             colormap = :turbo, colorrange = kappa_range, label = "κ",
                             show_images = true)
    save(joinpath(outdir, "compare_kappa_reconst_with_images.png"), fig)

    # prior kappa at z_s = 9
    fig = plot_quantity_row(names, datas, "prior_kappa_fine", "gridx_finefits", "gridy_finefits";
                             colormap = :turbo, colorrange = kappa_range, label = "κ_prior")
    save(joinpath(outdir, "compare_prior_kappa.png"), fig)

    # initial guess kappa at z_s = 9
    fig = plot_quantity_row(names, datas, "init_guess_fine", "gridx_finefits", "gridy_finefits";
                             colormap = :turbo, colorrange = kappa_range, label = "κ_init_guess")
    save(joinpath(outdir, "compare_init_guess_kappa.png"), fig)

    # reconstructed magnification at z_s = 9
    fig = plot_quantity_row(names, datas, "mag_fine", "gridx_finefits", "gridy_finefits";
                             colormap = :turbo, colorrange = magnif_range, label = "|μ|",
                             transform = abs, show_images = true)
    save(joinpath(outdir, "compare_reconst_mag_with_images.png"), fig)

    fig = plot_quantity_row(names, datas, "mag_fine", "gridx_finefits", "gridy_finefits";
                             colormap = :turbo, colorrange = magnif_range, label = "|μ|",
                             transform = abs)
    save(joinpath(outdir, "compare_reconst_mag.png"), fig)

    # κ_diff
    fig = plot_quantity_row(names, datas, "κ_diff", "gridx", "gridy";
                             colormap = :BrBG, symmetric = true, label = L"|κ_i - κ_{i-1}|")
    save(joinpath(outdir, "compare_kappa_diff.png"), fig)

    # κ_reldiff
    fig = plot_quantity_row(names, datas, "κ_reldiff", "gridx", "gridy";
                             colormap = :BrBG, symmetric = true, label = L"|κ_i - κ_{i-1}| / κ_{i-1}")
    save(joinpath(outdir, "compare_kappa_reldiff.png"), fig)

    # mag_reldev and kappa_reldev, and abs kappa_diff and kappa_reldiff in log
    for (i, name) in enumerate(names)
        d = datas[i]
        mag_reldev = (d["mag_fine"] .- d["mag_finefits"]) ./ d["mag_finefits"]
        kappa_reldev = (d["κ_fine"] .- d["kappa_finefits"]) ./ d["kappa_finefits"]
        kappa_diff_log = log10.(abs.(d["κ_diff"]))
        kappa_reldiff_log = log10.(abs.(d["κ_reldiff"]))
        diff_qty_tuple = d["diff_qty_tuple"]
        psi_diff, αx_diff, αy_diff, ψxx_diff, ψyy_diff, ψxy_diff = diff_qty_tuple
        datas[i]["__psi_diff"] = log10.(abs.(psi_diff))
        datas[i]["__αx_diff"] = log10.(abs.(αx_diff))
        datas[i]["__αy_diff"] = log10.(abs.(αy_diff))
        datas[i]["__ψxx_diff"] = log10.(abs.(ψxx_diff))
        datas[i]["__ψyy_diff"] = log10.(abs.(ψyy_diff))
        datas[i]["__ψxy_diff"] = log10.(abs.(ψxy_diff))
        datas[i]["__net_deflection_diff"] = log10.(sqrt.(αx_diff.^2 .+ αy_diff.^2))
        datas[i]["__mag_reldev"] = mag_reldev
        datas[i]["__kappa_reldev"] = kappa_reldev
        datas[i]["__kappa_diff_log"] = kappa_diff_log
        datas[i]["__kappa_reldiff_log"] = kappa_reldiff_log
        datas[i]["__kappa_diff_fine_log"] = log10.(abs.(d["κ_diff_fine"]))
        datas[i]["__kappa_reldiff_fine_log"] = log10.(abs.(d["κ_reldiff_fine"]))
    end

    fig = plot_quantity_row(names, datas, "__psi_diff", "gridx_finefits", "gridy_finefits";
                             colormap = :BrBG, colorrange = (-4.0, 1.0), symmetric = true, label = L"ψ_i - ψ_{i-1}", img_color = :blue, show_images = true)
    save(joinpath(outdir, "compare_psi_diff.png"), fig)
    fig = plot_quantity_row(names, datas, "__αx_diff", "gridx_finefits", "gridy_finefits";
                             colormap = :BrBG, colorrange = (-4.0, 1.0), symmetric = true, label = L"αx_i - αx_{i-1}", img_color = :blue, show_images = true)
    save(joinpath(outdir, "compare_αx_diff.png"), fig)
    fig = plot_quantity_row(names, datas, "__αy_diff", "gridx_finefits", "gridy_finefits";
                             colormap = :BrBG, colorrange = (-4.0, 1.0), symmetric = true, label = L"αy_i - αy_{i-1}", img_color = :blue, show_images = true)
    save(joinpath(outdir, "compare_αy_diff.png"), fig)
    fig = plot_quantity_row(names, datas, "__net_deflection_diff", "gridx_finefits", "gridy_finefits";
                             colormap = :BrBG, colorrange = (-4.0, 1.0), symmetric = true, label = L"|α_i - α_{i-1}|", img_color = :blue, show_images = true)
    save(joinpath(outdir, "compare_net_deflection_diff.png"), fig)
    fig = plot_quantity_row(names, datas, "__ψxx_diff", "gridx_finefits", "gridy_finefits";
                             colormap = :BrBG, colorrange = (-4.0, 1.0), symmetric = true, label = L"ψxx_i - ψxx_{i-1}", img_color = :blue, show_images = true)
    save(joinpath(outdir, "compare_ψxx_diff.png"), fig)
    fig = plot_quantity_row(names, datas, "__ψyy_diff", "gridx_finefits", "gridy_finefits";
                             colormap = :BrBG, colorrange = (-4.0, 1.0), symmetric = true, label = L"ψyy_i - ψyy_{i-1}", img_color = :blue, show_images = true)
    save(joinpath(outdir, "compare_ψyy_diff.png"), fig)
    fig = plot_quantity_row(names, datas, "__ψxy_diff", "gridx_finefits", "gridy_finefits";
                             colormap = :BrBG, colorrange = (-4.0, 1.0), symmetric = true, label = L"ψxy_i - ψxy_{i-1}", img_color = :blue, show_images = true)
    save(joinpath(outdir, "compare_ψxy_diff.png"), fig)

    fig = plot_quantity_row(names, datas, "__mag_reldev", "gridx_finefits", "gridy_finefits";
                             colormap = :BrBG, colorrange = (-1.0, 4.0), label = L"(|μ|- |μ|_t)/ |μ|_t")
    save(joinpath(outdir, "compare_mag_reldev.png"), fig)
    fig = plot_quantity_row(names, datas, "__mag_reldev", "gridx_finefits", "gridy_finefits";
                             colormap = :BrBG, colorrange = (-4.0, 4.0), label = L"(|μ|- |μ|_t)/ |μ|_t")
    save(joinpath(outdir, "compare_mag_reldevBrBG.png"), fig)

    fig = plot_quantity_row(names, datas, "__kappa_reldev", "gridx_finefits", "gridy_finefits";
                             colormap = :afmhot, colorrange = (-1.0, 2.0), label = L"(κ- κ_t)/ κ_t")
    save(joinpath(outdir, "compare_kappa_reldev.png"), fig)
    fig = plot_quantity_row(names, datas, "__kappa_reldev", "gridx_finefits", "gridy_finefits";
                             colormap = :BrBG, colorrange = (-2.0, 2.0), label = L"(κ- κ_t)/ κ_t")
    save(joinpath(outdir, "compare_kappa_reldevBrBG.png"), fig)
    fig = plot_quantity_row(names, datas, "__kappa_diff_log", "gridx", "gridy";
                             colormap = :afmhot, colorrange = (-4.0, 1.0), label = L"log10(|κ_i - κ_{i-1}|)",
                             show_images = true, img_color = :cyan)
    save(joinpath(outdir, "compare_kappa_diff_log.png"), fig)
    fig = plot_quantity_row(names, datas, "__kappa_diff_fine_log", "gridx_finefits", "gridy_finefits";
                             colormap = :afmhot, colorrange = (-4.0, 1.0), label = L"log10(|κ_i - κ_{i-1}|)",
                             show_images = true, img_color = :cyan)
    save(joinpath(outdir, "compare_kappa_diff_fine_log.png"), fig)
    fig = plot_quantity_row(names, datas, "__kappa_reldiff_fine_log", "gridx_finefits", "gridy_finefits";
                             colormap = :afmhot, colorrange = (-4.0, 1.0), label = L"log10(|κ_i - κ_{i-1}|)/ κ_{i-1}",
                             show_images = true, img_color = :cyan)
    save(joinpath(outdir, "compare_kappa_reldiff_fine_log.png"), fig)
    fig = plot_quantity_row(names, datas, "κ_diff_fine", "gridx_finefits", "gridy_finefits";
                             colormap = :BrBG, symmetric = true, label = L"|κ_i - κ_{i-1}|",
                             show_images = true, img_color = :cyan)
    save(joinpath(outdir, "compare_kappa_diff_fine.png"), fig)
    fig = plot_quantity_row(names, datas, "__kappa_reldiff_log", "gridx", "gridy";
                             colormap = :afmhot, colorrange = (-4.0, 1.0), label = L"log10(|κ_i - κ_{i-1}|)/ κ_{i-1}",
                             show_images = true, img_color = :cyan)
    save(joinpath(outdir, "compare_kappa_reldiff_log.png"), fig)

    fig = plot_summary(names, datas)
    save(joinpath(outdir, "compare_summary_stats.png"), fig)

    open(joinpath(outdir, "compare_summary.txt"), "w") do io
        for (i, name) in enumerate(names)
            d = datas[i]
            println(io, "run: ", name)
            println(io, "  RMS       = ", get(d, "RMS", "n/a"))
            println(io, "  count     = ", get(d, "count", "n/a"))
            println(io, "  total_img = ", get(d, "total_img", "n/a"))
            println(io, "  χ²        = ", get(d, "χ²", "n/a"))
            println(io, "  res       = ", get(d, "res", "n/a"), ", thres = ", get(d, "thres", "n/a"))
            println(io, "-"^40)
        end
    end

    println("comparison plots and summary written to: ", outdir)
end

main()