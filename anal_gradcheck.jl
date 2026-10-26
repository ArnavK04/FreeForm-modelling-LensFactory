# importing modules and files
using LensFactory
using LensFactory.Constants
using LensFactory.LensModel.LensModelIO
using LinearAlgebra
using FiniteDiff
using BenchmarkTools
using Test
using Random
using Printf

include("FreeFormLens.jl")
include("utility_functions.jl")

input_file = "../SMACS0723/smacs_j0723input.yaml"

model = LensModel.read_input(input_file)
param_ref = Dict(p.key => p.refer for p in model.parameters)
prior_kappa, gridx, gridy = LikelihoodFunctions.construct_prior(model; prior_flag=1, prior_value=0.5)
full_kernel = FreeFormLens.compute_fullkernel(model, gridx, gridy)

print("gridx = %d, gridy = %d, prior_kappa = %d\n", size(gridx), size(gridy), size(prior_kappa))
kappa_vec = vec(exp.(rand(size(prior_kappa,1), size(prior_kappa,2))))

# ---- functions (adjust the LikelihoodFunctions prefix if needed) ----
f_like(x) = LikelihoodFunctions.neg_loglikelihood_MEM(model,
    FreeFormLens.init_FreeFormLens(reshape(x, size(gridx)), gridx, gridy, true),
    param_ref, full_kernel)

g_an(x) = LikelihoodFunctions.loglikelihood_grad!(x, prior_kappa, gridx, gridy, model, param_ref, full_kernel)
g_old(x) = LikelihoodFunctions.loglikelihood_grad_fd!(x, prior_kappa, gridx, gridy, model, param_ref, full_kernel)
g_ref(x) = FiniteDiff.finite_difference_gradient(f_like, x, Val(:central))

# ---- compute gradients and time them ----
t_f   = minimum(@benchmark f_like($kappa_vec)).time / 1e9
b_an  = @benchmark g_an($kappa_vec)
t_an  = minimum(b_an).time / 1e9

t_old = @elapsed G_old = g_old(kappa_vec)      # slow: single run
t_ref = @elapsed G_ref = g_ref(kappa_vec)      # slowest: single run
G_an  = g_an(kappa_vec)

# ---- accuracy metrics ----
relerr(a, b) = norm(a - b) / norm(b)
cosine(a, b) = dot(a, b) / (norm(a) * norm(b))
maxabs(a, b) = maximum(abs.(a - b))

grads = Dict("analytic" => G_an, "old FD" => G_old, "FiniteDiff" => G_ref)
times = Dict("analytic" => t_an, "old FD" => t_old, "FiniteDiff" => t_ref)

println("\n=== Accuracy vs FiniteDiff central (reference) ===")
@printf("%-12s %12s %12s %12s\n", "method", "rel.err", "1-cosine", "max|diff|")
for name in ("analytic", "old FD")
    G = grads[name]
    @printf("%-12s %12.3e %12.3e %12.3e\n", name, relerr(G, G_ref), 1 - cosine(G, G_ref), maxabs(G, G_ref))
end

println("\n=== Pairwise: old FD vs analytic ===")
@printf("rel.err = %.3e   1-cosine = %.3e\n", relerr(G_old, G_an), 1 - cosine(G_old, G_an))

println("\n=== Speed ===")
@printf("%-12s %12s %18s\n", "method", "time (s)", "cost / f eval")
@printf("%-12s %12.4e %18s\n", "f_like", t_f, "1")
for name in ("analytic", "old FD", "FiniteDiff")
    @printf("%-12s %12.4e %18.1f\n", name, times[name], times[name] / t_f)
end
@printf("\nspeedup of analytic: %.0fx vs old FD, %.0fx vs FiniteDiff\n", t_old / t_an, t_ref / t_an)
@printf("analytic allocations: %d, %.1f MiB\n", b_an.allocs, b_an.memory / 2^20)

# ---- worst pixels for each method ----
for name in ("analytic", "old FD")
    d = abs.(grads[name] - G_ref)
    worst = sortperm(d, rev = true)[1:3]
    println("\nworst pixels for $name: ", worst)
    for i in worst
        @printf("  pixel %6d  method=% .6e  ref=% .6e\n", i, grads[name][i], G_ref[i])
    end
end