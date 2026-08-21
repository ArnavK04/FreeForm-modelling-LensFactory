using BenchmarkTools
using Test
using Random

# ---------------------------------------------------------------------
# Load your package/module. Adjust to match your project structure.
# e.g. `using YourPackageName` or `include("your_source_file.jl")`
# ---------------------------------------------------------------------
using LensFactory   
using LensFactory.Constants

include("FreeFormLens.jl")  # Adjust the path as necessary

# ---------------------------------------------------------------------
# Test setup
# ---------------------------------------------------------------------
Random.seed!(42)   # reproducible test data

gridx, gridy = Lenses.get_meshgrid(168, 168, 1)

n_sources = 100
θx_test = 50 .* randn(n_sources)
θy_test = 50 .* randn(n_sources)

println("Threads.nthreads() = ", Threads.nthreads())
if Threads.nthreads() == 1
    @warn "Julia is running with only 1 thread. Start Julia with `julia -t auto` (or -t N) to actually test multithreading."
end

# ---------------------------------------------------------------------
# 1. Correctness check
# ---------------------------------------------------------------------
println("\n--- Correctness check ---")

result_serial   = FreeFormLens.give_kernel(θx_test, θy_test, gridx, gridy)
result_threaded = FreeFormLens.give_kernel_threaded(θx_test, θy_test, gridx, gridy)

@testset "give_kernel: serial vs threaded" begin
    @test length(result_serial) == length(result_threaded)
    for k in eachindex(result_serial)
        for field in 1:6
            @test result_serial[k][field] ≈ result_threaded[k][field]
        end
    end
end

# Run multiple trials to catch nondeterministic races
println("\nRunning ", 10, " repeated trials to check for race conditions...")
all_match = true
for trial in 1:10
    r = FreeFormLens.give_kernel_threaded(θx_test, θy_test, gridx, gridy)
    match = all(result_serial[k][f] ≈ r[k][f] for k in eachindex(r), f in 1:6)
    global all_match &= match
    print(match ? "." : "X")
end
println()
println(all_match ? "All trials matched serial result. PASS" : "Mismatch detected across trials! Investigate for race condition.")

# ---------------------------------------------------------------------
# 2. Speed check
# ---------------------------------------------------------------------
println("\n--- Speed check ---")

b_serial   = @benchmark FreeFormLens.give_kernel($θx_test, $θy_test, $gridx, $gridy)
b_threaded = @benchmark FreeFormLens.give_kernel_threaded($θx_test, $θy_test, $gridx, $gridy)

println("\nSerial:")
show(stdout, MIME"text/plain"(), b_serial)
println("\n\nThreaded ($(Threads.nthreads()) threads):")
show(stdout, MIME"text/plain"(), b_threaded)

speedup = median(b_serial.times) / median(b_threaded.times)
println("\n\nMedian speedup: $(round(speedup, digits=2))x on $(Threads.nthreads()) threads")
println("Serial allocations:   $(b_serial.allocs), $(b_serial.memory) bytes")
println("Threaded allocations: $(b_threaded.allocs), $(b_threaded.memory) bytes")

# ---------------------------------------------------------------------
# 3. Sweep over problem size (optional, comment out if slow)
# ---------------------------------------------------------------------
println("\n--- Size sweep ---")
for n in [10, 50, 100, 500]
    θx_n, θy_n = 50 .* randn(n), 50 .* randn(n)
    t_s = @belapsed FreeFormLens.give_kernel($θx_n, $θy_n, $gridx, $gridy)
    t_t = @belapsed FreeFormLens.give_kernel_threaded($θx_n, $θy_n, $gridx, $gridy)
    println("n=$n:  serial=$(round(t_s*1000, digits=2))ms  threaded=$(round(t_t*1000, digits=2))ms  speedup=$(round(t_s/t_t, digits=2))x")
end