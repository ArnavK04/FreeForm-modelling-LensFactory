module FreeFormLens

using LensFactory
using LensFactory.Constants
using CairoMakie
using LensFactory.LFUtils

# Functions to export
export potential!
export deflection!
export jacobian!
export init_FreeFormLens
export compute_fullkernel

# functions to be used internally to get the lensing potential, deflection and jacobian for a free form lens 
# (check https://arxiv.org/abs/astro-ph/9707207v2)

function cell_integral(x::T, y::T) where T <: RV
    if x == 0.0 && y == 0.0
        return 0.0
    end
    return 0.5 * (x^2 * atan_cont(x, y) + y^2 * atan_cont(y, x) + x * y * log(x^2 + y^2) - 3*x*y)
end

function definite_integral(x::T, y::T, xc::T, yc::T, Δ::T) where T <: RV
   return cell_integral(x - (xc + 0.5*Δ), y - (yc + 0.5*Δ)) + cell_integral(x - (xc - 0.5*Δ), y - (yc - 0.5*Δ)) - cell_integral(x - (xc + 0.5*Δ), y - (yc - 0.5*Δ)) - cell_integral(x - (xc - 0.5*Δ), y - (yc + 0.5*Δ))
end

function deriv_x(x::T, y::T) where T <: RV
    return x * atan_cont(x, y) + 0.5 * y * log(x^2 + y^2) - y
end

function deriv_y(x::T, y::T) where T <: RV
    return y * atan_cont(y, x) + 0.5 * x * log(x^2 + y^2) - x
end

function deriv_xx(x::T, y::T) where T <: RV
   return atan_cont(x, y)
end

function deriv_yy(x::T, y::T) where T <: RV
   return atan_cont(y, x)
end

function definite_deriv_x(x::T, y::T, xc::T, yc::T, Δ::T) where T <: RV
   return deriv_x(x - (xc + 0.5*Δ), y - (yc + 0.5*Δ)) + deriv_x(x - (xc - 0.5*Δ), y - (yc - 0.5*Δ)) - deriv_x(x - (xc + 0.5*Δ), y - (yc - 0.5*Δ)) - deriv_x(x - (xc - 0.5*Δ), y - (yc + 0.5*Δ))
end

function definite_deriv_y(x::T, y::T, xc::T, yc::T, Δ::T) where T <: RV
   return deriv_y(x - (xc + 0.5*Δ), y - (yc + 0.5*Δ)) + deriv_y(x - (xc - 0.5*Δ), y - (yc - 0.5*Δ)) - deriv_y(x - (xc + 0.5*Δ), y - (yc - 0.5*Δ)) - deriv_y(x - (xc - 0.5*Δ), y - (yc + 0.5*Δ))
end

function definite_deriv_xx(x::T, y::T, xc::T, yc::T, Δ::T) where T <: RV
   return deriv_xx(x - (xc + 0.5*Δ), y - (yc + 0.5*Δ)) + deriv_xx(x - (xc - 0.5*Δ), y - (yc - 0.5*Δ)) - deriv_xx(x - (xc + 0.5*Δ), y - (yc - 0.5*Δ)) - deriv_xx(x - (xc - 0.5*Δ), y - (yc + 0.5*Δ))
end

function definite_deriv_yy(x::T, y::T, xc::T, yc::T, Δ::T) where T <: RV
   return deriv_yy(x - (xc + 0.5*Δ), y - (yc + 0.5*Δ)) + deriv_yy(x - (xc - 0.5*Δ), y - (yc - 0.5*Δ)) - deriv_yy(x - (xc + 0.5*Δ), y - (yc - 0.5*Δ)) - deriv_yy(x - (xc - 0.5*Δ), y - (yc + 0.5*Δ))
end

function deriv_xy(x::T, y::T) where T <: RV
    if x == 0.0 && y == 0.0
        return 0.0
    end
    return 0.5 * log(x^2 + y^2)
end

function definite_deriv_xy(x::T, y::T, xc::T, yc::T, Δ::T) where T <: RV
   return deriv_xy(x - (xc + 0.5*Δ), y - (yc + 0.5*Δ)) + deriv_xy(x - (xc - 0.5*Δ), y - (yc - 0.5*Δ)) - deriv_xy(x - (xc + 0.5*Δ), y - (yc - 0.5*Δ)) - deriv_xy(x - (xc - 0.5*Δ), y - (yc + 0.5*Δ))
end

# atan without the singularity
function atan_cont(x::T, y::T) where T <: RV
   if abs(x) < eps(T)
      return sign(y)*π/2
   end
   return atan(y/x)
end

"""
    potential!(ψ::T, θx::T, θy::T, κ::ROA, gridx::ROA, gridy::ROA) where T <: RV
    Gives potential at a point for the given convergence. The lensing potential is given as,
```math
ψ(θ_x, θ_y) = 1/π ∫∫ κ(θ'_x, θ'_y) ln|θ - θ'| d^2θ' = Σ κ_mn * 1/π∫∫_{mn} ln|θ - θ'| d^2θ' = Σ κ_mn * 1/π * \\psi_{mn}
where the integral is over the cell (m, n) of the grid. The convergence is assumed to be constant over a cell.
We use a method similar to that used in (https://arxiv.org/abs/astro-ph/9707207v2) to get the lensing potential from convergence.
Instead of assuming the integrand κ(θ'_x, θ'_y) ln|θ - θ'| is constant over a cell, we only assume κ is constant over a cell and integrate ln|θ - θ'| over the cell.
This is done by using the following formula,
```math
   \\tilde{\\psi_{mn}} (x,y) = 1\2π \\left( x^2 arctan (y/x) + y^2 arctan (x/y) + xy ln(x^2 + y^2) - 3 \\right)
```
This way the singularity at the cell center is avoided and the potential is calculated more accurately.
"""

function potential!(ψ::T, θx::T, θy::T, κ::M, gridx::M, gridy::M) where {T <: RV, M <: ROA}

   ψ_up = zero(ψ)

   ax1, ax2 = axes(gridx, 1), axes(gridx, 2)
   cell_size = gridx[2] - gridx[1]

   definite_integral_m = zeros(size(gridx, 1), size(gridx, 2))

   @inbounds Threads.@threads for j in ax2
      @inbounds for i in ax1
         xc, yc = gridx[i, 1], gridy[1, j]
         definite_integral_m[i,j] = definite_integral(θx, θy, xc, yc, cell_size)
      end
   end

   @inbounds for j in ax2
      @inbounds for i in ax1
         xc, yc = gridx[i, 1], gridy[1, j]
         ψ_up += κ[i, j] * definite_integral_m[i,j]
      end
   end
   ψ += (1/π) * ψ_up

   return ψ
end

"""function potential!(ψ::T, θx::T, θy::T, κ::M, gridx::M, gridy::M) where {T <: ROA, M <: ROA}

   θx_flat, θy_flat = vec(θx), vec(θy)
   θx_shape, θy_shape = size(θx), size(θy)

   ax1, ax2, ax3 = axes(gridx, 1), axes(gridx, 2), axes(θx_flat, 1)
   cell_size = gridx[2] - gridx[1]

   definite_integral_m = zeros(size(gridx, 1), size(gridx, 2), length(θx_flat))

   @inbounds for k in ax3
      @inbounds Threads.@threads for j in ax2
         @inbounds for i in ax1
            xc, yc = gridx[i, 1], gridy[1, j]
            definite_integral_m[i,j,k] = definite_integral(θx_flat[k], θy_flat[k], xc, yc, cell_size)
         end
      end
   end

   @inbounds for k in ax3
      @inbounds for j in ax2
         @inbounds for i in ax1
            xc, yc = gridx[i, 1], gridy[1, j]
            ψ[k] = ψ[k] + (1/π) * κ[i, j] * definite_integral_m[i,j,k]
         end
      end
   end

   θx = reshape(θx_flat, θx_shape)
   θy = reshape(θy_flat, θy_shape)

   return nothing
end"""

function potential!(ψ::T, θx::T, θy::T, κ::M, gridx::M, gridy::M) where {T <: ROA, M <: ROA}

   θx_flat, θy_flat = vec(θx), vec(θy)
   θx_shape, θy_shape = size(θx), size(θy)

   ax1, ax2, ax3 = axes(gridx, 1), axes(gridx, 2), axes(θx_flat, 1)
   cell_size = gridx[2] - gridx[1]

   @inbounds for k in ax3
      θxk, θyk = θx_flat[k], θy_flat[k]

      # split ax2 (grid columns) into at most nthreads() chunks
      chunks = Iterators.partition(ax2, cld(length(ax2), Threads.nthreads()))

      tasks = map(chunks) do chunk
         Threads.@spawn begin
            local_sum = 0.0
            @inbounds for j in chunk
               @inbounds for i in ax1
                  xc, yc = gridx[i, 1], gridy[1, j]
                  local_sum += κ[i, j] * definite_integral(θxk, θyk, xc, yc, cell_size)
               end
            end
            local_sum
         end
      end

      chunk_sums = fetch.(tasks)
      ψ[k] += (1/π) * sum(chunk_sums)
   end

   θx = reshape(θx_flat, θx_shape)
   θy = reshape(θy_flat, θy_shape)

   return nothing
end

function potential!(ψ::T, θx::T, θy::T, κ::M, gridx::M, gridy::M, kernel::Vector{NTuple{6, Matrix{Float64}}}) where {T <: ROA, M <: ROA}
   """
   Faster version of above potential! function. This needs pre-computed ψ_mn values for each cell in the grid.
   This is done by calling compute_kernel function beforehand. THe kernel received here would be a vector of Ntuples of matrices.
   """
   ψ_up = zero(ψ)
   
   ax1, ax2, ax3 = axes(gridx, 1), axes(gridx, 2), axes(θx, 1)

   @inbounds for k in ax3
      pot_kernel = kernel[k][1]     # a matrix of size (length(gridx), length(gridy)) containing the ψ_mn values for each cell in the grid for the k-th image position.
      @inbounds for j in ax2
         @inbounds for i in ax1
            ψ_up[k] += κ[i, j] * pot_kernel[i, j]
         end
      end
   end

   @. ψ += (1/π) * ψ_up

   return nothing

end

"""
    deflection!(ψx::T, ψy::T, θx::T, θy::T, κ::M, gridx::M, gridy::M) where {T <: RV, M <: ROA}
    Gives deflection at a point for the given convergence. The
    deflection is given as, 
```math
αx(θ_x, θ_y) = \\grad ψ = Σ κ_mn * 1/π * deriv (ψ_{mn}) 
where the ψ_{mn} is as defined before.
```
"""
function deflection!(ψx::T, ψy::T, θx::T, θy::T, κ::M, gridx::M, gridy::M) where {T <: RV, M <: ROA}

   ψx_up, ψy_up = zero(ψx), zero(ψy)

   ax1, ax2 = axes(gridx, 1), axes(gridx, 2)
   cell_size = gridx[2] - gridx[1]

   definite_deriv_x_m, definite_deriv_y_m = zeros(size(gridx, 1), size(gridx, 2)), zeros(size(gridx, 1), size(gridx, 2))

   @inbounds Threads.@threads for j in ax2
      @inbounds for i in ax1
         xc, yc = gridx[i, 1], gridy[1, j]
         definite_deriv_x_m[i,j] = definite_deriv_x(θx, θy, xc, yc, cell_size)
         definite_deriv_y_m[i,j] = definite_deriv_y(θx, θy, xc, yc, cell_size)
      end
   end
   
   @inbounds for j in ax2
      @inbounds for i in ax1
         xc, yc = gridx[i, 1], gridy[1, j]
         ψx_up += κ[i, j] * definite_deriv_x_m[i,j]
         ψy_up += κ[i, j] * definite_deriv_y_m[i,j]
      end
   end
   ψx += (1/π) * ψx_up
   ψy += (1/π) * ψy_up

   return ψx, ψy
end

"""function deflection!(ψx::T, ψy::T, θx::T, θy::T, κ::M, gridx::M, gridy::M) where {T <: ROA, M <: ROA}

   θx_flat, θy_flat = vec(θx), vec(θy)
   θx_shape, θy_shape = size(θx), size(θy)

   ax1, ax2, ax3 = axes(gridx, 1), axes(gridx, 2), axes(θx_flat, 1)
   cell_size = gridx[2] - gridx[1]

   definite_deriv_x_m, definite_deriv_y_m = zeros(size(gridx, 1), size(gridx, 2), length(θx_flat)), zeros(size(gridx, 1), size(gridx, 2), length(θx_flat))

   @inbounds for k in ax3
      @inbounds Threads.@threads for j in ax2
         @inbounds for i in ax1
            xc, yc = gridx[i, 1], gridy[1, j]
            definite_deriv_x_m[i,j,k] = definite_deriv_x(θx_flat[k], θy_flat[k], xc, yc, cell_size)
            definite_deriv_y_m[i,j,k] = definite_deriv_y(θx_flat[k], θy_flat[k], xc, yc, cell_size)
         end
      end
   end

   @inbounds for k in ax3
      @inbounds for j in ax2
         @inbounds for i in ax1
            xc, yc = gridx[i, 1], gridy[1, j]
            ψx[k] = ψx[k] + (1/π) * κ[i, j] * definite_deriv_x_m[i,j,k]
            ψy[k] = ψy[k] + (1/π) * κ[i, j] * definite_deriv_y_m[i,j,k]
         end
      end
   end

   θx = reshape(θx_flat, θx_shape)
   θy = reshape(θy_flat, θy_shape)

   return nothing
end"""

function deflection!(ψx::T, ψy::T, θx::T, θy::T, κ::M, gridx::M, gridy::M) where {T <: ROA, M <: ROA}

   θx_flat, θy_flat = vec(θx), vec(θy)
   θx_shape, θy_shape = size(θx), size(θy)

   ax1, ax2, ax3 = axes(gridx, 1), axes(gridx, 2), axes(θx_flat, 1)
   cell_size = gridx[2] - gridx[1]

   @inbounds for k in ax3
      θxk, θyk = θx_flat[k], θy_flat[k]

      chunks = Iterators.partition(ax2, cld(length(ax2), Threads.nthreads()))

      tasks = map(chunks) do chunk
         Threads.@spawn begin
            local_sum_x, local_sum_y = 0.0, 0.0
            @inbounds for j in chunk
               @inbounds for i in ax1
                  xc, yc = gridx[i, 1], gridy[1, j]
                  local_sum_x += κ[i, j] * definite_deriv_x(θxk, θyk, xc, yc, cell_size)
                  local_sum_y += κ[i, j] * definite_deriv_y(θxk, θyk, xc, yc, cell_size)
               end
            end
            (local_sum_x, local_sum_y)
         end
      end

      chunk_sums = fetch.(tasks)
      ψx[k] += (1/π) * sum(first(t) for t in chunk_sums)
      ψy[k] += (1/π) * sum(last(t) for t in chunk_sums)
   end

   θx = reshape(θx_flat, θx_shape)
   θy = reshape(θy_flat, θy_shape)

   return nothing
end

function deflection!(ψx::T, ψy::T, θx::T, θy::T, κ::M, gridx::M, gridy::M, kernel::Vector{NTuple{6, Matrix{Float64}}}) where {T <: ROA, M <: ROA}
   """
   Faster version of above function. This needs pre-computed ψ_mn values for each cell in the grid.
   """
   ψx_up, ψy_up = zero(ψx), zero(ψy)
   ax1, ax2, ax3 = axes(gridx, 1), axes(gridx, 2), axes(θx, 1)

   @inbounds for k in ax3
      defx_kernel = kernel[k][2]     # a matrix of size (length(gridx), length(gridy)) containing the ψ_mn values for each cell in the grid for the k-th image position.
      defy_kernel = kernel[k][3]     
      @inbounds for j in ax2
         @inbounds for i in ax1
            ψx_up[k] += κ[i, j] * defx_kernel[i, j]
            ψy_up[k] += κ[i, j] * defy_kernel[i, j]
         end
      end
   end

   @. ψx += (1/π) * ψx_up
   @. ψy += (1/π) * ψy_up

   return nothing
end

"""
    jacobian!(ψxx::T, ψyy::T, ψxy::T, θx::T, θy::T, κ::M, gridx::M, gridy::M) where {T <: RV, M <: ROA}
    Gives the Jacobian at a point for the given convergence. The Jacobian is given as,
```math
A(θ_x, θ_y) = \\grad^2 ψ = Σ κ_mn * 1/π * deriv (deriv(ψ_{mn})) 
where the ψ_{mn} is as defined before.
```
"""
function jacobian!(ψxx::T, ψyy::T, ψxy::T, θx::T, θy::T, κ::M, gridx::M, gridy::M) where {T <: RV, M <: ROA}

   ψxx_up, ψyy_up, ψxy_up = zero(ψxx), zero(ψyy), zero(ψxy)

   ax1, ax2 = axes(gridx, 1), axes(gridx, 2)
   cell_size = gridx[2] - gridx[1]

   definite_deriv_xx_m, definite_deriv_yy_m, definite_deriv_xy_m = zeros(size(gridx, 1), size(gridx, 2)), zeros(size(gridx, 1), size(gridx, 2)), zeros(size(gridx, 1), size(gridx, 2))

   @inbounds Threads.@threads for j in ax2
      @inbounds for i in ax1
         xc, yc = gridx[i, 1], gridy[1, j]
         definite_deriv_xx_m[i,j] = definite_deriv_xx(θx, θy, xc, yc, cell_size)
         definite_deriv_yy_m[i,j] = definite_deriv_yy(θx, θy, xc, yc, cell_size)
         definite_deriv_xy_m[i,j] = definite_deriv_xy(θx, θy, xc, yc, cell_size)
      end
   end

   @inbounds for j in ax2
      @inbounds for i in ax1
         xc, yc = gridx[i, 1], gridy[1, j]
         ψxx_up += κ[i, j] * definite_deriv_xx_m[i,j]
         ψyy_up += κ[i, j] * definite_deriv_yy_m[i,j]
         ψxy_up += κ[i, j] * definite_deriv_xy_m[i,j]
      end
   end
   ψxx += (1/π) * ψxx_up
   ψyy += (1/π) * ψyy_up
   ψxy += (1/π) * ψxy_up

   return ψxx, ψyy, ψxy
end

"""function jacobian!(ψxx::T, ψyy::T, ψxy::T, θx::T, θy::T, κ::M, gridx::M, gridy::M) where {T <: ROA, M <: ROA}
   
   θx_flat, θy_flat = vec(θx), vec(θy)
   θx_shape, θy_shape = size(θx), size(θy)

   ax1, ax2, ax3 = axes(gridx, 1), axes(gridx, 2), axes(θx_flat, 1)
   cell_size = gridx[2] - gridx[1]

   definite_deriv_xx_m, definite_deriv_yy_m, definite_deriv_xy_m = zeros(size(gridx, 1), size(gridx, 2), length(θx_flat)), zeros(size(gridx, 1), size(gridx, 2), length(θx_flat)), zeros(size(gridx, 1), size(gridx, 2), length(θx_flat))

   @inbounds for k in ax3
      @inbounds Threads.@threads for j in ax2
         @inbounds for i in ax1
            xc, yc = gridx[i, 1], gridy[1, j]
            definite_deriv_xx_m[i,j,k] = definite_deriv_xx(θx_flat[k], θy_flat[k], xc, yc, cell_size)
            definite_deriv_yy_m[i,j,k] = definite_deriv_yy(θx_flat[k], θy_flat[k], xc, yc, cell_size)
            definite_deriv_xy_m[i,j,k] = definite_deriv_xy(θx_flat[k], θy_flat[k], xc, yc, cell_size)
         end
      end
   end

   @inbounds for k in ax3
      @inbounds for j in ax2
         @inbounds for i in ax1
            ψxx[k] += (1/π) * κ[i, j] * definite_deriv_xx_m[i,j,k]
            ψyy[k] += (1/π) * κ[i, j] * definite_deriv_yy_m[i,j,k]
            ψxy[k] += (1/π) * κ[i, j] * definite_deriv_xy_m[i,j,k]
         end
      end
   end

   θx = reshape(θx_flat, θx_shape)
   θy = reshape(θy_flat, θy_shape)

   return nothing
end"""

function jacobian!(ψxx::T, ψyy::T, ψxy::T, θx::T, θy::T, κ::M, gridx::M, gridy::M) where {T <: ROA, M <: ROA}

   θx_flat, θy_flat = vec(θx), vec(θy)
   θx_shape, θy_shape = size(θx), size(θy)

   ax1, ax2, ax3 = axes(gridx, 1), axes(gridx, 2), axes(θx_flat, 1)
   cell_size = gridx[2] - gridx[1]

   @inbounds for k in ax3
      θxk, θyk = θx_flat[k], θy_flat[k]

      chunks = Iterators.partition(ax2, cld(length(ax2), Threads.nthreads()))

      tasks = map(chunks) do chunk
         Threads.@spawn begin
            local_xx, local_yy, local_xy = 0.0, 0.0, 0.0
            @inbounds for j in chunk
               @inbounds for i in ax1
                  xc, yc = gridx[i, 1], gridy[1, j]
                  local_xx += κ[i, j] * definite_deriv_xx(θxk, θyk, xc, yc, cell_size)
                  local_yy += κ[i, j] * definite_deriv_yy(θxk, θyk, xc, yc, cell_size)
                  local_xy += κ[i, j] * definite_deriv_xy(θxk, θyk, xc, yc, cell_size)
               end
            end
            (local_xx, local_yy, local_xy)
         end
      end

      chunk_sums = fetch.(tasks)
      ψxx[k] += (1/π) * sum(t[1] for t in chunk_sums)
      ψyy[k] += (1/π) * sum(t[2] for t in chunk_sums)
      ψxy[k] += (1/π) * sum(t[3] for t in chunk_sums)
   end

   θx = reshape(θx_flat, θx_shape)
   θy = reshape(θy_flat, θy_shape)

   return nothing
end

function jacobian!(ψxx::T, ψyy::T, ψxy::T, θx::T, θy::T, κ::M, gridx::M, gridy::M, kernel::Vector{NTuple{6, Matrix{Float64}}}) where {T <: ROA, M <: ROA}
   """
   Faster version of above function. This needs pre-computed ψ_mn values for each cell in the grid.
   """
   ψxx_up, ψyy_up, ψxy_up = zero(ψxx), zero(ψyy), zero(ψxy)
   ax1, ax2, ax3 = axes(gridx, 1), axes(gridx, 2), axes(θx, 1)

   @inbounds for k in ax3
      deriv_xx_kernel = kernel[k][4]     # a matrix of size (length(gridx), length(gridy)) containing the ψ_mn values for each cell in the grid for the k-th image position.
      deriv_yy_kernel = kernel[k][5]
      deriv_xy_kernel = kernel[k][6]
      @inbounds for j in ax2
         @inbounds for i in ax1
            ψxx_up[k] += κ[i, j] * deriv_xx_kernel[i, j]
            ψyy_up[k] += κ[i, j] * deriv_yy_kernel[i, j]
            ψxy_up[k] += κ[i, j] * deriv_xy_kernel[i, j]
         end
      end
   end

   @. ψxx += (1/π) * ψxx_up
   @. ψyy += (1/π) * ψyy_up
   @. ψxy += (1/π) * ψxy_up

   return nothing
end

# constructor for the free lens
@kwdef struct init_FreeFormLens <: Lenses.AbstractLens
    _lens_::Symbol = :FreeFormLens
    κ::ROA
    gridx::ROA
    gridy::ROA
    kernel_flag::Bool = false
end

"""# if kernel is not passed during lens init, then kernel_flag is set to false and kernel is set to nothing.
# Slower functions are used.
# This is general purpose lensing. For free-form lens modelling, use the kernel version.
init_FreeFormLens(κ::ROA, gridx::ROA, gridy::ROA) =
    init_FreeFormLens(_lens_ = :FreeFormLens, κ = κ, gridx = gridx, gridy = gridy, kernel_flag = false, kernel = nothing)

init_FreeFormLens(κ::RV, gridx::ROA, gridy::ROA) =
    init_FreeFormLens(_lens_ = :FreeFormLens, κ = fill(κ, size(gridx)), gridx = gridx, gridy = gridy, kernel_flag = false, kernel = nothing)"""

# kernel is passed during lens init, if it points to nothing, then slower functions are used (for general purpose lensing). If it points to a pre-computed kernel, then
# faster functions are used. But the kernel needs to be precomputed using compute_kernel function before lens init in this case.
# This is for lens modelling specifically for faster calculations.
init_FreeFormLens(κ::ROA, gridx::ROA, gridy::ROA, kernel_flag::Bool) =
    init_FreeFormLens(_lens_ = :FreeFormLens, κ = κ, gridx = gridx, gridy = gridy, kernel_flag = kernel_flag)

init_FreeFormLens(κ::RV, gridx::ROA, gridy::ROA, kernel_flag::Bool) =
    init_FreeFormLens(_lens_ = :FreeFormLens, κ = fill(κ, size(gridx)), gridx = gridx, gridy = gridy, kernel_flag = kernel_flag)

init_FreeFormLens(κ::ROA, gridx::ROA, gridy::ROA) =
    init_FreeFormLens(_lens_ = :FreeFormLens, κ = κ, gridx = gridx, gridy = gridy, kernel_flag = false)

function Lenses.potential_helper!(ψ::T, lens::init_FreeFormLens, θx::T, θy::T, kernel::Union{Nothing, Vector{NTuple{6, Matrix{Float64}}}} = nothing) where T <: Union{RV, ROA}
   if lens.kernel_flag && kernel != nothing
      return potential!(ψ, θx, θy, lens.κ, lens.gridx, lens.gridy, kernel)
   elseif !lens.kernel_flag  && kernel != nothing
      @warn "Kernel is passed but kernel_flag is false. calling original function."
      return potential!(ψ, θx, θy, lens.κ, lens.gridx, lens.gridy) 
   else
      return potential!(ψ, θx, θy, lens.κ, lens.gridx, lens.gridy)
   end
end

function Lenses.deflection_helper!(ψx::T, ψy::T, lens::init_FreeFormLens, θx::T, θy::T, kernel::Union{Nothing, Vector{NTuple{6, Matrix{Float64}}}} = nothing) where T <: Union{RV, ROA}
    if lens.kernel_flag && kernel != nothing
        return deflection!(ψx, ψy, θx, θy, lens.κ, lens.gridx, lens.gridy, kernel)
    elseif !lens.kernel_flag  && kernel != nothing
      @warn "Kernel is passed but kernel_flag is false. calling original function."
        return deflection!(ψx, ψy, θx, θy, lens.κ, lens.gridx, lens.gridy) 
    else
        return deflection!(ψx, ψy, θx, θy, lens.κ, lens.gridx, lens.gridy)
    end
end

function Lenses.jacobian_helper!(ψxx::T, ψyy::T, ψxy::T, lens::init_FreeFormLens, θx::T, θy::T, kernel::Union{Nothing, Vector{NTuple{6, Matrix{Float64}}}} = nothing) where T <: Union{RV, ROA}
    if lens.kernel_flag && kernel != nothing
        return jacobian!(ψxx, ψyy, ψxy, θx, θy, lens.κ, lens.gridx, lens.gridy, kernel)
    elseif !lens.kernel_flag  && kernel != nothing
      @warn "Kernel is passed but kernel_flag is false. calling original function."
        return jacobian!(ψxx, ψyy, ψxy, θx, θy, lens.κ, lens.gridx, lens.gridy) 
    else
        return jacobian!(ψxx, ψyy, ψxy, θx, θy, lens.κ, lens.gridx, lens.gridy)
    end
end

# The next new methods are needed so that composite lenses can work as well. 
# Will be needed for hybrid modelling in the future.

# new methods for Lenses module - they work with precomputed kernel for freeform lens.

function Lenses.get_potential(lens::Lenses.AbstractLens, θx::T, θy::T, kernel::Vector{NTuple{6, Matrix{Float64}}}) where T <: ROA
   # Check if the input coordinates are of the same size
   if size(θx) != size(θy)
      throw(ArgumentError("Input coordinates must be of the same size."))
   end

   # Promote both only if either is Int64
   if eltype(θx) === Int64 || eltype(θy) === Int64
      θx = Float64.(θx)
      θy = Float64.(θy)
   end

   # Initialize zero-valued potential array
   ψ = zero(θx)

   if lens._lens_ == :CompositeLens
      for component in lens._components_
         if component._lens_ == :FreeFormLens && kernel != nothing
            Lenses.potential_helper!(ψ, component, θx, θy, kernel)
         else
            Lenses.potential_helper!(ψ, component, θx, θy)
         end
      end
      return ψ
   else
      if lens._lens_ == :FreeFormLens && kernel != nothing
         Lenses.potential_helper!(ψ, lens, θx, θy, kernel)
      else
         Lenses.potential_helper!(ψ, lens, θx, θy)
      end
      return ψ
   end
end

function Lenses.get_deflection(lens::Lenses.AbstractLens, θx::T, θy::T, kernel::Vector{NTuple{6, Matrix{Float64}}}) where T <: ROA
   """
   Same except it takes kid as well. So that freefmlens during modelling knows which 
   sub-kernel to use for the given image positions. The kernel is a vector of vectors of Ntuples of matrices.
   """
   # Check if the input coordinates are of the same size
   if size(θx) != size(θy)
      throw(ArgumentError("Input coordinates must be of the same size."))
   end
   
   # Promote both only if either is Int64
   if eltype(θx) === Int64 || eltype(θy) === Int64
      θx = Float64.(θx)
      θy = Float64.(θy)
   end

   # Initialize zero-valued potential array
   ψx = zero(θx)
   ψy = zero(θx)

   if lens._lens_ == :CompositeLens
      for component in lens._components_
         if component._lens_ == :FreeFormLens && kernel != nothing
            Lenses.deflection_helper!(ψx, ψy, component, θx, θy, kernel)
         else
            Lenses.deflection_helper!(ψx, ψy, component, θx, θy)
         end
      end
      return ψx, ψy
   else
      if lens._lens_ == :FreeFormLens && kernel != nothing
         Lenses.deflection_helper!(ψx, ψy, lens, θx, θy, kernel)
      else
         Lenses.deflection_helper!(ψx, ψy, lens, θx, θy)
      end
      return ψx, ψy
   end
end

function Lenses.get_jacobian(lens::Lenses.AbstractLens, θx::T, θy::T, kernel::Vector{NTuple{6, Matrix{Float64}}}) where T <: ROA
   # Check if the input coordinates are of the same size
   if size(θx) != size(θy)
      throw(ArgumentError("Input coordinates must be of the same size."))
   end
   
   # Check if the input coordinates are of the same size
   if size(θx) != size(θy)
      throw(ArgumentError("Input coordinates must be of the same size."))
   end
   
   # Promote both only if either is Int64
   if eltype(θx) === Int64 || eltype(θy) === Int64
      θx = Float64.(θx)
      θy = Float64.(θy)
   end

   # Initialize zero-valued potential array
   ψxx = zero(θx)
   ψyy = zero(θy)
   ψxy = zero(θx)

   if lens._lens_ == :CompositeLens
      for component in lens._components_
         if component._lens_ == :FreeFormLens && kernel != nothing
            Lenses.jacobian_helper!(ψxx, ψyy, ψxy, component, θx, θy, kernel)
         else
            Lenses.jacobian_helper!(ψxx, ψyy, ψxy, component, θx, θy)
         end
      end
      return ψxx, ψyy, ψxy
   else
      if lens._lens_ == :FreeFormLens && kernel != nothing
         Lenses.jacobian_helper!(ψxx, ψyy, ψxy, lens, θx, θy, kernel)
      else
         Lenses.jacobian_helper!(ψxx, ψyy, ψxy, lens, θx, θy)
      end
      return ψxx, ψyy, ψxy
   end
end

# new methods for plotting functions. Helpful in diagnostics, where the image positions and lens are fixed.
# so no need to calculate the lens quantities again and again. Just use the pre-computed lens quantities.
# for these functions, only replaced lens argument with the tuple containing pre-computed lens quantities.

function Lenses.get_image(gridqty_tuple::NTuple{6, T}, θx::T, θy::T, adis::Float64, β::NTuple{2, RV}) where T <: Matrix{<:RV}
   # Get the potential gradient
   ψx, ψy = gridqty_tuple[2], gridqty_tuple[3]

   # Get grid for contour
   RXC = ContourFinder.get_contour(θx, θy, β[1] .- θx .+ adis .* ψx, 0.0)
   RYC = ContourFinder.get_contour(θx, θy, β[2] .- θy .+ adis .* ψy, 0.0)

   # Initialize empty Vector of tuples to store image positions
   image_position::Vector{NTuple{2, RV}} = []
   for contour_1 in RXC
      for contour_2 in RYC
         # Find the intersection points
         intersect_points = IntersectionFinder.get_intersection(first.(contour_1), last.(contour_1), first.(contour_2), last.(contour_2))
         
         # Store the intersection points in the image_position vector
         for point in intersect_points
            push!(image_position, point)
         end
      end
   end
   return image_position
end

function Lenses.get_critical_curve(gridqty_tuple::NTuple{6, T}, θx::T, θy::T, adis::Float64) where T <: Matrix{<:RV}
   # Get the jacobian components
   ψxx, ψyy, ψxy = gridqty_tuple[4], gridqty_tuple[5], gridqty_tuple[6]

   # Scale the deformation tensor
   ψxx_s = adis .* ψxx
   ψyy_s = adis .* ψyy
   ψxy_s = adis .* ψxy

   # Convergence and shear components
   κ  = 0.5 .* (ψxx_s .+ ψyy_s)
   γ1 = 0.5 .* (ψxx_s .- ψyy_s)
   γ2 = ψxy_s

   # Get the zero eigenvalue contours
   critical_tan = ContourFinder.get_contour(θx, θy, 1.0 .- κ .- sqrt.(γ1.^2 .+ γ2.^2), 0)
   critical_rad = ContourFinder.get_contour(θx, θy, 1.0 .- κ .+ sqrt.(γ1.^2 .+ γ2.^2), 0)

   return critical_tan, critical_rad   
end

function Lenses.get_caustic(lens::Lenses.AbstractLens, gridqty_tuple::NTuple{6, T}, θx::T, θy::T, adis::Float64) where T <: Matrix{<:RV}
   # Generate critical curves
   critical_tan, critical_rad = Lenses.get_critical_curve(gridqty_tuple, θx, θy, adis)

   # Get tangential caustics
   caustics_tan = Vector{Vector{Vector{Float64}}}(undef, length(critical_tan))
   for (idx, curve) in enumerate(critical_tan)
      ψ_x, ψ_y = Lenses.get_deflection(lens, first.(curve), last.(curve))
      src_x = first.(curve) .- adis .* ψ_x
      src_y =  last.(curve) .- adis .* ψ_y
      caustics_tan[idx] = [[x, y] for (x, y) in zip(src_x, src_y)]
   end
 
   # Get radial caustics
   caustics_rad = Vector{Vector{Vector{Float64}}}(undef, length(critical_rad))
   for (idx, curve) in enumerate(critical_rad)
      ψ_x, ψ_y = Lenses.get_deflection(lens, first.(curve), last.(curve))
      src_x = first.(curve) .- adis .* ψ_x
      src_y =  last.(curve) .- adis .* ψ_y
      caustics_rad[idx] = [[x, y] for (x, y) in zip(src_x, src_y)]
   end
   return caustics_tan, caustics_rad
end

function LensFactory.Lenses.plot_image_plane(lens::Lenses.AbstractLens,gridqty_tuple::NTuple{6, T}, θx::Matrix{<:RV}, θy::Matrix{<:RV}, adis::Float64;
                           two_panel::Bool=false,
                           plot_critical::Bool=true,
                           critical_tan_kws::NamedTuple=(color=:red, linewidth=2, linestyle=:solid),
                           critical_rad_kws::NamedTuple=(color=:red, linewidth=2, linestyle=:dash),
                           plot_caustic::Bool=true,
                           caustic_tan_kws::NamedTuple=(color=:green, linewidth=2, linestyle=:solid),
                           caustic_rad_kws::NamedTuple=(color=:green, linewidth=2, linestyle=:dash),
                           source::Union{Nothing, NTuple{2, RV}, Matrix{<:RV}} = nothing,
                           source_kws::NamedTuple=(color=:red, markersize=10, marker=:star5, heatmap=cgrad([:white, :blue])),
                           image_kws::NamedTuple=(color=:blue, markersize=10, marker=:star5, heatmap=cgrad([:white, :red])),
                           save_plot::Bool=false,
                           plot_name::String="image_plane.png",
                           resolution::Int64=2) where T <: Matrix{<:RV}

   if two_panel
      # Initialize empty figure
      fig = Figure(size=(800, 400), figure_padding=15, fontsize=20, fonts=(; regular="Times New Roman"))

      # Axis for source plane
      ax1 = Axis(fig[1, 1])

      # Plot source and its images
      if source !== nothing
         if isa(source, NTuple{2, RV})
            scatter!(ax1, source[1], source[2], color=source_kws.color, markersize=source_kws.markersize, marker=source_kws.marker)
         elseif isa(source, Matrix{<:RV})
            heatmap!(ax1, θx[:,1], θy[1,:], source, colormap=source_kws.heatmap)
         else
            error("Invalid source type: $(typeof(source)). Must be NTuple{2, RV} or Matrix{<:RV}.")
         end
      end

      # Get caustics and plot
      if plot_caustic
         # Get caustics
         caustic_tan, caustic_rad = Lenses.get_caustic(lens, gridqty_tuple, θx, θy, adis)

         # Plot tangential caustic
         for curve in caustic_tan
            lines!(ax1, first.(curve), last.(curve); caustic_tan_kws...)
         end

         # Plot radial caustic
         for curve in caustic_rad
            lines!(ax1, first.(curve), last.(curve); caustic_rad_kws...)
         end
      end

   
      # Set plot keywords
      LensFactory.Lenses.set_plotKws!(ax1)

      # Set axis labels and limits
      ax1.xlabel = L"\theta_1~\text{(in arcseconds)}"
      ax1.ylabel = L"\theta_2~\text{(in arcseconds)}"
      xlims!(minimum(θx), maximum(θx))
      ylims!(minimum(θy), maximum(θy))

      # Axis for image plane
      ax2 = Axis(fig[1, 2])

      # Plot source and its images
      if source !== nothing
         # Get the image positions
         image = Lenses.get_image(gridqty_tuple, θx, θy, adis, source)
         if isa(source, NTuple{2, RV})
            scatter!(ax2, first.(image), last.(image), color=image_kws.color, markersize=image_kws.markersize, marker=image_kws.marker)
         elseif isa(source, Matrix{<:RV})
            heatmap!(ax2, θx[:,1], θy[1,:], image, colormap=image_kws.heatmap)
         else
            ArgumentError("Invalid source type: $(typeof(source)). Must be NTuple{2, RV} or Matrix{<:RV}.")
         end
      end

      # Get critical curves
      if plot_critical
         # Get critical curves
         crit_tan, crit_rad = Lenses.get_critical_curve(gridqty_tuple, θx, θy, adis)

         # Plot tangential critical curve
         for curve in crit_tan
            lines!(ax2, first.(curve), last.(curve); critical_tan_kws...)
         end

         # Plot radial critical curve
         for curve in crit_rad
            lines!(ax2, first.(curve), last.(curve); critical_rad_kws...)
         end
      end

      # Set plot keywords
      LensFactory.Lenses.set_plotKws!(ax2)

      # Set axis labels and limits
      ax2.xlabel = L"\theta_1~\text{(in arcseconds)}"
      ax2.ylabel = L"\theta_2~\text{(in arcseconds)}"
      xlims!(minimum(θx), maximum(θx))
      ylims!(minimum(θy), maximum(θy))

      # Save plot
      if save_plot
         save(plot_name, fig, px_per_unit=resolution)
      end
      return fig, [ax1, ax2]
   else
      # Initialize empty figure
      fig = Figure(size=(400, 400), figure_padding=15, fontsize=20, fonts=(; regular="Times New Roman"))
      
      # Plot source + image plane
      ax = Axis(fig[1, 1])

      if source !== nothing
         # Get the image positions
         image = Lenses.get_image(gridqty_tuple, θx, θy, adis, source)

         if isa(source, NTuple{2, RV})
            scatter!(ax, source[1], source[2], color=source_kws.color, markersize=source_kws.markersize, marker=source_kws.marker)
            scatter!(ax, first.(image), last.(image), color=image_kws.color, markersize=image_kws.markersize, marker=image_kws.marker)
         elseif isa(source, Matrix{<:RV})
            heatmap!(ax, θx[:,1], θy[1,:], source, colormap=source_kws.heatmap, alpha=1.0)                                       
            heatmap!(ax, θx[:,1], θy[1,:], image, colormap=image_kws.heatmap, alpha=0.8)
         else
            error("Invalid source type: $(typeof(source)). Must be NTuple{2, RV} or Matrix{<:RV}.")
         end
      end

      if plot_caustic
         # Get caustics
         caustic_tan, caustic_rad = Lenses.get_caustic(gridqty_tuple, θx, θy, adis)

         # Plot tangential caustic
         for curve in caustic_tan
            lines!(ax, first.(curve), last.(curve); caustic_tan_kws...)
         end

         # Plot radial caustic
         for curve in caustic_rad
            lines!(ax, first.(curve), last.(curve); caustic_rad_kws...)
         end
      end

      if plot_critical
         # Get critical curves
         crit_tan, crit_rad = Lenses.get_critical_curve(gridqty_tuple, θx, θy, adis)

         # Plot tangential critical curve
         for curve in crit_tan
            lines!(ax, first.(curve), last.(curve); critical_tan_kws...)
         end

         # Plot radial critical curve
         for curve in crit_rad
            lines!(ax, first.(curve), last.(curve); critical_rad_kws...)
         end
      end


      # Set plot keywords
      LensFactory.Lenses.set_plotKws!(ax)

      # Set axis labels and limits
      ax.xlabel = L"\theta_1~\text{(in arcseconds)}"
      ax.ylabel = L"\theta_2~\text{(in arcseconds)}"
      xlims!(minimum(θx), maximum(θx))
      ylims!(minimum(θy), maximum(θy))

      if save_plot
         save(plot_name, fig, px_per_unit=resolution)
      end
      return fig, ax
   end
end

# New methods for LensModel that are compatible with the fast kernel based calculations.

function LensModel.LensModelUtils.lens_quantities(model::LensModel.ModelConfig, lens::Lenses.AbstractLens, full_kernel::Union{Vector{Vector{NTuple{6, Matrix{Float64}}}}})
   # Count the total number of knots in the lens model
   n_knots = sum(length(s.knots) for s in model.source_config.sources)

   # Allocate outputs (vector of vectors)
   ψ_all  = Vector{Vector{Float64}}(undef, n_knots)
   αx_all = Vector{Vector{Float64}}(undef, n_knots)
   αy_all = Vector{Vector{Float64}}(undef, n_knots)
   A_all  = Vector{NTuple{4, Vector{Float64}}}(undef, n_knots)

   kid = 1
   for src in model.source_config.sources
      for knot in src.knots
         # One image system knot positions
         x = knot.x
         y = knot.y

         sub_kernel = full_kernel[kid]

         # Potential
         ψ_all[kid] = Lenses.get_potential(lens, x, y, sub_kernel)

         # Deflection
         αx_all[kid], αy_all[kid] = Lenses.get_deflection(lens, x, y, sub_kernel)

         # Deformation tensor
         ψxx, ψyy, ψxy = Lenses.get_jacobian(lens, x, y, sub_kernel)
         A_all[kid] = (ψxx, ψxy, copy(ψxy), ψyy)

         # Increment
         kid = kid + 1
      end
   end
   return ψ_all, αx_all, αy_all, A_all
end

Lenses.lens_init_functions[:FreeFormLens] = (
    comp -> FreeFormLens.init_FreeFormLens(κ = comp.κ, gridx = comp.gridx, gridy = comp.gridy, kernel_flag = comp.kernel_flag)
)

""" Additional function. For given image positions, calculates the matrices corresponding to
definite_integral, definite_deriv_x, definite_deriv_y, definite_deriv_xx, definite_deriv_yy, definite_deriv_xy.
This makes the calculation of potential, deflection and jacobian much faster as we dont have to calculate atan and log
each time the functions are called. But it is memory intensive as the kernel needs to be stored always.

But need to make sure this function is called everytime the image positions are changed. Or better, use these functions only for 
free-form modelling. For general purpose lensing, use the original functions. 
"""

function give_kernel_serial(θx::T, θy::T, gridx::M, gridy::M) where {T <: ROA, M <: ROA}

    output_array = Vector{NTuple{6, Matrix{Float64}}}(undef, length(θx))
    ax1, ax2, ax3 = axes(gridx, 1), axes(gridx, 2), axes(θx, 1)
    cell_size = gridx[2] - gridx[1]
    n1, n2 = size(gridx, 1), size(gridx, 2)

    @inbounds for k in ax3
        k_integral = zeros(n1, n2)
        k_dx       = zeros(n1, n2)
        k_dy       = zeros(n1, n2)
        k_dxx      = zeros(n1, n2)
        k_dyy      = zeros(n1, n2)
        k_dxy      = zeros(n1, n2)
        @inbounds for j in ax2
            @inbounds for i in ax1
                xc, yc = gridx[i, 1], gridy[1, j]
                k_integral[i,j] = definite_integral(θx[k], θy[k], xc, yc, cell_size)
                k_dx[i,j]       = definite_deriv_x(θx[k], θy[k], xc, yc, cell_size)
                k_dy[i,j]       = definite_deriv_y(θx[k], θy[k], xc, yc, cell_size)
                k_dxx[i,j]      = definite_deriv_xx(θx[k], θy[k], xc, yc, cell_size)
                k_dyy[i,j]      = definite_deriv_yy(θx[k], θy[k], xc, yc, cell_size)
                k_dxy[i,j]      = definite_deriv_xy(θx[k], θy[k], xc, yc, cell_size)
            end
        end
        output_array[k] = (k_integral, k_dx, k_dy, k_dxx, k_dyy, k_dxy)
    end
    return output_array
end

function give_kernel(θx::T, θy::T, gridx::M, gridy::M) where {T <: ROA, M <: ROA}

    output_array = Vector{NTuple{6, Matrix{Float64}}}(undef, length(θx))
    ax1, ax2, ax3 = axes(gridx, 1), axes(gridx, 2), axes(θx, 1)
    cell_size = gridx[2] - gridx[1]
    n1, n2 = size(gridx, 1), size(gridx, 2)

    @inbounds for k in ax3
        k_integral = zeros(n1, n2)
        k_dx       = zeros(n1, n2)
        k_dy       = zeros(n1, n2)
        k_dxx      = zeros(n1, n2)
        k_dyy      = zeros(n1, n2)
        k_dxy      = zeros(n1, n2)
        @inbounds Threads.@threads for j in ax2
            @inbounds for i in ax1
                xc, yc = gridx[i, 1], gridy[1, j]
                k_integral[i,j] = definite_integral(θx[k], θy[k], xc, yc, cell_size)
                k_dx[i,j]       = definite_deriv_x(θx[k], θy[k], xc, yc, cell_size)
                k_dy[i,j]       = definite_deriv_y(θx[k], θy[k], xc, yc, cell_size)
                k_dxx[i,j]      = definite_deriv_xx(θx[k], θy[k], xc, yc, cell_size)
                k_dyy[i,j]      = definite_deriv_yy(θx[k], θy[k], xc, yc, cell_size)
                k_dxy[i,j]      = definite_deriv_xy(θx[k], θy[k], xc, yc, cell_size)
            end
        end
        output_array[k] = (k_integral, k_dx, k_dy, k_dxx, k_dyy, k_dxy)
    end
    return output_array
end

function compute_fullkernel(model::LensModel.ModelConfig, gridx::M, gridy::M) where {M <: ROA}

   n_knots = sum(length(s.knots) for s in model.source_config.sources)

   # Output vectors for the kernel
   # similar structure as that of various lens quantities that Lenses.lens_quantitites returns. Only that Float64 is replaced by NTuple{6, Matrix{Float64}}. 
   # This is because for each image position, we need to store the kernel values for all the cells in the grid for all 6 quantities (potential, deflection_x, deflection_y, jacobian_xx, jacobian_yy, jacobian_xy).
   kernel = Vector{Vector{NTuple{6,Matrix{Float64}}}}(undef, n_knots)     

   kid = 1
   for src in model.source_config.sources
      for knot in src.knots
         x = knot.x
         y = knot.y

         kernel[kid] = give_kernel(x, y, gridx, gridy)
         kid += 1
      end
   end
   return kernel
end

end # module ends
