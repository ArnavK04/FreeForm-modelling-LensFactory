module FreeFormLens

using LensFactory
using LensFactory.Constants
using Trapz

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
   @inbounds for j in ax2
      @inbounds for i in ax1
         xc, yc = gridx[i, 1], gridy[1, j]
         ψ_up += κ[i, j] * definite_integral(θx, θy, xc, yc, cell_size)
      end
   end
   ψ += (1/π) * ψ_up

   return ψ
end

function potential!(ψ::T, θx::T, θy::T, κ::M, gridx::M, gridy::M) where {T <: ROA, M <: ROA}

   ax1, ax2 = axes(gridx, 1), axes(gridx, 2)
   cell_size = gridx[2] - gridx[1]
   @inbounds for j in ax2
      @inbounds for i in ax1
         xc, yc = gridx[i, 1], gridy[1, j]
         ψ .= ψ .+ (1/π) .* κ[i, j] .* definite_integral.(θx, θy, xc, yc, cell_size)
      end
   end

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
   
   @inbounds for j in ax2
      @inbounds for i in ax1
         xc, yc = gridx[i, 1], gridy[1, j]
         ψx_up += κ[i, j] * definite_deriv_x(θx, θy, xc, yc, cell_size)
         ψy_up += κ[i, j] * definite_deriv_y(θx, θy, xc, yc, cell_size)
      end
   end
   ψx += (1/π) * ψx_up
   ψy += (1/π) * ψy_up

   return ψx, ψy
end

function deflection!(ψx::T, ψy::T, θx::T, θy::T, κ::M, gridx::M, gridy::M) where {T <: ROA, M <: ROA}

   ax1, ax2 = axes(gridx, 1), axes(gridx, 2)
   cell_size = gridx[2] - gridx[1]

   @inbounds for j in ax2
      @inbounds for i in ax1
         xc, yc = gridx[i, 1], gridy[1, j]
         ψx .= ψx .+ (1/π) .* κ[i, j] .* definite_deriv_x.(θx, θy, xc, yc, cell_size)
         ψy .= ψy .+ (1/π) .* κ[i, j] .* definite_deriv_y.(θx, θy, xc, yc, cell_size)
      end
   end

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

   @inbounds for j in ax2
      @inbounds for i in ax1
         xc, yc = gridx[i, 1], gridy[1, j]
         ψxx_up += κ[i, j] * definite_deriv_xx(θx, θy, xc, yc, cell_size)
         ψyy_up += κ[i, j] * definite_deriv_yy(θx, θy, xc, yc, cell_size)
         ψxy_up += κ[i, j] * definite_deriv_xy(θx, θy, xc, yc, cell_size)
      end
   end
   ψxx += (1/π) * ψxx_up
   ψyy += (1/π) * ψyy_up
   ψxy += (1/π) * ψxy_up

   return ψxx, ψyy, ψxy
end

function jacobian!(ψxx::T, ψyy::T, ψxy::T, θx::T, θy::T, κ::M, gridx::M, gridy::M) where {T <: ROA, M <: ROA}
   
   ax1, ax2 = axes(gridx, 1), axes(gridx, 2)
   cell_size = gridx[2] - gridx[1]
   
   @inbounds for j in ax2
      @inbounds for i in ax1
         xc, yc = gridx[i, 1], gridy[1, j]
         ψxx .= ψxx .+ (1/π) .* κ[i, j] .* definite_deriv_xx.(θx, θy, xc, yc, cell_size)
         ψyy .= ψyy .+ (1/π) .* κ[i, j] .* definite_deriv_yy.(θx, θy, xc, yc, cell_size)
         ψxy .= ψxy .+ (1/π) .* κ[i, j] .* definite_deriv_xy.(θx, θy, xc, yc, cell_size)
      end
   end

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
   if lens.kernel_flag     
      # I have cheated a bit here, so that not many changes are made to get_deflection function.
      # kid tells which sub-kernel to use for the given image positions. The kernel is a vector of vectors of Ntuples of matrices.
      return potential!(ψ, θx, θy, lens.κ, lens.gridx, lens.gridy, kernel)
   elseif !lens.kernel_flag  && kernel != nothing
      @warn "Kernel is passed but kernel_flag is false. not doing anything." 
   else
      return potential!(ψ, θx, θy, lens.κ, lens.gridx, lens.gridy)
   end
end

function Lenses.deflection_helper!(ψx::T, ψy::T, lens::init_FreeFormLens, θx::T, θy::T, kernel::Union{Nothing, Vector{NTuple{6, Matrix{Float64}}}} = nothing) where T <: Union{RV, ROA}
    if lens.kernel_flag
        return deflection!(ψx, ψy, θx, θy, lens.κ, lens.gridx, lens.gridy, kernel)
    elseif !lens.kernel_flag  && kernel != nothing
      @warn "Kernel is passed but kernel_flag is false. not doing anything." 
    else
        return deflection!(ψx, ψy, θx, θy, lens.κ, lens.gridx, lens.gridy)
    end
end

function Lenses.jacobian_helper!(ψxx::T, ψyy::T, ψxy::T, lens::init_FreeFormLens, θx::T, θy::T, kernel::Union{Nothing, Vector{NTuple{6, Matrix{Float64}}}} = nothing) where T <: Union{RV, ROA}
    if lens.kernel_flag
        return jacobian!(ψxx, ψyy, ψxy, θx, θy, lens.κ, lens.gridx, lens.gridy, kernel)
    elseif !lens.kernel_flag  && kernel != nothing
      @warn "Kernel is passed but kernel_flag is false. not doing anything." 
    else
        return jacobian!(ψxx, ψyy, ψxy, θx, θy, lens.κ, lens.gridx, lens.gridy)
    end
end

# The next new methods are needed so that composite lenses can work as well. 
# Will be needed for hybrid modelling in the future.

# new methods for Lenses module - they work with precomputed kernel for freeform lens.

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
         if component._lens_ == :FreeFormLens
            Lenses.deflection_helper!(ψx, ψy, component, θx, θy, kernel)
         else
            Lenses.deflection_helper!(ψx, ψy, component, θx, θy)
         end
      end
      return ψx, ψy
   else
      if lens._lens_ == :FreeFormLens
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
         if component._lens_ == :FreeFormLens
            Lenses.jacobian_helper!(ψxx, ψyy, ψxy, component, θx, θy, kernel)
         else
            Lenses.jacobian_helper!(ψxx, ψyy, ψxy, component, θx, θy)
         end
      end
      return ψxx, ψyy, ψxy
   else
      if lens._lens_ == :FreeFormLens
         Lenses.jacobian_helper!(ψxx, ψyy, ψxy, lens, θx, θy, kernel)
      else
         Lenses.jacobian_helper!(ψxx, ψyy, ψxy, lens, θx, θy)
      end
      return ψxx, ψyy, ψxy
   end
end

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
         if component._lens_ == :FreeFormLens
            Lenses.potential_helper!(ψ, component, θx, θy, kernel)
         else
            Lenses.potential_helper!(ψ, component, θx, θy)
         end
      end
      return ψ
   else
      if lens._lens_ == :FreeFormLens
         Lenses.potential_helper!(ψ, lens, θx, θy, kernel)
      else
         Lenses.potential_helper!(ψ, lens, θx, θy)
      end
      return ψ
   end
end

function Lenses.get_magnification_image(lens::Lenses.AbstractLens, θx::T, θy::T, adis::Float64, kernel::Vector{NTuple{6, Matrix{Float64}}}) where T <: ROA
   # Get the jacobian components
   ψxx, ψyy, ψxy = Lenses.get_jacobian(lens, θx, θy, kernel)

   # Scale the deformation tensor
   @. ψxx = adis * ψxx
   @. ψyy = adis * ψyy
   @. ψxy = adis * ψxy

   # μ = 1 / det(1 - A)
   return @. 1.0 / (1.0 + ψxx * ψyy - ψxx - ψyy - ψxy^2)
end

function Lenses.get_image(lens::Lenses.AbstractLens, θx::T, θy::T, adis::Float64, β::NTuple{2, RV}, kernel::Vector{NTuple{6, Matrix{Float64}}}) where T <: Matrix{<:RV}
   # Get the potential gradient
   ψx, ψy = Lenses.get_deflection(lens, θx, θy, kernel)

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