module FreeFormLens

using LensFactory
using LensFactory.Constants
using Trapz

# Functions to export
export potential!
export deflection!
export jacobian!
export init_FreeFormLens

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

   ψ_up = ψ

   ax1, ax2 = axes(gridx, 1), axes(gridx, 2)
   cell_size = gridx[2] - gridx[1]
   @inbounds for j in ax2
      @inbounds for i in ax1
         xc, yc = gridx[i, 1], gridy[1, j]
         ψ_up = ψ_up + (1/π) * κ[i, j] * definite_integral(θx, θy, xc, yc, cell_size)
      end
   end

   return ψ_up
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
   ψx_up, ψy_up = ψx, ψy

   ax1, ax2 = axes(gridx, 1), axes(gridx, 2)
   cell_size = gridx[2] - gridx[1]
   
   @inbounds for j in ax2
      @inbounds for i in ax1
         xc, yc = gridx[i, 1], gridy[1, j]
         ψx_up = ψx_up + (1/π) * κ[i, j] * definite_deriv_x(θx, θy, xc, yc, cell_size)
         ψy_up = ψy_up + (1/π) * κ[i, j] * definite_deriv_y(θx, θy, xc, yc, cell_size)
      end
   end

   return ψx_up, ψy_up
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

"""
    jacobian!(ψxx::T, ψyy::T, ψxy::T, θx::T, θy::T, κ::M, gridx::M, gridy::M) where {T <: RV, M <: ROA}
    Gives the Jacobian at a point for the given convergence. The Jacobian is given as,
```math
A(θ_x, θ_y) = \\grad^2 ψ = Σ κ_mn * 1/π * deriv (deriv(ψ_{mn})) 
where the ψ_{mn} is as defined before.
```
"""
function jacobian!(ψxx::T, ψyy::T, ψxy::T, θx::T, θy::T, κ::M, gridx::M, gridy::M) where {T <: RV, M <: ROA}

   ψxx_up, ψyy_up, ψxy_up = ψxx, ψyy, ψxy

   ax1, ax2 = axes(gridx, 1), axes(gridx, 2)
   cell_size = gridx[2] - gridx[1]

   @inbounds for j in ax2
      @inbounds for i in ax1
         xc, yc = gridx[i, 1], gridy[1, j]
         ψxx_up = ψxx_up + (1/π) * κ[i, j] * definite_deriv_xx(θx, θy, xc, yc, cell_size)
         ψyy_up = ψyy_up + (1/π) * κ[i, j] * definite_deriv_yy(θx, θy, xc, yc, cell_size)
         ψxy_up = ψxy_up + (1/π) * κ[i, j] * definite_deriv_xy(θx, θy, xc, yc, cell_size)
      end
   end
   return ψxx_up, ψyy_up, ψxy_up
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

# constructor for the free lens
@kwdef struct init_FreeFormLens <: Lenses.AbstractLens
    _lens_::Symbol = :FreeFormLens
    κ::ROA
    gridx::ROA
    gridy::ROA
end

init_FreeFormLens(κ::ROA, gridx::ROA, gridy::ROA) =
    init_FreeFormLens(_lens_ = :FreeFormLens, κ = κ, gridx = gridx, gridy = gridy)

init_FreeFormLens(κ::RV, gridx::ROA, gridy::ROA) =
    init_FreeFormLens(_lens_ = :FreeFormLens, κ = fill(κ, size(gridx)), gridx = gridx, gridy = gridy)

function Lenses.potential_helper!(ψ::T, lens::init_FreeFormLens, θx::T, θy::T) where T <: Union{RV, ROA}
    return potential!(ψ, θx, θy, lens.κ, lens.gridx, lens.gridy)
end

function Lenses.deflection_helper!(ψx::T, ψy::T, lens::init_FreeFormLens, θx::T, θy::T) where T <: Union{RV, ROA}
    return deflection!(ψx, ψy, θx, θy, lens.κ, lens.gridx, lens.gridy)
end

function Lenses.jacobian_helper!(ψxx::T, ψyy::T, ψxy::T, lens::init_FreeFormLens, θx::T, θy::T) where T <: Union{RV, ROA}
    return jacobian!(ψxx, ψyy, ψxy, θx, θy, lens.κ, lens.gridx, lens.gridy)
end

Lenses.lens_init_functions[:FreeFormLens] = (
    comp -> FreeFormLens.init_FreeFormLens(κ = comp.κ, gridx = comp.gridx, gridy = comp.gridy)
)

end



