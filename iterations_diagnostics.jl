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

include("FreeFormLens.jl")
include("utility_functions.jl")

"""
Things to add in this.

1. function to read file
2. rms vs iteeration table at best resolution
3. combined plot of various quantities vs iteration
"""