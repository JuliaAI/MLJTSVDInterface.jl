module MLJTSVDInterface

import TSVD
import MLJModelInterface
import ScientificTypesBase
using Random: MersenneTwister

const PKG = "TSVD"
const MMI = MLJModelInterface
const STB = ScientificTypesBase

"""
    TSVDTransformer()

Dimensionality reduction using truncated SVD.

This transformer performs linear dimensionality reduction by means of truncated singular value 
decomposition (SVD). Contrary to PCA, this estimator does not center the data before computing 
the singular value decomposition. This means it can work with sparse matrices efficiently.

"""
MMI.@mlj_model mutable struct TSVDTransformer <: MLJModelInterface.Unsupervised
    nvals::Int = 2
    maxiter::Int = 1000
    rng::Int = 123
end

struct TSVDTransformerResult
    singular_values::Vector{Float64}
    components::Matrix{Float64}
    is_table::Bool
end

as_matrix(X) = MMI.matrix(X)
as_matrix(X::AbstractArray) = X

function MMI.fit(transformer::TSVDTransformer, verbosity, Xuser)
    X = as_matrix(Xuser)
    U, s, V = TSVD.tsvd(
        X,
        transformer.nvals;
        maxiter=transformer.maxiter,
        initvec = convert(Vector{float(eltype(X))}, randn(MersenneTwister(transformer.rng), size(X,1)))
    )
    is_table = ~isa(Xuser, AbstractArray)
    fitresult = TSVDTransformerResult(s, V, is_table)
    cache = nothing

    return fitresult, cache, NamedTuple()
end

# for returning user-friendly form of the learned parameters:
function MMI.fitted_params(::TSVDTransformer, fitresult)
    singular_values = fitresult.singular_values
    components = fitresult.components
    is_table = fitresult.is_table
    return (singular_values = singular_values, components = components, is_table=is_table)
end

function MMI.transform(::TSVDTransformer, fitresult, Xuser)
    X = as_matrix(Xuser)
    Xtransformed = X * fitresult.components

    if fitresult.is_table
        Xtransformed = MMI.table(Xtransformed)
    end

    return Xtransformed
end


## META DATA

MMI.metadata_pkg(TSVDTransformer,
             name="$PKG",
             uuid="9449cd9e-2762-5aa3-a617-5413e99d722e",
             url="https://github.com/JuliaLinearAlgebra/TSVD.jl",
             is_pure_julia=true,
             license="MIT",
             is_wrapper=false
)

MMI.metadata_model(TSVDTransformer,
               input_scitype = Union{MMI.Table(STB.Continuous),AbstractMatrix{STB.Continuous}},
               output_scitype = Union{MMI.Table(STB.Continuous),AbstractMatrix{STB.Continuous}},
               docstring = "Truncated SVD dimensionality reduction",         # brief description
               path = "MLJTSVDInterface.TSVDTransformer"
               )

end # module
