module MLJTSVDInterface

import TSVD
import MLJModelInterface
import ScientificTypesBase

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
end

struct TSVDTransformerResult
    singular_values::Vector{Float64}
    components::Matrix{Float64}
end

function MMI.fit(transformer::TSVDTransformer, verbosity, X)
    U, s, V = TSVD.tsvd(X, transformer.nvals; maxiter=transformer.maxiter)
    fitresult = TSVDTransformerResult(s, V)
    cache = nothing

    return fitresult, cache, NamedTuple()
end

# for returning user-friendly form of the learned parameters:
function MMI.fitted_params(::TSVDTransformer, fitresult)
    singular_values = fitresult.singular_values
    components = fitresult.components
    return (singular_values = singular_values, components = components)
end

function MMI.transform(::TSVDTransformer, fitresult, X)
    Xtransformed = X * fitresult.components

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
