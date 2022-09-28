module MLJTSVDInterface

import TSVD
import MLJModelInterface
using Random: MersenneTwister, AbstractRNG, GLOBAL_RNG

const PKG = "TSVD"
const MMI = MLJModelInterface

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
    rng::Union{Int, AbstractRNG} = GLOBAL_RNG
end

struct TSVDTransformerResult
    singular_values::Vector{Float64}
    components::Matrix{Float64}
    is_table::Bool
end

as_matrix(X) = MMI.matrix(X)
as_matrix(X::AbstractArray) = X

_get_rng(rng::Int) = MersenneTwister(rng)
_get_rng(rng) = rng

function MMI.fit(transformer::TSVDTransformer, verbosity, Xuser)
    X = as_matrix(Xuser)
    rng = _get_rng(transformer.rng)

    U, s, V = TSVD.tsvd(
        X,
        transformer.nvals;
        maxiter=transformer.maxiter,
        initvec = convert(Vector{float(eltype(X))}, randn(rng, size(X,1)))
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
               input_scitype = Union{MMI.Table(MMI.Continuous),AbstractMatrix{MMI.Continuous}},
               output_scitype = Union{MMI.Table(MMI.Continuous),AbstractMatrix{MMI.Continuous}},
               docstring = "Truncated SVD dimensionality reduction",         # brief description
               path = "MLJTSVDInterface.TSVDTransformer"
               )

"""
$(MMI.doc_header(TSVDTransformer))
`TSVDTransformer`: Dimensionality reduction using truncated SVD. This transformer performs
linear dimensionality reduction by means of truncated singular value decomposition (SVD).
Contrary to PCA, this estimator does not center the data before computing the singular value
decomposition. This means it can work with sparse matrices efficiently.


# Training data
In MLJ or MLJBase, bind an instance `model` to data with
    mach = machine(model, X, y)
Here:

- `X` is any table of input features (eg, a `DataFrame`) whose columns
  are of scitype `Continuous`; check the column scitypes with `schema(X)`

Train the machine using `fit!(mach, rows=...)`.

# Operations

- `transform(mach, Xnew)`: return compressed representation of the target given new features `Xnew`, which
  should have the same scitype as `X` above.

# Hyper-parameters

- `nvals=2`: The number of singular values and vectors to compute.
- `maxiter=1000`: The maximum number if iterations to use. Defaults to 1000, however likely
  will finish before this
- `rng`: The random number generator to use, either an `Int` or an `AbstractRNG`.

# Fitted parameters

The fields of `fitted_params(mach)` are:

- `singular_values`: The estimated singular values, stored as a vector.
- `components`: The estimated component vectors, stored as a matrix.
- `is_table`: Whether or not the input data is a table.

# Examples

```julia
using MLJ
using Test
SVD = @load TSVDTransformer pkg=TSVD
X, y = @load_iris
svd = SVD(nvals=3)
mach = machine(svd, X) |> fit!
(; singular_values, components) =  fitted_params(mach)
preds = transform(mach, X)

to_matrix(x) = hcat(values(x)...)

@test sum(round.((to_matrix(preds) * components') - to_matrix(X))) == 0
```
"""
TSVDTransformer

end # module
