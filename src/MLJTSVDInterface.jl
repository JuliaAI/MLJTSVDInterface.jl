module MLJTSVDInterface

import TSVD
import MLJModelInterface
using Random: MersenneTwister, AbstractRNG, GLOBAL_RNG

const PKG = "TSVD"
const MMI = MLJModelInterface

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

MMI.metadata_pkg(
    TSVDTransformer,
    name="$PKG",
    uuid="9449cd9e-2762-5aa3-a617-5413e99d722e",
    url="https://github.com/JuliaLinearAlgebra/TSVD.jl",
    is_pure_julia=true,
    license="MIT",
    is_wrapper=false
)

MMI.metadata_model(
    TSVDTransformer,
    input_scitype = Union{MMI.Table(MMI.Continuous),AbstractMatrix{MMI.Continuous}},
    output_scitype = Union{MMI.Table(MMI.Continuous),AbstractMatrix{MMI.Continuous}},
    human_name = "truncated SVD transformer",
    docstring = "Truncated SVD dimensionality reduction",         # brief description
    path = "MLJTSVDInterface.TSVDTransformer"
)

"""
$(MMI.doc_header(TSVDTransformer))

This model performs linear dimension reduction. It differs from regular principal
component analysis in that data is not centered, so that sparsity, if present, can be
preserved during the computation. Text analysis is a common application.

The truncated SVD is computed by Lanczos bidiagonalization. The Lanczos vectors are
partially orthogonalized as described in R. M. Larsen, *Lanczos bidiagonalization with
partial reorthogonalization*, Department of Computer Science, Aarhus University, Technical
report, DAIMI PB-357, September 1998.

# Training data

In MLJ or MLJBase, bind an instance `model` to data with

    mach = machine(model, X, y)

Here:

- `X` is any table of input features (eg, a `DataFrame`) whose columns are of scitype
  `Continuous`; check the column scitypes with `schema(X)`; alternatively, `X` is any
  `AbstractMatrix` with `Continuous` elements; check the scitype with `scitype(X)`.

Train the machine using `fit!(mach, rows=...)`.

# Operations

- `transform(mach, Xnew)`: transform (project) observations in `Xnew` into their
  lower-dimensional representations; `Xnew` should have the same scitype as `X`
  above, and the object returned is a table or matrix according to the type of `X`.

# Hyper-parameters

- `nvals=2`: The output dimension (number of singular values)

- `maxiter=1000`: The maximum number if iterations.

- `rng=Random.GLOBAL_RNG`: The random number generator to use, either an `Int` seed, or an
  `AbstractRNG`.

# Fitted parameters

The fields of `fitted_params(mach)` are:

- `singular_values`: The estimated singular values, stored as a vector.

- `components`: The estimated component vectors, stored as a matrix.

- `is_table`: Whether or not `transform` returns a table or matrix.

# Examples

With tabular input:

```julia
using MLJ

SVD = @load TSVDTransformer pkg=TSVD
X, _ = @load_iris # `X`, a table
svd = SVD(nvals=3)
mach = machine(svd, X) |> fit!
(; singular_values, components) =  fitted_params(mach)
Xsmall = transform(mach, X) # a table

to_matrix(x) = hcat(values(x)...)

@assert sum(round.((to_matrix(Xsmall) * components') - to_matrix(X))) == 0
```

With sparse matrix input:

```julia
using MLJ
using SparseArrays

SVD = @load TSVDTransformer pkg=TSVD

# sparse matrix with 10 rows (observations):
I = rand(1:10, 100)
J = rand(1:10^6, 100)
K = rand(100)
X = sparse(I, J, K, 10, 10^6)

svd = SVD(nvals=4)
mach = machine(svd, X) |> fit!
Xsmall = transform(mach, X) # matrix with 10 rows but only 4 columns
```

"""
TSVDTransformer

end # module
