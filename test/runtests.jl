using MLJTSVDInterface # substitute for correct interface pkg name
using Test
using TSVD
using MLJBase
using SparseArrays
using StableRNGs # for RNGs stable across all julia versions
rng = StableRNGs.StableRNG(123)

@testset "tsvd transformer" begin
    n = 10
    p = 20
    prob_nonzero = 0.5

    # test with a sparse matrix
    X_sparse = sprand(rng, n, p, prob_nonzero)

    # use defaults - transform into an n x 2 dense matrix
    model = MLJTSVDInterface.TSVDTransformer()

    mach = machine(model, X_sparse)
    fit!(mach)
    X_transformed = transform(mach, X_sparse)

    # also do the raw transformation with TSVD library
    U, s, V = tsvd(X_sparse, 2)

    @test size(X_transformed) == (10, 2)
    @test isapprox(s, fitted_params(mach).singular_values)
    @test isapprox(abs.(V), abs.(fitted_params(mach).components))

    # test with a dense matrix
    X_dense = rand(rng, n, p)

    mach = machine(model, X_dense)
    fit!(mach)
    X_transformed = transform(mach, X_dense)

    # also do the raw transformation with TSVD library
    U, s, V = tsvd(X_dense, 2)

    @test size(X_transformed) == (10, 2)
    @test isapprox(s, fitted_params(mach).singular_values)
    @test isapprox(abs.(V), abs.(fitted_params(mach).components))
end
