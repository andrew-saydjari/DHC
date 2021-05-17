## Preloads
module eqws

    using FFTW
    using LinearAlgebra
    using SparseArrays
    using Statistics
    using Test
    using DSP
    using Interpolations
    using StatsBase

    import CUDA

    export
        fink_filter_bank,
        fink_filter_bank_p,
        fink_filter_list,
        fink_filter_hash,
        fink_filter_hash_p,
        fink_filter_bank_3dizer,
        wst_S1_deriv,
        wst_S20_deriv,
        wst_S20_deriv_sum,
        eqws_compute,
        eqws_compute_p,
        eqws_compute_3d,
        eqws_compute_RGB,
        eqws_compute_wrapper,
        eqws_compute_convmap,
        apodizer,
        wind_2d,
        transformMaker,
        S1_iso_matrix,
        S2_iso_matrix,
        S1_equiv_matrix,
        S2_equiv_matrix,
        S1_iso_matrix3d,
        S2_iso_matrix3d,
        pre_subdivide,
        post_subdivide,
        sub_strip,
        renorm

    include("compute.jl")
    include("filter_bank.jl")
    include("derivatives.jl")
    include("post_process.jl")
    include("pre_process.jl")

end # of module
