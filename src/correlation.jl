export autocor!, autocov!

check_lags(size_x::Int, τ::VecI{Int}) = (maximum(τ) < size_x || error("Lags must be less than the sample length."))

autodot(x::VecI{Tx}, T::Int, τ::Int) where Tx<:Real = dot(view(x, 1:(T-τ)), view(x, (τ+1):T))

#=
Compute the autocorrelation function (ACF) of a real-valued vector `x` with specifying the lags τ.
When `d` is provided, the mean of `x` would be subtracted from `x` before computing the ACF.
The output is normalized by the variance of `x`, i.e. so that the lag 0 autocorrelation is 1.
See `autocov!` for the unnormalized form.
=#

function autocor!(r::VecIO{Tr}, x::VecI{Tx}, τ::VecI{Int}, d::VecB{Tx}) where {Tr<:Real, Tx<:Real}
    size_x = length(x)
    size_x ≡ length(d) || throw(DimensionMismatch())
    size_τ = length(τ)
    size_τ ≡ length(r) || throw(DimensionMismatch())
    check_lags(size_x, τ)

    μ = mean(x)
    @simd for i in eachindex(d)
        @inbounds d[i] = x[i] - μ
    end
    σ = dot(d, d)
    @inbounds for i in 1:size_τ
        r[i] = autodot(d, size_x, abs(τ[i])) / σ
    end
    return r
end

function autocor!(r::VecIO{Tr}, x::VecI{Tx}, τ::VecI{Int}) where {Tr<:Real, Tx<:Real}
    size_x = length(x)
    size_τ = length(τ)
    size_τ ≡ length(r) || throw(DimensionMismatch())
    check_lags(size_x, τ)

    σ = dot(x, x)
    @inbounds for i in 1:size_τ
        r[i] = autodot(x, size_x, abs(τ[i])) / σ
    end
    return r
end

#=
Compute the autocovariance of a real-valued vector `x`, with specifying the lags τ.
When `d` is provided, the mean of `x` would be subtracted from `x` before computing the autocovariance.
The output is not normalized.
See `autocor!` for a function with normalization.
=#

function autocov!(r::VecIO{Tr}, x::VecI{Tx}, τ::VecI{Int}, d::VecB{Tx}) where {Tr<:Real, Tx<:Real}
    size_x = length(x)
    size_x ≡ length(d) || throw(DimensionMismatch())
    size_τ = length(τ)
    size_τ ≡ length(r) || throw(DimensionMismatch())
    check_lags(size_x, τ)

    μ = mean(x)
    @simd for i in eachindex(d)
        @inbounds d[i] = x[i] - μ
    end
    @inbounds for i in 1:size_τ
        r[i] = autodot(d, size_x, abs(τ[i])) / size_x
    end
    return r
end

function autocov!(r::VecIO{Tr}, x::VecI{Tx}, τ::VecI{Int}) where {Tr<:Real, Tx<:Real}
    size_x = length(x)
    size_τ = length(τ)
    size_τ ≡ length(r) || throw(DimensionMismatch())
    check_lags(size_x, τ)

    @inbounds for i in 1:size_τ
        r[i] = autodot(x, size_x, abs(τ[i])) / size_x
    end
    return r
end
