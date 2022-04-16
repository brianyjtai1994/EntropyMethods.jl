mean(x::VecI{T}) where T<:Real = sum(x) / length(x)

function dot(x::VecI{Tx}, y::VecI{Ty}, n::Int) where {Tx<:Real,Ty<:Real}
    r = 0.0
    m = mod(n, 5)
    if m ≠ 0
        @inbounds for i in 1:m
            r += x[i] * y[i]
        end
        n < 5 && return r
    end
    m += 1
    @inbounds for i in m:5:n
        r += x[i] * y[i] + x[i+1] * y[i+1] + x[i+2] * y[i+2] + x[i+3] * y[i+3] + x[i+4] * y[i+4]
    end
    return r
end

function dot(x::VecI{Tx}, y::VecI{Ty}) where {Tx<:Real,Ty<:Real}
    n = length(x)
    n == length(y) || error("dot: length(y) ≠ length(x) = $n.")
    return dot(x, y, n)
end
