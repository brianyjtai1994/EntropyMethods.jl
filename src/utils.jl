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

# Symmetric real-valued Toeplitz matrix solver
function levinson_durbin(r::VecI{T}) where T<:Real
    N = length(r)
    f = zeros(T, N) # forward vector
    b = zeros(T, N) # backward vector

    @inbounds f[1] = b[N] = inv(r[1]) # initialization
    
    for n in 2:N
        # ϵf = dot(view(r, n:-1:1), view(f, 1:n))
        # ϵb = dot(view(r, 1:n), view(b, N-n+1:N))

        m = mod(n, 5)
        j = n + 1 # start-index of forward vector
        k = N - n # start-index of backward vector

        ϵf = ϵb = 0.0
        if m ≠ 0
            @inbounds for i in 1:m
                ϵf += r[j-i] * f[i]
                ϵb += r[i] * b[k+i]
            end
        end

        if n ≥ 5
            m += 1
            @inbounds for i in m:5:n
                jj, kk = j - i, k + i
                ϵf += r[jj] * f[i] + r[jj-1] * f[i+1] + r[jj-2] * f[i+2] + r[jj-3] * f[i+3] + r[jj-4] * f[i+4]
                ϵb += r[i] * b[kk] + r[i+1] * b[kk+1] + r[i+2] * b[kk+2] + r[i+3] * b[kk+3] + r[i+4] * b[kk+4]
            end
        end

        de = 1.0 - ϵf * ϵb
        αf = βb = inv(de)
        βf = -ϵf / de
        αb = -ϵb / de

        @inbounds for i in 1:n
            fi, bi = f[i], b[k+i]
            f[i]   = αf * fi + βf * bi
            b[k+i] = αb * fi + βb * bi
        end
    end
    return f, b
end
