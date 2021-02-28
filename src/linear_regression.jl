export
    LinearRegressionModel,
    fit!,
    predict

mutable struct LinearRegressionModel{T,S}
    y::Vector{T}
    X::AbstractArray{S}
    argv::Vector{Float64}
    N::Integer
end

function LinearRegressionModel(df::DataFrame, label::Symbol, features::Vector{Symbol})
    N = nrow(df)
    y = df[!, label]
    X = Matrix(df[!, features])
    β = rand(length(features)+1)
    return LinearRegressionModel(y, X, β, N)
end

function LinearRegressionModel(df::DataFrame, label::Symbol, feature::Symbol)
    N = nrow(df)
    y = df[!, label]
    X = df[!, feature]
    β = rand(2)
    return LinearRegressionModel(y, X, β, N)
end

function predict(model::LinearRegressionModel, xs::Vector{T}) where {T<:Real}
    X = hcat(ones(T, size(xs, 1)), xs)

    return X * model.argv
end

function predict(model::LinearRegressionModel, X::Matrix{T}) where {T<:Real}
    X = hcat(ones(T, size(X, 1)), X)

    return X * model.argv
end

residual(model::LinearRegressionModel, i::Integer) = model.y[i] - predict(model, model.X[i,:])

residual(model::LinearRegressionModel) = model.y .- predict(model, model.X)

function loss(model::LinearRegressionModel)
    l = sum(x -> 0.5 * x^2, residual(model))

    return l/model.N
end

function ∇L(model::LinearRegressionModel, i)
    X = hcat(ones(size(model.X[i,:], 1)), model.X[i,:])
    return vec(sum(-residual(model) .* X, dims=1))
end

function ∇L(model::LinearRegressionModel)
    X = hcat(ones(size(model.X, 1)), model.X)
    return vec(sum(-residual(model) .* X, dims=1))
end

function fit!(model::LinearRegressionModel; method=gradient_descent, η::Real=1e-4, atol::Real=1e-6, show=false)
    β = method(model, η, atol, show)
    model.argv .= β
    return model
end

function gradient_descent(model::LinearRegressionModel, η::Real=1e-4, atol::Real=1e-6, show::Bool=false)
    β = model.argv
    while (l = loss(model)) > atol
        show && println("Loss: $l")
        β .-= η .* ∇L(model)
    end
    return β
end
