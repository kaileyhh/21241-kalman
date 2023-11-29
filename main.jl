using LinearAlgebra

function predict(X, P, A)
    # update the mean state estimate based on the transition matrix: X = AX
    X = *(A, X)
    # update covariance based on the transition matrix: P = APA^T
    P = *(A, *(P, transpose(A)))
    return (X, P)
end

dt = 0.1 # timestep

# X: mean state estimate of previous step
X = [0.0; 0.0; 0.1; 0.1]

# P: mean state covariance of previous step
P = Diagonal([0.01, 0.01, 0.01, 0.01])

# A transition matrix to curr step
A = [1 0 dt 0; 0 1 0 dt; 0 0 1 0; 0 0 0 1]

# Y: new measurement

predict(X, P, A)