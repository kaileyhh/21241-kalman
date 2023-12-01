using LinearAlgebra
using Plots

# prediction step
function predict(mu, P, A, Q)
    # update the mean state estimate based on the transition matrix: X = AX
    mu = *(A, mu)

    # update covariance based on the transition matrix: P = APA^T
    P = *(A, *(P, transpose(A))) + Q

    # return mean state, mean covariance
    return (mu, P)
end

# update step
function update(mu, P, X, H, R)
    # update the covariance of the predictive mean
    S = R + *(H, *(P, transpose(H)))

    # calculate Kalman gain
    K = *(*(P, transpose(H)), inv(S))
    
    # update the new expected sensor reading
    mu = mu + K * (X - *(H, mu))

    # update mean state covariant matrix
    P = P - *(K, *(H, P))

    # return mean state, mean covariance
    return (mu, P)
end

function run_kalman()

    # initializing variables
    dt = 0.1 # time step
    x = [rand(); rand()] # original state
    H = [1.0 0.0; 0.0 1.0] # measurement matrix
    P = [5.0 0; 0 5.0] # mean state covariance
    A = [1 dt; 0 1] # transition matrix
    Q = [0.1 0; 0 0.3] # noise

    R = *(H, *(P, transpose(H)))
    mu = *(H, x) # mean state vector

    x_arr = fill(0.0, 500) # true pos measurements for plotting
    y_arr = fill(0.0, 500) # true vel measurements for plotting

    
    pos_arr = fill(0.0, 500) # inits mu arr for filtered
    vel_arr = fill(0.0, 500) # inits vel arr for filitered

    time_arr = range(0, 50, 500)
    
    # loop over time to run Kalman filter
    for i in 1:500

        # predicts sparsely
        if (i % 15 == 0)
            (mu, P) = predict(mu, P, A, Q) # predicts
        end
        
        # updates with every new measurement
        (mu, P) = update(mu, P, x, H, R) # updates
        
        x_arr[i] = x[1] # fill in true position for comparison        
        y_arr[i] = x[2]  # fill in true vel for comparison
        pos_arr[i] = mu[1] # after predict/update, fill in mu_arr for plotting
        vel_arr[i] = mu[2] # after predict/update, fill in vel_arr for plotting
        
        x = [x[1] + 20*(rand() - 0.5); x[2] + 20*(rand() - 0.5)] # gets next random datapoint
        
    end

    # generate and save plots
    plot(time_arr, [x_arr, pos_arr], label=["true position" "filtered position"], dpi=1000)
    savefig("pos_plot.png") 
    plot(time_arr, [y_arr, vel_arr], label=["true velocity" "filtered velocity"], dpi= 1000)
    savefig("vel_plot.png")
    return
end

run_kalman()