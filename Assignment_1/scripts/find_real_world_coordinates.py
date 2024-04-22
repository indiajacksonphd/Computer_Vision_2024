import numpy as np
from scipy.optimize import least_squares

# Define camera calibration matrix K
K = np.array([[1476.8, 0, 987.9861],
              [0, 1479.3, 581.2310],
              [0, 0, 1]])

# Define distortion coefficients D
D = np.array([0.13783006, -0.96596213, 0.01385186, 0.00421161, 2.75232267])

# Define rotation matrix R and translation vector t
R = np.array([[0.38568274, -0.42523707, -0.81879317],
              [-0.27314195, 0.79504759, -0.54156514],
              [0.88127311, 0.43251909, 0.19048606]])

t = np.array([[0.55507626],
              [-0.83179141],
              [-0.00366016]])

# Observed pixel coordinates (u, v)
u_observed = 100
v_observed = 200

# Define the objective function to minimize the reprojection error
def objective_function(X, u_observed, v_observed, K, R, t):
    X_homogeneous = np.concatenate([X, np.ones((1, X.shape[1]))], axis=0)
    P = np.dot(K, np.concatenate([R, t], axis=1))
    projected = np.dot(P, X_homogeneous)
    projected /= projected[2, :]
    reprojection_error = np.sqrt((projected[0, :] - u_observed)**2 + (projected[1, :] - v_observed)**2)
    return reprojection_error

# Initial guess for 3D world coordinates (X, Y, Z)
X_initial_guess = np.array([[0.0], [0.0], [1.0]])

# Use least squares optimization to minimize the reprojection error
result = least_squares(objective_function, X_initial_guess, args=(u_observed, v_observed, K, R, t))

# Extract the optimized 3D world coordinates
X_optimized = result.x

print("Optimized 3D world coordinates:", X_optimized)
