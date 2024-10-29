import math
import numpy as np
import pandas as pd

EPSILON = 1e-8  # Small value to prevent division by zero

def conjugate_gradient_descent(func, grad_func, x_init, n_iterations=20, tol=1e-6):
    """
    Conjugate gradient descent with alpha calculated using Newton-Raphson method.
    """
    x = np.array(x_init, dtype=float)
    data = []  # Store iteration details

    # Initialize gradient and direction
    grad = np.array(grad_func(*x))
    direction = -grad  # Initial direction is the negative gradient

    def newton_raphson_alpha(x, direction, max_iter=20, tol=1e-6):
        """Find optimal alpha using Newton-Raphson method."""
        alpha = 0.8  # Initial guess for alpha
        for _ in range(max_iter):
            grad_value = grad_func(*(x + alpha * direction))
            grad_dot_direction = np.dot(grad_value, direction)

            # Second derivative (approximate) g''(alpha)
            grad_new = np.array(grad_func(*(x + (alpha + EPSILON) * direction)))
            grad_dot_grad_new = np.dot(grad_new, direction)

            if abs(grad_dot_direction) < tol:
                break  # Converged to optimal alpha

            # Update alpha using Newton-Raphson step
            alpha -= grad_dot_direction / (grad_dot_grad_new + EPSILON)
        if alpha > 1:
            alpha = 0.99
        
        return max(alpha, 0.3)  # Ensure alpha is non-negative

    for i in range(1, n_iterations + 1):
        y = func(*x)  # Current function value

        # Calculate optimal step size alpha using Newton-Raphson
        alpha = newton_raphson_alpha(x, direction)

        # Update parameters
        x_new = x - alpha * direction
        y_new = func(*x_new)

        # Calculate new gradient
        grad_new = np.array(grad_func(*x_new))

        # Calculate beta using Fletcher-Reeves formula
        beta = np.dot(grad_new, grad_new) / (np.dot(grad, grad) + EPSILON)

        # Calculate new conjugate direction
        direction_new = -grad_new + beta * direction

        # Calculate error term
        error_term = abs(y_new - y)

        # Store iteration details
        data.append([
            i, x[0], x[1], y, grad[0], grad[1], 
            x_new[0], x_new[1], y_new, error_term, alpha, beta
        ])

        # Update for next iteration
        x = x_new
        grad = grad_new
        direction = direction_new

        # Stop if the error is below tolerance
        if error_term < tol:
            break

    # Display results in a DataFrame
    columns = [
        'Iteration', 'x1', 'x2', 'Y', 'd(Y)/d(x1)', 'd(Y)/d(x2)',
        'x1_new', 'x2_new', 'Y_new', 'Error Term', 'Alpha', 'Beta'
    ]
    df = pd.DataFrame(data, columns=columns)
    print(df)

# Test functions remain the same
def objective_function(x1, x2):
    """Objective function: f(x1, x2) = x1*x2*(2-x1*x2)/(x1+x2)"""
    return x1 * x2 * (2 - x1 * x2) / (x1 + x2)

def gradient(x1, x2):
    """Gradient: Partial derivatives of the objective function."""
    df_dx1 = -(x2**2 * (-2 + x1**2 + 2 * x1 * x2)) / (x1 + x2)**2
    df_dx2 = -(x1**2 * (-2 + x2**2 + 2 * x1 * x2)) / (x1 + x2)**2
    return [df_dx1, df_dx2]

# Run the conjugate gradient descent
conjugate_gradient_descent(objective_function, gradient, x_init=[0.9, 0.6], n_iterations=20)
