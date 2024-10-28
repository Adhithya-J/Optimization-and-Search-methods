import math
import pandas as pd
import numpy as np

EPSILON = 1e-6

def conjugate_gradient_descent(func, grad_func, x_init, n_iterations=10, tol=1e-6):
    """
    Conjugate gradient descent implementation with line search for alpha
    and Fletcher-Reeves formula for beta calculation.
    """
    x = np.array(x_init, dtype=float)
    data = []  # Store iteration details
    
    # Initialize gradient and direction
    grad = np.array(grad_func(*x))
    direction = -grad  # Initial direction is negative gradient
    
    def line_search_alpha(x, direction, max_iter=10, tol=1e-6):
        """Find optimal alpha using line search."""
        alpha = 0.0
        beta = 1.0
        tau = 0.5  # reduction factor
        c = 0.5    # sufficient decrease parameter
        
        initial_value = func(*x)
        grad_dot_direction = np.dot(grad_func(*x), direction)
        
        # Backtracking line search
        for _ in range(max_iter):
            new_x = x + beta * direction
            new_value = func(*new_x)
            
            if new_value <= initial_value + c * beta * grad_dot_direction:
                alpha = beta
                break
            
            beta *= tau
        
        if alpha <= 0:
            return 0.1
        if alpha > 1:
            return 0.999
            
        return alpha

    for i in range(1, n_iterations + 1):
        y = func(*x)  # Current function value
        
        # Calculate optimal step size alpha using line search
        alpha = line_search_alpha(x, direction)
        
        # Update parameters with calculated alpha
        x_new = x + alpha * direction
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
    return x1*x2*(2-x1*x2)/(x1+x2)

def gradient(x1, x2):
    """Gradient: Partial derivatives of the objective function"""
    df_dx1 = -(x2**2 * (-2 + x1**2 + 2 * x1*x2))/(x1 + x2)**2
    df_dx2 = -(x1**2 * (-2 + x2**2 + 2 * x1*x2))/(x1 + x2)**2
    return [df_dx1, df_dx2]

# Run the conjugate gradient descent
conjugate_gradient_descent(objective_function, gradient, x_init=[0.9, 0.6], n_iterations=20)