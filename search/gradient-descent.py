import math
import pandas as pd
import numpy as np

EPSILON = 1e-6

def gradient_descent(func, grad_func, x_init, n_iterations=10, tol=1e-6):
    """
    Gradient descent where alpha is found using Newton-Raphson method on the equation of Y(alpha).
    """
    x = np.array(x_init, dtype=float)
    data = []  # Store iteration details

    def newton_raphson_alpha(grad, max_iter=20, tol=1e-6):
        """Find optimal alpha using Newton-Raphson method."""
        alpha = 0.8  # Initial guess for alpha
        for _ in range(max_iter):
            # First derivative g'(alpha) = grad . grad
            grad_dot_grad = np.dot(grad, grad)
            # Second derivative g''(alpha) = 0 in this quadratic case (since it's linear in alpha)
            grad_norm = np.linalg.norm(grad)**2  # Equivalent to g'(alpha)
            if grad_norm < tol:
                break
            # Update alpha using Newton-Raphson step
            alpha = alpha - grad_dot_grad / grad_norm

        if alpha > 1:
            return 0.999
        if alpha <= 0:
            return 0.1

        return alpha

    for i in range(1, n_iterations + 1):
        y = func(*x)  # Current function value
        grad = np.array(grad_func(*x))  # Gradient

        # Find optimal alpha using Newton-Raphson
        alpha = newton_raphson_alpha(grad)

        # Update parameters with calculated alpha
        x_new = x + alpha * grad
        y_new = func(*x_new)

        # Calculate error term
        error_term = abs(y_new - y)

        # Store iteration details
        data.append([
            i, x[0], x[1], y, alpha ,grad[0], grad[1], x_new[0], x_new[1], y_new, error_term
        ])

        # Update parameters for next iteration
        x = x_new

        # Stop if the error is below tolerance
        if error_term < tol:
            break

    # Display results in a DataFrame
    columns = [
        'Iteration', 'x1', 'x2', 'Y', 'alpha','d(Y)/d(x1)', 'd(Y)/d(x2)',
        'x1_new', 'x2_new', 'Y_new', 'Error Term (Y_new - Y)'
    ]
    df = pd.DataFrame(data, columns=columns)
    print(df)

# Test the gradient descent with a simple quadratic function
def objective_function(x1, x2):
    """Objective function: f(x1, x2) = (x1 - 2)^2 + (x2 - 3)^2"""
    return  x1*x2*(2-x1*x2)/(x1+x2) #10 + (x1** 2/2) + (2/(x1*x2)) + (6*x2)

def gradient(x1, x2):
    """Gradient: Partial derivatives of the objective function"""
    df_dx1 = -(x2**2 * (-2 + x1**2 + 2 * x1*x2))/(x1 + x2)**2 #x1 - (2/ (x1**2 * x2 + EPSILON))
    df_dx2 = -(x1**2 * (-2 + x2**2 + 2 * x1*x2))/(x1 + x2)**2 #6 - (2/(x1* x2**2 + EPSILON))
    return [df_dx1, df_dx2]

# Run the gradient descent with initial parameters [0, 0]
gradient_descent(objective_function, gradient, x_init=[0.9, 0.6], n_iterations=20)
