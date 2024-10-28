import math
import pandas as pd
import numpy as np

def fibonacci_search(func, a, b, n_iterations, tol=1e-5):
    # Generate Fibonacci numbers up to the required number of iterations
    fib = [1, 1]
    for i in range(2, n_iterations + 1):
        fib.append(fib[-1] + fib[-2])

    # Initialize search points
    k = 0
    x1 = a + (fib[n_iterations - 2] / fib[n_iterations]) * (b - a)
    x2 = a + (fib[n_iterations - 1] / fib[n_iterations]) * (b - a)
    f1, f2 = func(x1), func(x2)

    # Prepare a table to store iteration results
    data = []

    for k in range(1, n_iterations + 1):
        # Store current iteration details
        error_term = abs(f2 - f1) if k > 1 else np.nan
        reduction_ratio = (b - a) / (x2 - x1)
        data.append([k, reduction_ratio, a, b, f1, f2, error_term])

        # Narrow down the search space
        if f1 > f2:
            a = x1
            x1 = x2
            x2 = a + (fib[n_iterations - k - 1] / fib[n_iterations - k]) * (b - a)
            f1, f2 = f2, func(x2)
        else:
            b = x2
            x2 = x1
            x1 = a + (fib[n_iterations - k - 2] / fib[n_iterations - k]) * (b - a)
            f1, f2 = func(x1), f1

        # Stop if the interval is sufficiently small
        if abs(b - a) < tol:
            break

    # Create a DataFrame for the results
    df = pd.DataFrame(data, columns=['Iteration', 'Reduction Ratio', 'Lower Limit', 'Upper Limit',
                                     'Function Value 1', 'Function Value 2', 'Error Term'])
    print(df)

# Test the Fibonacci search
def test_func(x):
    return 2*math.pi*math.pow(x,2) + 8.2/x #(x - 2) ** 2 + 3  # A simple quadratic function

# Run the search on the test function with an interval [0, 5]
fibonacci_search(test_func, a=0.5, b=3.5, n_iterations=7)
