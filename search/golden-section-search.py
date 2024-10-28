import math
import pandas as pd
import numpy as np

def golden_section_search(func, a, b, n_iterations, tol=1e-5):
    """
    Implementation of Golden Section Search optimization algorithm.
    
    Args:
        func: Function to optimize
        a: Lower bound of search interval
        b: Upper bound of search interval
        n_iterations: Maximum number of iterations
        tol: Tolerance for convergence
        
    Returns:
        DataFrame containing iteration details
    """
    # Golden ratio
    golden_ratio = (1 + math.sqrt(5)) / 2
    golden_ratio_inv = 1 / golden_ratio
    
    # Initialize search points
    x1 = b - (b - a) * golden_ratio_inv
    x2 = a + (b - a) * golden_ratio_inv
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
            x2 = a + (b - a) * golden_ratio_inv
            f1, f2 = f2, func(x2)
        else:
            b = x2
            x2 = x1
            x1 = b - (b - a) * golden_ratio_inv
            f1, f2 = func(x1), f1
        
        # Stop if the interval is sufficiently small
        if abs(b - a) < tol:
            break
    
    # Create a DataFrame for the results
    df = pd.DataFrame(data, columns=['Iteration', 'Reduction Ratio', 'Lower Limit', 'Upper Limit', 
                                   'Function Value', 'Updated Function Value', 'Error Term'])
    print(df)
    
    # Return the best point found
    return x1 if f1 < f2 else x2

# Test function
def test_func(x):
    return 500 + 7.3*x + 0.0034*x**2 + 8*(850-x) + 0.0019*(850-x)**2 #2*math.pi*math.pow(x,2) + 8.2/x

# Run the search on the test function
if __name__ == "__main__":
    result = golden_section_search(test_func, a=300, b=400, n_iterations=12)
    print(f"\nOptimal x value found: {result}")
    print(f"Optimal function value: {test_func(result)}")