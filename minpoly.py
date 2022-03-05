import sympy as sp
import numpy as np

x = sp.symbols('x')

def subs_matrix(A, factors, powers):
    """Substitute matrix A into factors raised to powers and multiply result.
    
    Parameters
    ----------
    A : sympy.Matrix
        Matrix to substitute

    factors : list[sympy.expr]
        List of sympy expressions, e.g [x - 1, x**2 - 2*x, x - 3]

    powers: list[int]
        List of powers to raise the substituted factors to. e.g [1, 2, 3]    
    
    Returns
    -------
    Returns : sympy.Matrix
        E.g if factors = [x - 1, x**2 + 2] and powers = [3, 5],
        then return (A - 1*I)**3 * (A**2 + 2*I)**5
    """
    
    n, m = A.shape
    if n != m:
        raise ValueError("Dimension error, A should be a square matrix!")
    elif len(factors) != len(powers):
        raise ValueError("Dimension error, len(factors) should be equal to len(powers)!")
    else:
        subs_product = sp.Identity(n)  # Initialize the substituted product of factors
        for j, factor in enumerate(factors):
            terms = sp.PurePoly(factor).terms() 
            # e.g if factor is 3*x**2 - 1, then term_powers == [2, 0]
            terms_powers = [term[0][0] for term in terms]
            # e.g if factor is 3*x**2 - 1, then term_coeff == [3, -1]
            terms_coeffs = [term[1] for term in terms]
            subs_factor = sp.zeros(n, n)  # Initialize substituted factor, i.e f(A)
            for i in range(len(terms_coeffs)):
                subs_factor += terms_coeffs[i] * (A**terms_powers[i])

            subs_product *= (subs_factor)**powers[j]
    
    return subs_product


def generate_powers(alg_mult):
    """Return list of lists of all possible powers.
    
    Parameters
    ----------
    alg_mult : list
        The algebraic multiplicites of the power, i.e the maximum power
    
    Returns
    -------
    Returns : list
        List of lists of all possible combinations of powers, where each
        power has a min value of 1 and a max value of its corresponding alg_mult
        entry.
    """
    num_combinations = 1  # The total number of different combinations
    for power in alg_mult: num_combinations *= power
    possible_powers = [list(range(1, alg_mult[i]+1)) for i in range(len(alg_mult))]
    power_combinations = []
    for i in range(num_combinations):
        combination = []
        for j in range(len(possible_powers)):
            combination.append(possible_powers[j][i % len(possible_powers[j])])

        power_combinations.append(combination)
    
    # Sort the powers by length of array
    power_combinations.sort(key=lambda comb: np.linalg.norm(comb))
    return power_combinations
    

def get_min_poly(A):
    """Return the minimal polynomial for the matrix A."""
    n, m = A.shape  # Get dimensions of the matrix
    if n != m:
        raise ValueError("Dimension Error: A must be a square matrix!")
    else:
        char_poly = A.charpoly(x).as_expr()

        factor_list = sp.factor_list(char_poly)[1]
        factors = [factor_pair[0] for factor_pair in factor_list]
        # Algebraic multiplicities of each factor
        alg_mult = [factor_pair[1] for factor_pair in factor_list]
        power_combinations = generate_powers(alg_mult)
        min_poly_powers = []
        for power_list in power_combinations:
            # If we find a power combination that results in the zero matrix
            if subs_matrix(A, factors, power_list) == sp.zeros(n, n):
                min_poly_powers = power_list
                break
        if not min_poly_powers:
            raise ValueError("Minimal Polynomial could not be found!")

        min_poly = 1 
        for i in range(len(factors)):
            min_poly *= factors[i]**min_poly_powers[i]

        return min_poly

