import numpy as np
from sympy import integrate, Symbol

def integral_calculate(expression, x):

    result = integrate(expression, x)
    print(f"Result of indefinite integral: {result}")

    # Figure out how to do exceptions and write it out

    return result

def determine_expression(expr_string):
    expr_arr = np.array(list(expr_string))
    expression = ""
    x = Symbol('x')
    exp = []
    var_plus_coeff = []
    # Do some looping to determine how the expression should be formatted
    # Not correct, only works for coefficient * x, doesn't really work for exponents
    for i in range(expr_arr.shape[0]):
        if(i + 2 < expr_arr.shape[0]):
            if(expr_arr[i + 2] == 'x'):
                expression += expr_arr[i + 2] + "**" + expr_arr[i]
                i + 2
            elif expr_arr[i + 1] == 'x':
                expression += expr_arr[i] + "*" + expr_arr[i + 1]
                i + 1

    
    print(expr_arr)
    print(expr_arr.shape)

    print(expression)
    if expression != "":
        answer = integral_calculate(expression, x)


    return