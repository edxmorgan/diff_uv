from casadi import skew

def cross_product(v):
    S = skew(v)
    return S
