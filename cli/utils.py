import re

float_patern    = r"^-?\d+\.\d+$"
int_patern      = r"^-?\d+$"

def isfloat(x):
    return  (re.match(float_patern, x) != None) \
            or                                  \
            (re.match(int_patern, x) != None)

def isinteger(x):
    return re.match(int_patern, x) != None