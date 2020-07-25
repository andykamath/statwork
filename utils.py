import pandas as pd
import numpy as np
from sympy import *
from IPython.display import display, Math, Latex
import math

X = "x"
Y = "y"
XY = "xy"
YX = XY

printed = set()
def mprint(*args, printed=set()):
    toprint = ' '.join([str(a) for a in args])
    if toprint not in printed:
        display(Math(toprint))
        # printed.add(toprint)
    return printed

# Marginal Densities
def MD(var):
    def MY(df):
        mprint(f'MD({Y}) =', f'\sum_{Y}', 'p_{%s, %s}(a, %s)' % (X, Y, Y))
        mprint('')
        s = 0
        d = {X: [1]}
        for col in df.columns[1:]:
            d[col] = [df[col].sum()]
            mprint(f'{Y}_k = {col} \\to', ' + '.join([str(a) for a in df[col]]), '=', d[col])
        mprint('')
        tor = pd.DataFrame(d)
        mprint(f'MD({Y}) = ')
        display(tor)
        return tor
    
    def MX(df):
        mprint(f'MD({X}) =', f'\sum_{X}', 'p_{%s, %s}(a, %s)' % (X, Y, X))
        mprint('')
        s = 0
        d = {Y: [1]}
        for _, row in df.iterrows():
            col = row[X]
            d[col] = sum([row[i] for i in df.columns[1:]])
            mprint(f'{X}_k = {col} \\to', ' + '.join([str(a) for a in df.columns[1:]]), '=', d[col])
        tor = pd.DataFrame(d)
        mprint('')
        mprint(f'MD({X}) = ')
        display(tor)
        return tor
        
    varmap = {X: MX, Y: MY}
    return varmap[var]

# E(X)
def Exp(var):
    def EY(df):
        mprint(f'E({Y}) =', '\sum_{k=0}^n', f'k * p({Y} = k)')
        mprint('')
        s = 0
        sums = []
        for col in df.columns[1:]:
            sumval = col * df[col].sum()
            mprint(f'{Y}_k = {col} \\to', col, '*', '(' + ' + '.join([str(a) for a in df[col]]) + ')', '=', sumval)
            sums.append(col * df[col].sum())
        mprint('')
        tor = sum(sums)
        mprint(f'E({Y}) =', ' + '.join([str(a) for a in sums]), '= \\textbf{', tor, '}')
        return tor
    
    def EX(df):
        mprint(f'E({X}) =', '\sum_{k=0}^n', f'k * p({X} = k)')
        mprint('')
        s = 0
        sums = []
        for _, row in df.iterrows():
            sums.append(row[X] * sum([row[i] for i in df.columns[1:]]))
            mprint(f'{X}_k = {row[X]} \\to', row[X], "*", '('+' + '.join([str(row[i]) for i in df.columns[1:]])+')', "=", sums[-1])
        tor = sum(sums)
        mprint('')
        mprint(f'E({X}) =', ' + '.join([str(a) for a in sums]), '= \\textbf{', tor, '}')
        return tor
    
    def EXY(df):
        mprint(f'E({XY}) =', '\sum_{k=0}^n', f'k * p({XY} = k)')
        mprint('')
        sums = []
        for _, row in df.iterrows():
            for col in df.columns[1:]:
                prod = row[X] * col * row[col]
                mprint(f'{X}_k, {Y}_k = ({row[X]}, {col}) \\to', row[X], "*", col, "*", row[col], "=", prod)
                sums.append(prod)
        tor = sum(sums)
        mprint('')
        mprint(f'E({XY}) =', ' + '.join([str(a) for a in sums]), "= \\textbf{", tor, '}')
        return tor
        
    varmap = {X: EX, Y: EY, XY: EXY}
    return varmap[var]

# var(X)
def variance(var, func=None):
    def varY(df,):
        mprint(f"var({Y}) =", f"E[({Y} - \mu_{Y})^2]", "= \sum_{i=0}^n", f"({Y}_i - E({Y}))^2")
        mprint('')
        ey = Exp(Y)(df)
        mprint('')
        s = 0
        sums = []
        for col in df.columns[1:]:
            p = df[col].sum()
            n = int(col)
            sums.append(p * (n - ey)**2)
            mprint(f'{Y}_k = {n} \\to', f'{p}*({n} - {ey})^2 =', sums[-1])
        mprint('')
        tor = sum(sums)
        mprint(f"var({Y}) =", ' + '.join([str(a) for a in sums]), '= \\textbf{', tor, '}')
        return tor

    def varX(df):
        mprint(f"var({X}) =", f"E[({X} - \mu_{X})^2]", "= \sum_{i=0}^n", f"({X}_i - E({X}))^2")
        mprint('')
        ex = Exp(X)(df)
        mprint('')
        s = 0
        sums = []
        for _, row in df.iterrows():
            p = sum([row[i] for i in df.columns[1:]])
            n = int(row[X])
            sums.append(p * (n - ex)**2)
            mprint(f'{X}_k = {n} \\to', f'{p}*({n} - {ex})^2 =', sums[-1])
        tor = sum(sums)
        mprint('')
        mprint(f'var({X}) =', ' + '.join([str(a) for a in sums]), '= \\textbf{', tor, '}')
        return tor
    
    if func:
        coeff = func(1) - func(0)
        mprint('\\textbf{constants/intercepts get dropped}')
        mprint(f"var({coeff}{var}) = {coeff}^2*var({var}) = {coeff**2}*var({var})")
        def finish(df):
            v = variance(var)(df)
            tor = coeff**2 * v
            mprint(f"var({coeff}{var}) = {coeff**2}*{v} =", "\\textbf{%s}" % tor)
            return tor
        return finish
        
    varmap = {X: lambda df: varX(df), Y: lambda df: varY(df)}
    return varmap[var]


# std(Y)
def stdev(var, func=None):
    def stdX(df):
        mprint(f"\\sigma({X}) =", "\\sqrt{var(%s)}" % X)
        v = variance(X)(df)
        tor = math.sqrt(v)
        mprint(f"\\sigma({X}) =", "\\sqrt{%s}" % v, '=', tor)
        return tor

    def stdY(df):
        v = variance(Y)(df)
        tor = math.sqrt(v)
        mprint(f"\\sigma({Y}) =", "\\sqrt{%s}" % v, '=', tor)
        return tor
    
    if func:
        coeff = func(1) - func(0)
        mprint('\\textbf{constants/intercepts get dropped}')
        mprint(f"\\sigma({coeff}{var}) = |{coeff}|*var({var}) = {abs(coeff)}*\\sigma({var})")
        def finish(df):
            s = stdev(var)(df)
            print(s)
            tor = abs(coeff) * s
            mprint(f"\\sigma({coeff}{var}) = {abs(coeff)}*var({var}) =", "\\textbf{%s}" % tor)
            return tor
        return finish
    
    varmap = {X: stdX, Y: stdY}
    return varmap[var]


def cov(x, y, x_func=None, y_func=None):
    def covariance(df, x_coeff, y_coeff):
        exy = Exp(XY)(df)
        mprint('')
        ex = Exp(x)(df)
        mprint('')
        ey = Exp(y)(df)
        mprint('')
        tor = x_coeff*y_coeff*(exy - ex*ey)
        if y_coeff != 1 or x_coeff != 1:
            mprint(f"{x_coeff*y_coeff}cov({X}, {Y}) = {x_coeff*y_coeff} * ({exy} - {ex} * {ey}) =", "\\textbf{%s}" % tor)
        else:
            mprint(f"cov({x}, {y}) = {exy} - {ex} * {ey} =", "\\textbf{%s}" % tor)
        return tor
    
    if x == y:
        mprint(f"cov({x}, {y}) = var({x})")
        if y_func is not None:
            return variance(x, y_func)
        return variance(x, x_func)
    # DOES NOT WORK WHEN X OR Y ARE DIFFERENT DISTRIBUTIONS
    # this just gets the coefficient if you pass in, say lambda x: 2*x + 5, the 5 gets removed and the 2 stays
    x_coeff = 1
    y_coeff = 1
    if x_func:
        x_coeff *= (x_func(1) - x_func(0))
    if y_func:
        y_coeff *= (y_func(1) - y_func(0))
    if x_coeff != 1 and y_coeff != 1:
        mprint("\\textbf{constants get dropped}")
        mprint(f"cov({x_coeff}{x}, {y_coeff}{y}) = {x_coeff}*{y_coeff}*cov({x}, {y}) = {x_coeff*y_coeff}(E({x}{y}) - E({x})E({y}))")
    elif x_coeff != 1:
        mprint("\\textbf{constants get dropped}")
        mprint(f"cov({x_coeff}{x}, {y}) = {x_coeff}*cov({x}, {y}) = {x_coeff*y_coeff}(E({x}{y}) - E({x})E({y}))")
    elif y_coeff != 1:
        mprint("\\textbf{constants get dropped}")
        mprint(f"cov({x}, {y_coeff}{y}) = {y_coeff}*cov({x}, {y}) = {x_coeff*y_coeff}(E({x}{y}) - E({x})E({y}))")
    else:
        mprint(f"cov({x}, {y}) = (E({x}{y}) - E({x})E({y}))")
        
    mprint('')
    return lambda df: x_coeff * y_coeff * covariance(df, x_coeff, y_coeff)
    
def corr(x=X, y=Y):
    def correlation(df):
        mprint(f"corr({X}, {Y}) =", "\\frac{cov(%s, %s)}{\\sigma(%s)\\sigma(%s)}" % (X, Y, X, Y))
        covXY = cov(x, y)(df)
        sX = stdev(x)(df)
        sY = stdev(y)(df)
        tor = covXY/(sX*sY)
        mprint(f"corr({X}, {Y}) =", "\\frac{%s}{%s * %s}" % (covXY, sX, sY), "= \\textbf{%s}" % tor)
        return tor
        
    
    if callable(x) or callable(y):
        mprint("Coefficients and constants have no effect on correlation")
    return correlation