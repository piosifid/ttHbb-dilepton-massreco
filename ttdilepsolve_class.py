import math

class ttdilepsolve:
    def __init__(self):
        self.epsilon = 1.e-6  # numerical precision

    def solve(self, ETmiss, b, bb, lp, lm, mWp, mWm, mt, mtb, pnux, pnuy, pnuz, pnubx, pnuby, pnubz, cd_diff, cubic_single_root_cmplx):
        pass

    def quartic(self, poly, pnuy, cubic_single_root_cmplx):
        pass

    def cubic(self, poly, pnuy):
        pass

    def quadratic(self, poly, pnuy):
        pass

    def algebraic_pz(self, b, lp, mWp, mt, mb, mlp, pnux, pnuy, pnuz):
        pass

    def evalterm1(self, a1, pnux, pnuy):
        return a1[0] + a1[1] * pnux + a1[2] * pnuy

    def evalterm2(self, a2, pnux, pnuy):
        return a2[0] + a2[1] * pnux + a2[2] * pnuy + a2[3] * pnux**2 + a2[4] * pnux * pnuy + a2[5] * pnuy**2

def sqr(x):
    return x * x

def sign(a):
    return -1 if a < 0 else 1 if a > 0 else 0

def quad(x):
    return x**4
