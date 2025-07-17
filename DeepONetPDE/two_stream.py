import numpy as np

# Part 1 - Approximation
def qdr_olsonkunasz(I, src_prev, src_curr, src_next, dtau_prev, dtau_next, order):
    xp = np.exp(-dtau_prev) # Precalculate e^(-delta Tau)
    if order==1:
        u = 0.5 * (1 - xp)
        v = 0.5 * (1 - xp)
        w = 0.0
    elif order==2:
        u = (1 - (1 + dtau_prev) * xp) / dtau_prev
        v = (dtau_prev - 1 + xp) / dtau_prev
        w = 0.0
    elif order==3:
        e0 = 1 - xp
        e1 = dtau_prev - e0
        e2 = dtau_prev**2 - 2 * e1
        u = e0 + (e2 - (2 * dtau_prev + dtau_next) * e1) / (dtau_prev * (dtau_prev+dtau_next))
        v  = ((dtau_prev + dtau_next) * e1 - e2) / (dtau_prev * dtau_next)
        w  = (e2 - dtau_prev * e1) / (dtau_next * (dtau_prev + dtau_next))
    else:
        raise ValueError(f"Unknown integration order: {order}")
    I_new = I * xp + u * src_prev + v * src_curr + w * src_next
    return I_new, u, v, w

def ng_accel(sppprev, spprev, sprev, scur):
    q1 = scur - 2 * sprev + spprev
    q2 = scur - sprev - spprev + sppprev
    q3 = scur - sprev

    a1 = np.dot(q1, q1)
    b1 = np.dot(q1, q2)
    c1 = np.dot(q1, q3)
    a2 = np.dot(q2, q1)
    b2 = np.dot(q2, q2)
    c2 = np.dot(q2, q3)

    a  = (c1 * b2 - c2 * b1) / (a1 * b2 - a2 * b1)
    b  = (c2 * a1 - c1 * a2) / (a1 * b2 - a2 * b1)

    return (1 - a - b) * scur + a * sprev + b * spprev

def tridag(a, b, c, r):
    n = len(a)
    u = np.zeros(n, dtype=float)
    gamma = np.zeros(n, dtype=float)

    if b[0] == 0.0:
        raise ZeroDivisionError("tridag: rewrite equations, b[0] == 0")

    beta = b[0]
    u[0] = r[0] / beta

    for j in range(1, n):
        gamma[j] = c[j - 1] / beta
        beta = b[j] - a[j] * gamma[j]
        if beta == 0.0:
            raise ZeroDivisionError(f"tridag failed at j={j}")
        u[j] = (r[j] - a[j] * u[j - 1]) / beta

    for j in range(n - 2, -1, -1):
        u[j] -= gamma[j + 1] * u[j + 1]

    return u


def main():
    # Constants
    nzc = 49
    nzi = nzc + 1
    mu = 1.0 / np.sqrt(3.0)
    alpha0 = 1e5
    alpha1 = 1e-1

    # Input
    eps = float(input("Photon destruction probability (epsilon)? "))
    n_iter = int(input("Number of iterations? "))
    order = int(input("Order of integration (1, 2 or 3)? "))
    ali = int(input("Iteration method (0=LI, 1=ALI local, 2=ALI tridiag)? "))
    ng = int(input("Ng acceleration (0=no, 1=yes)? "))

    # Optical depth grid
    zi = np.linspace(0, 1, nzi)
    zc = 0.5 * (zi[:-1] + zi[1:])
    dz      = np.diff(zi)
    alpha   = np.exp(np.log(alpha0) + (np.log(alpha1) - np.log(alpha0)) * zc)
    bnu     = np.ones(nzi)
    dtau    = alpha * dz / mu
    dtaui   = np.zeros(nzi)
    dtaui[1:-1] = 0.5 * (dtau[1:] + dtau[:-1])
    dtaui[0] = 2 * dtaui[1] - dtaui[2]
    dtaui[-1] = 2 * dtaui[-2] - dtaui[-3]

    taui    = np.zeros(nzi)
    for i in reversed(range(nzc)):
        taui[i] = taui[i + 1] + dtau[i]
    np.savetxt("taui.out", taui)

    # Source Function
    dmt_a   = np.zeros(nzi)
    dmt_b   = np.ones(nzi)
    dmt_c   = np.zeros(nzi)
    drs     = np.ones(nzi)

    for i in range (1, nzi - 1):
        dmt_a[i] = -1.0 / (dtaui[i] * dtau[i - 1])
        dmt_b[i] = eps + 1.0 / (dtaui[i] * dtau[i - 1]) + 1.0 / (dtaui[i] * dtau[i])
        dmt_c[i] = -1.0 / (dtaui[i] * dtau[i])
        drs[i]   = eps * bnu[i]

    dmt_a[-1] = -1.0 / dtau[-1]
    dmt_b[-1] = 1.0 / dtau[-1] + 1.0
    dmt_c[-1] = 0.0
    drs[-1]   = 0.0

    j_diff = tridag(dmt_a, dmt_b, dmt_c, drs)
    s_diff = eps * bnu + (1 - eps) * j_diff
    np.savetxt("s_diff.out", s_diff)

    # (A)LI
    s_iter = eps * bnu.copy()
    s_bk = np.zeros((nzi, 4))
    ing = -1
    all_iters = []

    for iter in range(n_iter):
        lambda_a = np.zeros(nzi)
        lambda_b = np.zeros(nzi)
        lambda_c = np.zeros(nzi)
        j_iter  = np.zeros(nzi)
        intensity = bnu[0]
        j_iter[0] += 0.5 * intensity
        
        for i in range(1, nzi):
            sp = s_iter[i-1]
            sc = s_iter[i]
            dtp = dtau[i-1]
            if (i != nzi - 1):
                sn = s_iter[i + 1]
                dtn = dtau[i]
            else:
                sn = sc
                dtn = dtaui[i]
            
            intensity, u, v, w = qdr_olsonkunasz(intensity, sp, sc, sn, dtp, dtn, order)
            j_iter[i] += 0.5 * intensity
            lambda_a[i] += 0.5 * u
            lambda_b[i] += 0.5 * v
            lambda_c[i] += 0.5 * w
        
        mat_a = -(1 - eps) * lambda_a
        mat_b = 1 - (1 - eps) * lambda_b
        mat_c = -(1 - eps) * lambda_c

        if ali==0:
            s_iter = eps * bnu + (1 - eps) * j_iter
        elif ali==1:
            s_iter = (eps * bnu + (1 - eps) * (j_iter - lambda_b * s_iter)) / mat_b
        elif ali==2:
            j_tri = np.zeros(nzi)
            for i in range(1, nzi - 1):
                j_tri[i] = lambda_a[i] * s_iter[i - 1] + lambda_b[i] * s_iter[i] + lambda_c[i] * s_iter[i + 1]
            j_tri[0] = lambda_b[0] * s_iter[0] + lambda_c[0] * s_iter[1]
            j_tri[-1] = lambda_a[-1] * s_iter[-2] + lambda_b[-1] * s_iter[-1]

            rhs = eps * bnu + (1 - eps) * (j_iter - j_tri)
            s_iter = tridag(mat_a, mat_b, mat_c, rhs)
        else:
            raise ValueError(f"Unknown ALI case: {order}")

        # Ng acceleration
        if ng == 1:
            ing += 1
        if 0 <= ing < 4:
            s_bk[:, ing] = s_iter
        if ing == 3:
            s_iter = ng_accel(s_bk[:, 0], s_bk[:, 1], s_bk[:, 2], s_iter)
            ing = -1
        
        all_iters.append(s_iter.copy())
        
    np.savetxt("s_iter.txt", np.array(all_iters))
    np.savetxt("s_final.out", s_iter)
    
if __name__ == "__main__":
    main()
