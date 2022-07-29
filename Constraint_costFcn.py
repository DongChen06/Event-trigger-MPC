from time import time
import casadi as ca
import numpy as np
import math
from casadi import sin, cos, pi


def const_and_CostFcn_new(horizon, X, U, P, specs):
    Ts = specs[0]
    f = specs[1]
    n_states = specs[2]

    x_0 = P[:n_states]
    u_0 = U[:, 0]
    k1 = f(x_0, u_0)
    k2 = f(x_0 + Ts / 2 * k1, u_0)
    k3 = f(x_0 + Ts / 2 * k2, u_0)
    k4 = f(x_0 + Ts * k3, u_0)
    st_next_RK41 = x_0 + (Ts / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    g = X[:, 0] - st_next_RK41

    cost_fn = 2.0 * (x_0[2] - 4 * sin((2 * pi * x_0[0]) / 50)) ** 2 \
              + 1e-6 * u_0[0] ** 2 \
              + u_0[1] ** 2

    # cost_fn = 0

    # cost_fn = 10.0 * (x_0[0] - P[n_states]) ** 2 + 10.0 * (x_0[2] - P[1 + n_states]) ** 2 \
    #           + 10 * (x_0[1] - 10) ** 2 \
    #           + 1 * u_0[0] ** 2 \
    #           + 1 * u_0[1] ** 2

    # runge kutta
    for k in range(horizon - 1):
        st = X[:, k]
        con = U[:, k + 1]

        cost_fn = cost_fn \
                  + 2.0 * (st[2] - 4 * sin((2 * pi * st[0]) / 50)) ** 2 \
                  + 1e-6 * con[0] ** 2 \
                  + con[1] ** 2

        # cost_fn = cost_fn \
        #           + 10.0 * (st[0] - P[2 * (k+1) + n_states]) ** 2 + 10.0 * (st[2] - P[2 * (k+1) + 1 + n_states]) ** 2 \
        #           + 10 * (st[1] - 10) ** 2\
        #           + 1 * con[0] ** 2 \
        #           + 1 * con[1] ** 2

        st_next = X[:, k + 1]
        k1 = f(st, con)
        k2 = f(st + Ts / 2 * k1, con)
        k3 = f(st + Ts / 2 * k2, con)
        k4 = f(st + Ts * k3, con)
        st_next_RK4 = st + (Ts / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        g = ca.vertcat(g, st_next - st_next_RK4)

    # cost_fn = cost_fn + 2.0 * (X[0, horizon-1] - P[-2]) ** 2 + 2.0 * (X[2, horizon-1] - P[-1]) ** 2 \
    #           + 10 * (X[1, horizon-1] - 10) ** 2

    cost_fn = cost_fn + 2.0 * (X[2, horizon-1] - 4 * sin((2 * pi * X[0, horizon-1]) / 50)) ** 2


    return cost_fn, g
