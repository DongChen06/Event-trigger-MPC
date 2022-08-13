# -------------------------------------------------------------#
# This is the implementation of cloud-local MPC. In this script
# the local mpc is explicitly formulated and cloud mpc is called
# this code. Then with a predefined alpha the control is applied
# to the plant.
# --------------------------------------------------------------#
from time import time
import casadi as ca
import numpy as np
import math
from casadi import sin, cos, pi
import matplotlib.pyplot as plt
from dynamics_linear import *
from dynamics_nonlinear import *
from cloud_mpc_file import *
from Constraint_costFcn import *
from shift import *
from newarguments import *
from Linearized_dynamics import *


def cloud_local_mpc(ct, N, alpha, t0, u0_bar, u_hat_traj, X0_2, state_init, actual_state, M, all_states_cloud, u_all):
    Ts = 0.2  # time between steps in seconds
    horizon = N - (ct % N)

    # state symbolic variables
    x1 = ca.SX.sym('x1')
    x2 = ca.SX.sym('x2')
    x3 = ca.SX.sym('x3')
    x4 = ca.SX.sym('x4')
    x5 = ca.SX.sym('x5')
    x6 = ca.SX.sym('x6')

    states = ca.vertcat(
        x1,
        x2,
        x3,
        x4,
        x5,
        x6
    )
    n_states = states.numel()

    # w = -0.01 + 0.15 * np.random.rand(n_states, 150)  # local
    # w_x = -0.02 + 0.2 * np.random.rand(n_states, 150)  # cloud
    # seed = 66
    # np.random.seed(seed)
    # random.seed(seed)
    w = 0.1 * np.random.rand(n_states, 300)  # Original
    w_x = 0 * np.random.rand(n_states, 300)  # Original

    # control symbolic variables
    u1 = ca.SX.sym('u1')
    u2 = ca.SX.sym('u2')
    controls = ca.vertcat(
        u1,
        u2
    )
    n_controls = controls.numel()

    P = ca.SX.sym('P', n_states)  # column vector for storing initial state
    X = ca.SX.sym('X', n_states, horizon)
    U = ca.SX.sym('U', n_controls, horizon)

    OPT_variables = ca.vertcat(
        X.reshape((-1, 1)),  # Example: 3x11 ---> 33x1 where 3=states, 11=N+1
        U.reshape((-1, 1))
    )

    # Casadi symbolic functions
    # f = ca.Function('f', [states, controls], [veh_rhs_linear(states, controls)])  #veh_rhs_linear_Model
    f_nonlinear = ca.Function('f1', [states, controls], [veh_rhs_nonlinear(states, controls)])  # nonlinear model

    # Linearization about the state at time ct and last control input
    if state_init[2, :]-actual_state[2, -1] > 0:

        xref = ca.DM([20.7784, 9.38292, 2.03305, 0.0163441, 0.0801592, -0.0960399])
        # xref = ca.DM([103.793, 9.0518, 2.06196, 0.00665512, -0.0900298, -0.044337])

    else:
        # xref = ca.DM([20.7784, 9.38292, 2.03305, 0.0163441, 0.0801592, -0.0960399])
        xref = ca.DM([103.793, 9.0518, 2.06196, 0.00665512, -0.0900298, -0.044337])

    # xref = state_init
    # xref = ca.DM([0, 10, 0, -0.0691, 0.2343, -0.0123])
    uref = u0_bar[:, 0]
    A, B = linearize(xref, uref)

    def veh_rhs_linearized(x, u):
        return A @ (x - xref) + B @ (u - uref) + veh_rhs_nonlinear(xref, uref)

    f = ca.Function('f', [states, controls], [veh_rhs_linearized(states, controls)])  # veh_rhs_linear_Model

    plant = ca.Function('plant', [states, controls], [veh_rhs_nonlinear(states, controls)])  # Actual plant
    specs = [Ts, f, n_states, n_controls]
    cost_fn, g = const_and_CostFcn_new(horizon, X, U, P, specs)
    args = newargs_new(horizon, specs)

    nlp_prob = {
        'f': cost_fn,
        'x': OPT_variables,
        'g': g,
        'p': P}

    opts = {
        'ipopt': {
            'max_iter': 2000,
            'print_level': 0,
            'acceptable_tol': 1e-8,
            'acceptable_obj_change_tol': 1e-6
        },
        'print_time': 0
    }

    solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

    if (ct + 1) % N == 1:  # condition for triggering cloud mpc
        # x_traj_cloud is the state trajectory calculated by the cloud
        state_init_hat = state_init + w_x[:, ct:ct + 1]
        if ct == 0:
            u_hat_traj, pred_states_cloud = cloud_mpc(N, state_init_hat, u0_bar[:, -1])
            all_states_cloud = ca.horzcat(all_states_cloud, pred_states_cloud)
        else:
            u_hat_traj, pred_states_cloud = cloud_mpc(N, state_init_hat, u_hat_traj[:, -1])
            all_states_cloud = ca.horzcat(all_states_cloud, pred_states_cloud[:, 1:])

    args['p'] = ca.vertcat(state_init)
    # optimization variable current state
    args['x0'] = ca.vertcat(
        ca.reshape(X0_2, n_states * horizon, 1),
        ca.reshape(u0_bar, n_controls * horizon, 1)
    )

    sol = solver(
        x0=args['x0'],
        lbx=args['lbx'],
        ubx=args['ubx'],
        lbg=args['lbg'],
        ubg=args['ubg'],
        p=args['p']
    )

    u_bar_traj = ca.reshape(sol['x'][n_states * horizon:], n_controls, horizon)
    X0 = ca.reshape(sol['x'][: n_states * horizon], n_states, horizon)

    u_bar = u_bar_traj[:, 0]
    u_hat = u_hat_traj[:, ct % N]
    u = alpha * u_hat + (1 - alpha) * u_bar
    # u_all = ca.horzcat(u_all, u)
    u_all = ca.horzcat(u_all, ca.vertcat(u, u_hat, u_bar))

    buffer_linear_model_difference = []
    difference_value = []

    # cost = (state_init.T @ Q @ state_init) + (u @ R @ u)
    cost = 2.0 * (state_init[2] - 4 * sin(2 * pi / 50 * state_init[0])) ** 2 \
           + 1e-6 * u[0] ** 2 + u[1] ** 2

    t0, state_init = shift_timestep(Ts, t0, state_init, u, plant)  # applying the control to the plant
    state_init += w[:, ct: ct + 1]
    # state_init += (state_init / 5000) * np.random.rand(1)  # states are noisy now
    actual_state = ca.horzcat(actual_state, state_init)

    if ct + 1 < M - 1:
        buffer_states_cloud = ca.horzcat(ca.DM.zeros(n_states, M - (ct + 2)), all_states_cloud[:, 0: ct + 2])
        buffer_actual_states = ca.horzcat(ca.DM.zeros(n_states, M - (ct + 2)), actual_state[:, 0: ct + 2])

    else:
        buffer_states_cloud = all_states_cloud[:, ct - (M - 2):ct + 2]
        buffer_actual_states = actual_state[:, ct - (M - 2):ct + 2]

    if ct + 1 < M:
        for i in range(ct + 1):
            difference_value = ca.horzcat(difference_value, f_nonlinear(actual_state[:, i], u_all[0:2, i])
                                          - f(actual_state[:, i], u_all[0:2, i]))
        buffer_linear_model_difference = ca.horzcat(ca.DM.zeros(n_states, M - (ct + 1)), difference_value)
    else:
        for i in range(M):
            difference_value = f_nonlinear(actual_state[:, ct - i], u_all[0:2, ct - i]) \
                               - f(actual_state[:, ct - i], u_all[0:2, ct - i])
            buffer_linear_model_difference = ca.horzcat(difference_value, buffer_linear_model_difference)

    u0_bar = u_bar_traj[:, 1:]
    X0_2 = X0[:, 1:]

    if (ct + 1) % N == 0:
        u0_bar = u_bar_traj[:, 0] * ca.DM.ones((n_controls, N))
        X0_2 = ca.repmat(state_init, 1, N)

    if (ct + 1) % N == N - 1:
        u0_bar = u_bar_traj[:, 0]

    return state_init, cost, u, t0, u_hat_traj, X0_2, u0_bar, actual_state, all_states_cloud, u_all, \
           buffer_linear_model_difference, buffer_states_cloud, buffer_actual_states
