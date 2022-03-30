from mpc import *
import matplotlib.pyplot as plt


state_init = ca.DM([0, 10, 0, -0.0691, 0.2343, -0.0123])
actual_state_seq = state_init
ct = 0
alpha = 1
t0 = 0
N = 5
u0_bar = 1e-4 * ca.DM.ones((2, N))
u_hat_traj = 1e-4 * ca.DM.ones((2, N))
X0_2 = ca.repmat(state_init, 1, N)
buffer_size = 2
all_states_cloud = []
u_all = []
state_init, cost, u, t0, u_hat_traj, X0_2, u0_bar, actual_state_seq, all_states_cloud, u_all, lin_model_diff, \
buffer_states_cloud, buffer_actual_states = cloud_local_mpc(ct, N, alpha, t0, u0_bar, u_hat_traj, X0_2,
                                                            state_init, actual_state_seq, buffer_size,
                                                            all_states_cloud, u_all)


class PendEnv:
    def __init__(self, max_ct=100, N=5):
        self.max_ct = max_ct
        self.N = N

    def reset(self):
        self.t0 = 0
        self.u0_bar = 1e-3 * ca.DM.ones((2, N))
        self.u_hat_traj = 1e-3 * ca.DM.ones((2, N))
        self.X0_2 = ca.repmat(state_init, 1, N)
        self.x0 = ca.DM([0, 10, 0, -0.0691, 0.2343, -0.0123])
        self.all_actual_states = self.x0
        self.all_cloud_states = []
        self.all_u = []
        self.ct = 0  # ct is the time step,

        # self.buffer_lin_model_diff = []

        return print("The Plant is reset to initial conditions at time zero")

    def step(self, alpha):

        next_state, cost_val, con, t, u_hat, X0, u_bar, actual_state_sequence, all_states_cloud, u_all, \
        buffer_lin_model, buffer_states_cloud, \
        buffer_actual_states \
            = cloud_local_mpc(
            self.ct, self.N, alpha, self.t0, self.u0_bar,
            self.u_hat_traj, self.X0_2, self.x0,
            self.all_actual_states, buffer_size,
            self.all_cloud_states, self.all_u)

        reward = -cost_val

        self.ct += 1
        self.t0 = t
        self.u_hat_traj = u_hat
        self.X0_2 = X0
        self.u0_bar = u_bar
        self.x0 = next_state
        self.all_actual_states = actual_state_sequence
        self.all_cloud_states = all_states_cloud
        self.all_u = u_all
        # self.buffer_lin_model_diff = buffer_lin_model

        # if self.ct > self.max_ct:
        #     print('Maximum time step exceeded, Plant is reset to initial conditions')
        #     self.reset()
        # if next_state[0] < -10 or next_state[0] > 10:
        #     reward = -ca.inf
        #     self.reset()
        #     print('Constraints violated, - inf reward ')

        return next_state, con, reward, buffer_actual_states, buffer_states_cloud, buffer_lin_model


Pen = PendEnv()
# Pen.reset()
# for i in range(50):
#     Pen.step(0)
# plt.plot(Pen.all_actual_states[0, :].T,Pen.all_actual_states[2, :].T)
Pen.reset()
for i in range(50):
    Pen.step(1)
plt.plot(Pen.all_actual_states[0, :].T,Pen.all_actual_states[2, :].T)
plt.plot(Pen.all_actual_states[0, :].T, 4 * sin(2 * pi / 50 * Pen.all_actual_states[0, :]).T)
plt.show()

# # # plt.subplot(2, 2, 1)
# # plt.subplot(2, 2, 2)
# plt.plot(Pen.all_actual_states[1, :].T)
# plt.subplot(2, 2, 3)
# plt.plot(Pen.all_actual_states[2, :].T)
# plt.subplot(2, 2, 4)
# plt.plot(Pen.all_actual_states[3, :].T)

# plt.figure()
# plt.plot(Pen.all_u[1, :].T)