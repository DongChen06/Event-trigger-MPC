from mpc import *
import matplotlib.pyplot as plt
import os

state_init = ca.DM([0, 10, 0, -0.0691, 0.2343, -0.0123])
# actual_state_seq = state_init
# ct = 0
# alpha = 1
# t0 = 0
N = 5
# u0_bar = 1e-4 * ca.DM.ones((2, N))
# u_hat_traj = 1e-4 * ca.DM.ones((2, N))
# X0_2 = ca.repmat(state_init, 1, N)
buffer_size = 2
# all_states_cloud = []
# u_all = []
# state_init, cost, u, t0, u_hat_traj, X0_2, u0_bar, actual_state_seq, all_states_cloud, u_all, lin_model_diff, \
# buffer_states_cloud, buffer_actual_states = cloud_local_mpc(ct, N, alpha, t0, u0_bar, u_hat_traj, X0_2,
#                                                             state_init, actual_state_seq, buffer_size,
#                                                             all_states_cloud, u_all)


class VehEnv:
    def __init__(self, max_ct=100, N=5):
        self.max_ct = max_ct
        self.N = N
        self.obs_dim = 6
        self.observation_space = int(self.obs_dim + buffer_size * self.obs_dim * 2)
        self.action_space = 1
        self.action_high = 1
        self.threshold_error = 100

    def reset(self):
        self.t0 = 0
        self.u0_bar = 1e-3 * ca.DM.ones((2, N))
        self.u_hat_traj = 1e-3 * ca.DM.ones((2, N))
        self.X0_2 = ca.repmat(state_init, 1, N)
        self.x0 = ca.DM([0, 10, 0, -0.0691, 0.2343, -0.0123])  # x, vx, y, vy, phi, r
        self.all_actual_states = self.x0
        self.all_cloud_states = []
        self.all_u = []
        self.ct = 0  # ct is the time step
        print("The Plant is reset to initial conditions at time zero")
        # self.buffer_lin_model_diff = []
        return np.concatenate((np.array(self.x0).flatten(), np.zeros(self.obs_dim * buffer_size), np.zeros(self.obs_dim  * buffer_size)))

    def step(self, alpha):
        alpha = np.clip(alpha, 0, 1)
        next_state, cost_val, con, t, u_hat, X0, u_bar, actual_state_sequence, all_states_cloud, u_all, \
        buffer_lin_model, buffer_states_cloud, \
        buffer_actual_states \
            = cloud_local_mpc(
            self.ct, self.N, alpha, self.t0, self.u0_bar,
            self.u_hat_traj, self.X0_2, self.x0,
            self.all_actual_states, buffer_size,
            self.all_cloud_states, self.all_u)

        reward = np.array(-cost_val)[0][0]
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

        tracking_error = (next_state[2] - 4 * np.sin(2 * np.pi / 50 * np.array(next_state[0]))) ** 2
        if self.ct > self.max_ct:  # final time stop condition
            # print('Maximum time step exceeded, Plant is reset to initial conditions')
            done = True
        # elif tracking_error > self.threshold_error:
        #     done = True
        #     reward = -100
        else:
            done = False

        # if self.ct > self.max_ct:
        #     print('Maximum time step exceeded, Plant is reset to initial conditions')
        #     self.reset()
        # if next_state[0] < -10 or next_state[0] > 10:
        #     reward = -ca.inf
        #     self.reset()
        #     print('Constraints violated, - inf reward ')
        state = np.concatenate((next_state.flatten(), np.array(buffer_actual_states - buffer_states_cloud).flatten(),
                                np.array(buffer_lin_model).flatten()))

        return state, reward, done, self.all_u

    def close(self):
        pass


if __name__ == "__main__":
    seed = 66
    np.random.seed(seed)
    random.seed(seed)

    if not os.path.exists('runs' + '/Veh/local'):
        os.mkdir('runs' + '/Veh/local')

    if not os.path.exists('runs' + '/Veh/cloud'):
        os.mkdir('runs' + '/Veh/cloud')

    Pen = VehEnv()
    Pen.reset()
    done = False
    local_return = 0
    local_obs, local_a = [], []

    # test the local mpc
    while not done:
        state, reward, done, all_u = Pen.step(0)
        local_return += reward
        local_obs.append(state)
        local_a.append(0)

    print("Return for Local MPC:", local_return)
    np.save('runs' + '/Veh/local/ob_history', local_obs)
    np.save('runs' + '/Veh/local/actions', local_a)
    np.save('runs' + '/Veh/local/numpy_u', np.array(all_u))

    # test the cloud mpc
    Pen.reset()
    done = False
    cloud_return = 0
    cloud_obs, cloud_a = [], []

    # test the local mpc
    while not done:
        state, reward, done, all_u = Pen.step(1)
        cloud_return += reward
        cloud_obs.append(state)
        cloud_a.append(1)

    print("Return for Cloud MPC:", cloud_return)
    np.save('runs' + '/Veh/cloud/ob_history', cloud_obs)
    np.save('runs' + '/Veh/cloud/actions', cloud_a)
    np.save('runs' + '/Veh/cloud/numpy_u', np.array(all_u))

    # # # plt.subplot(2, 2, 1)
    # # plt.subplot(2, 2, 2)
    # plt.plot(Pen.all_actual_states[1, :].T)
    # plt.subplot(2, 2, 3)
    # plt.plot(Pen.all_actual_states[2, :].T)
    # plt.subplot(2, 2, 4)
    # plt.plot(Pen.all_actual_states[3, :].T)

# plt.figure()
# plt.plot(Pen.all_u[1, :].T)