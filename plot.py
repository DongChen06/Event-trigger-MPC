import numpy as np
import matplotlib.pyplot as plt


def smooth(x, timestamps=9):
    # last 100
    n = len(x)
    y = np.zeros(n)
    for i in range(n):
        start = max(0, i - timestamps)
        y[i] = float(x[start:(i + 1)].sum()) / (i - start + 1)
    return y


# def plot_Pendulum():
#     path = '/home/reza/Downloads/Pend Env/runs/Pendulum/sac_s_0_t_2022_02_25_16_16_14/'
#     # obs_0 = np.load('ob_history_0.npy')
#     # obs_1 = np.load('ob_history_1.npy')
#     obs = np.load(path + 'ob_history.npy')
#     actions = np.load(path + 'actions.npy')
#     episode_returns = np.load(path + 'episode_returns.npy')
#     eval_returns = np.load(path + 'eval_returns.npy')
#
#     plt.figure(1)
#     # plt.title('Epoch Returns')
#     # plt.xlabel('Training epochs')
#     # plt.ylabel('Average return')
#     # plt.xlim([0, epochs])
#     # plt.ylim([-2000, 8000])
#     # Plot the smoothed returns
#     # episode_rewards = smooth(episode_rewards)
#     plt.plot(obs[:, 0], label='Cart position')
#     plt.legend()
#     plt.show()
#
#     plt.figure(2)
#     # plt.title('Epoch Returns')
#     # plt.xlabel('Training epochs')
#     # plt.ylabel('Average return')
#     # plt.xlim([0, epochs])
#     # plt.ylim([-2000, 8000])
#     # Plot the smoothed returns
#     # episode_rewards = smooth(episode_rewards)
#     plt.plot(obs[:, 1], label='Cart velocity')
#     plt.legend()
#     plt.show()
#
#     plt.figure(3)
#     # plt.title('Epoch Returns')
#     # plt.xlabel('Training epochs')
#     # plt.ylabel('Average return')
#     # plt.xlim([0, epochs])
#     # plt.ylim([-2000, 8000])
#     # Plot the smoothed returns
#     # episode_rewards = smooth(episode_rewards)
#     plt.plot(obs[:, 2], label='Pendulum angle')
#     plt.legend()
#     plt.show()
#
#     plt.figure(4)
#     # plt.title('Epoch Returns')
#     # plt.xlabel('Training epochs')
#     # plt.ylabel('Average return')
#     # plt.xlim([0, epochs])
#     # plt.ylim([-2000, 8000])
#     # Plot the smoothed returns
#     # episode_rewards = smooth(episode_rewards)
#     plt.plot(obs[:, 3], label='Pendulum velocity')
#     plt.legend()
#     plt.show()
#
#     plt.figure(5)
#     # plt.title('Epoch Returns')
#     # plt.xlabel('Training epochs')
#     # plt.ylabel('Average return')
#     # plt.xlim([0, epochs])
#     # plt.ylim([-2000, 8000])
#     # Plot the smoothed returns
#     # episode_rewards = smooth(episode_rewards)
#     plt.plot(actions, label='actions')
#     plt.plot(np.clip(actions, 0, 1), label='actions_clipped_0.5')
#     plt.legend()
#     plt.show()
#
#     plt.figure(6)
#     # plt.title('Epoch Returns')
#     # plt.xlabel('Training epochs')
#     # plt.ylabel('Average return')
#     # plt.xlim([0, epochs])
#     # plt.ylim([-2000, 8000])
#     # Plot the smoothed returns
#     # episode_rewards = smooth(episode_rewards)
#     plt.plot(episode_returns, label='episode_returns_0.5')
#     plt.legend()
#     plt.show()
#
#     plt.figure(7)
#     # plt.title('Epoch Returns')
#     # plt.xlabel('Training epochs')
#     # plt.ylabel('Average return')
#     # plt.xlim([0, epochs])
#     # plt.ylim([-2000, 8000])
#     # Plot the smoothed returns
#     # episode_rewards = smooth(episode_rewards)
#     plt.plot(eval_returns, label='eval_returns_0.5')
#     plt.legend()
#     plt.show()


def plot_Pendulum():
    model_path = 'sac_s_0_t_2022_07_28_13_07_53'
    path = '/home/reza/Downloads/Pend Env/runs/Pendulum/' + model_path
    obs = np.load(path + '/ob_history.npy')
    actions = np.load(path + '/actions.npy')
    u = np.load(path + '/numpy_u.npy')

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(obs[:, 0])
    axs[0, 0].set_title('Cart position')
    axs[0, 1].plot(obs[:, 1], 'tab:orange')
    axs[0, 1].set_title('Cart velocity')
    axs[1, 0].plot(obs[:, 2], 'tab:green')
    axs[1, 0].set_title('Pendulum angle')
    axs[1, 1].plot(obs[:, 3], 'tab:red')
    axs[1, 1].set_title('Pendulum velocity')

    for ax in axs.flat:
        ax.set(xlabel='time steps')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    # for ax in axs.flat:
    #     ax.label_outer()

    plt.tight_layout()
    plt.show()

    plt.figure(2)
    plt.plot(actions, label='actions')
    plt.plot(np.clip(actions, 0, 1), label='actions_clipped_0.5')
    plt.legend()
    plt.show()

    plt.figure(3)
    plt.plot(u[0, :], 'r', label='u')
    plt.plot(u[1, :], '--' , label='u_local')
    plt.plot(u[2, :],  '-.', c='k', label='u_cloud')
    # plt.xlim([0, 20])
    # plt.ylim([-0.01, 0.001])
    plt.legend()
    plt.show()


def plot_Veh():
    model_path = 'sac_s_0_t_2022_07_28_13_07_53'
    path = 'runs/Veh/' + model_path
    obs = np.load(path + '/ob_history.npy')
    actions = np.load(path + '/actions.npy')
    u = np.load(path + '/numpy_u.npy')

    fig, axs = plt.subplots(2, 3, figsize=(10, 6))
    axs[0, 0].plot(obs[:, 0], obs[:, 2], label='RL')
    axs[0, 0].plot(obs[:, 0], 4 * np.sin(2 * np.pi / 50 * np.array(obs[:, 0])), label='ground truth')
    axs[0, 0].set_title('Tracking Performance')
    axs[0, 0].set_xlabel('x')
    axs[0, 0].set_ylabel('y')

    axs[0, 1].plot(obs[:, 2] - 4 * np.sin(2 * np.pi / 50 * np.array(obs[:, 0])), 'tab:orange')
    axs[0, 1].set_title('Tracking error')
    axs[0, 1].set_xlabel('Time steps')

    axs[0, 2].plot(actions, c='green', label='actions')
    axs[0, 2].plot(np.clip(actions, 0, 1), c='red', label='actions_clipped')
    axs[0, 2].set_title('Action')
    axs[0, 2].set_xlabel('Time steps')
    axs[0, 2].legend()

    axs[1, 0].plot(u[0, :], label='u', c='red')
    axs[1, 0].plot(u[2, :], linestyle='--', c='green', label='u_cloud')
    axs[1, 0].plot(u[4, :], linestyle='-.', c='k', label='u_local')
    axs[1, 0].set_title('Control Variable (Torque)')
    axs[1, 0].set_xlabel('Time steps')
    axs[1, 0].legend()

    axs[1, 1].plot(u[1, :],  label='u', c='red')
    axs[1, 1].plot(u[3, :],  linestyle='--', c='green', label='u_cloud')
    axs[1, 1].plot(u[5, :],  linestyle='-.', c='k', label='u_local')
    axs[1, 1].set_title('Control Variable (Steering))')
    axs[1, 1].set_xlabel('Time steps')
    axs[1, 1].legend()

    axs[1, 2].axis('off')

    # for ax in axs.flat:
    #     ax.set(xlabel='time steps')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    # for ax in axs.flat:
    #     ax.label_outer()

    plt.tight_layout()
    plt.legend()
    plt.show()

    # plt.figure(2)
    # plt.plot(actions, label='actions')
    # plt.plot(np.clip(actions, 0, 1), label='actions_clipped')
    # plt.legend()
    # plt.show()

    # plt.figure(3)
    # plt.plot(u[0, :], 'r', label='u')
    # plt.plot(u[1, :], '--' , label='u_local')
    # plt.plot(u[2, :],  '-.', c='k', label='u_cloud')
    # # plt.xlim([0, 20])
    # # plt.ylim([-0.01, 0.001])
    # plt.legend()
    # plt.show()


if __name__ == '__main__':
    # plot_Pendulum()
    plot_Veh()