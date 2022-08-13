import random
import argparse
import numpy as np
import torch
from env_veh import VehEnv
import os



# Configurations
parser = argparse.ArgumentParser(description='RL algorithms with PyTorch in Pendulum environment')
parser.add_argument('--env', type=str, default='Veh',
                    help='pendulum environment')
parser.add_argument('--algo', type=str, default='ppo',
                    help='select an algorithm among vpg, npg, trpo, ppo, ddpg, td3, sac, asac, tac, atac')
parser.add_argument('--seed', type=int, default=0, 
                    help='seed for random number generators')
parser.add_argument('--max_step', type=int, default=100,
                    help='max episode step')
parser.add_argument('--gpu_index', type=int, default=1)
args = parser.parse_args()
device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')

if args.algo == 'vpg':
    from agents.vpg import Agent
elif args.algo == 'npg':
    from agents.trpo import Agent
elif args.algo == 'trpo':
    from agents.trpo import Agent
elif args.algo == 'ppo':
    from agents.ppo import Agent
elif args.algo == 'ddpg':
    from agents.ddpg import Agent
elif args.algo == 'td3':
    from agents.td3 import Agent
elif args.algo == 'sac':
    from agents.sac import Agent
elif args.algo == 'asac': # Automating entropy adjustment on SAC
    from agents.sac import Agent
elif args.algo == 'tac': 
    from agents.sac import Agent
elif args.algo == 'atac': # Automating entropy adjustment on TAC
    from agents.sac import Agent


def main():
    # Initialize environment
    env = VehEnv(max_ct=args.max_step)
    obs_dim = env.observation_space
    act_dim = env.action_space
    act_limit = env.action_high

    # Create an agent
    if args.algo == 'ddpg' or args.algo == 'td3':
        agent = Agent(env, args, device, obs_dim, act_dim, act_limit)
    elif args.algo == 'sac':
        agent = Agent(env, args, device, obs_dim, act_dim, act_limit, 
                      alpha=0.5)
    elif args.algo == 'asac':
        agent = Agent(env, args, device, obs_dim, act_dim, act_limit, 
                      automatic_entropy_tuning=True)
    elif args.algo == 'tac':
        agent = Agent(env, args, device, obs_dim, act_dim, act_limit, 
                      alpha=0.5,
                      log_type='log-q', 
                      entropic_index=1.2)
    elif args.algo == 'atac':
        agent = Agent(env, args, device, obs_dim, act_dim, act_limit, 
                      log_type='log-q', 
                      entropic_index=1.2, 
                      automatic_entropy_tuning=True)
    else: # vpg, npg, trpo, ppo
        agent = Agent(env, args, device, obs_dim, act_dim, act_limit)

    returns = []
    best_return = -1e6
    best_model_index = 0
    seed = 66

    # specify the model directory here
    model_path = 'ppo_s_0_t_2022_08_11_18_50_31'

    """finding the best model"""
    print("------Finding the best model------")
    for i in range(18000, 20001, 5):
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

        pretrained_model_path = 'runs/Veh/' + model_path
        pretrained_model = torch.load(pretrained_model_path + '/save_model/Veh_ppo_s_0_i_' + str(i) + '.pt', map_location=device)
        agent.policy.load_state_dict(pretrained_model)

        # Run one episode
        eval_step_length, eval_episode_return, ob_history, actions = agent.test(args.max_step)
        print("Model Index:", i, "Evaluation Return:", eval_episode_return, "Episode Length:", eval_step_length)
        returns.append(eval_episode_return)
        if eval_episode_return >= best_return:
            best_model_index = i
            best_return = eval_episode_return

    print("Best model is", best_model_index)
    print("Best return is", max(returns))
    print("------------End------------")

    # """test a particular model"""
    print("------Testing the Best Model------")
    print("Model path:", model_path)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    pretrained_model_path = 'runs/Veh/' + model_path
    pretrained_model = torch.load(pretrained_model_path + '/save_model/Veh_ppo_s_0_i_' + str(best_model_index) + '.pt', map_location=device)
    agent.policy.load_state_dict(pretrained_model)

    # Run one episode
    eval_step_length, eval_episode_return, ob_history, (actions, u) = agent.test(args.max_step)
    print("Evaluation Return", eval_episode_return)
    np.save(pretrained_model_path + '/ob_history', ob_history)
    np.save(pretrained_model_path + '/actions', actions)
    np.save(pretrained_model_path + '/numpy_u', u)
    print("------------End------------")


if __name__ == "__main__":
    main()