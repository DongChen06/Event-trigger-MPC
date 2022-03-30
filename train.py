import os
import gym
import time
import argparse
import datetime
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from env_veh import VehEnv
from shutil import copy


# Configurations
parser = argparse.ArgumentParser(description='RL algorithms with PyTorch in Pendulum environment')
parser.add_argument('--env', type=str, default='Veh',
                    help='pendulum environment')
parser.add_argument('--algo', type=str, default='sac',
                    help='select an algorithm among vpg, trpo, ppo, ddpg, td3, sac, asac')
parser.add_argument('--phase', type=str, default='train',
                    help='choose between training phase and testing phase')
parser.add_argument('--render', action='store_true', default=False,
                    help='if you want to render, set this to True')
parser.add_argument('--load', type=str, default=None,
                    help='copy & paste the saved model name, and load it')
parser.add_argument('--seed', type=int, default=0, 
                    help='seed for random number generators')
parser.add_argument('--iterations', type=int, default=500,
                    help='iterations to run and train agent')
parser.add_argument('--eval_per_train', type=int, default=5,
                    help='evaluation number per training')
parser.add_argument('--max_step', type=int, default=100,
                    help='max episode step')
parser.add_argument('--tensorboard', action='store_true', default=True)
parser.add_argument('--gpu_index', type=int, default=1)
args = parser.parse_args()
device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')

if args.algo == 'vpg':
    from deep_rl.agents.vpg import Agent
elif args.algo == 'npg':
    from deep_rl.agents.trpo import Agent
elif args.algo == 'trpo':
    from deep_rl.agents.trpo import Agent
elif args.algo == 'ppo':
    from deep_rl.agents.ppo import Agent
elif args.algo == 'ddpg':
    from deep_rl.agents.ddpg import Agent
elif args.algo == 'td3':
    from deep_rl.agents.td3 import Agent
elif args.algo == 'sac':
    from deep_rl.agents.sac import Agent
elif args.algo == 'asac': # Automating entropy adjustment on SAC
    from deep_rl.agents.sac import Agent
elif args.algo == 'tac': 
    from deep_rl.agents.sac import Agent
elif args.algo == 'atac': # Automating entropy adjustment on TAC
    from deep_rl.agents.sac import Agent


def main():
    """Main."""
    # Initialize environment
    env = VehEnv(max_ct=args.max_step)
    obs_dim = env.observation_space
    act_dim = env.action_space
    act_limit = env.action_high
    
    print('---------------------------------------')
    print('Environment:', args.env)
    print('Algorithm:', args.algo)
    print('State dimension:', obs_dim)
    print('Action dimension:', act_dim)
    print('Action limit:', act_limit)
    print('---------------------------------------')

    # Set a random seed
    # env.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

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

    # If we have a saved model, load it
    if args.load is not None:
        pretrained_model_path = os.path.join('./save_model/' + str(args.load))
        pretrained_model = torch.load(pretrained_model_path, map_location=device)
        agent.policy.load_state_dict(pretrained_model)

    # Create a SummaryWriter object by TensorBoard
    if args.tensorboard and args.load is None:
        dir_name = 'runs/' + args.env + '/' \
                           + args.algo \
                           + '_s_' + str(args.seed) \
                           + '_t_' + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        writer = SummaryWriter(log_dir=dir_name)

    # save log files
    copy('train.py', dir_name)
    copy('env_veh.py', dir_name)
    copy('mpc.py', dir_name)
    copy('cloud_mpc_file.py', dir_name)

    start_time = time.time()

    if not os.path.exists(dir_name + '/save_model'):
        os.mkdir(dir_name + '/save_model')

    train_num_steps = 0
    train_sum_returns = 0.
    train_num_episodes = 0
    episode_returns = []
    eval_returns = []

    for i in range(args.iterations):
        if args.phase == 'train':
            agent.eval_mode = False
            
            # Run one episode
            train_step_length, train_episode_return = agent.run(args.max_step)
            train_num_steps += train_step_length
            train_sum_returns += train_episode_return
            train_num_episodes += 1
            train_average_return = train_sum_returns / train_num_episodes if train_num_episodes > 0 else 0.0
            episode_returns.append(train_episode_return)

            # Log experiment result for training episodes
            if args.tensorboard and args.load is None:
                writer.add_scalar('Train/AverageReturns', train_average_return, i)
                writer.add_scalar('Train/EpisodeReturns', train_episode_return, i)
                writer.add_scalar('Train/EpisodeStep', train_step_length, i)
                if args.algo == 'asac' or args.algo == 'atac':
                    writer.add_scalar('Train/Alpha', agent.alpha, i)

        # Perform the evaluation phase -- no learning
        if (i + 1) % args.eval_per_train == 0:
            eval_sum_returns = 0.
            eval_num_episodes = 0
            agent.eval_mode = True

            for _ in range(1):
                # Run one episode
                eval_step_length, eval_episode_return, ob_history, _ = agent.test(args.max_step)

                eval_sum_returns += eval_episode_return
                eval_num_episodes += 1

            eval_average_return = eval_sum_returns / eval_num_episodes if eval_num_episodes > 0 else 0.0
            eval_returns.append(eval_average_return)

            # Save the trained model
            ckpt_path = os.path.join(dir_name + '/save_model/' + args.env + '_' + args.algo \
                                         + '_s_' + str(args.seed) \
                                         + '_i_' + str(i + 1) + '.pt')
            torch.save(agent.policy.state_dict(), ckpt_path)

            # Log experiment result for evaluation episodes
            if args.tensorboard and args.load is None:
                writer.add_scalar('Eval/AverageReturns', eval_average_return, i)
                writer.add_scalar('Eval/EpisodeReturns', eval_episode_return, i)
                writer.add_scalar('Eval/EpisodeStep', eval_step_length, i)

            np.save(dir_name + '/episode_returns', episode_returns)
            np.save(dir_name + '/eval_returns', eval_returns)

            if args.phase == 'train':
                print('---------------------------------------')
                print('Iterations:', i + 1)
                print('Steps:', train_num_steps)
                print('Episodes:', train_num_episodes)
                print('EpisodeReturn:', round(train_episode_return, 2))
                print('AverageReturn:', round(train_average_return, 2))
                print('EvalEpisodes:', eval_num_episodes)
                print('EvalEpisodeReturn:', round(eval_episode_return, 2))
                print('EvalAverageReturn:', round(eval_average_return, 2))
                print('OtherLogs:', agent.logger)
                print('Time:', int(time.time() - start_time))
                print('---------------------------------------')

            elif args.phase == 'test':
                print('---------------------------------------')
                print('EvalEpisodes:', eval_num_episodes)
                print('EvalEpisodeReturn:', round(eval_episode_return, 2))
                print('EvalAverageReturn:', round(eval_average_return, 2))
                print('Time:', int(time.time() - start_time))
                print('---------------------------------------')


if __name__ == "__main__":
    main()
