from legged_gym import *
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry, copy_task_env_to_log_dir


def train(args):
    env, _ = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    copy_task_env_to_log_dir(args.task, ppo_runner.log_dir)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)


if __name__ == '__main__':
    train(get_args())
