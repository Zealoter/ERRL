"""
# @Author: JuQi
# @Time  : 2022/1/26 8:18 PM
# @E-mail: 18672750887@163.com
"""

from __future__ import print_function
import time
import ray
from ray import tune
from MoCoPPOTrainer import MoCoPPOTrainer
from ray.rllib.agents.impala import ImpalaTrainer
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.utils.framework import try_import_torch
torch, nn = try_import_torch()

if __name__ == "__main__":
    ray.shutdown()
    ray.init(ignore_reinit_error=True)
    config = {
        'log_level'                  : 'ERROR',
        "use_critic"                 : True,
        "use_gae"                    : True,
        'gamma'                      : 1,
        'lr'                         : 1e-4,
        'env'                        : 'CartPole-v0',
        'num_workers'                : 3,
        'num_gpus'                   : 0,
        'framework'                  : 'torch',
        'train_batch_size'           : 20000,
        "sgd_minibatch_size"         : 512,
        "shuffle_sequences"          : True,
        "num_sgd_iter"               : 1,
        "batch_mode"                 : "complete_episodes",
        "history_model_save_upper"   : 0,
        "history_model_save_interval": 10,
        "elo_compare_time"           : 1,
        'moco_pool_upper'            : 50000,
        "elo_eta"                    : 4,
        'elo_K'                      : 1.6,

        "model"                      : {
            'fcnet_hiddens': [32, 32, 32],
            "use_lstm"     : True,
            # "lstm_cell_size" : 16,
            # "max_seq_len"    : 20,
        },
    }
    stop = {
        'episode_reward_mean': 2000
    }
    st = time.time()
    results = tune.run(
        MoCoPPOTrainer,  # Specify the algorithm to train
        config=config,
        stop=stop,
        # keep_checkpoints_num=10,
        # checkpoint_freq=50,
        checkpoint_at_end=True,
        # local_dir=''
    )
    print('elapsed time=', time.time() - st)
    ray.shutdown()

