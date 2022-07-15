import logging
from typing import Dict, List, Type, Union

import ray
from ray.rllib.agents.ppo.ppo_tf_policy import setup_config
from ray.rllib.evaluation.postprocessing import compute_gae_for_sample_batch, \
    Postprocessing
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.policy.sample_batch import SampleBatch
from MoCoTorchPolicy import EntropyCoeffSchedule, \
    LearningRateSchedule, TorchPolicy
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.torch_utils import apply_grad_clipping, \
    explained_variance, sequence_mask
from ray.rllib.utils.typing import TensorType
import copy
import numpy as np

torch, nn = try_import_torch()

logger = logging.getLogger(__name__)


class MoCoPPOTorchPolicy(TorchPolicy, LearningRateSchedule, EntropyCoeffSchedule):
    """PyTorch policy class used with PPOTrainer."""

    def __init__(self, observation_space, action_space, config):
        config = dict(ray.rllib.agents.ppo.ppo.DEFAULT_CONFIG, **config)
        setup_config(self, observation_space, action_space, config)

        TorchPolicy.__init__(
            self,
            observation_space,
            action_space,
            config,
            max_seq_len=config["model"]["max_seq_len"])

        EntropyCoeffSchedule.__init__(self, config["entropy_coeff"],
                                      config["entropy_coeff_schedule"])
        LearningRateSchedule.__init__(self, config["lr"],
                                      config["lr_schedule"])

        # The current KL value (as python float).
        self.kl_coeff = self.config["kl_coeff"]
        # Constant target value.
        self.kl_target = self.config["kl_target"]
        self.count_train = 0  # 训练纪次
        self.history_model_save_interval = self.config["history_model_save_interval"]  # 历史模型保存间隔
        self.history_model_save_upper = self.config["history_model_save_upper"]  # 历史模型保存上限
        self.history_model = []  # 历史模型集合
        self.moco_pool_upper = self.config["moco_pool_upper"]  # moco 蓄水池大小
        self.moco_reward_pool = np.array([])  # 历史收益纪录
        self.moco_elo_pool = np.array([])  # 历史elo纪录
        self.elo_compare_time = self.config["elo_compare_time"]  # 比较次数
        self.elo_eta = self.config["elo_eta"]
        self.elo_K = self.config["elo_K"]
        # TODO: Don't require users to call this manually.
        self._initialize_loss_from_dummy_batch()

    @override(TorchPolicy)
    def postprocess_trajectory(self,
                               sample_batch,
                               other_agent_batches=None,
                               episode=None):
        # Do all post-processing always with no_grad().
        # Not using this here will introduce a memory leak
        # in torch (issue #6962).
        # TODO: no_grad still necessary?
        with torch.no_grad():
            return compute_gae_for_sample_batch(self, sample_batch,
                                                other_agent_batches, episode)

    # TODO: Add method to Policy base class (as the new way of defining loss
    #  functions (instead of passing 'loss` to the super's constructor)).
    @override(TorchPolicy)
    def loss(self, model: ModelV2, dist_class: Type[ActionDistribution],
             train_batch: SampleBatch) -> Union[TensorType, List[TensorType]]:
        """Constructs the loss for Proximal Policy Objective.

        Args:
            model: The Model to calculate the loss for.
            dist_class: The action distr. class.
            train_batch: The training data.

        Returns:
            The PPO loss tensor given the input batch.
        """

        def get_elo_point(elo_value, elo_reward, moco_value, moco_reward):
            """
            计算新的elo分
            :param elo_value:
            :param elo_reward:
            :param moco_value:
            :param moco_reward:
            :return:
            """
            compare_ans = np.float64(
                elo_reward > moco_reward
            )
            compare_ans_tmp = np.float64(
                elo_reward == moco_reward
            )
            compare_ans = compare_ans + 0.5 * compare_ans_tmp
            op_compare_ans = 1 - compare_ans
            # TODO：探索时间长度和结果之间的关系
            time_step = elo_reward % 1000
            op_time_step = moco_reward % 1000
            time_step = 100
            op_time_step = 100

            gap_value_fn_out = moco_value * op_time_step - elo_value * time_step
            win_rate_evaluation = 1 / (1 + (10 ** (gap_value_fn_out / self.elo_eta)))
            op_win_rate_evaluation = 1 - win_rate_evaluation

            next_elo_value = elo_value + self.elo_K * (compare_ans - win_rate_evaluation) / time_step
            op_next_elo_value = moco_value + self.elo_K * (op_compare_ans - op_win_rate_evaluation) / op_time_step

            return next_elo_value, op_next_elo_value

        def get_compare_result(compare_time, elo_value, elo_reward, moco_elo_pool, moco_reward_pool):
            # elo_value2=copy.deepcopy(elo_value)
            for _ in range(compare_time):
                tmp_shuffle_order = np.arange(len(moco_elo_pool))
                np.random.shuffle(tmp_shuffle_order)
                tmp_shuffle_order = tmp_shuffle_order[:len(elo_value)]
                tmp_moco_elo = moco_elo_pool[tmp_shuffle_order]
                tmp_moco_reward = moco_reward_pool[tmp_shuffle_order]

                elo_value[:], moco_value = get_elo_point(elo_value, elo_reward, tmp_moco_elo, tmp_moco_reward)
                # moco_elo_pool[tmp_shuffle_order] = moco_value
            # assert self.count_train < 200, str(elo_reward) + '\n' + str(elo_value) + '\n' + str(elo_value2)

        # 间隔一定时间 保存历史模型
        self.count_train += 1
        if self.count_train % self.history_model_save_interval == 0 and self.history_model_save_upper != 0:
            self.history_model.append(model)
            self.history_model = self.history_model[-self.history_model_save_upper:]

        logits, state = model(train_batch)
        curr_action_dist = dist_class(logits, model)
        # assert False,logits
        # RNN case: Mask away 0-padded chunks at end of time axis.
        if state:
            B = len(train_batch[SampleBatch.SEQ_LENS])
            max_seq_len = logits.shape[0] // B
            mask = sequence_mask(
                train_batch[SampleBatch.SEQ_LENS],
                max_seq_len,
                time_major=model.is_time_major())
            mask = torch.reshape(mask, [-1])
            num_valid = torch.sum(mask)

            def reduce_mean_valid(t):
                return torch.sum(t[mask]) / num_valid

        # non-RNN case: No masking.
        else:
            mask = None
            reduce_mean_valid = torch.mean

        prev_action_dist = dist_class(
            train_batch[SampleBatch.ACTION_DIST_INPUTS], model)

        logp_ratio = torch.exp(
            curr_action_dist.logp(train_batch[SampleBatch.ACTIONS]) -
            train_batch[SampleBatch.ACTION_LOGP]
        )
        # 和之前的kl散度
        action_kl = prev_action_dist.kl(curr_action_dist)
        mean_kl_loss = reduce_mean_valid(action_kl)

        # 现在的动作熵
        curr_entropy = curr_action_dist.entropy()
        mean_entropy = reduce_mean_valid(curr_entropy)

        # policy loss
        surrogate_loss = torch.min(
            train_batch[Postprocessing.ADVANTAGES] * logp_ratio,
            train_batch[Postprocessing.ADVANTAGES] * torch.clamp(
                logp_ratio,
                1 - self.config["clip_param"],
                1 + self.config["clip_param"]
            )
        )
        mean_policy_loss = reduce_mean_valid(-surrogate_loss)

        # Compute a value function loss.
        if self.config["use_critic"]:
            prev_value_fn_out = train_batch[SampleBatch.VF_PREDS]
            value_fn_out = model.value_function()  # 现在的value 不能变
            moco_target = value_fn_out.clone()  # 计算后的value 需要变
            if self.config['num_gpus'] == 0:
                next_value_fn_out = moco_target.detach().numpy()
                true_reward = train_batch[Postprocessing.VALUE_TARGETS].detach().numpy()
            else:
                next_value_fn_out = moco_target.detach().cpu().numpy()
                true_reward = train_batch[Postprocessing.VALUE_TARGETS].detach().cpu().numpy()

            last_num = true_reward[0]
            if true_reward[0] == 0:
                for tmp_i in range(len(true_reward)):
                    if true_reward[tmp_i] != 0:
                        last_num = true_reward[tmp_i]
                        break
            # 消除lstm的0

            for tmp_i in range(len(true_reward)):
                if true_reward[tmp_i] == 0:
                    true_reward[tmp_i] = last_num
                else:
                    last_num = true_reward[tmp_i]

            moco_elo_value_fn_out = copy.deepcopy(next_value_fn_out)
            for tmp_model in self.history_model:
                tmp_model(train_batch)
                tmp_elo_value_fn_out = tmp_model.value_function()
                if self.config['num_gpus'] == 0:
                    tmp_elo_value_fn_out = tmp_elo_value_fn_out.detach().numpy()
                else:
                    tmp_elo_value_fn_out = tmp_elo_value_fn_out.detach().cpu().numpy()
                moco_elo_value_fn_out += tmp_elo_value_fn_out

            moco_elo_value_fn_out = moco_elo_value_fn_out / (len(self.history_model) + 1)

            self.moco_elo_pool = np.append(self.moco_elo_pool, moco_elo_value_fn_out)
            self.moco_reward_pool = np.append(self.moco_reward_pool, true_reward)
            self.moco_elo_pool = self.moco_elo_pool[-self.moco_pool_upper:]
            self.moco_reward_pool = self.moco_reward_pool[-self.moco_pool_upper:]

            get_compare_result(
                self.elo_compare_time,
                next_value_fn_out,
                true_reward,
                self.moco_elo_pool,
                self.moco_reward_pool
            )

            # if self.count_train > 3000:
            #     assert False, 'moco_elo_value_fn_out' + '\n' \
            #                   + str(moco_elo_value_fn_out) + '\n' \
            #                   + 'next_value_fn_out' + '\n' \
            #                   + str(next_value_fn_out) + '\n' \
            #                   + 'true_reward' + '\n' \
            #                   + str(true_reward) + '\n'

            vf_loss1 = torch.pow(
                value_fn_out - moco_target,
                2.0
            )
            vf_clipped = prev_value_fn_out + torch.clamp(
                value_fn_out - prev_value_fn_out,
                -self.config["vf_clip_param"],
                self.config["vf_clip_param"]
            )
            vf_loss2 = torch.pow(
                vf_clipped - moco_target,
                2.0
            )
            vf_loss = torch.max(vf_loss1, vf_loss2)
            mean_vf_loss = reduce_mean_valid(vf_loss)

        # Ignore the value function.
        else:
            vf_loss = mean_vf_loss = 0.0

        total_loss = reduce_mean_valid(-surrogate_loss +
                                       self.kl_coeff * action_kl +
                                       self.config["vf_loss_coeff"] * vf_loss -
                                       self.entropy_coeff * curr_entropy)

        # Store values for stats function in model (tower), such that for
        # multi-GPU, we do not override them during the parallel loss phase.
        model.tower_stats["total_loss"] = total_loss
        model.tower_stats["mean_policy_loss"] = mean_policy_loss
        model.tower_stats["mean_vf_loss"] = mean_vf_loss
        model.tower_stats["vf_explained_var"] = explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS], model.value_function())
        model.tower_stats["mean_entropy"] = mean_entropy
        model.tower_stats["mean_kl_loss"] = mean_kl_loss

        return total_loss

    def _value(self, **input_dict):
        # When doing GAE, we need the value function estimate on the
        # observation.
        if self.config["use_gae"]:
            # Input dict is provided to us automatically via the Model's
            # requirements. It's a single-timestep (last one in trajectory)
            # input_dict.
            input_dict = self._lazy_tensor_dict(input_dict)
            model_out, _ = self.model(input_dict)
            # [0] = remove the batch dim.
            return self.model.value_function()[0].item()
        # When not doing GAE, we do not require the value function's output.
        else:
            return 0.0

    def update_kl(self, sampled_kl):
        # Update the current KL value based on the recently measured value.
        if sampled_kl > 2.0 * self.kl_target:
            self.kl_coeff *= 1.5
        elif sampled_kl < 0.5 * self.kl_target:
            self.kl_coeff *= 0.5
        # Return the current KL value.
        return self.kl_coeff

    # TODO: Make this an event-style subscription (e.g.:
    #  "after_actions_computed").
    @override(TorchPolicy)
    def extra_action_out(self, input_dict, state_batches, model, action_dist):
        # Return value function outputs. VF estimates will hence be added to
        # the SampleBatches produced by the sampler(s) to generate the train
        # batches going into the loss function.
        return {
            SampleBatch.VF_PREDS: model.value_function(),
        }

    # TODO: Make this an event-style subscription (e.g.:
    #  "after_gradients_computed").
    @override(TorchPolicy)
    def extra_grad_process(self, local_optimizer, loss):
        return apply_grad_clipping(self, local_optimizer, loss)

    # TODO: Make this an event-style subscription (e.g.:
    #  "after_losses_computed").
    @override(TorchPolicy)
    def extra_grad_info(self,
                        train_batch: SampleBatch) -> Dict[str, TensorType]:
        return convert_to_numpy({
            "cur_kl_coeff"    : self.kl_coeff,
            "cur_lr"          : self.cur_lr,
            "total_loss"      : torch.mean(
                torch.stack(self.get_tower_stats("total_loss"))),
            "policy_loss"     : torch.mean(
                torch.stack(self.get_tower_stats("mean_policy_loss"))),
            "vf_loss"         : torch.mean(
                torch.stack(self.get_tower_stats("mean_vf_loss"))),
            "vf_explained_var": torch.mean(
                torch.stack(self.get_tower_stats("vf_explained_var"))),
            "kl"              : torch.mean(
                torch.stack(self.get_tower_stats("mean_kl_loss"))),
            "entropy"         : torch.mean(
                torch.stack(self.get_tower_stats("mean_entropy"))),
            "entropy_coeff"   : self.entropy_coeff,
        })

    # TODO: Make lr-schedule and entropy-schedule Plugin-style functionalities
    #  that can be added (via the config) to any Trainer/Policy.
    @override(TorchPolicy)
    def on_global_var_update(self, global_vars):
        super().on_global_var_update(global_vars)
        if self._lr_schedule:
            self.cur_lr = self._lr_schedule.value(global_vars["timestep"])
            for opt in self._optimizers:
                for p in opt.param_groups:
                    p["lr"] = self.cur_lr
        if self._entropy_coeff_schedule is not None:
            self.entropy_coeff = self._entropy_coeff_schedule.value(
                global_vars["timestep"])
