"""
# @Author: JuQi
# @Time  : 2022/2/7 7:20 PM
# @E-mail: 18672750887@163.com
"""

import numpy as np
import torch
import random

a = np.array([0.3,-0.5])
b=(a-np.floor(a))*1000
print(b)

# a=np.random.random(3)
# print(a)
# b=torch.from_numpy(a)
# print(b)
# a[0]=1
# print(a)
# print(b)
# value_fn_out = np.random.random(10)
# true_reward = np.random.random(10)
#
# shuffle_order = np.arange(10)
# np.random.shuffle(shuffle_order)
#
# print(value_fn_out)
# print(shuffle_order)
#
# compare_value_fn_out = value_fn_out[shuffle_order]
# compare_true_reward = true_reward[shuffle_order]
#
# print(value_fn_out)
# print(compare_value_fn_out)
#
# compare_ans = np.float64(true_reward > compare_true_reward)
# print(compare_ans)
#
# gap_value_fn_out = compare_value_fn_out - value_fn_out
# win_rate_evaluation = 1 / (1 + np.e ** gap_value_fn_out)
#
# value_fn_out = value_fn_out + 0.03 * (compare_ans - win_rate_evaluation)
# print(value_fn_out)
