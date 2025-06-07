ERRL
---
此环境的灵感来源与ELO Ranking，既然对比学习可以用在图像或者NLP领域。
而强化学习中显然会有一个trail比另一个trail好的情况，
那么是不是可以把对比学习的思想用在强化学习中？
# 安装
```shell
conda create -n errl python=3.8
conda activate errl
pip install -r requirements.txt 
```

# 运行
```shell
python mcrl_code/train_base.py
```
# 介绍
## MoCoEnv
rllib游戏环境构建，在这里的设定是只有最后的状态才有收益
## MoCoPPOPolicy
修改的重点，参考MoCo的方法，着重修改了loss的计算方法。