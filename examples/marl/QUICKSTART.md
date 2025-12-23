# MARL训练快速开始指南

## 前置要求

1. **环境准备**
   - 已安装RLinf框架
   - 配置好多智能体环境（或使用单智能体环境自动适配）
   - 准备初始模型checkpoint（可选，可从零开始训练）

2. **硬件要求**
   - 根据智能体数量配置足够的GPU
   - 独立策略模式：每个智能体建议至少1个GPU
   - 共享策略模式：可以共享GPU资源

## 快速开始

### 步骤1: 准备配置文件

复制并修改配置文件：

```bash
cd examples/marl
cp config/mappo_example.yaml config/my_marl_config.yaml
```

编辑 `config/my_marl_config.yaml`，修改以下关键配置：

```yaml
# 1. MARL配置
marl:
  num_agents: 2                    # 修改为你的智能体数量
  agent_ids: ["agent_0", "agent_1"]
  algorithm: "mappo"              # 选择算法: mappo, ippo, maddpg, qmix
  policy_sharing: "independent"   # 或 "shared"

# 2. 环境配置
env:
  train:
    simulator_type: "your_marl_env"  # 替换为你的环境类型
    total_num_envs: 64

# 3. 模型配置
actor:
  model:
    model_path: "/path/to/your/model"  # 初始模型路径
    model_type: "mlp_policy"          # 根据你的模型类型修改

# 4. 输出配置
runner:
  logger:
    log_path: "./logs"
    experiment_name: "my_marl_experiment"
```

### 步骤2: 准备多智能体环境

确保你的环境实现了多智能体接口：

```python
class YourMARLEnv:
    def reset(self):
        # 返回: obs_dict, infos
        return {
            "agent_0": obs_0_tensor,
            "agent_1": obs_1_tensor,
        }, infos
    
    def step(self, actions):
        # actions: {agent_id: action_tensor}
        # 返回: obs_dict, rewards_dict, dones_dict, infos
        return obs_dict, rewards_dict, dones_dict, infos
```

**注意**: 如果使用单智能体环境，框架会自动适配，但可能需要调整奖励分配策略。

### 步骤3: 运行训练

**方式1: 使用脚本（推荐）**

```bash
cd examples/marl
bash run_marl.sh config/my_marl_config.yaml
```

**方式2: 直接运行Python**

```bash
cd examples/marl
python main_marl.py --config config/my_marl_config.yaml
```

### 步骤4: 监控训练

训练过程中会：
- 在控制台显示进度条和指标
- 保存checkpoint到: `{log_path}/{experiment_name}/checkpoints/global_step_X/`
- 记录日志到配置的日志系统

### 步骤5: 恢复训练（可选）

如果需要从checkpoint恢复：

```yaml
runner:
  resume_dir: "./logs/my_marl_experiment/checkpoints/global_step_100"
```

## 配置说明

### 关键配置项

1. **智能体数量** (`marl.num_agents`)
   - 必须与实际环境中的智能体数量一致

2. **策略共享模式** (`marl.policy_sharing`)
   - `independent`: 每个智能体独立策略（适合异构智能体）
   - `shared`: 所有智能体共享策略（适合同构智能体，节省资源）

3. **算法选择** (`marl.algorithm`)
   - `mappo`: 集中训练分散执行（推荐）
   - `ippo`: 完全独立学习
   - `maddpg`: 连续动作空间
   - `qmix`: 值分解方法

4. **GPU资源配置** (`placement`)
   - 根据策略共享模式调整 `actor.num_workers`
   - 独立策略：`num_workers = num_agents`
   - 共享策略：`num_workers = 1`

## 常见问题

### Q1: 如何适配单智能体环境？

框架会自动检测并适配。如果环境返回单智能体格式，会自动转换为多智能体格式。奖励会平均分配给各个智能体（可在代码中自定义分配策略）。

### Q2: 如何选择策略共享模式？

- **独立策略**: 当智能体角色不同、动作空间不同时使用
- **共享策略**: 当智能体同构（相同角色、相同动作空间）时使用，可以显著节省内存和计算

### Q3: 如何添加新的MARL算法？

1. 在 `rlinf/algorithms/marl_algorithms.py` 中添加优势计算函数
2. 使用 `@register_advantage("your_algorithm")` 装饰器
3. 在配置中使用 `algorithm.advantage_type: "your_algorithm"`

### Q4: Checkpoint保存在哪里？

Checkpoint保存在：
```
{log_path}/{experiment_name}/checkpoints/global_step_{step}/
├── actor/
│   ├── actor_agent_0/  (独立策略)
│   └── actor_agent_1/
└── actor/
    └── shared_actor/   (共享策略)
```

### Q5: 如何调整训练参数？

主要参数在配置文件的 `algorithm` 部分：
- `gamma`: 折扣因子（默认0.99）
- `gae_lambda`: GAE平滑因子（默认0.95）
- `ratio_clip_eps`: PPO裁剪比例（默认0.2）
- `learning_rate`: 学习率（默认3e-4）

## 下一步

- 查看 [README.md](README.md) 了解详细文档
- 查看 [MARL_FRAMEWORK.md](../../rlinf/MARL_FRAMEWORK.md) 了解框架设计
- 参考配置文件示例了解所有可用选项

## 获取帮助

如果遇到问题：
1. 检查配置文件格式是否正确
2. 确认环境接口实现正确
3. 查看日志文件 `marl_training.log`
4. 检查GPU资源是否充足

