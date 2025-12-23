# MARL训练框架配置完成

## ✅ 已完成的配置和修改

### 1. 核心代码实现

- ✅ **MultiAgentEnvWorker** (`rlinf/workers/env/multi_agent_env_worker.py`)
  - 多智能体环境Worker实现
  - 支持多智能体格式的观察、动作、奖励处理
  - 自动适配单智能体环境

- ✅ **MARLRunner** (`rlinf/runners/marl_runner.py`)
  - 完整的MARL训练运行器
  - 支持多种训练范式
  - 修复了数据流和指标聚合问题

- ✅ **MultiAgentActorGroup** (`rlinf/runners/marl_runner.py`)
  - 多智能体Actor组管理
  - 支持独立和共享策略模式

- ✅ **MARL算法模块** (`rlinf/algorithms/marl_algorithms.py`)
  - MAPPO、IPPO、MADDPG、QMIX算法实现
  - 多智能体优势计算和损失函数

### 2. 训练脚本和配置

- ✅ **主训练脚本** (`examples/marl/main_marl.py`)
  - 完整的训练入口
  - 支持FSDP Actor（可扩展为Megatron）
  - 自动创建Worker组

- ✅ **配置文件** (`examples/marl/config/mappo_example.yaml`)
  - 完整的MAPPO配置示例
  - 包含所有必要的配置项
  - 添加了cluster和group_name配置

- ✅ **运行脚本** (`examples/marl/run_marl.sh`)
  - 便捷的训练启动脚本
  - 自动日志记录

### 3. 文档

- ✅ **快速开始指南** (`examples/marl/QUICKSTART.md`)
- ✅ **详细文档** (`examples/marl/README.md`)
- ✅ **框架设计文档** (`rlinf/MARL_FRAMEWORK.md`)

### 4. 代码修复

- ✅ 修复了MARL runner中的数据流问题
- ✅ 修复了指标聚合逻辑
- ✅ 添加了必要的导入和错误处理
- ✅ 更新了`__init__.py`以导出MARL相关类

## 🚀 开始训练

### 快速开始

1. **准备配置文件**
   ```bash
   cd examples/marl
   cp config/mappo_example.yaml config/my_config.yaml
   # 编辑配置文件，修改环境类型、模型路径等
   ```

2. **运行训练**
   ```bash
   bash run_marl.sh config/my_config.yaml
   # 或
   python main_marl.py --config config/my_config.yaml
   ```

### 配置检查清单

在运行前，请确保：

- [ ] 配置文件中的 `marl.num_agents` 与实际环境一致
- [ ] `env.train.simulator_type` 指向正确的环境类型
- [ ] `actor.model.model_path` 指向有效的模型路径（如果从checkpoint开始）
- [ ] `placement.actor.num_workers` 根据策略共享模式正确设置
- [ ] 所有 `group_name` 配置项都已设置
- [ ] `cluster.num_nodes` 和GPU配置正确

### 环境要求

- 多智能体环境需要返回字典格式：
  ```python
  obs = {"agent_0": obs_0, "agent_1": obs_1, ...}
  rewards = {"agent_0": r_0, "agent_1": r_1, ...}
  dones = {"agent_0": d_0, "agent_1": d_1, ...}
  ```

- 如果使用单智能体环境，框架会自动适配

## 📝 关键配置说明

### 策略共享模式

**独立策略** (`policy_sharing: "independent"`):
```yaml
placement:
  actor:
    num_workers: 2  # 等于智能体数量
```

**共享策略** (`policy_sharing: "shared"`):
```yaml
placement:
  actor:
    num_workers: 1  # 所有智能体共享
```

### 算法选择

- **MAPPO** (推荐): `marl.algorithm: "mappo"` + `algorithm.advantage_type: "mappo"`
- **IPPO**: `marl.algorithm: "ippo"` + `algorithm.advantage_type: "ippo"`
- **MADDPG**: `marl.algorithm: "maddpg"` + `algorithm.advantage_type: "maddpg"`
- **QMIX**: `marl.algorithm: "qmix"` + 自定义混合网络

## 🔧 自定义和扩展

### 添加新的MARL算法

1. 在 `rlinf/algorithms/marl_algorithms.py` 中添加：
   ```python
   @register_advantage("your_algorithm")
   def compute_your_algorithm_advantages(...):
       # 实现算法逻辑
       return advantages, returns
   ```

2. 在配置中使用：
   ```yaml
   algorithm:
     advantage_type: "your_algorithm"
   ```

### 使用Megatron Actor

修改 `examples/marl/main_marl.py` 中的 `create_actor_worker_group` 函数：

```python
from rlinf.workers.actor.megatron_actor_worker import MegatronActor

def create_actor_worker_group(cfg, agent_id, cluster, placement):
    actor_group = MegatronActor.create_group(cfg, component_placement).launch(
        cluster,
        name=f"{cfg.actor.group_name}_{agent_id}",
        placement_strategy=placement,
    )
    return actor_group
```

## 📊 训练监控

训练过程中会记录：
- 环境指标: `env/...`
- Rollout指标: `rollout/{agent_id}/...`
- 训练指标: `train/{agent_id}/...`
- 时间指标: `time/...`

Checkpoint保存在：
```
{log_path}/{experiment_name}/checkpoints/global_step_{step}/actor/
```

## 🐛 故障排除

### 常见错误

1. **"agent_ids长度必须等于num_agents"**
   - 检查 `marl.agent_ids` 列表长度是否等于 `marl.num_agents`

2. **"resume_dir does not exist"**
   - 检查checkpoint路径是否正确
   - 确保checkpoint目录结构正确

3. **环境接口错误**
   - 确保环境返回字典格式
   - 检查观察、奖励、done的键名是否与 `agent_ids` 一致

4. **GPU资源不足**
   - 减少 `placement.actor.num_workers`
   - 使用共享策略模式
   - 减少 `env.train.total_num_envs`

## 📚 参考文档

- [快速开始指南](QUICKSTART.md)
- [详细文档](README.md)
- [框架设计文档](../../rlinf/MARL_FRAMEWORK.md)

## ✨ 下一步

1. 根据你的具体环境修改配置文件
2. 实现或注册你的多智能体环境
3. 运行训练并监控结果
4. 根据需要调整超参数

祝训练顺利！🎉

