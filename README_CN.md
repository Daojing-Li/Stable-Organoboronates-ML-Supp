# 有机硼化合物反应性预测的机器学习工具

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)]()
[![License](https://img.shields.io/badge/License-MIT-green)]()

[English](README.md) | [中文](README_CN.md)

这个项目是我们研究论文"**Organometallic-type reactivity of stable organoboronates for selective (hetero)arene C−H/C-halogen borylation and beyond**"的机器学习部分的补充材料。它包含了用于预测有机硼化合物反应产率的机器学习模型和分析工具。

## 目录

1. [系统要求](#系统要求)
2. [安装指南](#安装指南)
3. [工作流程概述](#工作流程)
4. [详细使用指南](#详细使用指南)
   - [数据准备](#数据准备)
   - [模型训练与特征筛选](#模型训练与特征筛选)
   - [进一步特征筛选](#进一步特征筛选)
   - [模型评估与可视化](#模型评估与可视化)
   - [外部验证与预测](#外部验证与预测)
5. [预期运行时间](#预期运行时间)
6. [结果复现说明](#结果复现说明)
7. [许可证](#许可证)

## 系统要求

### 软件依赖和操作系统

本项目的主要依赖库：

- Python >= 3.9
- pandas >= 2.0
- numpy >= 1.20
- scikit-learn >= 1.0
- matplotlib >= 3.5
- seaborn >= 0.12
- xgboost >= 1.5
- lightgbm >= 3.3
- optuna >= 3.0
- shap >= 0.40
- joblib >= 1.1

详细的依赖列表请参见[requirements.txt](requirements.txt)文件。

### 已测试的操作系统

- Linux:  Arch Linux
- Windows: Windows 10, Windows 11

### 硬件要求

- CPU: 任何现代多核处理器（推荐4核以上）
- RAM: 至少4GB，推荐8GB以上
- 存储: 至少500MB可用空间
- GPU: 不需要，但如果安装了支持CUDA的GPU，一些模型（如XGBoost）可能会加速

## 安装指南

### 安装步骤

1. 克隆或下载本仓库：

```bash
git clone https://github.com/Daojing-Li/Stable-Organoboronates-ML-Supp
cd Stable-Organoboronates-ML-Supp
```

2. 设置环境并安装依赖：

#### 选项1：使用uv（推荐 - 更快）

uv是一个用Rust编写的高性能Python包管理器，可以显著加速依赖安装。

**典型安装时间：约30秒**（在标准台式电脑上）

```bash
# 如果没有安装uv，先安装它
# 在macOS和Linux上
curl -LsSf https://astral.sh/uv/install.sh | sh

# 在Windows上
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 使用单个命令创建环境并安装所有依赖
uv sync
```

这个单一命令会自动：
- 在`.venv`创建虚拟环境
- 安装requirements.txt中的所有依赖
- 通过并行下载和缓存优化安装过程

激活虚拟环境：
```bash
# 在Linux/macOS上
source .venv/bin/activate

# 在Windows上
.venv\Scripts\activate
```

#### 选项2：使用标准工具

**典型安装时间：3-5分钟**（在标准台式电脑上）

```bash
# 创建虚拟环境
python -m venv .venv

# 激活环境
# 在Linux/macOS上
source .venv/bin/activate
# 在Windows上
.venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

## 工作流程

本项目提供了一个完整的工作流程，用于构建和评估有机硼化合物反应性预测模型：

### 1. 数据准备

输入数据应为CSV格式，包含物理化学特性和目标产率值。查看`example/data_round2.csv`文件了解数据格式示例。

### 2. 模型筛选与特征优化

使用主脚本`main.py`进行初步模型训练和特征筛选：

```bash
python main.py --n_trials 100 --mae_threshold 20 --min_features 5
```

这个步骤将：
- 训练多种机器学习模型（SVR、随机森林、XGBoost等）
- 执行超参数优化
- 进行初步特征选择
- 保存最佳模型和性能指标

### 3. 进一步特征筛选

如果初步筛选后的特征仍然较多，可使用`example/model_feature_filter.py`进行进一步筛选：

```bash
# 首先修改脚本中的数据路径和所需的最大特征数量
python example/model_feature_filter.py
```

这个步骤允许您指定最多保留的特征数量，进一步简化模型，提高可解释性。

### 4. 模型评估与可视化

使用`example/example_pic.ipynb`生成全面的模型评估和可视化结果：

- 实际值与预测值的对比散点图
- 留一交叉验证评估（LOO）
- 模型在100个随机划分上的稳定性评估
- 特征重要性分析

### 5. 外部验证与新数据预测

使用`example/prediction_round2.ipynb`对新数据进行预测和外部验证：

- 加载训练好的最佳模型
- 对新数据进行预处理
- 生成预测结果
- 评估预测性能（如果有实际值可比较）

## 详细使用指南

### 数据准备

输入数据应为CSV格式，包含以下列：

1. 化合物识别符（如`sub_H`, `sub_B`）
2. 各种物理化学特性，包括但不限于：
   - `pka_H`: pKa值
   - `dipole_H`: 偶极矩
   - `homo_H`: 最高占据分子轨道能量
   - `lumo_H`: 最低未占据分子轨道能量
   - 其他各种物理化学特性
3. 目标变量`yield`（产率，百分比）

### 模型训练与特征筛选

#### 使用main.py脚本

`main.py`是项目的主要入口，用于训练多种回归模型、优化超参数并进行初步特征筛选：

```bash
python main.py --n_trials 100 --mae_threshold 20 --min_features 5
```

#### 参数说明

- `--n_trials`: Optuna优化的试验次数（默认100）
  - 更高的值可能导致更好的模型性能，但需要更长的训练时间
  - 推荐值范围：50-200

- `--mae_threshold`: MAE阈值，用于筛选好的模型（默认20）
  - 低于此阈值的模型被认为是"好"模型
  - 推荐值范围：15-25，取决于您的数据和期望精度

- `--min_features`: 要保留的最小特征数量（默认5）
  - 即使特征选择过程认为可以删除更多特征，也会保留这么多特征
  - 推荐值范围：5-10

- `--n_jobs`: 使用的CPU核心数（默认-1，表示使用所有核心）
  - 可以加速训练过程，特别是在使用树基模型时
  - 在资源受限的环境中，可以设置为一个较小的值

- `--keep_versions`: 每个模型保留的版本数（默认2）
  - 控制为每个模型类型保存的最佳模型版本数量
  - 较高的值会占用更多磁盘空间

#### 训练过程

1. 脚本将自动开始训练多种模型，包括SVR、随机森林、XGBoost等。
2. 对于每个模型，会执行超参数优化以找到最佳配置。
3. 脚本会通过迭代特征选择来提高模型性能。
4. 训练过程的日志将输出到控制台和保存在`models`目录下。

#### 训练结果

训练完成后，最佳模型和相关结果将保存在`models`目录下，每个模型类型有一个子目录：

```
models/
├── SVR/
│   ├── svr_final_20230615123456.joblib  # 保存的模型
│   ├── svr_final_20230615123456_metrics.txt  # 模型性能指标
│   └── ... 
├── RandomForest/
│   ├── randomforest_final_20230615123456.joblib
│   ├── randomforest_final_20230615123456_metrics.txt
│   └── ...
└── ...
```

### 进一步特征筛选

如果初步筛选后的特征数量仍然较多，可以使用`example/model_feature_filter.py`脚本进一步减少特征数量：

#### 使用model_feature_filter.py

1. 打开脚本并修改以下参数：
   - 数据文件路径
   - 目标最大特征数（`max_features`参数）
   - 模型类型（取消注释或注释相应的模型）

2. 运行脚本：

```bash
python example/model_feature_filter.py
```

3. 脚本将输出经过筛选后的最优特征集合和相应的模型性能。

#### 特征筛选原理

该脚本使用递归特征消除与交叉验证(RFECV)方法，通过以下步骤筛选特征：

1. 初始化模型并设置超参数
2. 训练初始模型并评估性能
3. 递归地移除最不重要的特征，并重新评估性能
4. 找到能够提供最佳性能的最小特征子集

### 模型评估与可视化

使用`example/example_pic.ipynb` Jupyter笔记本进行详细的模型评估和可视化：

笔记本中包含以下主要功能：
   - 加载训练好的最佳模型
   - 生成实际值与预测值的对比散点图
   - 执行留一交叉验证(LOO)评估模型稳健性
   - 在100个随机数据集划分上评估模型稳定性
   - 分析特征重要性

### 外部验证与预测

使用`example/prediction_round2.ipynb`进行外部验证和新数据预测：

笔记本中包含以下主要功能：
   - 加载训练好的最佳模型
   - 加载新的验证数据集
   - 预处理数据，使其适合模型输入
   - 生成预测结果
   - 评估预测性能（如果有实际值可比较）
   - 可视化预测结果

#### 预测新数据

要预测全新的数据，您需要：

1. 准备包含相同特征的新数据文件
2. 在笔记本中修改数据加载部分，指向您的新数据
3. 运行笔记本的预测部分
4. 分析和导出预测结果

## 预期运行时间

工作流程中不同步骤的运行时间因硬件配置而异。以下是在标准台式计算机上的典型运行时间估计：

1. **数据准备**：可忽略不计（几秒钟）

2. **模型筛选与特征优化**（步骤2）：
   - 使用推荐参数进行完整模型训练需要**3-4天**的时间（在个人电脑上）
   - 如果仅用于测试目的，可以将`n_trials`减少到10-20，这样只需几个小时即可完成

3. **进一步特征筛选**（步骤3）：
   - 运行时间主要取决于`max_features`参数
   - 使用`max_features=7`（我们论文中使用的值）：在个人电脑上需要**7天以上**
   - 如果仅用于测试目的，设置`max_features=3-5`将显著减少运行时间

4. **模型评估与可视化**：取决于模型的复杂度和数量，从几分钟到几小时不等

5. **外部验证与预测**：几分钟

## 结果复现说明

要复现我们论文中的确切结果，请按照以下步骤操作：

1. 按照[安装指南](#安装指南)完成安装

2. 使用推荐参数运行初始模型训练：
   ```bash
   python main.py --n_trials 100 --mae_threshold 20 --min_features 5
   ```

3. 对于进一步的特征筛选，修改`example/model_feature_filter.py`脚本：
   - 在脚本中设置`max_features = 7`
   - 运行脚本：
   ```bash
   python example/model_feature_filter.py
   ```
   **注意**：此步骤需要大量计算时间（在个人电脑上需要7天以上）

4. 使用`example`目录中的笔记本生成评估可视化结果

5. 对于外部验证，使用`example/prediction_round2.ipynb`笔记本

这些步骤将复现我们论文中呈现的结果。如果只是为了快速测试或探索，可以通过以下方式减少计算时间：
- 在步骤2中使用更少的试验次数（例如，`--n_trials 20`）
- 在步骤3中设置较小的`max_features`值（3-5）

## 许可证

本项目采用MIT许可证。详情请参见[LICENSE](LICENSE)文件。
