# Week14: 多标签蛋白功能分类 — 工程骨架

## 目录结构
```
week14/
├── tools/
│   ├── set_seed.py               # 统一种子控制，强制确定性算子
│   ├── checkpoint_io.py          # 统一checkpoint存取格式
│   ├── verify_determinism.py     # 每日开工前确定性自检
│   ├── fetch_uniprot_data.py     # 从UniProt拉取原始数据
│   └── analyze_label_frequency.py # 标签频次分析+Top-K过滤
├── data/
│   ├── raw/          # 原始拉取数据
│   └── processed/    # 过滤后的multilabel数据 + 标签空间清单
├── models/           # 模型定义
├── configs/          # 训练配置
├── checkpoints/       # 模型权重
├── tests/
│   └── test_day1_data.py  # Day1数据正确性校验
├── notes/            # 学习笔记
└── logs/             # 训练日志
```

## Day1 执行步骤
```bash
cd week14

# 1. 拉取原始数据（2000条人类已审核蛋白）
python tools/fetch_uniprot_data.py --n 2000

# 2. 分析标签频次，确定Top-K标签空间（默认K=50）
python tools/analyze_label_frequency.py --k 50

# 3. 运行单元测试，校验数据格式
pytest tests/test_day1_data.py -v

# 4. 确定性自检（每天开工前必须先跑这个）
python tools/verify_determinism.py
```

## 设计原则
1. **单一入口**：种子控制、checkpoint存取都只有一个函数入口，禁止散落调用。
2. **强制确定性**：`torch.use_deterministic_algorithms(True, warn_only=False)`，
   遇到不支持确定性实现的算子直接报错，而不是静默降级。
3. **shift-left测试**：数据构造阶段就做正确性校验，不等训练完看指标异常再排查。
4. **自包含**：每周文件夹独立，可脱离其他周目录单独复现。
