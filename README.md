# 🧠 大模型基础与应用 —— 期中作业

> 📚 *本项目为《大模型的基础与应用》课程的期中作业，内容包括模型训练、评估与实验管理。*

---

## 📂 项目结构
├── data/ # 数据集（IWSLT2017 英德数据）

├── src/ # 源代码：模型定义、训练与评估

├── scripts/ # 辅助脚本（模型训练脚本）

├── train-process/ # 训练流程记录

├── results/ #  loss 与 BELU 变化曲线 

├── logs/ # 训练与评估日志

├── requirements.txt # 依赖包

└── README.md # 项目说明文档


---

## ⚙️ 环境配置

### 1️⃣ 创建虚拟环境
```bash
conda create -p lkr/conda/lm python=3.10
```

### 2️⃣ 安装依赖
```bash
pip install -r requirements.txt
```

## 🚀 模型训练
运行主训练脚本（注意力头数为4）
```bash
bash run-h4.sh --epochs 10 --batch_size 64 --seed 42
```
📝 训练日志保存在 logs/ 中，模型权重保存在 src/checkpoints/。

测试训练好的模型（注意力头数为4）
```bash
python src/translate.py
```


