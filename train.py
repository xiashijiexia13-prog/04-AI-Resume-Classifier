"""
train.py
========
功能：训练多个机器学习模型，自动比较并选择最佳模型。

训练流程（企业标准ML模型开发流程）：
    1. 加载TF-IDF特征矩阵和标签
    2. 训练3个模型：
       - Logistic Regression（逻辑回归）—— 线性分类基线
       - LinearSVC（线性支持向量机）—— 高维稀疏数据表现优秀
       - Random Forest（随机森林）—— 非线性集成方法
    3. 每个模型输出完整的评估指标
    4. 生成混淆矩阵可视化
    5. 自动选择最佳模型并保存

为什么选择这3个模型？

    Logistic Regression：
        - 文本分类的经典基线模型
        - 训练快、可解释性强（能看到每个词的权重）
        - 与TF-IDF配合效果好（线性模型+稀疏特征）
        - 面试常考点：多分类用softmax还是OVR？

    LinearSVC：
        - 在TF-IDF文本分类任务中通常表现最佳
        - 最大化类别间隔，对高维稀疏数据鲁棒
        - hinge loss → 稀疏解 → 泛化能力强
        - 面试常考点：SVM的核技巧、支持向量概念

    Random Forest：
        - 非线性模型，可能捕获词之间的交互关系
        - Bagging集成 → 降低方差
        - 输出特征重要性 → 可解释
        - 面试常考点：Bagging vs Boosting、随机性来源

评估指标说明（多分类场景）：

    Accuracy（准确率）：
        预测正确的比例。在不均衡数据上可能误导
        例：如果90%数据是IT类，全猜IT也有90%准确率

    Precision（精确率）：
        预测为某类的样本中，真正是该类的比例
        = TP / (TP + FP)
        "说我适合这个岗位，我真的适合吗？"

    Recall（召回率）：
        某类的所有样本中，被正确找出的比例
        = TP / (TP + FN)
        "这个岗位的简历，都被我找到了吗？"

    F1-score：
        Precision和Recall的调和平均数
        = 2 * P * R / (P + R)
        综合衡量模型在每个类别上的表现

    Weighted Avg：
        按每个类别的样本数加权平均
        在不均衡数据上比macro avg更有参考价值

涉及的AI/ML知识点：
    1. 多分类模型训练
    2. 类别权重处理（class_weight='balanced'）
    3. 分类报告（Classification Report）
    4. 混淆矩阵（Confusion Matrix）
    5. 模型选择与比较
    6. 模型持久化（joblib）
"""
import numpy as np
import pandas as pd
import joblib
import json
import os
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)
import warnings
warnings.filterwarnings('ignore')

# 图表样式
sns.set_style("whitegrid")
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})


def load_features():
    """加载之前保存的特征矩阵和标签。"""
    print("[加载] 特征和标签...")
    X_train = joblib.load("models/X_train.pkl")
    X_test = joblib.load("models/X_test.pkl")
    y_train = joblib.load("models/y_train.pkl")
    y_test = joblib.load("models/y_test.pkl")

    print(f"  X_train: {X_train.shape}")
    print(f"  X_test:  {X_test.shape}")
    print(f"  y_train: {len(y_train)}")
    print(f"  y_test:  {len(y_test)}")

    return X_train, X_test, y_train, y_test


def train_logistic_regression(X_train, y_train, X_test, y_test):
    """
    训练逻辑回归模型。

    参数说明：
        multi_class='multinomial':
            使用softmax进行多分类（而非OVR一对多）
            softmax输出每个类别的概率，总和为1

        solver='saga':
            支持multinomial + L1/L2/elasticnet的大规模求解器
            saga在大规模稀疏数据上表现好

        max_iter=2000:
            最大迭代次数。TF-IDF特征维度高，需要更多迭代收敛

        class_weight='balanced':
            自动给少数类更高的权重
            公式: weight = n_samples / (n_classes * n_samples_per_class)
            BPO只有22条，权重 ≈ 2481 / (24*22) ≈ 4.7
            IT有120条，权重 ≈ 2481 / (24*120) ≈ 0.86
    """
    print("\n" + "=" * 60)
    print("[模型1] Logistic Regression")
    print("=" * 60)

    model = LogisticRegression(
        solver='saga',
        max_iter=2000,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    # 训练
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    # 预测
    y_pred = model.predict(X_test)

    # 评估
    metrics = evaluate_model(y_test, y_pred, model, train_time)
    return model, metrics


def train_linearsvc(X_train, y_train, X_test, y_test):
    """
    训练LinearSVC模型。

    为什么LinearSVC需要CalibratedClassifierCV包裹？

    问题：LinearSVC本身不输出概率（没有predict_proba方法）
         我们Flask页面需要显示"80%匹配IT岗位"这样的置信度
         没有概率就没法做这个功能

    解决：CalibratedClassifierCV
         用 Platt Scaling（一种后处理校准方法）
         将SVM的决策函数值 → 转化为0~1之间的概率
         原理类似把分数映射到sigmoid曲线上

    参数说明：
        C=1.0:
            正则化强度的倒数。C越小，正则化越强
            在高维TF-IDF特征上，默认C=1.0通常表现不错

        dual=False:
            n_samples > n_features时推荐False
            我们1984样本 vs 5000特征，所以False

        class_weight='balanced':
            同逻辑回归，处理类别不均衡
    """
    print("\n" + "=" * 60)
    print("[模型2] LinearSVC")
    print("=" * 60)

    # 基础SVM模型
    svm = LinearSVC(
        C=1.0,
        dual=False,
        class_weight='balanced',
        random_state=42,
        max_iter=3000,
    )

    # 用CalibratedClassifierCV包裹，使其能输出概率
    model = CalibratedClassifierCV(
        svm,
        method='sigmoid',  # Platt Scaling
        cv=3               # 3折交叉验证做校准
    )

    # 训练
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    # 预测
    y_pred = model.predict(X_test)

    # 评估
    metrics = evaluate_model(y_test, y_pred, model, train_time)
    return model, metrics


def train_random_forest(X_train, y_train, X_test, y_test):
    """
    训练随机森林模型。

    随机森林原理（面试高频考点）：
        1. Bootstrap采样：从训练集中有放回地随机抽取N个样本
        2. 每棵树只用随机选的一部分特征（max_features）
        3. 训练多棵决策树，每棵树不做剪枝（充分生长）
        4. 预测时：所有树投票，少数服从多数

    两个随机性来源（面试重点）：
        - 样本随机：每棵树用不同的Bootstrap采样
        - 特征随机：每个节点分裂时只考虑部分特征
        → 两个随机性 → 树之间相关性低 → 集成效果好

    参数说明：
        n_estimators=200:
            200棵树。越多越稳定，但训练越慢

        max_depth=50:
            限制树深度防止过拟合
            TF-IDF的5000个特征很多是噪音，深度太大会学到噪音

        min_samples_split=5:
            节点至少5个样本才能继续分裂
            防止树对个别样本过拟合

        class_weight='balanced':
            同前两个模型
    """
    print("\n" + "=" * 60)
    print("[模型3] Random Forest")
    print("=" * 60)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=50,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=0
    )

    # 训练
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    # 预测
    y_pred = model.predict(X_test)

    # 评估
    metrics = evaluate_model(y_test, y_pred, model, train_time)
    return model, metrics


def evaluate_model(y_true, y_pred, model, train_time):
    """
    全面评估模型性能。

    返回的指标字典包含：
        - accuracy: 整体准确率
        - precision_weighted: 加权精确率
        - recall_weighted: 加权召回率
        - f1_weighted: 加权F1分数
        - train_time: 训练耗时
        - classification_report: 完整分类报告(dict)
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # 完整的分类报告
    report = classification_report(y_true, y_pred, output_dict=True,
                                   zero_division=0)

    metrics = {
        'accuracy': round(acc, 4),
        'precision_weighted': round(prec, 4),
        'recall_weighted': round(rec, 4),
        'f1_weighted': round(f1, 4),
        'train_time_seconds': round(train_time, 2),
        'classification_report': report,
        'predictions': y_pred.tolist() if hasattr(y_pred, 'tolist') else list(y_pred),
    }

    print(f"\n  --- 评估结果 ---")
    print(f"  Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Precision: {prec:.4f}  (weighted avg)")
    print(f"  Recall:    {rec:.4f}  (weighted avg)")
    print(f"  F1-score:  {f1:.4f}  (weighted avg)")
    print(f"  训练耗时:  {train_time:.2f} 秒")

    return metrics


def plot_confusion_matrix(y_true, y_pred, title, save_path, label_names=None):
    """
    绘制混淆矩阵。

    混淆矩阵解读：
        - 第i行第j列：真实类别i被预测为类别j的样本数
        - 对角线：正确预测的样本
        - 非对角线：错误预测 → 可以看出模型把哪些类别搞混了

    为什么用百分比而不是绝对数量？
        各类别样本数差异大（120 vs 22）
        用百分比更公平地反映每个类别的分类准确率
    """
    cm = confusion_matrix(y_true, y_pred)

    # 按行归一化（每行之和 = 100%）
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    cm_normalized = np.nan_to_num(cm_normalized)

    # 只显示前12个类别（全部24个太密看不清）
    n_display = 12

    if label_names is None:
        tick_labels = [str(i) for i in range(n_display)]
    else:
        tick_labels = [label_names.get(str(i), str(i)) for i in range(n_display)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # 左图：原始数量
    sns.heatmap(cm[:n_display, :n_display],
                annot=True, fmt='d', cmap='Blues',
                xticklabels=tick_labels,
                yticklabels=tick_labels,
                ax=ax1, linewidths=0.5,
                cbar_kws={'label': 'Count'})
    ax1.set_title(f'{title} - Count (Top {n_display})', fontweight='bold')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('True')

    # 右图：百分比
    sns.heatmap(cm_normalized[:n_display, :n_display],
                annot=True, fmt='.0%', cmap='YlOrRd',
                xticklabels=tick_labels,
                yticklabels=tick_labels,
                ax=ax2, linewidths=0.5,
                vmin=0, vmax=1,
                cbar_kws={'label': 'Percentage'})
    ax2.set_title(f'{title} - Normalized (Top {n_display})', fontweight='bold')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('True')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  混淆矩阵已保存: {save_path}")


def plot_model_comparison(all_metrics, save_path):
    """绘制模型对比柱状图。"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    model_names = list(all_metrics.keys())
    colors = ['#3498db', '#2ecc71', '#e74c3c']

    # Accuracy对比
    accs = [all_metrics[m]['accuracy'] for m in model_names]
    axes[0].bar(model_names, accs, color=colors, edgecolor='white')
    axes[0].set_title('Accuracy Comparison', fontweight='bold')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_ylim(0, 1)
    for i, v in enumerate(accs):
        axes[0].text(i, v + 0.01, f'{v:.4f}', ha='center', fontweight='bold')

    # F1-score对比
    f1s = [all_metrics[m]['f1_weighted'] for m in model_names]
    axes[1].bar(model_names, f1s, color=colors, edgecolor='white')
    axes[1].set_title('F1-Score Comparison (Weighted)', fontweight='bold')
    axes[1].set_ylabel('F1-Score')
    axes[1].set_ylim(0, 1)
    for i, v in enumerate(f1s):
        axes[1].text(i, v + 0.01, f'{v:.4f}', ha='center', fontweight='bold')

    # 训练时间对比
    times = [all_metrics[m]['train_time_seconds'] for m in model_names]
    axes[2].bar(model_names, times, color=colors, edgecolor='white')
    axes[2].set_title('Training Time Comparison', fontweight='bold')
    axes[2].set_ylabel('Seconds')
    for i, v in enumerate(times):
        axes[2].text(i, v + 0.1, f'{v:.1f}s', ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  模型对比图已保存: {save_path}")


def select_best_model(all_models, all_metrics):
    """
    选择最佳模型。

    选择标准：F1-score (weighted)
        - 为什么不是Accuracy？ 因为数据不均衡
        - 为什么是Weighted不是Macro？ Weighted考虑了各类样本量
        - F1兼顾了Precision和Recall，单一指标太片面

    Returns:
        (best_model, best_model_name)
    """
    best_name = max(all_metrics, key=lambda k: all_metrics[k]['f1_weighted'])
    best_model = all_models[best_name]
    best_f1 = all_metrics[best_name]['f1_weighted']

    print("\n" + "=" * 60)
    print(f"最佳模型: {best_name}")
    print(f"F1-score (weighted): {best_f1:.4f}")
    print("=" * 60)

    return best_model, best_name


def save_best_model(model, model_name, metrics):
    """保存最佳模型及其元数据。"""
    os.makedirs("models", exist_ok=True)

    # 保存模型
    model_path = "models/best_model.pkl"
    joblib.dump(model, model_path)

    # 保存模型元信息
    meta = {
        "model_name": model_name,
        "metrics": {
            "accuracy": metrics['accuracy'],
            "precision_weighted": metrics['precision_weighted'],
            "recall_weighted": metrics['recall_weighted'],
            "f1_weighted": metrics['f1_weighted'],
            "train_time_seconds": metrics['train_time_seconds'],
        },
        "features": {
            "method": "TF-IDF",
            "max_features": 5000,
            "ngram_range": "(1, 2)",
            "vectorizer_path": "models/tfidf_vectorizer.pkl",
        },
        "data": {
            "total_samples": 2481,
            "num_categories": 24,
            "train_samples": 1984,
            "test_samples": 497,
        }
    }

    meta_path = "models/model_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\n  模型已保存: {model_path}")
    print(f"  元数据已保存: {meta_path}")


def main():
    """执行完整的模型训练和评估流程。"""
    print("=" * 60)
    print("简历分类器 — 模型训练与评估")
    print("=" * 60)

    # Step 1: 加载特征
    X_train, X_test, y_train, y_test = load_features()

    # Step 2: 训练3个模型
    all_models = {}
    all_metrics = {}

    # 模型1: Logistic Regression
    lr_model, lr_metrics = train_logistic_regression(
        X_train, y_train, X_test, y_test
    )
    all_models['LogisticRegression'] = lr_model
    all_metrics['LogisticRegression'] = lr_metrics

    # 模型2: LinearSVC
    svm_model, svm_metrics = train_linearsvc(
        X_train, y_train, X_test, y_test
    )
    all_models['LinearSVC'] = svm_model
    all_metrics['LinearSVC'] = svm_metrics

    # 模型3: Random Forest
    rf_model, rf_metrics = train_random_forest(
        X_train, y_train, X_test, y_test
    )
    all_models['RandomForest'] = rf_model
    all_metrics['RandomForest'] = rf_metrics

    # Step 3: 加载标签名（用于混淆矩阵）
    with open("models/label_map.json", "r") as f:
        label_map = json.load(f)

    # Step 4: 绘制混淆矩阵
    print("\n" + "=" * 60)
    print("生成混淆矩阵可视化")
    print("=" * 60)
    for name, metrics in all_metrics.items():
        plot_confusion_matrix(
            y_test, metrics['predictions'], name,
            f"static/confusion_matrix_{name}.png",
            label_map
        )

    # Step 5: 模型对比图
    plot_model_comparison(all_metrics, "static/model_comparison.png")

    # Step 6: 选择最佳模型
    best_model, best_name = select_best_model(all_models, all_metrics)
    best_metrics = all_metrics[best_name]

    # Step 7: 保存最佳模型
    save_best_model(best_model, best_name, best_metrics)

    # Step 8: 打印完整评估报告
    print("\n" + "=" * 60)
    print("完整模型评估总结")
    print("=" * 60)

    print(f"\n{'模型':<25} {'Accuracy':>10} {'Precision':>10} "
          f"{'Recall':>10} {'F1':>10} {'Time(s)':>10}")
    print("-" * 75)
    for name, m in all_metrics.items():
        marker = " <<<" if name == best_name else ""
        print(f"{name:<25} {m['accuracy']:>10.4f} {m['precision_weighted']:>10.4f} "
              f"{m['recall_weighted']:>10.4f} {m['f1_weighted']:>10.4f} "
              f"{m['train_time_seconds']:>10.2f}{marker}")

    print(f"\n结论: {best_name} 表现最佳，已保存为最佳模型。")
    print("=" * 60)


if __name__ == "__main__":
    main()
