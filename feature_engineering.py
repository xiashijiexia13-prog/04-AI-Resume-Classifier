"""
feature_engineering.py
======================
功能：文本向量化与特征工程 —— 将文本转换为机器学习模型可处理的数值特征。

处理流程（企业标准NLP特征工程Pipeline）：
    1. 加载清洗后的数据
    2. 划分训练集/测试集（80/20，分层抽样保持类别比例）
    3. TF-IDF向量化（文本 → 数值矩阵）
    4. 保存TF-IDF向量器（预测时需要复用）
    5. 输出特征矩阵信息

核心概念：TF-IDF（Term Frequency - Inverse Document Frequency）

    为什么要用TF-IDF而不是简单的词频（CountVectorizer）？
        - 词频只统计"这个词出现了几次"
        - TF-IDF额外考虑"这个词在多少文档中出现"
        - 高频但到处都出现的词（如 "the", "and"）→ TF-IDF得分低
        - 高频但只在少数文档出现的词（如 "ledger", "reconciliations"）→ TF-IDF得分高
        - 这正是我们需要的：找到对特定岗位有区分度的词！

    TF-IDF公式：
        TF(t,d)  = 词t在文档d中出现的次数 / 文档d的总词数
        IDF(t)   = log(总文档数 / 包含词t的文档数)
        TF-IDF   = TF × IDF

涉及的AI/ML知识点：
    1. 文本向量化（Text Vectorization） —— NLP的基础操作
    2. TF-IDF原理 —— 信息检索领域的经典算法
    3. 训练集/测试集划分 —— 防止过拟合，评估模型泛化能力
    4. 分层抽样（Stratified Split） —— 保持类别分布，解决不均衡问题
    5. 停用词（Stop Words） —— 过滤无意义高频词

涉及的Python知识点：
    1. sklearn.feature_extraction.text.TfidfVectorizer
    2. sklearn.model_selection.train_test_split
    3. scipy.sparse —— 稀疏矩阵存储（节省内存）
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from typing import Tuple
import os


def load_cleaned_data(filepath: str = "data/cleaned_resume.csv") -> pd.DataFrame:
    """
    加载清洗后的数据。

    Args:
        filepath: 清洗后CSV文件路径

    Returns:
        pd.DataFrame
    """
    df = pd.read_csv(filepath)
    print(f"[加载] 清洗后数据: {len(df):,} 条记录")
    print(f"  列: {list(df.columns)}")
    return df


def split_train_test(
    df: pd.DataFrame,
    text_col: str = "cleaned_text",
    label_col: str = "label",
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[list, list, list, list]:
    """
    划分训练集和测试集。

    关键参数 stratify：
        - 使用分层抽样，保证训练集和测试集中各类别比例一致
        - 如果不用stratify，可能出现测试集中某个类别完全没有样本
        - 在类别不均衡场景下尤其重要（我们的数据5.5:1不均衡）

    为什么random_state=42？
        - 固定随机种子，确保每次运行结果一致
        - 42是ML社区的非官方"梗"（《银河系漫游指南》）
        - 企业项目中必须固定种子，否则结果不可复现

    Args:
        df: 数据框
        text_col: 文本列名
        label_col: 标签列名
        test_size: 测试集比例（默认20%）
        random_state: 随机种子

    Returns:
        Tuple[list, list, list, list]:
            X_train, X_test, y_train, y_test
    """
    print(f"\n[划分] 训练集/测试集 (test_size={test_size})...")

    X = df[text_col].tolist()
    y = df[label_col].tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # 分层抽样：保持各类别在训练/测试集中的比例
    )

    print(f"  训练集: {len(X_train):,} 条 ({len(X_train)/len(df)*100:.0f}%)")
    print(f"  测试集: {len(X_test):,} 条 ({len(X_test)/len(df)*100:.0f}%)")

    # 验证分层抽样效果：检查各类别在两集合中的比例
    from collections import Counter
    train_dist = Counter(y_train)
    test_dist = Counter(y_test)

    print(f"\n  分层抽样验证（示例前3类）:")
    total = len(df)
    for label_id in sorted(train_dist.keys())[:3]:
        train_pct = train_dist[label_id] / len(y_train) * 100
        test_pct = test_dist[label_id] / len(y_test) * 100
        orig_pct = (train_dist[label_id] + test_dist[label_id]) / total * 100
        print(f"    类别{label_id}: 原始{orig_pct:.1f}%, "
              f"训练{train_pct:.1f}%, 测试{test_pct:.1f}%")

    return X_train, X_test, y_train, y_test


def create_tfidf_features(
    X_train: list,
    X_test: list,
    max_features: int = 5000,
    ngram_range: Tuple[int, int] = (1, 2),
    max_df: float = 0.7,
    min_df: int = 2
) -> Tuple[np.ndarray, np.ndarray, TfidfVectorizer]:
    """
    使用TF-IDF将文本转换为数值特征矩阵。

    参数选择依据（基于前面EDA的发现）：

    max_features=5000：
        - 我们的数据有24个类别、2,481条简历
        - 5000个特征足够覆盖各类别的区分性词汇
        - 太多特征 → 训练慢、容易过拟合
        - 太少特征 → 丢失重要信息

    ngram_range=(1, 2)：
        - unigrams(1-gram): 单个词，如 "machine", "learning"
        - bigrams(2-gram):  连续两个词，如 "machine learning", "project management"
        - bigrams能捕获短语语义，通常提升1-3%准确率

    max_df=0.7：
        - 忽略在70%以上文档中出现的词
        - 这些词太普遍，没有区分度
        - 如我们的数据中 "state", "city", "company" 可能被过滤

    min_df=2：
        - 忽略只在1个文档中出现的词
        - 这些词太罕见，模型无法学习其规律
        - 也避免拼写错误等噪音

    Args:
        X_train: 训练集文本列表
        X_test: 测试集文本列表
        max_features: 最大特征数
        ngram_range: n-gram范围
        max_df: 最大文档频率阈值
        min_df: 最小文档频率阈值

    Returns:
        Tuple[np.ndarray, np.ndarray, TfidfVectorizer]:
            X_train_tfidf, X_test_tfidf, tfidf_vectorizer
    """
    print(f"\n[TF-IDF] 文本向量化...")
    print(f"  max_features={max_features}")
    print(f"  ngram_range={ngram_range}")
    print(f"  max_df={max_df}")
    print(f"  min_df={min_df}")

    # 创建TF-IDF向量器
    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        max_df=max_df,
        min_df=min_df,
        stop_words='english',     # 自动去除英文停用词
        sublinear_tf=True,        # 使用 1+log(TF) 代替原始TF，抑制高频词
        strip_accents='unicode',  # 去除重音符号
        lowercase=True            # 再次确保小写（防御性编程）
    )

    # 在训练集上fit（学习词汇表），然后transform（转换为矩阵）
    # 重要：测试集只能用transform，不能用fit！
    # 原因：如果测试集参与fit，就等于"偷看答案"，泛化能力评估就不准了
    print("\n  正在fit训练集词汇表并转换为TF-IDF矩阵...")
    X_train_tfidf = tfidf.fit_transform(X_train)

    print("  正在transform测试集（使用训练集的词汇表）...")
    X_test_tfidf = tfidf.transform(X_test)

    # 输出特征矩阵信息
    print(f"\n  === TF-IDF特征矩阵 ===")
    print(f"  训练集矩阵: {X_train_tfidf.shape}")
    print(f"  测试集矩阵: {X_test_tfidf.shape}")
    print(f"  词汇表大小: {len(tfidf.vocabulary_):,}")

    # 稀疏矩阵密度（非零值比例）
    sparsity = 1.0 - (X_train_tfidf.nnz /
                       (X_train_tfidf.shape[0] * X_train_tfidf.shape[1]))
    print(f"  稀疏度: {sparsity:.2%} (文本数据的典型特征)")

    # 内存占用估算
    memory_mb = X_train_tfidf.data.nbytes / (1024 * 1024)
    print(f"  训练集内存: {memory_mb:.1f} MB")

    # 展示TF-IDF特征词Top10（按IDF排序，IDF越大说明越稀有=越可能有区分度）
    print("\n  === TF-IDF词汇示例 ===")
    feature_names = tfidf.get_feature_names_out()
    # 找出IDF最高的前10个词（最稀有的词）
    idf_scores = tfidf.idf_
    top_idf_indices = np.argsort(idf_scores)[-10:][::-1]
    print("  Top 10 稀有词汇（IDF最高 → 区分度可能最高）:")
    for idx in top_idf_indices[:10]:
        print(f"    {feature_names[idx]:<25} IDF={idf_scores[idx]:.2f}")

    # 展示全局最高TF-IDF值的词
    tfidf_means = np.array(X_train_tfidf.mean(axis=0)).flatten()
    top_tfidf_indices = np.argsort(tfidf_means)[-10:][::-1]
    print("\n  Top 10 平均TF-IDF最高词汇:")
    for idx in top_tfidf_indices[:10]:
        print(f"    {feature_names[idx]:<25} Avg_TFIDF={tfidf_means[idx]:.6f}")

    return X_train_tfidf, X_test_tfidf, tfidf


def save_vectorizer(tfidf: TfidfVectorizer, output_dir: str = "models") -> str:
    """
    保存TF-IDF向量器，供后续预测使用。

    为什么要保存？
        模型训练时的词汇表和IDF值必须与预测时完全一致
        否则：训练时 "accountant" 是第5号特征
              预测时 "accountant" 是第3号特征 → 模型输入错位，结果全错

    Args:
        tfidf: 训练好的TF-IDF向量器
        output_dir: 保存目录

    Returns:
        str: 保存路径
    """
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "tfidf_vectorizer.pkl")
    joblib.dump(tfidf, save_path)

    file_size = os.path.getsize(save_path) / 1024
    print(f"\n[保存] TF-IDF向量器: {save_path} ({file_size:.0f} KB)")
    return save_path


def main():
    """执行完整的特征工程流程。"""
    print("=" * 60)
    print("简历数据集 — 特征工程 (TF-IDF)")
    print("=" * 60)

    # Step 1: 加载数据
    df = load_cleaned_data()

    # Step 2: 划分训练/测试集
    X_train, X_test, y_train, y_test = split_train_test(df)

    # Step 3: TF-IDF特征提取
    X_train_tfidf, X_test_tfidf, tfidf = create_tfidf_features(
        X_train, X_test
    )

    # Step 4: 保存向量器（模型预测时必须用到）
    save_vectorizer(tfidf)

    # Step 5: 保存划分后的数据（供训练模块使用）
    print("\n[保存] 特征矩阵和标签...")
    joblib.dump(X_train_tfidf, "models/X_train.pkl")
    joblib.dump(X_test_tfidf, "models/X_test.pkl")
    joblib.dump(y_train, "models/y_train.pkl")
    joblib.dump(y_test, "models/y_test.pkl")
    print(f"  models/X_train.pkl ({os.path.getsize('models/X_train.pkl')/1024:.0f} KB)")
    print(f"  models/X_test.pkl  ({os.path.getsize('models/X_test.pkl')/1024:.0f} KB)")
    print(f"  models/y_train.pkl ({os.path.getsize('models/y_train.pkl')/1024:.0f} KB)")
    print(f"  models/y_test.pkl  ({os.path.getsize('models/y_test.pkl')/1024:.0f} KB)")

    print("\n" + "=" * 60)
    print("特征工程完成！")
    print("=" * 60)
    print(f"\n数据流向: 文本({len(X_train)}+{len(X_test)}条)")
    print(f"  → TF-IDF向量化")
    print(f"  → 训练集矩阵: {X_train_tfidf.shape}")
    print(f"  → 测试集矩阵: {X_test_tfidf.shape}")
    print(f"  → 准备好了，可以开始模型训练!")
    print(f"\n所有中间文件已保存到 models/ 目录")


if __name__ == "__main__":
    main()
