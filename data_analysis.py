"""
data_analysis.py
================
功能：对清洗后的简历数据进行多维度分析和可视化。

分析维度（企业AI项目标准EDA流程）：
    1. 类别分布分析 —— 检查数据是否均衡
    2. 文本长度分析 —— 了解数据质量分布
    3. 词频统计 —— 发现高频关键词
    4. 各类别特征词分析 —— 每个岗位的独特关键词
    5. 生成所有图表保存到 static/ 目录

涉及的AI/ML知识点：
    1. EDA（Exploratory Data Analysis，探索性数据分析）
       —— 建模前最重要的一步，理解数据特征
    2. 类别不平衡检测 —— 影响模型评估指标选择
    3. TF-IDF思想 —— 词频分析是TF-IDF的基础
    4. 文本分布分析 —— 决定模型是否需要处理长尾分布

涉及的Python知识点：
    1. matplotlib —— 绑图核心库
    2. seaborn —— 统计图表高级接口
    3. collections.Counter —— 词频统计
    4. matplotlib中文字体处理
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，服务器环境必备
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from typing import List, Dict, Tuple
import os
import json
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# 全局样式配置
# ============================================================
# 设置seaborn样式 —— 让图表更专业
sns.set_style("whitegrid")
sns.set_palette("husl")

# 设置matplotlib全局参数
plt.rcParams.update({
    'figure.dpi': 150,           # 高分辨率输出
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',     # 自动裁剪白边
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
})

# 输出目录
OUTPUT_DIR = "static"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. 类别分布分析
# ============================================================
def plot_category_distribution(df: pd.DataFrame) -> str:
    """
    绘制岗位类别分布柱状图。

    为什么做这个分析？
        - 检查数据是否类别均衡
        - 严重不均衡会导致模型偏向多数类
        - 决定是否需要过采样/欠采样/类别权重调整

    Args:
        df: 包含 'Category' 列的数据框

    Returns:
        str: 保存的图片路径
    """
    print("\n[1/5] 分析类别分布...")

    # 统计各类别数量，按降序排列
    cat_counts = df['Category'].value_counts()

    # 计算不均衡指标
    max_count = cat_counts.max()
    min_count = cat_counts.min()
    imbalance_ratio = max_count / min_count
    mean_count = cat_counts.mean()
    std_count = cat_counts.std()

    print(f"  最多: {cat_counts.idxmax()} ({max_count}条)")
    print(f"  最少: {cat_counts.idxmin()} ({min_count}条)")
    print(f"  不均衡比: {imbalance_ratio:.1f}:1")
    print(f"  均值: {mean_count:.0f}, 标准差: {std_count:.0f}")

    # 绘图
    fig, ax = plt.subplots(figsize=(14, 8))

    # 颜色渐变：数量越多颜色越深
    colors = plt.cm.Blues(0.3 + 0.7 * (cat_counts.values - min_count) /
                          (max_count - min_count))

    bars = ax.barh(range(len(cat_counts)), cat_counts.values, color=colors,
                   edgecolor='white', linewidth=0.5)

    # 标注数值
    for i, (count, cat) in enumerate(zip(cat_counts.values, cat_counts.index)):
        ax.text(count + 1, i, str(count), va='center', fontsize=9)

    # 标签和标题
    ax.set_yticks(range(len(cat_counts)))
    ax.set_yticklabels(cat_counts.index, fontsize=9)
    ax.invert_yaxis()  # 最大的在上面
    ax.set_xlabel('Sample Count', fontsize=12)
    ax.set_title(f'Resume Category Distribution\n'
                 f'(Total: {len(df):,} resumes, {df["Category"].nunique()} categories, '
                 f'Imbalance Ratio: {imbalance_ratio:.1f}:1)',
                 fontsize=14, fontweight='bold')

    # 添加均值线
    ax.axvline(mean_count, color='red', linestyle='--', alpha=0.5,
               label=f'Mean ({mean_count:.0f})')
    ax.legend(loc='lower right')

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "category_distribution.png")
    plt.savefig(save_path)
    plt.close()
    print(f"  图表已保存: {save_path}")

    return save_path


# ============================================================
# 2. 文本长度分析
# ============================================================
def plot_text_length_distribution(df: pd.DataFrame) -> str:
    """
    绘制文本长度分布直方图和箱线图。

    为什么做这个分析？
        - 文本长度影响模型输入处理
        - 过长文本可能需要截断
        - 过短文本可能信息不足
        - 了解数据分布有助于选择max_features等参数

    Args:
        df: 包含 'cleaned_text' 列的数据框

    Returns:
        str: 保存的图片路径
    """
    print("\n[2/5] 分析文本长度分布...")

    # 计算文本长度（字符数和词数）
    df_temp = df.copy()
    df_temp['char_len'] = df_temp['cleaned_text'].str.len()
    df_temp['word_count'] = df_temp['cleaned_text'].str.split().str.len()

    # 统计信息
    for metric, name in [('char_len', '字符'), ('word_count', '词')]:
        series = df_temp[metric]
        print(f"  平均{name}数: {series.mean():.0f}")
        print(f"  中位数{name}数: {series.median():.0f}")
        print(f"  最小{name}数: {series.min():.0f}")
        print(f"  最大{name}数: {series.max():.0f}")

    # 创建双图布局
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 图1: 字符长度直方图
    ax1 = axes[0, 0]
    ax1.hist(df_temp['char_len'], bins=50, color='steelblue', edgecolor='white',
             alpha=0.8)
    ax1.axvline(df_temp['char_len'].median(), color='red', linestyle='--',
                label=f"Median: {df_temp['char_len'].median():.0f}")
    ax1.set_xlabel('Character Count')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Character Length Distribution')
    ax1.legend()

    # 图2: 词数直方图
    ax2 = axes[0, 1]
    ax2.hist(df_temp['word_count'], bins=50, color='darkseagreen',
             edgecolor='white', alpha=0.8)
    ax2.axvline(df_temp['word_count'].median(), color='red', linestyle='--',
                label=f"Median: {df_temp['word_count'].median():.0f}")
    ax2.set_xlabel('Word Count')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Word Count Distribution')
    ax2.legend()

    # 图3: 各类别平均文本长度
    ax3 = axes[1, 0]
    cat_len = df_temp.groupby('Category')['char_len'].mean().sort_values()
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(cat_len)))
    ax3.barh(range(len(cat_len)), cat_len.values, color=colors,
             edgecolor='white')
    ax3.set_yticks(range(len(cat_len)))
    ax3.set_yticklabels(cat_len.index, fontsize=8)
    ax3.invert_yaxis()
    ax3.set_xlabel('Avg Characters')
    ax3.set_title('Average Resume Length by Category')

    # 图4: 文本长度箱线图(Top 10类别)
    ax4 = axes[1, 1]
    top10_cats = df_temp['Category'].value_counts().head(10).index
    df_top10 = df_temp[df_temp['Category'].isin(top10_cats)]
    df_top10_ordered = df_top10.sort_values('Category')
    data_for_box = [df_top10_ordered[df_top10_ordered['Category'] == cat]
                    ['char_len'].values for cat in top10_cats]
    ax4.boxplot(data_for_box, vert=False, showfliers=False)
    ax4.set_yticks(range(1, len(top10_cats) + 1))
    ax4.set_yticklabels(top10_cats, fontsize=8)
    ax4.set_xlabel('Character Count')
    ax4.set_title('Text Length Boxplot (Top 10 Categories)')

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "text_length_analysis.png")
    plt.savefig(save_path)
    plt.close()
    print(f"  图表已保存: {save_path}")

    return save_path


# ============================================================
# 3. 词频分析
# ============================================================
def get_top_words(texts: List[str], n: int = 30,
                  stop_words: set = None) -> List[Tuple[str, int]]:
    """
    统计所有文本中的高频词。

    Args:
        texts: 文本列表
        n: 返回前N个高频词
        stop_words: 停用词集合

    Returns:
        List[Tuple[str, int]]: [(单词, 频次), ...]
    """
    if stop_words is None:
        stop_words = set()

    all_words = []
    for text in texts:
        words = str(text).split()
        all_words.extend([w for w in words
                         if len(w) > 2  # 过滤过短词（a, an, is...)
                         and w not in stop_words])

    counter = Counter(all_words)
    return counter.most_common(n)


def plot_top_words(df: pd.DataFrame) -> str:
    """
    绘制高频词柱状图。

    为什么做这个分析？
        - 了解简历数据中哪些词最常见
        - 识别可能的停用词
        - 初步判断哪些词可能对分类有区分度

    Args:
        df: 包含 'cleaned_text' 列的数据框

    Returns:
        str: 保存的图片路径
    """
    print("\n[3/5] 分析高频词...")

    # 基础英文停用词（NLTK标准停用词的精简版）
    basic_stopwords = {
        'the', 'and', 'for', 'with', 'that', 'this', 'are', 'from',
        'was', 'have', 'has', 'had', 'not', 'but', 'you', 'all',
        'can', 'been', 'its', 'will', 'would', 'could', 'should',
        'may', 'been', 'who', 'what', 'when', 'where', 'which',
        'how', 'then', 'than', 'just', 'also', 'very', 'too',
        'any', 'some', 'each', 'every', 'both', 'few', 'more',
        'most', 'other', 'into', 'over', 'such', 'only', 'own',
        'same', 'new', 'now', 'well', 'one', 'two', 'per'
    }

    # 统计全局高频词
    all_texts = df['cleaned_text'].tolist()
    top_words = get_top_words(all_texts, n=30, stop_words=basic_stopwords)

    print(f"  Top 5 高频词: {top_words[:5]}")

    # 绘图
    fig, ax = plt.subplots(figsize=(12, 8))

    words, counts = zip(*reversed(top_words))  # 反转使最大的在上

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(words)))
    ax.barh(range(len(words)), counts, color=colors, edgecolor='white')

    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words, fontsize=10)
    ax.set_xlabel('Frequency', fontsize=12)
    ax.set_title('Top 30 Most Common Words in Resumes\n'
                 '(after removing basic stopwords)',
                 fontsize=14, fontweight='bold')

    # 在柱上标注数值
    for i, count in enumerate(counts):
        ax.text(count + 50, i, str(count), va='center', fontsize=8)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "top_words.png")
    plt.savefig(save_path)
    plt.close()
    print(f"  图表已保存: {save_path}")

    return save_path


# ============================================================
# 4. 各类别特征词分析
# ============================================================
def plot_category_keywords(df: pd.DataFrame, top_n: int = 5) -> str:
    """
    找出每个岗位类别最有区分度的关键词，绘制热力图。

    为什么要做这个分析？
        - 验证每个类别的文本是否有明显的词差异
        - 如果各类别用词高度重叠，分类难度大
        - 为特征工程提供方向

    方法：
        对每个类别，使用TF-IDF思想找出该类别特有的高频词
        即：在该类中出现频繁，但在其他类中不常见的词

    Args:
        df: 包含 'Category' 和 'cleaned_text' 列的数据框
        top_n: 每个类别展示前N个特征词

    Returns:
        str: 保存的图片路径
    """
    print(f"\n[4/5] 分析各类别特征词...")

    # 构建各类别的词频字典
    categories = sorted(df['Category'].unique())
    cat_word_freq = {}  # {category: Counter}
    global_word_freq = Counter()  # 全局词频

    for cat in categories:
        cat_texts = df[df['Category'] == cat]['cleaned_text'].tolist()
        cat_words = []
        for text in cat_texts:
            words = [w for w in str(text).split() if len(w) > 2]
            cat_words.extend(words)

        cat_word_freq[cat] = Counter(cat_words)
        global_word_freq.update(cat_words)

    # 计算每个类别每个词的 TF-IDF 风格得分
    # Score = (词在该类中的频率) / (词在全局的频率 + 1)
    # 这样能找出该类别特有的词，而非全局通用词
    cat_keywords = {}
    for cat in categories:
        scores = {}
        for word, freq in cat_word_freq[cat].most_common(200):
            global_freq = global_word_freq.get(word, 0) + 1
            # 特有词得分 = 类内频率 * log(总类别数 / 出现该词的类别数)
            n_cats_with_word = sum(1 for c in categories
                                   if word in cat_word_freq[c])
            tfidf_like = freq * np.log(len(categories) / (n_cats_with_word + 1))
            scores[word] = tfidf_like

        top_keywords = sorted(scores.items(), key=lambda x: x[1],
                              reverse=True)[:top_n]
        cat_keywords[cat] = [w for w, _ in top_keywords]

    # 打印示例
    print("  各类别Top特征词示例:")
    for cat in list(categories)[:5]:
        print(f"    {cat}: {', '.join(cat_keywords[cat][:5])}")

    # 生成热力图数据
    # 选择展示前12个类别（热力图太大看不清）
    selected_cats = sorted(categories, key=lambda c: len(df[df['Category'] == c]),
                           reverse=True)[:12]
    all_kw = []
    for cat in selected_cats:
        all_kw.extend(cat_keywords[cat][:3])
    all_kw = list(dict.fromkeys(all_kw))  # 去重保序

    # 构建热力图矩阵
    heatmap_data = np.zeros((len(selected_cats), len(all_kw)))
    for i, cat in enumerate(selected_cats):
        for j, word in enumerate(all_kw):
            heatmap_data[i, j] = cat_word_freq[cat].get(word, 0)

    # 归一化（按行）
    row_sums = heatmap_data.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # 防止除零
    heatmap_norm = heatmap_data / row_sums

    # 绘图
    fig, ax = plt.subplots(figsize=(16, 8))
    sns.heatmap(heatmap_norm,
                xticklabels=all_kw,
                yticklabels=selected_cats,
                cmap='YlOrRd',
                annot=False,
                linewidths=0.5,
                linecolor='white',
                ax=ax,
                cbar_kws={'label': 'Normalized Frequency'})

    ax.set_title('Category-Specific Keywords Heatmap (Top 12 Categories)',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Keywords', fontsize=12)
    ax.set_ylabel('Job Category', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=9)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "category_keywords_heatmap.png")
    plt.savefig(save_path)
    plt.close()
    print(f"  图表已保存: {save_path}")

    return save_path


# ============================================================
# 5. 数据概览统计表
# ============================================================
def generate_statistics_summary(df: pd.DataFrame) -> str:
    """
    生成数据统计摘要，保存为JSON。

    Args:
        df: 数据框

    Returns:
        str: 保存路径
    """
    print("\n[5/5] 生成统计摘要...")

    stats = {
        "total_samples": len(df),
        "num_categories": df['Category'].nunique(),
        "categories": df['Category'].value_counts().to_dict(),
        "text_stats": {
            "avg_chars": round(df['cleaned_text'].str.len().mean(), 1),
            "median_chars": round(df['cleaned_text'].str.len().median(), 1),
            "min_chars": int(df['cleaned_text'].str.len().min()),
            "max_chars": int(df['cleaned_text'].str.len().max()),
            "avg_words": round(df['cleaned_text'].str.split().str.len().mean(), 1),
            "median_words": round(df['cleaned_text'].str.split().str.len().median(), 1),
        },
        "imbalance_ratio": round(
            df['Category'].value_counts().max() /
            df['Category'].value_counts().min(), 1
        ),
        "cleaning_info": {
            "original_rows": 2484,
            "cleaned_rows": len(df),
            "removed_rows": 2484 - len(df),
            "text_size_reduction_pct": "6.2%",
        }
    }

    save_path = os.path.join(OUTPUT_DIR, "data_statistics.json")
    with open(save_path, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"  统计摘要已保存: {save_path}")
    return save_path


# ============================================================
# 主流程
# ============================================================
def main():
    """执行完整的数据分析流程。"""
    print("=" * 60)
    print("简历数据集 — 探索性数据分析 (EDA)")
    print("=" * 60)

    # 加载清洗后的数据
    df = pd.read_csv("data/cleaned_resume.csv")
    print(f"\n数据加载完成: {len(df):,} 条简历, "
          f"{df['Category'].nunique()} 个类别")

    # 执行5个分析维度
    plot_category_distribution(df)
    plot_text_length_distribution(df)
    plot_top_words(df)
    plot_category_keywords(df)
    generate_statistics_summary(df)

    # 总结
    print("\n" + "=" * 60)
    print("数据分析完成！所有图表已保存到 static/ 目录")
    print("=" * 60)
    print("\n生成的文件列表:")
    for f in os.listdir(OUTPUT_DIR):
        filepath = os.path.join(OUTPUT_DIR, f)
        size = os.path.getsize(filepath) / 1024
        print(f"  static/{f} ({size:.1f} KB)")


if __name__ == "__main__":
    main()
