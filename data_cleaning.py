"""
data_cleaning.py
================
功能：对简历数据集进行清洗和预处理。

清洗步骤（企业标准NLP数据预处理流程）：
    1. 缺失值检查与处理
    2. 重复数据检测
    3. 文本清洗：
       - 移除URL链接
       - 移除HTML标签
       - 移除特殊字符（保留有意义的标点）
       - 统一空白字符（换行、制表符 → 单个空格）
       - 去除多余空格
    4. 文本质量检查（空文本、异常短文本等）
    5. 类别标签编码（字符串 → 数字ID）
    6. 保存清洗后的数据

涉及的AI/ML知识点：
    1. 数据清洗（Data Cleaning）—— ML项目中占60-80%工作量的环节
    2. 正则表达式（Regex）—— 文本清洗的核心工具
    3. 标签编码（Label Encoding）—— 将分类标签转为模型可处理的数值
    4. 数据质量保证（Data Quality）—— 防止 Garbage In, Garbage Out

涉及的Python知识点：
    1. re 模块 —— 正则表达式
    2. pandas 数据处理
    3. sklearn.preprocessing.LabelEncoder —— 标签编码
"""
import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Tuple


def load_raw_data(filepath: str = "data/resume_dataset.csv") -> pd.DataFrame:
    """
    加载原始数据集。

    Args:
        filepath: CSV文件路径

    Returns:
        pd.DataFrame: 原始数据
    """
    df = pd.read_csv(filepath)
    print(f"[加载] 原始数据: {df.shape[0]} 行 × {df.shape[1]} 列")
    return df


def check_missing_values(df: pd.DataFrame) -> None:
    """
    检查并报告缺失值情况。

    Args:
        df: 数据框
    """
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100

    print("\n--- 缺失值检查 ---")
    if missing.sum() == 0:
        print("  [OK] 无缺失值")
    else:
        for col in df.columns:
            if missing[col] > 0:
                print(f"  [!!] {col}: {missing[col]} ({missing_pct[col]:.1f}%)")


def check_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    检查并移除重复数据。

    Args:
        df: 数据框

    Returns:
        pd.DataFrame: 去重后的数据框
    """
    n_before = len(df)
    df = df.drop_duplicates(subset=['Resume_str'], keep='first')
    n_after = len(df)
    n_dupes = n_before - n_after

    print(f"\n--- 重复值检查 ---")
    if n_dupes == 0:
        print("  [OK] 无重复数据")
    else:
        print(f"  [!!] 发现 {n_dupes} 条重复简历，已移除")
    return df


def clean_text(text: str) -> str:
    """
    清洗单条文本。

    清洗流程：
        1. 转小写（英文NLP标准做法，减少词汇表大小）
        2. 移除URL（http/https链接对分类无意义）
        3. 移除HTML标签
        4. 移除特殊字符，保留字母、数字和基本标点
        5. 合并空白字符（多个空格/换行 → 单个空格）
        6. 去除首尾空格

    Args:
        text: 原始文本字符串

    Returns:
        str: 清洗后的文本
    """
    if not isinstance(text, str):
        return ""

    # Step 1: 转小写
    # 为什么？英文中 "Engineer" 和 "engineer" 应被视为同一个词
    # 这步减少了词汇量，避免模型学习到大小写噪音
    text = text.lower()

    # Step 2: 移除URL
    # URL对岗位分类没有贡献，属于噪音
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)

    # Step 3: 移除HTML标签
    # 部分简历文本中混入了HTML标签（如<br>、<p>等）
    text = re.sub(r'<[^>]+>', ' ', text)

    # Step 4: 移除特殊字符
    # 保留：字母(a-z)、数字(0-9)、空格、基本标点(. , ; : ! ? -)
    # 移除：emoji、特殊符号、非英文字符等
    # 注意：这一步也会移除 @ # $ % 等，这些对岗位分类意义不大
    text = re.sub(r'[^a-z0-9\s.,;:!?\'\"()-]', ' ', text)

    # Step 5: 合并多个空白字符为单个空格
    # 换行符、制表符、多个连续空格 → 单个空格
    text = re.sub(r'\s+', ' ', text)

    # Step 6: 去除首尾空格
    text = text.strip()

    return text


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    清洗整个数据框的文本列。

    Args:
        df: 原始数据框

    Returns:
        pd.DataFrame: 清洗后的数据框
    """
    print("\n--- 文本清洗 ---")

    # 只使用 Resume_str 列（文本内容），Resume_html 是HTML格式的重复信息
    df = df[['Resume_str', 'Category']].copy()
    df.rename(columns={'Resume_str': 'resume_text'}, inplace=True)

    # 清洗每一行文本
    print("  正在清洗文本...")
    df['cleaned_text'] = df['resume_text'].apply(clean_text)

    # 清洗后检查
    empty_count = (df['cleaned_text'].str.strip() == '').sum()
    short_count = (df['cleaned_text'].str.len() < 50).sum()

    print(f"  清洗后空文本: {empty_count} 条")
    print(f"  清洗后过短文本(<50字符): {short_count} 条")

    # 移除清洗后为空的文本
    if empty_count > 0:
        df = df[df['cleaned_text'].str.strip() != '']
        print(f"  已移除 {empty_count} 条空文本")

    # 计算原始与清洗后的文本长度变化
    orig_len = df['resume_text'].str.len().mean()
    clean_len = df['cleaned_text'].str.len().mean()
    print(f"  原始平均长度: {orig_len:.0f} 字符")
    print(f"  清洗后平均长度: {clean_len:.0f} 字符")
    print(f"  缩减比例: {(1 - clean_len/orig_len) * 100:.1f}%")

    return df


def encode_labels(df: pd.DataFrame) -> Tuple[pd.DataFrame, LabelEncoder, dict]:
    """
    将类别标签编码为数值ID。

    为什么需要标签编码？
        机器学习模型（如Logistic Regression, Random Forest）
        只能处理数值型目标变量，不能直接处理字符串标签。
        例如："INFORMATION-TECHNOLOGY" → 0, "HR" → 1, ...

    Args:
        df: 包含 'Category' 列的数据框

    Returns:
        Tuple[pd.DataFrame, LabelEncoder, dict]:
            - 添加了 label 列的数据框
            - 训练好的 LabelEncoder 对象
            - {label_id: category_name} 映射字典
    """
    print("\n--- 标签编码 ---")

    le = LabelEncoder()
    df['label'] = le.fit_transform(df['Category'])

    # 构建映射字典
    label_map = {idx: category for idx, category in enumerate(le.classes_)}

    print(f"  类别总数: {len(le.classes_)}")
    print(f"  编码示例（前5个）:")
    for idx in range(min(5, len(le.classes_))):
        print(f"    {le.classes_[idx]} → {idx}")

    return df, le, label_map


def save_cleaned_data(df: pd.DataFrame, output_path: str = "data/cleaned_resume.csv") -> str:
    """
    保存清洗后的数据。

    Args:
        df: 清洗后的数据框
        output_path: 输出文件路径

    Returns:
        str: 保存路径
    """
    df.to_csv(output_path, index=False)
    file_size_kb = df.memory_usage(deep=True).sum() / 1024
    print(f"\n[保存] 清洗后数据: {output_path}")
    print(f"  行数: {len(df):,}, 列数: {len(df.columns)}")
    print(f"  内存占用: {file_size_kb:.0f} KB")
    return output_path


def cleaning_summary(df_raw: pd.DataFrame, df_clean: pd.DataFrame) -> None:
    """
    输出数据清洗总结报告。

    Args:
        df_raw: 原始数据框
        df_clean: 清洗后数据框
    """
    print("\n" + "=" * 60)
    print("数据清洗总结报告")
    print("=" * 60)

    print(f"\n{'指标':<25} {'清洗前':>12} {'清洗后':>12}")
    print("-" * 50)
    print(f"{'样本数':<25} {len(df_raw):>12,} {len(df_clean):>12,}")
    print(f"{'类别数':<25} {df_raw['Category'].nunique():>12} {df_clean['Category'].nunique():>12}")

    # 显示前3条清洗示例
    print("\n--- 清洗效果示例 ---")
    for i in range(min(3, len(df_clean))):
        orig = df_clean.iloc[i]['resume_text'][:120].replace('\n', ' ')
        clean = df_clean.iloc[i]['cleaned_text'][:120]
        cat = df_clean.iloc[i]['Category']
        print(f"\n  [{i+1}] 类别: {cat}")
        print(f"  原始: {orig}...")
        print(f"  清洗: {clean}...")


def main():
    """主函数：执行完整的数据清洗流程。"""
    print("=" * 60)
    print("简历数据集 — 数据清洗")
    print("=" * 60)

    # Step 1: 加载原始数据
    df = load_raw_data()

    # Step 2: 检查缺失值
    check_missing_values(df)

    # Step 3: 检查并移除重复
    df = check_duplicates(df)

    # Step 4: 清洗文本
    df = clean_dataframe(df)

    # Step 5: 编码类别标签
    df, label_encoder, label_map = encode_labels(df)

    # Step 6: 保存清洗结果
    save_cleaned_data(df)

    # Step 7: 打印清洗总结
    cleaning_summary(pd.read_csv("data/resume_dataset.csv"), df)

    # Step 8: 保存标签映射（后续Web展示需要）
    import json
    with open("models/label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"\n[保存] 标签映射: models/label_map.json")

    print("\n" + "=" * 60)
    print("数据清洗完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
