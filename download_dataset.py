"""
download_dataset.py
===================
功能：从Kaggle自动下载简历分类数据集，并复制到项目 data/ 目录。

数据集信息：
    - 名称：snehaanbhawal/resume-dataset
    - 样本数：约2,500条简历
    - 类别数：25个岗位类别
    - 来源：Kaggle公开数据集

涉及的AI/ML知识点：
    1. 数据获取（Data Acquisition）—— ML项目流程的第一步
    2. Kaggle API认证机制
    3. 数据集缓存机制（避免重复下载）
"""
import os
import shutil
import kagglehub


def download_resume_dataset(output_dir: str = "data") -> str:
    """
    从Kaggle下载简历数据集并复制到项目目录。

    Args:
        output_dir: 项目内数据存储目录，默认为 data/

    Returns:
        str: 数据集CSV文件的本地路径
    """
    print("=" * 60)
    print("开始下载简历数据集...")
    print("=" * 60)

    # Step 1: 使用 kagglehub 下载数据集（自动使用缓存的认证信息）
    # kagglehub 会自动检查 ~/.kaggle/kaggle.json 进行认证
    # 如果数据集已下载过，会直接使用缓存，不会重复下载
    print("\n[1/3] 正在从Kaggle下载数据集...")
    dataset_path = kagglehub.dataset_download(
        "snehaanbhawal/resume-dataset"
    )
    print(f"  缓存路径: {dataset_path}")

    # Step 2: 定位CSV文件
    # 数据集解压后，CSV文件在 Resume/Resume.csv
    csv_source = os.path.join(dataset_path, "Resume", "Resume.csv")
    if not os.path.exists(csv_source):
        raise FileNotFoundError(f"未找到CSV文件: {csv_source}")

    print(f"\n[2/3] 找到CSV文件: {csv_source}")

    # Step 3: 复制到项目 data/ 目录
    os.makedirs(output_dir, exist_ok=True)
    csv_dest = os.path.join(output_dir, "resume_dataset.csv")

    shutil.copy2(csv_source, csv_dest)
    file_size_mb = os.path.getsize(csv_dest) / (1024 * 1024)
    print(f"\n[3/3] 数据集已复制到: {csv_dest}")
    print(f"  文件大小: {file_size_mb:.2f} MB")

    return csv_dest


def preview_dataset(csv_path: str) -> None:
    """
    预览数据集的基本信息。

    Args:
        csv_path: CSV文件路径
    """
    import pandas as pd

    print("\n" + "=" * 60)
    print("数据集预览")
    print("=" * 60)

    df = pd.read_csv(csv_path)

    print(f"\n总样本数: {len(df):,}")
    print(f"特征列: {list(df.columns)}")
    print(f"岗位类别数: {df['Category'].nunique()}")

    print("\n--- 各类别样本分布 ---")
    print(df['Category'].value_counts().to_string())

    print("\n--- 前3条简历摘要 ---")
    for i in range(3):
        text = str(df.iloc[i]['Resume_str']).replace('\n', ' ')[:150]
        cat = df.iloc[i]['Category']
        print(f"\n  [{i+1}] 类别={cat}")
        print(f"  内容: {text}...")


if __name__ == "__main__":
    # 下载数据集
    csv_path = download_resume_dataset()

    # 预览数据集
    preview_dataset(csv_path)

    print("\n" + "=" * 60)
    print("数据集下载完成！")
    print("=" * 60)
