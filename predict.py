"""
predict.py
==========
功能：加载训练好的最佳模型，对新简历文本进行岗位类别预测。

预测流程（企业ML推理Pipeline）：
    1. 加载TF-IDF向量器（必须与训练时一致）
    2. 加载最佳分类模型
    3. 加载标签映射（数字ID → 岗位名称）
    4. 文本清洗（与训练时保持完全一致的清洗逻辑）
    5. TF-IDF向量化
    6. 模型预测 → 输出岗位类别和置信度

为什么预测流程也必须和训练一致？
    - 训练时对文本做了"转小写→去URL→去特殊字符→TF-IDF"
    - 预测时也必须完全相同的处理步骤
    - 任何差异都会导致特征空间不匹配 → 预测结果错误

涉及的AI/ML知识点：
    1. 模型持久化与加载（joblib）
    2. 推理Pipeline（Inference Pipeline）
    3. 概率校准（CalibratedClassifierCV）
    4. 模型版本管理
"""
import re
import joblib
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import os


class ResumeClassifier:
    """
    简历岗位分类器。

    封装了完整的推理Pipeline：
        文本清洗 → TF-IDF向量化 → 模型预测 → 结果格式化

    使用示例：
        classifier = ResumeClassifier()
        result = classifier.predict("5 years experience in Python...")
        print(result['category'])     # 'INFORMATION-TECHNOLOGY'
        print(result['confidence'])   # 0.85
    """

    def __init__(self, model_dir: str = "models"):
        """
        初始化分类器：加载向量器、模型和标签映射。

        Args:
            model_dir: 模型文件所在目录
        """
        self.model_dir = model_dir

        # 加载TF-IDF向量器
        vectorizer_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")
        if not os.path.exists(vectorizer_path):
            raise FileNotFoundError(
                f"TF-IDF向量器未找到: {vectorizer_path}\n"
                "请先运行 feature_engineering.py 生成向量器"
            )
        self.vectorizer = joblib.load(vectorizer_path)
        print(f"[加载] TF-IDF向量器: {vectorizer_path}")

        # 加载最佳模型
        model_path = os.path.join(model_dir, "best_model.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"最佳模型未找到: {model_path}\n"
                "请先运行 train.py 训练模型"
            )
        self.model = joblib.load(model_path)
        print(f"[加载] 最佳模型: {model_path}")

        # 加载标签映射
        label_map_path = os.path.join(model_dir, "label_map.json")
        if not os.path.exists(label_map_path):
            raise FileNotFoundError(
                f"标签映射未找到: {label_map_path}"
            )
        with open(label_map_path, "r") as f:
            # JSON的key是字符串，转为int
            raw_map = json.load(f)
            self.label_map = {int(k): v for k, v in raw_map.items()}
        print(f"[加载] 标签映射: {len(self.label_map)} 个类别")

        # 加载模型元数据
        meta_path = os.path.join(model_dir, "model_metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                self.metadata = json.load(f)
            print(f"[加载] 模型元数据: {self.metadata['model_name']}, "
                  f"F1={self.metadata['metrics']['f1_weighted']:.4f}")

        print(f"\n分类器初始化完成！")

    def clean_text(self, text: str) -> str:
        """
        清洗输入文本（必须与data_cleaning.py中的clean_text完全一致）。

        Args:
            text: 原始简历文本

        Returns:
            str: 清洗后的文本
        """
        if not isinstance(text, str):
            return ""

        # 转小写
        text = text.lower()

        # 移除URL
        text = re.sub(r'https?://\S+|www\.\S+', ' ', text)

        # 移除HTML标签
        text = re.sub(r'<[^>]+>', ' ', text)

        # 移除特殊字符（保留字母、数字、基本标点）
        text = re.sub(r'[^a-z0-9\s.,;:!?\'\"()-]', ' ', text)

        # 合并空白字符
        text = re.sub(r'\s+', ' ', text)

        # 去首尾空格
        text = text.strip()

        return text

    def predict(self, resume_text: str, top_k: int = 3) -> Dict:
        """
        预测简历对应的岗位类别。

        Args:
            resume_text: 简历文本内容
            top_k: 返回概率最高的K个类别

        Returns:
            Dict: {
                'category': str,        # 最佳匹配岗位
                'confidence': float,    # 置信度 (0~1)
                'top_k': [              # Top-K个可能岗位
                    {'category': str, 'confidence': float},
                    ...
                ],
                'cleaned_text': str,    # 清洗后的文本（调试用）
            }
        """
        # Step 1: 文本清洗
        cleaned = self.clean_text(resume_text)

        # 输入验证
        if len(cleaned) < 50:
            return {
                'error': True,
                'message': '简历文本过短（<50字符），请提供更完整的简历内容。',
            }

        # Step 2: TF-IDF向量化
        # 只transform，不fit！使用训练集的词汇表
        tfidf_matrix = self.vectorizer.transform([cleaned])

        # Step 3: 模型预测概率
        # predict_proba返回每个类别的概率，shape为(1, 24)
        probabilities = self.model.predict_proba(tfidf_matrix)[0]

        # 获取概率最高的K个类别索引
        top_k_indices = np.argsort(probabilities)[-top_k:][::-1]

        # Step 4: 格式化结果
        top_category_id = top_k_indices[0]
        top_confidence = probabilities[top_category_id]

        top_k_results = []
        for idx in top_k_indices:
            top_k_results.append({
                'category': self.label_map[idx],
                'confidence': round(float(probabilities[idx]), 4),
                'label_id': int(idx),
            })

        return {
            'error': False,
            'category': self.label_map[top_category_id],
            'confidence': round(float(top_confidence), 4),
            'top_k': top_k_results,
            'cleaned_text': cleaned[:300] + '...' if len(cleaned) > 300 else cleaned,
        }

    def batch_predict(self, texts: List[str]) -> pd.DataFrame:
        """
        批量预测多份简历。

        Args:
            texts: 简历文本列表

        Returns:
            pd.DataFrame: 包含原始文本、清洗文本、预测结果、置信度的DataFrame
        """
        results = []
        for text in texts:
            result = self.predict(text)
            results.append(result)

        df = pd.DataFrame(results)
        return df


def main():
    """演示预测功能。"""
    print("=" * 60)
    print("简历分类器 — 预测演示")
    print("=" * 60)

    # 初始化分类器
    classifier = ResumeClassifier()

    # 测试案例1: IT类简历
    test_resume_1 = """
    SOFTWARE ENGINEER

    Summary:
    Experienced software engineer with 5+ years of experience in Python,
    Java, and cloud technologies. Strong background in machine learning
    and data engineering.

    Technical Skills:
    - Python, Java, JavaScript, SQL
    - AWS, Docker, Kubernetes
    - TensorFlow, PyTorch, Scikit-learn
    - Git, CI/CD, Agile methodologies

    Experience:
    Senior Software Engineer | ABC Tech Corp | 2019-2024
    - Developed microservices architecture handling 1M+ requests/day
    - Built ML pipeline for customer churn prediction
    - Led migration from monolith to Kubernetes-based microservices
    - Improved system performance by 40% through optimization

    Junior Developer | XYZ Solutions | 2017-2019
    - Developed REST APIs using Python Flask and Django
    - Implemented database schemas and optimized SQL queries
    - Collaborated in Agile team of 8 developers

    Education:
    BS in Computer Science, Stanford University
    """

    # 测试案例2: Healthcare类简历
    test_resume_2 = """
    REGISTERED NURSE

    Summary:
    Compassionate registered nurse with 8 years of experience in critical
    care and emergency medicine. Strong patient advocacy skills and
    excellent clinical judgment.

    Clinical Skills:
    - Critical care nursing
    - Emergency room procedures
    - Patient assessment and care planning
    - Medication administration
    - EMR/EHR systems (Epic, Cerner)

    Experience:
    Senior Registered Nurse | City General Hospital | 2018-2024
    - Managed care for 15+ patients per shift in ICU
    - Coordinated with multidisciplinary teams for treatment plans
    - Mentored junior nurses and nursing students
    - Implemented new patient monitoring protocols

    Registered Nurse | Community Health Center | 2016-2018
    - Provided primary care services to underserved communities
    - Conducted health education workshops
    - Managed vaccination programs

    Education:
    BSN, Johns Hopkins University School of Nursing

    Licenses:
    - RN License, State of California
    - ACLS and BLS Certified
    """

    # 测试案例3: Finance类简历
    test_resume_3 = """
    FINANCIAL ANALYST

    Summary:
    Detail-oriented financial analyst with 6 years of experience in
    investment banking and corporate finance. Expert in financial
    modeling, valuation, and market analysis.

    Skills:
    - Financial modeling and valuation
    - DCF, LBO, M&A analysis
    - Bloomberg Terminal, FactSet
    - Advanced Excel, VBA, Python for finance
    - Risk assessment and portfolio management

    Experience:
    Senior Financial Analyst | Goldman Sachs | 2020-2024
    - Built financial models for $500M+ M&A transactions
    - Conducted due diligence on investment opportunities
    - Presented investment recommendations to senior management
    - Managed team of 3 junior analysts

    Financial Analyst | JP Morgan | 2018-2020
    - Analyzed market trends and prepared investment reports
    - Assisted in portfolio management for high-net-worth clients
    - Developed automated reporting tools reducing manual work by 60%

    Education:
    MBA, Finance, Wharton School, University of Pennsylvania
    CFA Charterholder
    """

    test_cases = [
        ("IT/Software", test_resume_1),
        ("Healthcare", test_resume_2),
        ("Finance", test_resume_3),
    ]

    for category_expected, resume_text in test_cases:
        print(f"\n{'='*60}")
        print(f"测试: 期望 = {category_expected}")
        print(f"{'='*60}")

        result = classifier.predict(resume_text)

        if result.get('error'):
            print(f"  错误: {result['message']}")
            continue

        print(f"  预测岗位: {result['category']}")
        print(f"  置信度: {result['confidence']:.2%}")
        print(f"\n  Top-3候选:")
        for i, candidate in enumerate(result['top_k'], 1):
            bar = '█' * int(candidate['confidence'] * 50)
            print(f"    {i}. {candidate['category']:<30} "
                  f"{candidate['confidence']:.2%} {bar}")

    print(f"\n{'='*60}")
    print("预测演示完成！")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
