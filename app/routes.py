"""
Flask路由模块。

定义Web页面的URL路由和API接口：
    GET  /          → 首页（简历输入表单）
    POST /predict   → 预测API（JSON输入，JSON输出）

架构设计：
    - 分类器在应用启动时加载一次（全局单例）
    - 避免每次请求都重新加载模型（模型文件可能很大）
    - 加载失败时应用仍可启动，但API返回错误信息而非崩溃
"""
import sys
import os

# 确保项目根目录在Python路径中（便于导入predict模块）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Blueprint, render_template, request, jsonify

# 创建蓝图
bp = Blueprint('main', __name__)

# 全局分类器实例（延迟加载）
_classifier = None


def get_classifier():
    """
    获取分类器单例。

    延迟加载（Lazy Loading）模式：
        - 首次调用时加载模型
        - 后续调用直接返回已加载的实例
        - 好处：启动快、节省内存

    Returns:
        ResumeClassifier 或 None
    """
    global _classifier
    if _classifier is None:
        try:
            from predict import ResumeClassifier
            _classifier = ResumeClassifier()
        except Exception as e:
            print(f"[错误] 分类器加载失败: {e}")
            return None
    return _classifier


@bp.route('/')
def index():
    """首页：显示简历输入表单。"""
    # 尝试加载分类器，检查模型状态
    classifier = get_classifier()
    model_ready = classifier is not None

    model_info = {}
    if model_ready and hasattr(classifier, 'metadata'):
        model_info = {
            'model_name': classifier.metadata.get('model_name', 'Unknown'),
            'f1_score': classifier.metadata['metrics']['f1_weighted'],
            'num_categories': len(classifier.label_map),
            'categories': list(classifier.label_map.values()),
        }

    return render_template('index.html',
                           model_ready=model_ready,
                           model_info=model_info)


@bp.route('/predict', methods=['POST'])
def predict():
    """
    预测API接口。

    请求格式（JSON）：
        {
            "resume_text": "完整的简历文本内容..."
        }

    响应格式（JSON）：
        成功：{
            "success": true,
            "category": "INFORMATION-TECHNOLOGY",
            "confidence": 0.85,
            "top_k": [
                {"category": "...", "confidence": 0.85},
                ...
            ]
        }
        失败：{
            "success": false,
            "error": "错误信息"
        }
    """
    # Step 1: 获取分类器
    classifier = get_classifier()
    if classifier is None:
        return jsonify({
            'success': False,
            'error': '模型未加载。请确保已运行 train.py 训练模型。'
        }), 500

    # Step 2: 获取请求数据
    data = request.get_json(silent=True)
    if not data or 'resume_text' not in data:
        return jsonify({
            'success': False,
            'error': '请提供 resume_text 参数。'
        }), 400

    resume_text = data['resume_text'].strip()
    if len(resume_text) < 50:
        return jsonify({
            'success': False,
            'error': '简历文本过短（<50字符），请提供更完整的内容。'
        }), 400

    # Step 3: 预测
    try:
        result = classifier.predict(resume_text)

        if result.get('error'):
            return jsonify({
                'success': False,
                'error': result['message']
            }), 400

        return jsonify({
            'success': True,
            'category': result['category'],
            'confidence': result['confidence'],
            'top_k': result['top_k'],
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'预测过程出错: {str(e)}'
        }), 500


@bp.route('/health')
def health():
    """健康检查接口。"""
    classifier = get_classifier()
    return jsonify({
        'status': 'ok' if classifier else 'model not loaded',
        'model_ready': classifier is not None,
    })
