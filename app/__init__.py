"""
Flask应用工厂模块。

使用应用工厂模式（Application Factory Pattern）：
    - 延迟创建Flask实例
    - 便于测试（可以传入不同配置）
    - 符合Flask官方推荐的最佳实践
"""
from flask import Flask


def create_app():
    """
    创建并配置Flask应用。

    Returns:
        Flask: 配置好的Flask应用实例
    """
    app = Flask(__name__, template_folder='../templates', static_folder='../static')

    # 配置
    app.config['SECRET_KEY'] = 'ai-resume-classifier-2024'
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 限制请求大小为16MB

    # 注册路由
    from app import routes
    app.register_blueprint(routes.bp)

    return app
