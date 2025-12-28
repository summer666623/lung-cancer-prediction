from flask import Flask
from flask_cors import CORS
from app.routes import api_bp


def create_app():
    """
    创建并配置 Flask 应用
    """
    app = Flask(__name__)

    # 允许前端跨域访问
    CORS(app)

    # 注册 API 蓝图
    app.register_blueprint(api_bp, url_prefix="/api")

    return app
