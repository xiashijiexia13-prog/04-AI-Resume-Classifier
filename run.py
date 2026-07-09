"""
run.py
======
Flask Web应用启动入口。

启动方式：
    python run.py

然后浏览器访问: http://127.0.0.1:5000
"""
from app import create_app

app = create_app()

if __name__ == '__main__':
    print("=" * 60)
    print("AI Resume Classifier - Web应用启动")
    print("=" * 60)
    print(f"\n访问地址: http://127.0.0.1:5000")
    print(f"按 Ctrl+C 停止服务\n")

    # debug=True: 开发模式，代码修改后自动重启
    # 生产环境应设为False
    app.run(host='0.0.0.0', port=5000, debug=True)
