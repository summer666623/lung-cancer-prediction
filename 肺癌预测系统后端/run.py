from app.app_loader import create_app

app = create_app()

if __name__ == "__main__":
    # 本地开发用
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
