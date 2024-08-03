import subprocess

if __name__ == "__main__":
    # 运行 create_db.py
    subprocess.run(["python", "create_db.py"], check=True)
    
    # 运行 fzu_with_history_langserve.py
    subprocess.run(["python", "fzu_with_history_langserve.py"], check=True)