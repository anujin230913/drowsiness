from flask import Flask
import subprocess

app = Flask(__name__)

@app.route("/start")
def start_system():
    subprocess.Popen(["python", "main.py"])
    return "System started"

if __name__ == "__main__":
    app.run(debug=True)
