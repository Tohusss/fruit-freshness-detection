from flask import Flask, render_template, request
import os
from src.predict import predict_result

# Tell Flask where templates & static folders are
app = Flask(
    __name__,
    template_folder="web/templates",
    static_folder="web/static"
)

UPLOAD_FOLDER = "web/static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def upload_image():
    if "image" not in request.files:
        return "No file uploaded", 400

    file = request.files["image"]
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    fruit, freshness, fruit_conf, freshness_conf = predict_result(filepath)

    return render_template(
        "index.html",
        fruit=fruit,
        freshness=freshness,
        fruit_conf=fruit_conf,
        freshness_conf=freshness_conf
    )

if __name__ == "__main__":
    app.run(debug=True)
