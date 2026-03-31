from flask import Flask, request, render_template
from ultralytics import YOLO
import os

app = Flask(__name__)

# Load YOLOv8 model
model = YOLO("model/best.pt")

@app.route("/", methods=["GET", "POST"])
def index():
    result_img = None

    if request.method == "POST":
        file = request.files["image"]

        if file:
            # Save uploaded image
            input_path = os.path.join("static", "input.jpg")
            file.save(input_path)

            # Run detection
            results = model(input_path)

            # Save result image
            output_path = os.path.join("static", "result.jpg")
            results[0].save(filename=output_path)

            result_img = output_path

    return render_template("index.html", result_img=result_img)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
