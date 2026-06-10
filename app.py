from flask import Flask, request, render_template
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")  # IMPORTANT for servers
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# Load your trained model
model = joblib.load("iris_model.pkl")


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        sl = float(request.form["sepal_length"])
        sw = float(request.form["sepal_width"])
        pl = float(request.form["petal_length"])
        pw = float(request.form["petal_width"])
        values = [sl, sw, pl, pw]

        if any(v < 0 for v in values):
           return render_template(
             "index.html",
              prediction_text=" No negative values allowed"
             )

        features = np.array([[sl, sw, pl, pw]])

        prediction = model.predict(features)[0]
        proba = model.predict_proba(features)[0]

        classes = ["Setosa", "Versicolor", "Virginica"]
        result = classes[int(prediction)]

        # 🔥 CREATE BAR CHART
        plt.figure(figsize=(5, 3))
        plt.bar(classes, proba)
        plt.ylim(0, 1)
        plt.title("Prediction Confidence")

        img_path = "static/prob.png"
        plt.savefig(img_path)
        plt.close()

        return render_template(
            "index.html",
            prediction_text=f"🌸 Prediction: {result}",
            confidence_text=f"🔥 Confidence generated!",
            image_path=img_path
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"❌ Error: {str(e)}"
        )


if __name__ == "__main__":
    app.run()