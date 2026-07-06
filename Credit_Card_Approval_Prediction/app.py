from flask import Flask, render_template, request
import pickle, numpy as np

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    features = [float(x) for x in request.form.values()]
    scaled = scaler.transform([features])
    pred = model.predict(scaled)[0]
    result = "Approved ✅" if pred == 1 else "Rejected ❌"
    return render_template("result.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
