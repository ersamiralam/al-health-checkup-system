from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# =========================
# TRAIN SIMPLE ML MODEL
# =========================

data = {
    "age":[25,60,45,70,35,55,65,28,50,75],
    "bp":[120,160,130,180,110,150,170,115,145,190],
    "sugar":[90,180,100,220,95,170,210,85,160,240],
    "risk":[0,1,0,1,0,1,1,0,1,1]
}

df = pd.DataFrame(data)

X = df[["age","bp","sugar"]]
y = df["risk"]

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X,y)

# =========================
# HOME
# =========================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )

# =========================
# PREDICT
# =========================

@app.route("/predict", methods=["POST"])
def predict():

    name = request.form["name"]

    age = int(request.form["age"])

    gender = request.form["gender"]

    height = float(
        request.form["height"]
    )

    weight = float(
        request.form["weight"]
    )

    bp = int(
        request.form["bp"]
    )

    sugar = int(
        request.form["sugar"]
    )

    smoking = request.form["smoking"]

    alcohol = request.form["alcohol"]

    exercise = request.form["exercise"]

    bmi = round(
        weight / ((height/100)**2),
        2
    )

    prediction = model.predict(
        [[age,bp,sugar]]
    )[0]

    score = 100

    if bp > 140:
        score -= 20

    if sugar > 140:
        score -= 20

    if bmi > 30:
        score -= 15

    if smoking == "Yes":
        score -= 10

    if alcohol == "Yes":
        score -= 10

    if exercise == "No":
        score -= 10

    score = max(score,20)

    if prediction == 1:
        risk = "High Risk"
    else:
        risk = "Low Risk"

    return render_template(
        "index.html",
        result=True,
        name=name,
        age=age,
        gender=gender,
        bmi=bmi,
        bp=bp,
        sugar=sugar,
        score=score,
        risk=risk
    )

# =========================

if __name__ == "__main__":
    app.run(debug=True)