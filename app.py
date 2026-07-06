from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model and scaler
model = joblib.load("heart_disease_model.pkl")
scaler = joblib.load("scaler.pkl")


# ---------------- Home Page ----------------
@app.route("/")
def home():
    return render_template("index.html")


# ---------------- Prediction Route ----------------
@app.route("/predict", methods=["POST"])
def predict():

    # Get input values from form
    age = int(request.form["age"])
    sex = int(request.form["sex"])
    chest_pain = int(request.form["chest_pain_type"])
    resting_bp = float(request.form["resting_blood_pressure"])
    cholesterol = float(request.form["serum_cholesterol_mg_per_dl"])
    fasting = int(request.form["fasting_blood_sugar_gt_120_mg_per_dl"])
    ekg = int(request.form["resting_ekg_results"])
    max_hr = float(request.form["max_heart_rate_achieved"])
    angina = int(request.form["exercise_induced_angina"])
    oldpeak = float(request.form["oldpeak_eq_st_depression"])
    slope = int(request.form["slope_of_peak_exercise_st_segment"])
    vessels = int(request.form["num_major_vessels"])
    thal_normal = int(request.form["thal_normal"])
    thal_reversible_defect = int(request.form["thal_reversible_defect"])

    # Create DataFrame with model input features
    input_data = pd.DataFrame({
        "slope_of_peak_exercise_st_segment": [slope],
        "resting_blood_pressure": [resting_bp],
        "chest_pain_type": [chest_pain],
        "num_major_vessels": [vessels],
        "fasting_blood_sugar_gt_120_mg_per_dl": [fasting],
        "resting_ekg_results": [ekg],
        "serum_cholesterol_mg_per_dl": [cholesterol],
        "oldpeak_eq_st_depression": [oldpeak],
        "sex": [sex],
        "age": [age],
        "max_heart_rate_achieved": [max_hr],
        "exercise_induced_angina": [angina],
        "thal_normal": [thal_normal],
        "thal_reversible_defect": [thal_reversible_defect]
    })

    # Continuous columns for scaling
    continuous_cols = [
        "age",
        "resting_blood_pressure",
        "serum_cholesterol_mg_per_dl",
        "max_heart_rate_achieved",
        "oldpeak_eq_st_depression"
    ]

    # Apply StandardScaler
    input_data[continuous_cols] = scaler.transform(
        input_data[continuous_cols]
    )

    # Predict disease
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    # Prediction result
    if prediction[0] == 1:
        result = "Heart Disease Detected"
    else:
        result = "No Heart Disease Detected"

    # Prediction confidence
    confidence = round(max(probability[0]) * 100, 2)

    # Risk level and recommendations
    if prediction[0] == 1:
        risk = "HIGH"
        recommendation = [
            "Consult a Cardiologist",
            "Take ECG/Echo Test",
            "Reduce Cholesterol",
            "Exercise Regularly",
            "Avoid Smoking"
        ]
    else:
        risk = "LOW"
        recommendation = [
            "Maintain Healthy Lifestyle",
            "Regular Health Check-up",
            "Balanced Diet",
            "Daily Exercise"
        ]

    return render_template(

    "result.html",

    result=result,

    confidence=confidence,

    risk=risk,

    recommendation=recommendation,

    age=age,

    sex=sex,

    chest_pain=chest_pain,

    resting_bp=resting_bp,

    cholesterol=cholesterol,

    fasting=fasting,

    ekg=ekg,

    max_hr=max_hr,

    angina=angina,

    oldpeak=oldpeak,

    slope=slope,

    vessels=vessels,

    thal_normal=thal_normal,

    thal_reversible_defect=thal_reversible_defect

)


# ---------------- Run Flask App ----------------
if __name__ == "__main__":
    app.run(debug=True)