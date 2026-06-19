from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load model and encoders
model = pickle.load(open("model.pkl", "rb"))
encoders = pickle.load(open("encoder.pkl", "rb"))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    age = int(request.form['age'])
    gender = request.form['gender'].strip()
    occupation = request.form['occupation'].strip()
    screen_time = float(request.form['screen_time'])
    app_used = request.form['app_used'].strip()
    checks = request.form['checks'].strip()
    usage_time = request.form['usage_time'].strip()
    sleep = float(request.form['sleep'])
    distracted = request.form['distracted'].strip()
    work_hours = float(request.form['work_hours'])
    productivity = int(request.form['productivity'])
    mood = int(request.form['mood'])

    input_data = pd.DataFrame([[
        age,
        gender,
        occupation,
        screen_time,
        app_used,
        checks,
        usage_time,
        sleep,
        distracted,
        work_hours,
        productivity,
        mood
    ]], columns=[
        'Age',
        'Gender',
        'Occupation',
        'Daily screen time (hours)',
        'Most used social media app',
        'How often do you check social media daily?',
        'Social media usage mainly during',
        'Average sleep duration (hours)',
        'Do you feel distracted because of social media?',
        'Study/Work hours per day',
        'Productivity level (1-10)',
        'Mood level (1-5)'
    ])

    # Encode categorical columns
    for col in input_data.columns:
        if col in encoders:
            try:
                input_data[col] = input_data[col].astype(str).str.strip()
                input_data[col] = encoders[col].transform(input_data[col])
            except Exception as e:
                return render_template(
                    'index.html',
                    prediction_text=f"Error in column '{col}': {e}"
                )

    # Make prediction
    prediction = model.predict(input_data)

    # Decode prediction
    result = encoders[
        'How addicted do you feel to social media?'
    ].inverse_transform(prediction)[0]

    return render_template(
        'index.html',
        prediction_text=f'Addiction Level : {result}'
    )


if __name__ == "__main__":
    app.run(debug=True)