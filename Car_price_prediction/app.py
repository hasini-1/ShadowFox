from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load Trained Model
model = joblib.load("car_price_model.pkl")


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    present_price = float(request.form['present_price'])
    kms_driven = float(request.form['kms_driven'])

    owner = int(request.form['owner'])

    manufacturing_year = int(
        request.form['manufacturing_year']
    )

    car_age = 2025 - manufacturing_year

    fuel_type = request.form['fuel_type']
    seller_type = request.form['seller_type']
    transmission = request.form['transmission']

    fuel_type_diesel = 1 if fuel_type == "Diesel" else 0
    fuel_type_petrol = 1 if fuel_type == "Petrol" else 0

    seller_type_individual = (
        1 if seller_type == "Individual"
        else 0
    )

    transmission_manual = (
        1 if transmission == "Manual"
        else 0
    )

    input_data = pd.DataFrame([[
        present_price,
        kms_driven,
        owner,
        car_age,
        fuel_type_diesel,
        fuel_type_petrol,
        seller_type_individual,
        transmission_manual
    ]], columns=[
        'Present_Price',
        'Kms_Driven',
        'Owner',
        'Car_Age',
        'Fuel_Type_Diesel',
        'Fuel_Type_Petrol',
        'Seller_Type_Individual',
        'Transmission_Manual'
    ])

    prediction = model.predict(
        input_data
    )[0]

    return render_template(
        'index.html',
        prediction=round(prediction, 2),
        present_price=present_price,
        kms_driven=kms_driven,
        owner=owner,
        manufacturing_year=manufacturing_year,
        fuel_type=fuel_type,
        seller_type=seller_type,
        transmission=transmission
    )


if __name__ == "__main__":
    app.run(debug=True)