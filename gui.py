import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
import streamlit as st

class HousePricePredictor:
    def __init__(self, data_file="C:\project\House Price India.csv"):
        self.data_file = data_file
        self.data = None
        self.model = None

    def load_data(self):
        self.data = pd.read_csv(self.data_file)
        self.data.dropna(inplace=True)

    def prep_data(self):
        X = self.data[['number of bedrooms', 'living area', 'number of floors']]
        y = self.data['Price']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def train_model(self):
        self.model = LinearRegression()
        self.model.fit(self.X_train, self.y_train)

    def predict_price(self, num_bedrooms, living_area, num_floors):
        data = pd.DataFrame({'number of bedrooms': [num_bedrooms], 'living area': [living_area], 'number of floors': [num_floors]})
        return self.model.predict(data)[0]

# Set the background image
background_image = """
<style>
[data-testid="stAppViewContainer"] > .main {
    background-image: url("https://unsplash.com/photos/3d-render-of-building-exterior-L5nd7rPrEic");
    background-size: 100vw 100vh;  # This sets the size to cover 100% of the viewport width and height
    background-position: center;  
    background-repeat: no-repeat;
}
</style>
"""

st.markdown(background_image, unsafe_allow_html=True)

input_style = """
<style>
input[type="text"] {
    background-color: transparent;
    color: #a19eae;  // This changes the text color inside the input box
}
div[data-baseweb="base-input"] {
    background-color: transparent !important;
}
[data-testid="stAppViewContainer"] {
    background-color: transparent !important;
}
</style>
"""
st.markdown(input_style, unsafe_allow_html=True)
def main():
    st.title("House Price Predictor")

    hpp = HousePricePredictor()
    hpp.load_data()
    hpp.prep_data()
    hpp.train_model()

    num_bedrooms = st.number_input("Enter the number of bedrooms:", min_value=1, value=1)
    living_area = st.number_input("Enter the living area in sqft:", min_value=1.0, value=1000.0)
    num_floors = st.number_input("Enter the number of floors:", min_value=1, value=1)

    if st.button("Predict Price"):
        price = hpp.predict_price(num_bedrooms, living_area, num_floors)
        st.write(f'Predicted price for a house with {num_bedrooms} bedrooms, living area of {living_area} sqft, and {num_floors} floors is: {price:.2f}')

if __name__ == "__main__":
    main()
