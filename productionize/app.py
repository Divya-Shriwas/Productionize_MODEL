import streamlit as st
import numpy as np
from pickle import load
import pandas as pd

Std_scaler = load(open(r"D:\models\scaler_std.pkl", 'rb'))
ohe_encoder = load(open(r"D:\models\ohe_encoder.pkl", 'rb'))
RF_model = load(open(r"D:\models\RF_model.pkl", 'rb'))

def main():
    st.title("Medical Cost Prediction")
    html_temp ="""
    <div style="background-color:#025246 ;padding:10x">
    <h2 style="color:white;text-align:center;">Laptop Price Prediction</h2>
    </div>
    """

    age = st.text_input("AGE", placeholder="Enter value",)
    sex = st.text_input("GENDER", placeholder="Enter Gender")
    bmi = st.text_input("BMI", placeholder="Enter value")
    smoker = st.text_input("SMOKER", placeholder="Enter")



    btn_click = st.button("Predict")

    if btn_click == True:
        if age and sex and bmi and smoker:
            query_point1 = np.array([ int(bmi),int(age)]).reshape(1, -1)
            query_point2 = np.array([str(sex),str(smoker)]).reshape(1, -1)
            query_point_trans1 = Std_scaler.transform(query_point1)
        
            query_point_trans2 =ohe_encoder.transform(query_point2)
            query_point_tran = np.concatenate((query_point_trans1,query_point_trans2), axis =1)


            pred =RF_model.predict(query_point_tran)

            st.success(pred)
        else:
            st.error("Enter the values properly.")



if __name__=='__main__':
    main()