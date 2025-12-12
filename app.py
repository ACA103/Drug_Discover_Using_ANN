# -*- coding: utf-8 -*-
"""Drug Discovery Using ANN - Hugging Face Deployment"""

import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow import keras
import gradio as gr


features = ['Molecular Weight','LogP','HBA','HBD','TPSA']


rf = joblib.load("models/rf_model.joblib")
scaler = joblib.load("models/scaler.joblib")
le = joblib.load("models/le.joblib")
keras_model = keras.models.load_model("models/keras_mlp.h5")


def predict_protein(mol_weight, logp, hba, hbd, tpsa):
    sample = pd.DataFrame([{
        'Molecular Weight': mol_weight,
        'LogP': logp,
        'HBA': hba,
        'HBD': hbd,
        'TPSA': tpsa
    }])
    
   
    s_scaled = scaler.transform(sample[features].values)
    
   
    pred_rf_idx = rf.predict(s_scaled)[0]
    pred_rf_class = le.inverse_transform([pred_rf_idx])[0]
    
  
    pred_prob = keras_model.predict(s_scaled)[0]
    top3_idx = np.argsort(pred_prob)[-3:][::-1]
    top3_predictions = [(le.inverse_transform([i])[0], float(pred_prob[i])) for i in top3_idx]
    
   
    top3_df = pd.DataFrame(top3_predictions, columns=["Protein", "Probability"])
    top3_df["Probability %"] = (top3_df["Probability"] * 100).round(2)
    
    return pred_rf_class, top3_df


inputs = [
    gr.Slider(0, 1000, step=0.1, label="Molecular Weight", info="Weight of molecule"),
    gr.Slider(-5, 10, step=0.1, label="LogP", info="Lipid solubility"),
    gr.Slider(0, 20, step=1, label="HBA", info="Hydrogen bond acceptors"),
    gr.Slider(0, 10, step=1, label="HBD", info="Hydrogen bond donors"),
    gr.Slider(0, 200, step=0.1, label="TPSA", info="Topological Polar Surface Area")
]


outputs = [
    gr.Textbox(label="Random Forest Prediction"),
    gr.Dataframe(headers=["Protein", "Probability", "Probability %"], label="Top 3 Keras Predictions")
]

description_md = """
# 🧬 Drug Discovery Protein Predictor

Predict protein targets for molecules using **Random Forest** and **Keras MLP**.

- Adjust molecular descriptors using sliders.
- Random Forest shows the top predicted protein.
- Keras MLP shows top 3 predictions with probabilities.
"""


iface = gr.Interface(
    fn=predict_protein,
    inputs=inputs,
    outputs=outputs,
    title="Drug Discovery Protein Predictor",
    description=description_md
)

iface.launch(share=True)
