## 🚀 Project Overview

**DrugDiscovery-ProteinPredictor** is a full-stack machine learning system designed for predicting molecular bioactivity and protein targets. Leveraging **Random Forest, Keras MLP, and RL-enhanced PyTorch MLP models**, the project showcases:

- Advanced **feature engineering**: molecular descriptors (MW, LogP, HBA, HBD, TPSA)
- **Reinforcement learning fine-tuning** for adaptive model improvement
- Multi-model evaluation with **accuracy, top-K prediction, and interpretability**
- **Interactive Gradio interface** for real-time user input and prediction comparison

This system bridges cutting-edge ML techniques with real-world bioinformatics applications, enabling researchers to test hypotheses and compare predictive models efficiently.

---

## 🎯 Motivation

Drug discovery involves screening thousands of compounds, which is costly and time-consuming. By predicting protein targets computationally:

- Reduce experimental cost and effort
- Prioritize high-potential compounds for wet-lab validation
- Accelerate early-stage drug development

Integrating **reinforcement learning** allows the system to iteratively improve predictions based on rewards, mimicking adaptive decision-making in experimental design.

---

## 🛠 Methodology

1. **Data Preparation**
   - Dataset: Multi-target bioactivity dataset from ChEMBL
   - Features: Molecular weight, LogP, HBA, HBD, TPSA
   - Preprocessing: `LabelEncoder`, `StandardScaler`, handling missing values

2. **Model Development**
   - **Random Forest Classifier**: Baseline performance and top-1 predictions
   - **Keras MLP**: Dropout layers and class weighting for robust multi-label predictions
   - **RL-PyTorch MLP Agent**: Each prediction is treated as an action; positive rewards for correct classifications, boosting top-K accuracy

3. **Evaluation & Benchmarking**
   - Accuracy improvement of **85.62%** with RL fine-tuning
   - Comparison of top predictions between Random Forest, baseline MLP, and RL-enhanced MLP

4. **Deployment**
   - Gradio interface with clear tabs for model outputs
   - Interactive predictions with visual comparison of models
   - Persisted trained models for reproducibility

---

## 📈 Results

- **Random Forest**: Reliable baseline predictions
- **Keras MLP**: Improved multi-label prediction accuracy
- **RL-Enhanced MLP**: Adaptive learning achieves **85.62% top-1 accuracy**, outperforming baseline models
- User-friendly interface enables seamless selection and comparison of models for research exploration

---
