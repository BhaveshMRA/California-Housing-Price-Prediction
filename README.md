
🏡 California Housing Price Prediction (Linear Regression from Scratch)

This project implements Linear Regression from scratch to predict house prices using the California Housing dataset.
Two training approaches are compared:
	•	Closed-form Normal Equation
	•	Stochastic Gradient Descent (SGD) with learning-rate decay

The goal is to understand optimization behavior, convergence, and generalization without relying on high-level ML libraries.


📌 Project Overview
	•	Dataset: California Housing (Scikit-learn)
	•	Task: Regression
	•	Target Variable: Median house value
	•	Models Implemented:
	•	Linear Regression via Normal Equation
	•	Linear Regression via Stochastic Gradient Descent



🧠 Key Concepts Covered
	•	Feature normalization (z-score standardization)
	•	Bias term handling
	•	Train / Validation / Test split (60% / 20% / 20%)
	•	Mean Squared Error (MSE) evaluation
	•	Learning rate decay
	•	Convergence analysis using loss curves



🛠️ Tech Stack
	•	Python
	•	NumPy – numerical computation
	•	Matplotlib – loss visualization
	•	Scikit-learn – dataset loading only (no models used)



⚙️ Implementation Details

1️⃣ Data Preprocessing
	•	Loaded California Housing dataset
	•	Normalized all features using z-score normalization
	•	Added a bias (intercept) term manually

2️⃣ Data Splitting
	•	60% Training
	•	20% Validation
	•	20% Testing
(Randomized using fixed seed for reproducibility)


📐 Models Implemented

🔹 Normal Equation

A closed-form solution:

\theta = (X^TX)^{-1}X^Ty
	•	Fast convergence
	•	Requires matrix inversion
	•	Suitable for smaller datasets



🔹 Stochastic Gradient Descent (SGD)
	•	Updates parameters one sample at a time
	•	Includes learning rate decay
	•	Tracks training and validation MSE per epoch

θ ← θ − α (ŷ − y) x




📊 Evaluation Metric
	•	Mean Squared Error (MSE)

Computed separately for:
	•	Training set
	•	Validation set
	•	Test set



📈 Results & Observations
	•	Normal Equation achieves stable performance quickly
	•	SGD converges gradually with proper learning rate tuning
	•	Validation loss helps monitor overfitting
	•	Learning rate decay improves SGD stability



📉 Visualization
	•	Training vs Validation loss curves
	•	Clear convergence behavior for SGD
	•	Helps diagnose underfitting / overfitting



🎯 Learning Outcomes
	•	Gained deep understanding of linear regression internals
	•	Implemented optimization without ML libraries
	•	Learned trade-offs between analytical and iterative solutions
	•	Practiced proper ML evaluation workflows



🔮 Future Improvements
	•	Add Mini-Batch Gradient Descent
	•	Compare with Ridge / Lasso Regression
	•	Add R² score evaluation
	•	Hyperparameter tuning for learning rate



👤 Author

Bhavesh Maurya
Machine Learning & Data Science Enthusiast

