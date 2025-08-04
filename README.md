# JacketSizePredictor-Stochastic-Gradient-Descent
Predicting jacket sizes (in inches) based on a person’s age, height (cm), and weight (kg) using a ML algorithm called Stochastic Gradient Descent (SGD). The purpose is to create a simple, yet effective, predictive model suitable for applications in fashion technology, e-commerce, and custom tailoring, where precise sizing can enhance customer needs
Algorithm: Stochastic Gradient Descent Regressor
The core algorithm used is SGD Regressor, which is a variant of linear regression. Instead of processing the entire dataset at once (as in traditional linear regression), SGD updates the model step-by-step using small batches of data, making it efficient and scalable.
Key features:
Learning Rate (eta0) control for optimization speed.
Max Iterations (max_iter) to define training cycles.
Applied feature scaling using StandardScaler to normalize input featur
