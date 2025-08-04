import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

# 🔹 1. Load your CSV data
df = pd.read_csv('jacket_sizes_data.csv')

# 🔹 2. Prepare features and target
X = df[['Age', 'Height_cm', 'Weight_kg']]
y = df['Jacket_Size_in']

# 🔹 3. Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 🔹 4. Train SGD Regressor
model = SGDRegressor(max_iter=2000, learning_rate='invscaling', eta0=0.005)
model.fit(X_scaled, y)

# 🔹 5. Evaluate the model
y_pred = model.predict(X_scaled)
print("R² Score:", r2_score(y, y_pred))
print("Mean Squared Error:", mean_squared_error(y, y_pred))

# ----------------------------------------
# 🔮 6. Predict Jacket Size for New Person
# Example: Age = 25, Height = 170 cm, Weight = 65 kg
new_input = [[25, 174, 65]]

# Scale the new input same as training data
new_input_scaled = scaler.transform(new_input)

# Predict jacket size
predicted_size = model.predict(new_input_scaled)
print(f"Predicted Jacket Size: {predicted_size[0]:.2f} inches")
