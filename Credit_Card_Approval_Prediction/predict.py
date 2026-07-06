import pickle, numpy as np
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

sample = np.array([[1, 0, 1, 0, 200000, 2, 1, 0, 1, 35, 5, 1, 1, 1, 0, 2]])
scaled = scaler.transform(sample)
print("Prediction:", "Approved" if model.predict(scaled)[0] == 1 else "Rejected")
