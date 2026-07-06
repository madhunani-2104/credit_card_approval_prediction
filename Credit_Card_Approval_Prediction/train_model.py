import pandas as pd, numpy as np, pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

app_df = pd.read_csv("dataset/application_record.csv")
credit_df = pd.read_csv("dataset/credit_record.csv")

credit_df["target"] = credit_df["STATUS"].apply(lambda x: 1 if x in ["C", "X", "0"] else 0)
target = credit_df.groupby("ID")["target"].min().reset_index()
df = app_df.merge(target, on="ID", how="inner").drop(columns=["ID"])

for col in df.select_dtypes(include="object").columns:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

X = df.drop("target", axis=1)
y = df["target"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
print("Accuracy:", accuracy_score(y_test, model.predict(X_test)))
print(classification_report(y_test, model.predict(X_test)))

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
