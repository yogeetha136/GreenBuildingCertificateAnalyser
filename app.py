import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib

try:
    # Step 1: Load dataset
    file_path = "building_data.csv"
    df = pd.read_csv(file_path)

    # Step 2: Preprocess datetime column
    if 'Last_Inspection_Date' in df.columns:
        df['Last_Inspection_Date'] = pd.to_datetime(df['Last_Inspection_Date'], format='%d-%m-%Y', errors='coerce')
        df['Last_Inspection_Date_Timestamp'] = df['Last_Inspection_Date'].astype('int64') // 10**9
        df = df.drop(columns=['Last_Inspection_Date'])

    # Step 3: Handle numeric and categorical columns
    # Identify categorical and numeric columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    # Apply Label Encoding to categorical columns
    label_encoders = {}
    for col in categorical_cols:
        if col != 'Green_Certified':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le

    # Handle Number_of_Floors column
    if 'Number_of_Floors' in df.columns:
        try:
            df['Number_of_Floors'] = pd.to_numeric(df['Number_of_Floors'], errors='coerce')
        except ValueError:
            print(f"Error converting 'Number_of_Floors' to numeric.")

    # Step 4: Prepare features and target
    X = df.drop(columns=["Green_Certified"])
    y = df["Green_Certified"]

    # Step 5: Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 6: Train XGBoost Model
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        'objective': 'binary:logistic',
        'max_depth': 6,
        'learning_rate': 0.1,
        'eval_metric': 'logloss'
    }

    bst = xgb.train(params, dtrain, num_boost_round=200)

    # Step 7: Evaluate Model
    y_train_pred = (bst.predict(dtrain) > 0.5).astype(int)
    print(f"Training Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")

    y_test_pred = (bst.predict(dtest) > 0.5).astype(int)
    print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred))

    # Step 8: Save Model and Encoders
    joblib.dump(bst, "models/xgboost_green_certified_full_model.pkl")
    joblib.dump(label_encoders, "models/label_encoders.pkl")
    print("Model and encoders saved successfully.")

except Exception as e:
    print(f"Error: {e}")
