from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import io

app = FastAPI()

@app.post("/train")
def train_model():
    # Load and clean data
    df = pd.read_csv("train_timeseries.csv")
    df = df.dropna()
    df = df.drop('date', axis=1)
    df['score'] = df['score'].astype(int)

    # Define features and labels
    x = df[['fips', 'PRECTOT', 'PS', 'QV2M', 'T2M', 'T2MDEW', 'T2MWET',
            'T2M_MAX', 'T2M_MIN', 'T2M_RANGE', 'TS', 'WS10M', 'WS10M_MAX',
            'WS10M_MIN', 'WS10M_RANGE', 'WS50M', 'WS50M_MAX', 'WS50M_MIN',
            'WS50M_RANGE']]
    y = df['score']

    # Split the data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    # Train model
    model = RandomForestClassifier(n_estimators=70)
    model.fit(x_train, y_train)

    # Evaluate
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)

    # Plot feature importances
    plt.figure(figsize=(10, 6))
    plt.bar(x.columns, model.feature_importances_)
    plt.xticks(rotation=90)
    plt.title(f"Feature Importances (Accuracy: {acc:.2f})")
    plt.tight_layout()

    # Convert plot to image stream
    img_bytes = io.BytesIO()
    plt.savefig(img_bytes, format="png")
    plt.close()
    img_bytes.seek(0)

    return StreamingResponse(img_bytes, media_type="image/png")
