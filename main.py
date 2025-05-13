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
    # Load data
    df = pd.read_csv("soil_data.csv")
    
    # Clean and select features
    df = df.drop(['lon'], axis=1)
    df = df.drop(['SQ2', 'SQ3', 'SQ4', 'SQ5', 'SQ6', 'SQ7'], axis=1)
    df = df.drop(['slope1', 'slope2', 'slope3', 'slope4', 'slope5', 'slope6', 'slope7', 'slope8'], axis=1)
    df = df.dropna()

    # Define X and y
    x = df[['fips', 'lat', 'elevation', 'aspectN', 'aspectE', 'aspectS', 'aspectW',
            'aspectUnknown', 'WAT_LAND', 'NVG_LAND', 'URB_LAND', 'GRS_LAND',
            'FOR_LAND', 'CULTRF_LAND', 'CULTIR_LAND', 'CULT_LAND']]
    y = df['SQ1']

    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    # Train model
    model = RandomForestClassifier()
    model.fit(x_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)

    # Plot feature importances
    plt.figure(figsize=(10, 6))
    plt.bar(x.columns, model.feature_importances_)
    plt.xticks(rotation=90)
    plt.title(f"Feature Importances (Accuracy: {acc:.2f})")
    plt.tight_layout()

    # Return image
    img_bytes = io.BytesIO()
    plt.savefig(img_bytes, format="png")
    plt.close()
    img_bytes.seek(0)

    return StreamingResponse(img_bytes, media_type="image/png")
