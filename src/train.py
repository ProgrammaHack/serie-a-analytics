import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from src.data_loader import load_data
from src.preprocessing import preprocess
from src.engine import build_training_frame, FEATURE_COLUMNS


def main():
    df = load_data("data/raw")
    df = preprocess(df)

    train_df, tracker = build_training_frame(df)

    print("Training frame shape:", train_df.shape)
    print("NaN check:")
    print(train_df[FEATURE_COLUMNS + ["FTR"]].isna().sum())

    train_df = train_df.dropna(subset=FEATURE_COLUMNS + ["FTR"]).copy()

    if len(train_df) == 0:
        raise ValueError("Dataset vuoto dopo feature engineering.")

    X = train_df[FEATURE_COLUMNS]
    y = train_df["FTR"]

    # split temporale: più realistico per il calcio
    split_idx = int(len(train_df) * 0.8)
    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]

    model = RandomForestClassifier(
        n_estimators=700,
        max_depth=14,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    print("\nAccuracy:", accuracy_score(y_test, preds))
    print("\nClassification report:")
    print(classification_report(y_test, preds))

    joblib.dump(model, "models/model.pkl")
    tracker.elo.save()

    print("Model saved")
    print("Elo saved")


if __name__ == "__main__":
    main()