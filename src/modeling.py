from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd

def run_models(train):
    X = train.drop("SalePrice", axis=1)
    y = np.log1p(train["SalePrice"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=10),
        "Lasso": Lasso(alpha=0.001, max_iter=20000),
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=20,
            min_samples_split=5,
            random_state=42
        )
    }

    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        preds_exp = np.expm1(preds)
        y_test_exp = np.expm1(y_test)

        rmse = np.sqrt(mean_squared_error(y_test_exp, preds_exp))
        mae = mean_absolute_error(y_test_exp, preds_exp)
        r2 = r2_score(y_test_exp, preds_exp)

        results.append([name, rmse, mae, r2])

        print(f"\n{name}")
        print("RMSE:", rmse)
        print("MAE:", mae)
        print("R2:", r2)

    df = pd.DataFrame(results, columns=["Model", "RMSE", "MAE", "R2"])
    df.to_csv("outputs/results.txt", index=False)

    return df
