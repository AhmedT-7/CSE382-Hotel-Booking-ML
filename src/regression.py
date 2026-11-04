from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from cleaning import clean_data
from preprocessing import prepare_supervised_data


def evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test):
    """Fits a model and evaluates it using MAE, RMSE, and R2 on the test set."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    return {"MAE": mae, "RMSE": rmse, "R2": r2}, model


def main():
    project_dir = Path(__file__).resolve().parent.parent
    dataset_path = project_dir / "data" / "H2.csv"
    
    print("Loading and cleaning dataset for Regression task...")
    df = pd.read_csv(dataset_path)
    df = clean_data(df)
    
    print("Preparing features and supervised splits...")
    splits, preprocessor = prepare_supervised_data(df, "regression")

    X_train = preprocessor.fit_transform(splits["X_train"])
    X_val = preprocessor.transform(splits["X_val"])
    X_test = preprocessor.transform(splits["X_test"])
    
    y_train = splits["y_train"]
    y_val = splits["y_val"]
    y_test = splits["y_test"]
    
    results = []
    
    models = {
        "Linear Regression": {
            "model": LinearRegression(),
            "params": {} 
        },
        "Decision Tree": {
            "model": DecisionTreeRegressor(random_state=42),
            "params": {
                "max_depth": [5, 10, 20, None],
                "min_samples_split": [2, 10, 20]
            }
        },
        "Random Forest": {
            "model": RandomForestRegressor(random_state=42, n_jobs=-1),
            "params": {
                "n_estimators": [50],
                "max_depth": [10, 20]
            }
        }
    }
    
    artifacts_dir = project_dir / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    
    for name, config in models.items():
        print(f"\n---------------------------------------------")
        print(f"Evaluating: {name}")
        print(f"---------------------------------------------")
        
        metrics_base, fitted_base = evaluate_model(
            config["model"], X_train, y_train, X_val, y_val, X_test, y_test
        )
        results.append({
            "Model": f"{name} (Baseline)",
            **metrics_base
        })
        print(f"-> Baseline | MAE: {metrics_base['MAE']:.4f} | RMSE: {metrics_base['RMSE']:.4f} | R2: {metrics_base['R2']:.4f}")
        
        if config["params"]:
            grid_search = GridSearchCV(
                estimator=config["model"],
                param_grid=config["params"],
                scoring="neg_root_mean_squared_error",
                cv=3,
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            
            metrics_tuned, _ = evaluate_model(
                best_model, X_train, y_train, X_val, y_val, X_test, y_test
            )
            results.append({
                "Model": f"{name} (Tuned)",
                **metrics_tuned
            })
            print(f"-> Tuned Best Params: {grid_search.best_params_}")
            print(f"-> Tuned    | MAE: {metrics_tuned['MAE']:.4f} | RMSE: {metrics_tuned['RMSE']:.4f} | R2: {metrics_tuned['R2']:.4f}")
            
            filename = f"reg_{name.lower().replace(' ', '_')}_tuned.pkl"
            joblib.dump(best_model, artifacts_dir / filename)
        else:
            filename = f"reg_{name.lower().replace(' ', '_')}_baseline.pkl"
            joblib.dump(fitted_base, artifacts_dir / filename)

    results_df = pd.DataFrame(results)
    print("\n========================================================")
    print("FINAL REGRESSION MODEL COMPARISON RESULTS")
    print("========================================================")
    print(results_df.to_string(index=False))
    
    joblib.dump(preprocessor, artifacts_dir / "regression_preprocessor.pkl")
    print("\n[SUCCESS] Preprocessor and model artifacts successfully saved to /artifacts/")


if __name__ == "__main__":
    main()