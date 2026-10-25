# Predicting Car Price (AT82.03 Machine Learning)

## A1: Predicting Car Price
Predict a used car's selling price so Chaky's company can price its cars, and serve the model on a web page.

- `A1/A1.ipynb`: EDA, cleaning, model selection, and report
- `A1/app_a1.py`: Dash web app (any field can be left empty; it is filled with the training median/mode)
- `A1/models_a1/`: saved model, scaler, and column info

**Result:** Random Forest was the best of 7 models (5-fold CV). Test R² = 0.943 (log price), about 14% average error. `year` and `max_power` matter most.

## A2: Linear Regression from scratch + MLflow
- `A2/A2.ipynb`: the LinearRegression class (r2, Xavier init, momentum, feature importance), the experiment, and the report
- `A2/app_a2.py`: two-page web app (old A1 model + new A2 model)
- `A2/mlflow.db`: all runs. View them with `cd A2 && mlflow ui --backend-store-uri sqlite:///mlflow.db`

144 combinations (Normal/Lasso/Ridge/Polynomial × momentum × batch/mini/sto × zeros/xavier × 3 learning rates), 3-fold CV.
**Best:** Polynomial, mini-batch, zeros init, lr 0.01, no momentum. Test R² = 0.913 (log price).

## A3: Classification (Logistic Regression from scratch)
- `A3/A3.ipynb`: price split into 4 classes; LogisticRegression with accuracy, precision, recall, F1 (per class, macro, weighted) checked against sklearn, and a Ridge penalty
- `A3/app_a3.py`: final three-page web app (A1, A2, A3 models)
- `A3/tests/`: unit tests run by GitHub Actions

18 MLflow runs (method × learning rate × penalty). **Best:** batch, lr 0.1, no penalty.
Test accuracy 0.760, macro F1 0.761. Registered as `st127120-a3-model` (Staging).

### Run
```bash
docker compose up --build    # then open http://localhost:8050
```
Without Docker: `cd A1 && pip install -r requirements.txt && python app_a1.py`