# Predicting Car Price (AT82.03 Machine Learning)

## A1: Predicting Car Price
Predict a used car's selling price so Chaky's company can price its cars, and serve the model on a web page.

- `A1/A1.ipynb`: EDA, cleaning, model selection, and report
- `A1/app_a1.py`: Dash web app (any field can be left empty; it is filled with the training median/mode)
- `A1/models_a1/`: saved model, scaler, and column info

**Result:** Random Forest was the best of 7 models (5-fold CV). Test R² = 0.943 (log price), about 14% average error. `year` and `max_power` matter most.

### Run
```bash
docker compose up --build    # then open http://localhost:8050
```
Without Docker: `cd A1 && pip install -r requirements.txt && python app_a1.py`