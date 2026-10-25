"""A1 web app: predicts a car's selling price with the Random Forest trained in A1.ipynb.

Run locally:  python app_a1.py   then open http://localhost:8050
"""
import os

import joblib
import numpy as np
import pandas as pd
import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, dcc, html

# ---------- load the model files saved at the end of A1.ipynb ----------
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models_a1')
model = joblib.load(os.path.join(MODEL_DIR, 'car_price_model.pkl'))
scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
meta = joblib.load(os.path.join(MODEL_DIR, 'model_columns.pkl'))
FILL = meta['fill_values']  # training-set median/mode, used when the user skips a field


def predict_price(inputs):
    """Turn raw form values into one model-ready row and return the price in rupees.

    Any value that is None is replaced by the training-set median (numbers) or mode (categories),
    exactly like the imputation in the notebook.
    """
    row = {col: (FILL[col] if inputs.get(col) in (None, '') else inputs[col])
           for col in meta['num_cols'] + meta['cat_cols']}
    df = pd.DataFrame([row])
    # Same one-hot encoding as training; reindex adds the missing dummy columns as 0
    df = pd.get_dummies(df, columns=meta['cat_cols'], dtype=int)
    df = df.reindex(columns=meta['columns'], fill_value=0)
    df[meta['num_cols']] = scaler.transform(df[meta['num_cols']].astype(float))
    # The model predicts log(price) -> exp() turns it back into a price
    return float(np.exp(model.predict(df)[0]))


# ---------- layout ----------
def field(label, component, hint):
    return dbc.Col([dbc.Label(label), component, dbc.FormText(hint)], md=4, className='mb-3')


def num_input(id_, placeholder, step='any'):
    return dbc.Input(id=id_, type='number', placeholder=placeholder, step=step)


def dropdown(id_, options):
    return dcc.Dropdown(id=id_, options=options, placeholder='Skip to use the most common')


form = dbc.Row([
    field('Brand', dropdown('brand', meta['brands']), f"default: {FILL['brand']}"),
    field('Year', num_input('year', 'e.g. 2015', 1), f"default: {FILL['year']:.0f}"),
    field('Kilometres driven', num_input('km_driven', 'e.g. 60000', 1), f"default: {FILL['km_driven']:,.0f}"),
    field('Fuel', dropdown('fuel', ['Diesel', 'Petrol']), f"default: {FILL['fuel']}"),
    field('Transmission', dropdown('transmission', ['Manual', 'Automatic']), f"default: {FILL['transmission']}"),
    field('Seller type', dropdown('seller_type', ['Individual', 'Dealer', 'Trustmark Dealer']),
          f"default: {FILL['seller_type']}"),
    field('Owner', dropdown('owner', [{'label': l, 'value': v} for l, v in
                                       [('First', 1), ('Second', 2), ('Third', 3), ('Fourth & above', 4)]]),
          'default: First'),
    field('Mileage (kmpl)', num_input('mileage', 'e.g. 19.3'), f"default: {FILL['mileage']}"),
    field('Engine (CC)', num_input('engine', 'e.g. 1248', 1), f"default: {FILL['engine']:.0f}"),
    field('Max power (bhp)', num_input('max_power', 'e.g. 82.85'), f"default: {FILL['max_power']}"),
    field('Seats', num_input('seats', 'e.g. 5', 1), f"default: {FILL['seats']:.0f}"),
])

INPUT_IDS = meta['num_cols'] + meta['cat_cols']

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], title='Car Price Predictor')
server = app.server  # for gunicorn in Docker

app.layout = dbc.Container([
    html.H2('Car Price Predictor', className='mt-4'),
    dbc.Alert([
        html.B('How it works. '),
        'Fill in what you know about the car and press Predict. You can leave any field empty: '
        'the model then uses the typical value from the training data (shown under each field). '
        'The prediction comes from a Random Forest trained on 8,000 used cars, and is usually '
        'within about 14% of the real price. Year and max power matter most.',
    ], color='info'),
    form,
    dbc.Button('Predict', id='submit', color='primary', className='mb-3'),
    html.Div(id='result'),
], style={'maxWidth': '900px'})


@app.callback(Output('result', 'children'),
              Input('submit', 'n_clicks'),
              [State(i, 'value') for i in INPUT_IDS],
              prevent_initial_call=True)
def on_submit(_, *values):
    price = predict_price(dict(zip(INPUT_IDS, values)))
    skipped = [i for i, v in zip(INPUT_IDS, values) if v in (None, '')]
    note = f" (filled in for you: {', '.join(skipped)})" if skipped else ''
    return dbc.Alert([html.H4(f'Predicted price: {price:,.0f}'), html.Small(note)], color='success')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8050, debug=False)
