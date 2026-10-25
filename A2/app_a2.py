"""A2 web app: two pages, the old A1 model (Random Forest) and the new A2 model (our own
LinearRegression class).

Run locally (from the A2 folder):  python app_a2.py   then open http://localhost:8050
"""
import os
import sys

import joblib
import numpy as np
import pandas as pd
import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, dcc, html

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # the A2 model is an instance of linear_regression.py, pickle needs to import it

MODEL_DIRS = {
    'a1': os.path.join(HERE, '..', 'A1', 'models_a1'),
    'a2': os.path.join(HERE, 'models_a2'),
}


def load(name):
    d = MODEL_DIRS[name]
    return {'model': joblib.load(os.path.join(d, 'car_price_model.pkl')),
            'scaler': joblib.load(os.path.join(d, 'scaler.pkl')),
            'meta': joblib.load(os.path.join(d, 'model_columns.pkl'))}


MODELS = {name: load(name) for name in MODEL_DIRS}


def predict_price(name, inputs):
    """Raw form values -> imputed, encoded, scaled row -> price. Same steps as the notebooks."""
    m = MODELS[name]
    meta, fill = m['meta'], m['meta']['fill_values']
    row = {c: (fill[c] if inputs.get(c) in (None, '') else inputs[c])
           for c in meta['num_cols'] + meta['cat_cols']}
    # The A2 polynomial model extrapolates badly outside the training data -> clip to the seen range
    for c, (lo, hi) in meta.get('num_ranges', {}).items():
        row[c] = min(max(float(row[c]), lo), hi)
    df = pd.get_dummies(pd.DataFrame([row]), columns=meta['cat_cols'], dtype=int)
    df = df.reindex(columns=meta['columns'], fill_value=0)
    df[meta['num_cols']] = m['scaler'].transform(df[meta['num_cols']].astype(float))
    pred_log = m['model'].predict(df.values.astype(float) if name == 'a2' else df)
    return float(np.exp(pred_log[0]))  # both models predict log(price)


INPUT_IDS = ['year', 'km_driven', 'owner', 'mileage', 'engine', 'max_power', 'seats',
             'fuel', 'seller_type', 'transmission', 'brand']


def form(prefix):
    """The same input form for both pages; ids are prefixed so the pages don't clash."""
    meta = MODELS[prefix]['meta']
    fill = meta['fill_values']

    def field(label, comp, hint):
        return dbc.Col([dbc.Label(label), comp, dbc.FormText(hint)], md=4, className='mb-3')

    def num(id_, ph, step='any'):
        return dbc.Input(id=f'{prefix}-{id_}', type='number', placeholder=ph, step=step)

    def dd(id_, opts):
        return dcc.Dropdown(id=f'{prefix}-{id_}', options=opts, placeholder='Skip to use the most common')

    owners = [{'label': l, 'value': v} for l, v in [('First', 1), ('Second', 2), ('Third', 3), ('Fourth & above', 4)]]
    return html.Div([
        dbc.Row([
            field('Brand', dd('brand', meta['brands']), f"default: {fill['brand']}"),
            field('Year', num('year', 'e.g. 2015', 1), f"default: {fill['year']:.0f}"),
            field('Kilometres driven', num('km_driven', 'e.g. 60000', 1), f"default: {fill['km_driven']:,.0f}"),
            field('Fuel', dd('fuel', ['Diesel', 'Petrol']), f"default: {fill['fuel']}"),
            field('Transmission', dd('transmission', ['Manual', 'Automatic']), f"default: {fill['transmission']}"),
            field('Seller type', dd('seller_type', ['Individual', 'Dealer', 'Trustmark Dealer']), f"default: {fill['seller_type']}"),
            field('Owner', dd('owner', owners), 'default: First'),
            field('Mileage (kmpl)', num('mileage', 'e.g. 19.3'), f"default: {fill['mileage']}"),
            field('Engine (CC)', num('engine', 'e.g. 1248', 1), f"default: {fill['engine']:.0f}"),
            field('Max power (bhp)', num('max_power', 'e.g. 82.85'), f"default: {fill['max_power']}"),
            field('Seats', num('seats', 'e.g. 5', 1), f"default: {fill['seats']:.0f}"),
        ]),
        dbc.Button('Predict', id=f'{prefix}-submit', color='primary', className='mb-3'),
        html.Div(id=f'{prefix}-result'),
    ])


a2_cfg = MODELS['a2']['meta']['config']
PAGES = {
    '/': ('Home', html.Div([
        html.H2('Car Price Predictor'),
        html.P('Two models predict the selling price of a used car. Pick one in the menu above.'),
        html.Ul([
            html.Li([html.B('Old model (A1): '), 'a Random Forest from scikit-learn.']),
            html.Li([html.B('New model (A2): '), 'a linear regression we wrote from scratch, '
                     'chosen from 144 experiments tracked with MLflow.']),
        ]),
    ])),
    '/a1': ('Old model (A1)', html.Div([
        html.H2('Old model: Random Forest (A1)'),
        dbc.Alert('Fill in what you know and press Predict. Empty fields are filled with the typical '
                  'value from the training data (shown under each field). Test R² ≈ 0.94.', color='info'),
        form('a1'),
    ])),
    '/a2': ('New model (A2)', html.Div([
        html.H2('New model: Linear Regression from scratch (A2)'),
        dbc.Alert([
            html.P('How to use: fill in what you know and press Predict. Empty fields are filled with the '
                   'typical value from the training data, exactly like the old page.'),
            html.P(['What is new: this model is our own gradient-descent linear regression '
                    f"({a2_cfg['regression']}, {a2_cfg['method']} gradient descent, {a2_cfg['init']} init, "
                    f"learning rate {a2_cfg['lr']}, momentum {a2_cfg['momentum']}), picked as the best of 144 "
                    'cross-validated MLflow runs. It is far smaller and faster than the Random Forest, and every '
                    'prediction can be explained: each feature has one coefficient that says how much it raises or '
                    f"lowers the price. Test R² ≈ {MODELS['a2']['meta']['test_r2']:.2f}, a little below the old "
                    'model, which captures non-linear effects better.'], className='mb-0'),
        ], color='info'),
        form('a2'),
    ])),
}

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], title='Car Price Predictor',
           suppress_callback_exceptions=True)
server = app.server  # for gunicorn

navbar = dbc.NavbarSimple(
    [dbc.NavItem(dbc.NavLink(title, href=path, active='exact')) for path, (title, _) in PAGES.items() if path != '/'],
    brand='Car Price', brand_href='/', color='dark', dark=True)

# every page is in the layout from the start (hidden when not active), so all callbacks are valid
app.layout = html.Div([
    dcc.Location(id='url'),
    navbar,
    dbc.Container([html.Div(content, id=f'page{path.replace("/", "-")}') for path, (_, content) in PAGES.items()],
                  className='mt-4', style={'maxWidth': '900px'}),
])


@app.callback([Output(f'page{p.replace("/", "-")}', 'style') for p in PAGES], Input('url', 'pathname'))
def show_page(path):
    path = path if path in PAGES else '/'
    return [{'display': 'block' if p == path else 'none'} for p in PAGES]


def register(prefix):
    @app.callback(Output(f'{prefix}-result', 'children'),
                  Input(f'{prefix}-submit', 'n_clicks'),
                  [State(f'{prefix}-{i}', 'value') for i in INPUT_IDS],
                  prevent_initial_call=True)
    def on_submit(_, *values):
        price = predict_price(prefix, dict(zip(INPUT_IDS, values)))
        skipped = [i for i, v in zip(INPUT_IDS, values) if v in (None, '')]
        note = f"Filled in for you: {', '.join(skipped)}" if skipped else ''
        return dbc.Alert([html.H4(f'Predicted price: {price:,.0f}'), html.Small(note)], color='success')


register('a1')
register('a2')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8050)), debug=False)
