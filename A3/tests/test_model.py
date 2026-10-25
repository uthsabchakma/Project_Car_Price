"""Unit tests for the A3 model (run from the repo root:  pytest A3/tests)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from app_a3 import MODELS, prepare_input, predict_class  # noqa: E402

SAMPLE = {'brand': 'Maruti', 'year': 2014, 'km_driven': 145500, 'fuel': 'Diesel', 'seller_type': 'Individual',
          'transmission': 'Manual', 'owner': 1, 'mileage': 23.4, 'engine': 1248, 'max_power': 74, 'seats': 5}


def test_model_takes_expected_input():
    """The model accepts a row with exactly the columns it was trained on, and rejects anything else."""
    meta, model = MODELS['a3']['meta'], MODELS['a3']['model']
    X = prepare_input('a3', SAMPLE)
    assert list(X.columns) == meta['columns']
    assert X.shape == (1, len(meta['columns']))
    model.predict(X.values.astype(float))  # must not raise
    with pytest.raises(ValueError):
        model.predict(np.zeros((1, len(meta['columns']) - 1)))  # wrong number of features


def test_model_output_shape():
    """One prediction per input row, each one a class between 0 and 3."""
    model = MODELS['a3']['model']
    X = np.vstack([prepare_input('a3', SAMPLE).values, prepare_input('a3', {}).values]).astype(float)
    pred = model.predict(X)
    assert pred.shape == (2,)
    assert set(pred) <= {0, 1, 2, 3}
    assert model.predict_proba(X).shape == (2, 4)


def test_skipped_fields_are_imputed():
    """An empty form still gives a valid class (the app fills skipped fields with training medians/modes)."""
    assert predict_class({}) in (0, 1, 2, 3)
