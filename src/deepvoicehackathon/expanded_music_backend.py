"""Expanded MUSIC17 inference, with explicit full/fold training receipts.

Invalid evidence or model metadata raises ValueError; the runner owns fallback.
"""
from . import music_backend as music

FEATURES = music.FEATURES
SCORE_FIELDS = music.SCORE_FIELDS
EPSILON = music.EPSILON
VERSION = 3
MODEL_TYPE = 'gradient_boosted_trees_music17_expanded'
PROTOCOL_SHA = '4e02efea855c4e86f34df6d72d27ceadbfdf17681a81efe6a10ca15c77e87e22'
SOURCE44_SHA = '4c143dcf09155b24990bb0d638cb1432b90dfc061b6cef654b0e1ad408c9c841'
BACKEND44_SHA = '19816ba06811cb3b385802ea35f837a9a219891525f7dc2b02c71d098d6be9f7'
RECIPE = 'fixed64-depth2-lr0.05-leaf24-music17-expanded'


def features(rows):
    return music.features(rows)


def _legacy_model(model):
    return dict(model, version=music.VERSION, model_type=music.MODEL_TYPE,
                protocol_sha256=music.PROTOCOL_SHA, recipe=music.RECIPE,
                training_rows=1680, training_groups=40)


def validate_model(model, *, allow_fold=False):
    """Release accepts only the full receipt; fold inference is explicit opt-in."""
    try:
        fixed = dict(version=VERSION, model_type=MODEL_TYPE,
                     protocol_sha256=PROTOCOL_SHA, recipe=RECIPE,
                     training_rows=2115, training_groups=69,
                     training_class_counts={'0': 927, '1': 1188},
                     source44_sha256=SOURCE44_SHA, backend44_sha256=BACKEND44_SHA)
        if not isinstance(model, dict) or any(model.get(k) != v for k, v in fixed.items()):
            raise ValueError('incompatible expanded MUSIC17 metadata')
        for key in ('version', 'training_rows', 'training_groups', 'fit_rows', 'fit_groups'):
            if type(model[key]) is not int:
                raise ValueError('integer population metadata required')
        for key in ('training_class_counts', 'fit_class_counts'):
            counts = model[key]
            if (not isinstance(counts, dict) or set(counts) != {'0', '1'}
                    or any(type(v) is not int or v <= 0 for v in counts.values())):
                raise ValueError('explicit binary class counts required')
        receipt = (model['fit_rows'], model['fit_groups'],
                   model['fit_class_counts']['0'], model['fit_class_counts']['1'])
        full = (2115, 69, 927, 1188)
        folds = {(1695, 59, 717, 978), (1905, 64, 822, 1083),
                 (1995, 61, 903, 1092), (2010, 62, 906, 1104)}
        if model['fit_role'] == 'full':
            if receipt != full:
                raise ValueError('full model requires actual full training receipt')
        elif model['fit_role'] != 'fold' or not allow_fold or receipt not in folds:
            raise ValueError('fold model is not a release model')
        music.validate_model(_legacy_model(model))
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError('malformed expanded MUSIC17 tree export') from error


def predict(model, rows, *, allow_fold=False):
    validate_model(model, allow_fold=allow_fold)
    return music.predict(_legacy_model(model), rows)
