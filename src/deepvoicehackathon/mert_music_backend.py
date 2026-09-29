"""Portable frozen MUSIC785 logistic inference; invalid evidence raises ValueError.

Only the caller may retain the incumbent on failure. This module performs no
fitting, model loading, audio processing, or batch-dependent normalization.
"""
from __future__ import annotations

import json

import numpy as np
from scipy.special import expit

from . import music_backend as music

VERSION = 1
MODEL_TYPE = 'logistic_music785_mert'
SCORE_FIELDS = (
    'df_voice_mean', 'df_music_mean', 'df_music_max', 'spectra', 'nii',
    'artifactnet', 'lofcz', 'current_voice', 'current_music', 'voice_present',
    'music_present', 'current_file', 'file_music',
)
FEATURES = (*music.FEATURES, *(f'mert:{i:03d}' for i in range(768)))
EPSILON = 1e-6
PROTOCOL_SHA = 'd8fcd4a972c868bca75c303379cb7729163f978c8ee58031deac006d8507e92d'
SOURCE50_SHA = 'f3494938d11ae1948dd319de0c40aab197eb78c381089007914bf6daba21dc07'
MERT_REVISION = '12af15fef9d0ac838c3f475bfbbf26d2060dd4f5'
MERT_WEIGHTS_SHA = 'a2b8b747f72c06e0595aeae41ae5473f4364938c6b39b2c58be38c48e6bd3fcd'
RECIPE = 'weighted-standardscaler-logistic-C1-l2-lbfgs-music785-mert95m-v1'
IMPLEMENTATIONS = ('src/deepvoicehackathon/mert_music_backend.py',
                   'tools/train_sub51_mert_music.py')
MAX_MODEL_BYTES = 1_000_000


def model_metadata():
    return dict(version=VERSION, model_type=MODEL_TYPE, task='MUSIC_FAKE only',
        feature_names=list(FEATURES), epsilon=EPSILON, input_dtype='float64',
        embedding_dtype='float32', C=1.0, penalty='l2', solver='lbfgs',
        max_iter=2000, tol=1e-4, fit_intercept=True, class_weight=None,
        seed=20260905, training_rows=1680, training_groups=40,
        view_weights=dict(clean=.5, mp3_64=.25, telephone=.25),
        class_balance='fold-local n/(2*n_class); same weights for scaler and classifier',
        protocol_sha256=PROTOCOL_SHA, source50_sha256=SOURCE50_SHA,
        mert_model='m-a-p/MERT-v1-95M', mert_revision=MERT_REVISION,
        mert_weights_sha256=MERT_WEIGHTS_SHA, recipe=RECIPE)


def _sha(value):
    return (type(value) is str and len(value) == 64
            and all(c in '0123456789abcdef' for c in value))


def _real(value):
    return type(value) in (int, float) and np.isfinite(value)


def _same_type_value(actual, expected):
    if type(actual) is not type(expected):
        return False
    if type(expected) is dict:
        return (set(actual) == set(expected)
                and all(_same_type_value(actual[k], v) for k, v in expected.items()))
    if type(expected) is list:
        return (len(actual) == len(expected)
                and all(_same_type_value(a, b) for a, b in zip(actual, expected)))
    return actual == expected


def validate_receipt(receipt):
    """Compact receipt binds the full identities/weights retained in the fit journal."""
    keys = {'fit_id', 'role', 'scheme', 'fold', 'training_rows', 'training_groups',
            'validation_rows', 'validation_groups', 'training_labels',
            'training_identities_sha256', 'validation_identities_sha256',
            'weights_sha256', 'weights_sum', 'scaler_training_rows',
            'scaler_weight_sum', 'classifier_n_iter'}
    if type(receipt) is not dict or set(receipt) != keys:
        raise ValueError('strict fit receipt required')
    for key in ('fit_id', 'fold', 'training_rows', 'training_groups', 'validation_rows',
                'validation_groups', 'scaler_training_rows', 'classifier_n_iter'):
        if type(receipt[key]) is not int:
            raise ValueError('integer fit receipt fields required')
    if not (1 <= receipt['fit_id'] <= 17 and 0 < receipt['training_rows'] <= 1680
            and 0 < receipt['training_groups'] <= 40
            and 0 <= receipt['validation_rows'] <= 1680
            and 0 <= receipt['validation_groups'] <= 40
            and receipt['scaler_training_rows'] == receipt['training_rows']
            and 1 <= receipt['classifier_n_iter'] <= 2000):
        raise ValueError('invalid fit receipt counts')
    if receipt['role'] == 'full':
        if (receipt['scheme'], receipt['fold'], receipt['fit_id'], receipt['training_rows'],
                receipt['training_groups'], receipt['validation_rows'], receipt['validation_groups']) != (
                'full', -1, 17, 1680, 40, 0, 0):
            raise ValueError('full receipt must describe exactly the full fit')
    elif receipt['role'] == 'oof_fold':
        scheme = receipt['scheme']
        if (type(scheme) is not str or scheme not in ('20260905', '20260908', 'generator_out')
                or not 0 <= receipt['fold'] < (8 if scheme == 'generator_out' else 4)
                or receipt['fit_id'] != (0 if scheme == '20260905' else 4 if scheme == '20260908' else 8) + receipt['fold'] + 1
                or receipt['training_rows'] + receipt['validation_rows'] != 1680
                or receipt['training_groups'] + receipt['validation_groups'] != 40
                or receipt['validation_rows'] == 0 or receipt['validation_groups'] == 0):
            raise ValueError('fold receipt must describe a disjoint frozen split')
    else:
        raise ValueError('unknown fit role')
    labels = receipt['training_labels']
    if (type(labels) is not dict or set(labels) != {'0', '1'}
            or any(type(v) is not int or v <= 0 for v in labels.values())
            or sum(labels.values()) != receipt['training_rows']):
        raise ValueError('actual MUSIC training label counts required')
    for key in ('training_identities_sha256', 'validation_identities_sha256', 'weights_sha256'):
        if not _sha(receipt[key]):
            raise ValueError('receipt SHA256 required')
    if (not _real(receipt['weights_sum']) or receipt['weights_sum'] <= 0
            or not _real(receipt['scaler_weight_sum'])
            or not np.isclose(receipt['weights_sum'], receipt['scaler_weight_sum'], rtol=1e-12, atol=1e-12)):
        raise ValueError('scaler must consume the same training weights')


def validate_model(model):
    try:
        fixed = model_metadata()
        if (type(model) is not dict or set(model) != set(fixed) |
                {'mean', 'scale', 'coef', 'intercept', 'implementation_sha256', 'fit_receipt'}):
            raise ValueError('strict model keys required')
        for key, value in fixed.items():
            if not _same_type_value(model[key], value):
                raise ValueError('fixed recipe metadata mismatch: ' + key)
        pins = model['implementation_sha256']
        if type(pins) is not dict or set(pins) != set(IMPLEMENTATIONS) or not all(map(_sha, pins.values())):
            raise ValueError('portable two-file implementation SHA256 map required')
        for key in ('mean', 'scale', 'coef'):
            values = model[key]
            if type(values) is not list or len(values) != 785 or not all(map(_real, values)):
                raise ValueError('finite JSON real array of length785 required')
        if any(v <= 0 for v in model['scale']) or not _real(model['intercept']):
            raise ValueError('positive scales and finite intercept required')
        validate_receipt(model['fit_receipt'])
        if len(json.dumps(model, allow_nan=False, separators=(',', ':')).encode('utf-8')) >= MAX_MODEL_BYTES:
            raise ValueError('model must be under1MB')
    except (KeyError, TypeError, IndexError, ValueError, OverflowError) as error:
        raise ValueError('malformed MUSIC785 logistic export') from error


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key: ' + key)
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError('nonfinite JSON constant: ' + value)


def loads(raw):
    """Strict JSON boundary, including duplicate keys and raw byte size."""
    if type(raw) not in (str, bytes) or len(raw.encode('utf-8') if isinstance(raw, str) else raw) >= MAX_MODEL_BYTES:
        raise ValueError('model JSON must be under1MB')
    try:
        model = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
        validate_model(model)
        return model
    except (UnicodeError, TypeError, ValueError, OverflowError) as error:
        raise ValueError('invalid MUSIC785 JSON') from error


def features(rows, embeddings):
    """Exact MUSIC17 float64 prefix plus finite768 float32 values promoted to64."""
    try:
        prefix = music.features(rows)
        embeddings = np.asarray(embeddings)
        if (not isinstance(embeddings, np.ndarray) or embeddings.dtype != np.dtype('float32')
                or embeddings.shape != (len(rows), 768) or not np.isfinite(embeddings).all()):
            raise ValueError('finite float32 [rows,768] embedding matrix required')
        result = np.column_stack((prefix, embeddings.astype(np.float64)))
        if result.dtype != np.float64 or result.shape != (len(rows), 785) or not np.isfinite(result).all():
            raise ValueError('finite float64 MUSIC785 features required')
        return result
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError('invalid MUSIC785 original probabilities/embeddings') from error


def predict(model, rows, embeddings):
    validate_model(model)
    x = features(rows, embeddings)
    mean, scale, coefficient = (np.asarray(model[k], dtype=np.float64)
                                for k in ('mean', 'scale', 'coef'))
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        raw = ((x - mean) / scale) @ coefficient + model['intercept']
    if not np.isfinite(raw).all():
        raise ValueError('nonfinite MUSIC785 log odds')
    return expit(raw)
