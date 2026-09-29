"""Isolated19-feature peak-aware FILE export; legacy17-feature contract unchanged."""
import numpy as np
from scipy.special import expit

from .file_backend import FEATURES as BASE_FEATURES, EPSILON, features as base_features

FEATURES=(*BASE_FEATURES,'logit:df_voice_max','logit:voice_present*df_voice_max')


def features(rows):
    original=base_features(rows)
    try:
        maximum=np.asarray([float(r['df_voice_max']) for r in rows],dtype=np.float64)
        presence=np.asarray([float(r['voice_present']) for r in rows],dtype=np.float64)
    except (KeyError,TypeError,ValueError,OverflowError) as error:
        raise ValueError('missing/invalid peak feature input') from error
    if not np.isfinite(maximum).all() or ((maximum<0)|(maximum>1)).any():
        raise ValueError('peak probability must be finite in[0,1]')
    bounded=np.clip(np.column_stack((maximum,presence*maximum)),EPSILON,1-EPSILON)
    return np.column_stack((original,np.log(bounded)-np.log1p(-bounded)))


def predict(model,rows):
    if (model.get('version')!=4 or model.get('model_type')!='gradient_boosted_trees_voice_peak'
            or model.get('feature_names')!=list(FEATURES) or model.get('epsilon')!=EPSILON
            or model.get('input_dtype')!='float32'):
        raise ValueError('incompatible peak-aware export')
    x=features(rows).astype(np.float32)
    raw=np.full(len(rows),model['initial_log_odds'],dtype=np.float64)
    for tree in model['trees']:
        for i,values in enumerate(x):
            node=0
            while tree['children_left'][node]!=-1:
                node=(tree['children_left'][node] if float(values[tree['feature'][node]])<=float(tree['threshold'][node])
                      else tree['children_right'][node])
            raw[i]+=model['learning_rate']*tree['value'][node]
    if not np.isfinite(raw).all():raise ValueError('nonfinite tree log odds')
    return expit(raw)
