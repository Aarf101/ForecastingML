"""Handler Chronos-2 commun à l’évaluation et au Batch Transform CPU."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from retail_forecast.evaluate.benchmark_chronos import context_frames, MODEL_REVISION
from retail_forecast.features.build import active_catalog

CONTEXT_LENGTH = 256
CHUNK_SIZE = 32


def requests_from_sales(sales, chunk_size=CHUNK_SIZE):
    """Sérialise des historiques quotidiens complets, sans inventer de date manquante."""
    sales = active_catalog(sales)
    origin = pd.Timestamp(sales.date.max())
    expected = pd.date_range(end=origin, periods=CONTEXT_LENGTH)
    requests, block = [], []
    for (store, item), history in sales.groupby(['store_id', 'item_id'], sort=True):
        history = history.sort_values('date').tail(CONTEXT_LENGTH)
        if not pd.DatetimeIndex(history.date).equals(expected):
            raise ValueError(f'historique incomplet : {store}/{item}')
        values = history.units.to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError('ventes invalides')
        block.append({'series_id':f'{store}/{item}', 'origin_date':origin.strftime('%Y-%m-%d'), 'target':values.tolist()})
        if len(block) == chunk_size:
            requests.append({'series':block});block=[]
    if block:
        requests.append({'series':block})
    return requests


def input_fn(body, content_type, context=None):
    """Chaque ligne contient au plus 32 séries, chacune avec 256 observations."""
    if content_type.split(';')[0] != 'application/json':
        raise ValueError('application/json requis')
    document = json.loads(body)
    if set(document) != {'series'} or not isinstance(document['series'], list) or not 1 <= len(document['series']) <= CHUNK_SIZE:
        raise ValueError('lot de séries invalide')
    ids, origins = set(), set()
    for series in document['series']:
        if set(series) != {'series_id','origin_date','target'}:
            raise ValueError('schéma de série invalide')
        identifier = series['series_id']
        if not isinstance(identifier, str) or len(identifier.split('/')) != 2 or identifier in ids:
            raise ValueError('identifiant invalide ou dupliqué')
        values = np.asarray(series['target'],dtype=float)
        if values.shape != (CONTEXT_LENGTH,) or not np.isfinite(values).all() or (values < 0).any():
            raise ValueError('contexte quotidien incomplet ou invalide')
        ids.add(identifier);origins.add(pd.Timestamp(series['origin_date']))
    if len(origins) != 1 or any(pd.isna(origin) for origin in origins):
        raise ValueError('origine commune valide requise')
    return document


def model_fn(model_dir, context=None):
    """Charge les poids figés ; aucun téléchargement ni entraînement à l'inférence."""
    import torch
    from chronos import Chronos2Pipeline
    torch.set_num_threads(2)
    root = Path(model_dir)
    metadata = json.loads((root/'chronos_metadata.json').read_text())
    if metadata['revision'] != MODEL_REVISION or metadata['context_length'] != CONTEXT_LENGTH:
        raise ValueError('configuration non évaluée')
    pipeline = Chronos2Pipeline.from_pretrained(root/'weights', device_map='cpu', dtype=torch.float32, local_files_only=True)
    return {'pipeline':pipeline,'metadata':metadata}


def predict_fn(document, model, context=None):
    """Même calendrier, médiane et quantiles que le benchmark évalué."""
    frames=[]
    origin = pd.Timestamp(document['series'][0]['origin_date'])
    for series in document['series']:
        frames.append(pd.DataFrame({'series_id':series['series_id'], 'date':pd.date_range(end=origin,periods=CONTEXT_LENGTH), 'units':series['target']}))
    history, future = context_frames(pd.concat(frames,ignore_index=True), origin, CONTEXT_LENGTH, 'calendar')
    prediction = model['pipeline'].predict_df(history, future_df=future, id_column='series_id', timestamp_column='date', target='target',
        prediction_length=16, quantile_levels=[.1,.5,.9], context_length=CONTEXT_LENGTH, batch_size=128, cross_learning=False, freq='D')
    result=prediction.rename(columns={'date':'target_date','predictions':'forecast_units','0.1':'p10','0.9':'p90'}).copy()
    result[['store_id','item_id']]=result.series_id.str.split('/',expand=True)
    result['origin_date']=origin;result['horizon']=(result.target_date-origin).dt.days
    result[['forecast_units','p10','p90']]=result[['forecast_units','p10','p90']].clip(lower=0)
    low, high=result.p10.copy(), result.p90.copy()
    result['p10'],result['p90']=np.minimum(low,high),np.maximum(low,high)
    if (len(result)!=16*len(document['series']) or result.duplicated(['series_id','horizon']).any()
            or set(result.horizon)!=set(range(1,17)) or set(result.series_id)!=set(history.series_id)
            or not np.isfinite(result[['forecast_units','p10','p90']]).all().all()
            or not (result.forecast_units.ge(result.p10-1e-5)&result.forecast_units.le(result.p90+1e-5)).all()):
        raise ValueError('sortie Chronos incomplète ou quantiles incohérents')
    for name in ['origin_date','target_date']:
        result[name]=result[name].dt.strftime('%Y-%m-%d')
    return result[['store_id','item_id','origin_date','target_date','horizon','forecast_units','p10','p90']].to_dict('records')


def output_fn(predictions, accept, context=None):
    """JSON strict : aucune valeur non finie n'est publiable."""
    if accept.split(';')[0] != 'application/json':
        raise ValueError('application/json requis')
    return json.dumps(predictions,allow_nan=False)
