"""Compare Chronos-2 au modèle actif, sur les mêmes couples origine/cible."""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import tarfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from retail_forecast.evaluate.metrics import wape, rmsle, pinball_loss
from retail_forecast.data.simulate import HOLIDAYS_MMDD
from retail_forecast.features.build import active_catalog, build_features, add_target, time_split, TARGET

KEYS = ['store_id', 'item_id', 'date', 'horizon']
MODEL_REVISION = '29ec3766d36d6f73f0696f85560a422f50e8498c'


def evaluation_panel(sales: pd.DataFrame, archive: Path) -> pd.DataFrame:
    """Recalcule les prédictions évaluées sans réentraîner ni modifier l'artefact."""
    import lightgbm as lgb
    with tarfile.open(archive) as tar:
        metadata = json.load(tar.extractfile('metadata.json'))
        sales = active_catalog(sales)
        # 180 jours suffisent aux lags et aux frontières validation/test inchangées.
        recent = sales[sales.date > sales.date.max() - pd.Timedelta(days=180)]
        base = build_features(recent, horizon=1, dropna=False, levels=metadata['categorical_levels'])
        features = metadata['features']
        parts = []
        for horizon in metadata['horizons']:
            data = add_target(base.drop(columns=[TARGET]), horizon)
            data = data.dropna(subset=[*features, TARGET, 'seasonal_naive'])
            _, _, test = time_split(data, **metadata['split'])
            test['horizon'] = horizon
            result = test[[*KEYS, 'target_date', TARGET, 'seasonal_naive']].copy()
            for column, name in [('lightgbm', f'model_{horizon}.txt'),
                                 ('lightgbm_p10', f'quantile_{horizon}_10.txt'),
                                 ('lightgbm_p90', f'quantile_{horizon}_90.txt')]:
                booster = lgb.Booster(model_str=tar.extractfile(name).read().decode())
                result[column] = np.clip(booster.predict(test[features], num_threads=2), 0, None)
            lo, hi = result.lightgbm_p10.copy(), result.lightgbm_p90.copy()
            result['lightgbm_p10'], result['lightgbm_p90'] = np.minimum(lo, hi), np.maximum(lo, hi)
            parts.append(result)
        return pd.concat(parts, ignore_index=True)


def select_series(sales: pd.DataFrame, limit: int) -> list[tuple]:
    """Sous-échantillon pilote déterministe réparti sur les magasins ; jamais un split aléatoire."""
    pairs = sales[['store_id', 'item_id']].drop_duplicates().sort_values(['store_id', 'item_id'])
    if limit and limit < len(pairs):
        # Sélection par identifiants uniquement, équilibrée entre magasins.
        pairs = pairs.assign(rank=[hashlib.sha256(f'{s}/{i}'.encode()).hexdigest()
                                   for s, i in pairs.itertuples(index=False, name=None)])
        pairs = pairs.sort_values(['store_id','rank'])
        pairs['within_store'] = pairs.groupby('store_id').cumcount()
        pairs = pairs.sort_values(['within_store','store_id']).head(limit)
    return list(pairs[['store_id','item_id']].itertuples(index=False, name=None))


def context_frames(sales, origin, context_length, covariates):
    """N'utilise aucune vente, promotion ou prix observé après l'origine."""
    history = sales[(sales.date <= origin) & (sales.date > origin-pd.Timedelta(days=context_length))]
    context = history[['series_id', 'date', 'units']].rename(columns={'units': 'target'}).copy()
    ids = sorted(context.series_id.unique())
    future = pd.MultiIndex.from_product([ids, pd.date_range(origin+pd.Timedelta(days=1), periods=16)], names=['series_id','date']).to_frame(index=False)
    if covariates != 'none':
        for frame in [context, future]:
            for name, values in [('dow', frame.date.dt.dayofweek), ('month', frame.date.dt.month), ('day', frame.date.dt.day)]:
                frame[name] = values.astype('float32')
            frame['is_holiday'] = frame.date.dt.strftime('%m-%d').isin(HOLIDAYS_MMDD).astype('float32')
        if covariates == 'retail':
            for name in ['promo', 'unit_price', 'lead_time_days']:
                context[name] = history[name].to_numpy(dtype='float32')
    return context.sort_values(['series_id','date']), future.sort_values(['series_id','date'])


def score(panel, predictions):
    """Mêmes observations, WAPE moyen des horizons et couverture des deux modèles."""
    matched = panel.merge(predictions, on=KEYS, how='left', validate='one_to_one')
    if matched[['chronos','chronos_p10','chronos_p90']].isna().any().any():
        raise ValueError('prévisions Chronos manquantes : comparaison bloquée')
    report = {}
    for name in ['lightgbm', 'chronos', 'seasonal_naive']:
        horizons = {}
        for h, frame in matched.groupby('horizon'):
            y, pred = frame[TARGET].to_numpy(), frame[name].to_numpy()
            metrics = {'wape': wape(y, pred), 'rmsle': rmsle(y, pred), 'rows': len(frame)}
            if name != 'seasonal_naive':
                lo, hi = frame[name+'_p10'].to_numpy(), frame[name+'_p90'].to_numpy()
                metrics.update(coverage_80=float(np.mean((y >= lo) & (y <= hi))),
                               pinball_p10=pinball_loss(y, lo, .1), pinball_p90=pinball_loss(y, hi, .9))
            horizons[str(h)] = metrics
        report[name] = {'mean_wape': float(np.mean([m['wape'] for m in horizons.values()])),
                        'mean_rmsle': float(np.mean([m['rmsle'] for m in horizons.values()])), 'by_horizon': horizons}
        if name != 'seasonal_naive':
            report[name]['mean_coverage_80'] = float(np.mean([m['coverage_80'] for m in horizons.values()]))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/raw/sales.parquet'))
    parser.add_argument('--artifact', type=Path, default=Path('data/benchmark/active-model.tar.gz'))
    parser.add_argument('--model-path', default='amazon/chronos-2')
    parser.add_argument('--context-length', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    parser.add_argument('--series-limit', type=int, default=0)
    parser.add_argument('--origin-limit', type=int, default=0)
    parser.add_argument('--covariates', choices=['none','calendar','retail'], default='calendar')
    parser.add_argument('--output', type=Path, default=Path('reports/chronos2_benchmark.json'))
    args = parser.parse_args(argv)
    started = time.monotonic()
    sales = pd.read_parquet(args.data);sales['date'] = pd.to_datetime(sales.date)
    digest = hashlib.sha256(args.artifact.read_bytes()+args.data.read_bytes()).hexdigest()
    cache = Path('data/benchmark');cache.mkdir(exist_ok=True)
    panel_path = cache/f'panel-{digest[:12]}.parquet'
    if not panel_path.exists():
        print('Recomputing reference evaluation panel', flush=True)
        evaluation_panel(sales, args.artifact).to_parquet(panel_path, index=False)
    panel = pd.read_parquet(panel_path)
    pairs = select_series(sales, args.series_limit)
    selection = pd.MultiIndex.from_tuples(pairs, names=['store_id','item_id'])
    sales = sales[sales.set_index(['store_id','item_id']).index.isin(selection)].copy()
    panel = panel[panel.set_index(['store_id','item_id']).index.isin(selection)].copy()
    origins = sorted(panel.date.unique())
    if args.origin_limit and args.origin_limit < len(origins):
        origins = [origins[i] for i in np.linspace(0,len(origins)-1,args.origin_limit,dtype=int)]
        panel = panel[panel.date.isin(origins)].copy()
    sales['series_id'] = sales.store_id + '/' + sales.item_id
    config = {'model': 'amazon/chronos-2', 'revision': MODEL_REVISION,
              'context_length': args.context_length, 'covariates': args.covariates, 'calendar_holidays': True,
              'cross_learning': False, 'batch_size': args.batch_size, 'threads': args.threads,
              'series': len(pairs), 'origins': len(origins), 'rows_scored': len(panel),
              'dataset_and_artifact_sha256': digest, 'point_statistic': 'Chronos median vs LightGBM regression mean',
              'device': args.device, 'fine_tuned': False, 'protocol': 'matched_rolling_origin_purged_v2'}
    directory = cache/('chronos-'+hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:12]);directory.mkdir(exist_ok=True)
    gc.collect()
    import torch
    from chronos import Chronos2Pipeline
    torch.set_num_threads(args.threads)
    pipeline = Chronos2Pipeline.from_pretrained(args.model_path, device_map=args.device, dtype=torch.float32, revision=MODEL_REVISION)
    print(f'Loaded Chronos-2; {len(pairs)} series, {len(origins)} origins, {len(panel)} matched observations',flush=True)
    inference_time = 0.0
    for i, value in enumerate(origins):
        origin = pd.Timestamp(value);path = directory/(origin.strftime('%Y-%m-%d')+'.parquet')
        if path.exists():continue
        context, future = context_frames(sales, origin, args.context_length, args.covariates)
        tick = time.monotonic()
        pred = pipeline.predict_df(context, future_df=future if args.covariates!='none' else None,
            id_column='series_id', timestamp_column='date', target='target', prediction_length=16,
            quantile_levels=[.1,.5,.9], context_length=args.context_length,
            batch_size=args.batch_size, cross_learning=False, freq='D')
        elapsed = time.monotonic()-tick;inference_time += elapsed
        pred = pred.rename(columns={'date':'target_date','predictions':'chronos','0.1':'chronos_p10','0.9':'chronos_p90'})
        pred['date'] = origin;pred['horizon'] = (pred.target_date-origin).dt.days
        pred[['store_id','item_id']] = pred.series_id.str.split('/',expand=True)
        pred[['chronos','chronos_p10','chronos_p90']] = pred[['chronos','chronos_p10','chronos_p90']].clip(lower=0)
        if not np.isfinite(pred[['chronos','chronos_p10','chronos_p90']]).all().all():raise ValueError('prédictions non finies')
        lower, upper = pred.chronos_p10.copy(), pred.chronos_p90.copy()
        pred['chronos_p10'], pred['chronos_p90'] = np.minimum(lower,upper), np.maximum(lower,upper)
        pred[[*KEYS,'chronos','chronos_p10','chronos_p90']].to_parquet(path,index=False)
        path.with_suffix('.json').write_text(json.dumps({'inference_seconds': elapsed}))
        print(f'origin {i+1}/{len(origins)} {origin.date()} · {elapsed:.2f}s',flush=True)
    predictions = pd.concat([pd.read_parquet(directory/(pd.Timestamp(o).strftime('%Y-%m-%d')+'.parquet')) for o in origins],ignore_index=True)
    report = {'config':config, 'metrics':score(panel,predictions), 'inference_seconds_this_invocation': inference_time,
              'wall_seconds':time.monotonic()-started,
              'total_inference_seconds':sum(json.loads(f.read_text())['inference_seconds'] for f in directory.glob('*.json')),
              'checkpoint_directory':str(directory)}
    args.output.parent.mkdir(exist_ok=True,parents=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print({name:values['mean_wape'] for name,values in report['metrics'].items()},flush=True)
    print(args.output,flush=True)


if __name__ == '__main__':
    main()
