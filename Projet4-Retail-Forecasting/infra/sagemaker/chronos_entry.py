"""Évalue Chronos sur SageMaker puis prépare son artefact d'inférence figé."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
from importlib.metadata import version
from pathlib import Path


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    parser.add_argument('--threads',type=int,default=16)
    parser.add_argument('--origin-shard',type=int,default=0)
    parser.add_argument('--origin-shards',type=int,default=1)
    args=parser.parse_args(argv)
    import torch
    import pandas as pd
    from huggingface_hub import snapshot_download
    from chronos import Chronos2Pipeline
    from retail_forecast.evaluate import benchmark_chronos as benchmark
    from chronos_inference import requests_from_sales, input_fn, predict_fn
    if args.device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('GPU attendu : arrêt au lieu de lancer un calcul CPU long')
    output=Path('/opt/ml/output/data');output.mkdir(parents=True,exist_ok=True)
    model=Path('/opt/ml/model');model.mkdir(parents=True,exist_ok=True)
    snapshot_download('amazon/chronos-2',revision=benchmark.MODEL_REVISION,local_dir=model/'weights',allow_patterns=['config.json','*.safetensors'])
    data=Path('/opt/ml/input/data/sales/sales.parquet')
    artifact=Path('/opt/ml/input/data/reference/model.tar.gz')
    benchmark.main(['--data',str(data),'--artifact',str(artifact),'--model-path',str(model/'weights'),
        '--series-limit','0','--batch-size','128','--threads',str(args.threads),'--device',args.device,
        '--origin-shard',str(args.origin_shard),'--origin-shards',str(args.origin_shards),'--cache-dir','/opt/ml/checkpoints',
        '--output',str(output/'full_benchmark.json')])
    report=json.loads((output/'full_benchmark.json').read_text())
    if report['config']['series']!=1200 or report['config']['rows_scored']<=0:
        raise ValueError('le catalogue complet attendu n’a pas été évalué')
    directory=Path(report['checkpoint_directory'])
    predictions=pd.concat([pd.read_parquet(p) for p in sorted(directory.glob('*.parquet'))],ignore_index=True)
    predictions.to_parquet(output/'backtest_predictions.parquet',index=False)
    panel=Path('/opt/ml/checkpoints')/('panel-'+report['config']['dataset_and_artifact_sha256'][:12]+'.parquet')
    shutil.copy2(panel,output/'reference_panel.parquet')
    metadata={'model_id':'amazon/chronos-2','revision':benchmark.MODEL_REVISION,'context_length':256,
        'point_statistic':'median','covariates':'calendar','cross_learning':False,'fine_tuned':False,
        'run_id':args.run_id,'dataset_sha256':hashlib.file_digest(data.open('rb'),'sha256').hexdigest(),
        'reference_artifact_sha256':hashlib.file_digest(artifact.open('rb'),'sha256').hexdigest(),
        'weights_sha256':hashlib.file_digest((model/'weights/model.safetensors').open('rb'),'sha256').hexdigest(),
        'versions':{name:version(name) for name in ['torch','chronos-forecasting','transformers','pandas','lightgbm']}}
    (model/'chronos_metadata.json').write_text(json.dumps(metadata,indent=2))
    shutil.copy2(model/'chronos_metadata.json',output/'chronos_metadata.json')
    if args.origin_shard != 0:
        shutil.rmtree(model/'weights')
        print('[chronos] partition évaluée, poids identiques au modèle candidat de la partition 0',flush=True)
        return
    gc.collect();torch.cuda.empty_cache()
    pipeline=Chronos2Pipeline.from_pretrained(model/'weights',device_map=args.device,dtype=torch.float32,local_files_only=True)
    sales=pd.read_parquet(data);sales['date']=pd.to_datetime(sales.date)
    requests=requests_from_sales(sales)
    (output/'batch_input.jsonl').write_text('\n'.join(json.dumps(r,allow_nan=False) for r in requests)+'\n')
    current=[]
    for request in requests:
        current.extend(predict_fn(input_fn(json.dumps(request),'application/json'),{'pipeline':pipeline,'metadata':metadata}))
    forecast=pd.DataFrame(current)
    if len(forecast)!=19200 or forecast.duplicated(['store_id','item_id','horizon']).any():
        raise ValueError('lot courant incomplet')
    forecast.to_parquet(output/'reference_forecast.parquet',index=False)
    code=model/'code';code.mkdir(exist_ok=True)
    shutil.copy2(Path(__file__).with_name('chronos_inference.py'),code/'chronos_inference.py')
    shutil.copytree(Path(__file__).parent/'retail_forecast',code/'retail_forecast',dirs_exist_ok=True)
    shutil.copy2(Path(__file__).with_name('requirements.txt'),code/'requirements.txt')
    print('[chronos] catalogue complet évalué ; 19 200 prévisions courantes et artefact prêts',flush=True)


if __name__=='__main__':
    main()
