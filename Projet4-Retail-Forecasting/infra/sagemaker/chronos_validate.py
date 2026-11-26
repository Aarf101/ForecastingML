"""Collecte et vérifie indépendamment les sorties Chronos, sans promotion active."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

from infra.sagemaker.chronos_benchmark import session, EVIDENCE
from retail_forecast.config import ROOT, SETTINGS
from retail_forecast.evaluate.benchmark_chronos import score, KEYS

LOCAL=ROOT/'data/benchmark/full-aws'


def collect(shard):
    """Ne consomme que l'archive du job désigné, après son succès."""
    file=EVIDENCE.with_name(f'chronos_full_aws_s{shard}.json');evidence=json.loads(file.read_text())
    s=session();sm=s.client('sagemaker');s3=s.client('s3')
    description=sm.describe_training_job(TrainingJobName=evidence['job'])
    evidence.update(status=description['TrainingJobStatus'],failure_reason=description.get('FailureReason',''))
    for key in ['TrainingTimeInSeconds','BillableTimeInSeconds','TrainingStartTime','TrainingEndTime']:
        if key in description:evidence[key]=str(description[key]) if 'Time' in key and 'Seconds' not in key else description[key]
    file.write_text(json.dumps(evidence,indent=2)+'\n')
    if evidence['status']!='Completed':return None
    uri=description['ModelArtifacts']['S3ModelArtifacts']
    evidence['model_artifact']=uri;output_uri=uri.rsplit('/',1)[0]+'/output.tar.gz'
    bucket,key=output_uri.removeprefix('s3://').split('/',1)
    directory=LOCAL/f's{shard}';directory.mkdir(parents=True,exist_ok=True)
    archive=directory/'output.tar.gz'
    if not archive.exists():
        s3.download_file(bucket,key,str(archive))
    with tarfile.open(archive) as tar:tar.extractall(directory,filter='data')
    required=['full_benchmark.json','backtest_predictions.parquet','chronos_metadata.json']
    if shard==0:required+=['reference_forecast.parquet','batch_input.jsonl']
    if any(not (directory/name).exists() for name in required):raise ValueError('sortie AWS incomplète')
    metadata=json.loads((directory/'chronos_metadata.json').read_text())
    if metadata['dataset_sha256']!=evidence['dataset_sha256']:
        raise ValueError('données AWS différentes du jeu envoyé')
    evidence['output_archive_uri']=output_uri
    if shard==0 and not evidence.get('batch_input_uri'):
        key=evidence['prefix']+'/serving/input.jsonl'
        s3.upload_file(str(directory/'batch_input.jsonl'),SETTINGS.bucket,key)
        evidence['batch_input_uri']=f's3://{SETTINGS.bucket}/{key}'
    file.write_text(json.dumps(evidence,indent=2)+'\n')
    print(f'Collected shard {shard}',flush=True)
    return directory


def combine():
    """Recalcule les métriques sur 537 600 couples à partir des prédictions brutes."""
    folders=[collect(i) for i in range(6)]
    if any(folder is None for folder in folders):raise ValueError('attendre les six jobs terminés')
    reports=[json.loads((folder/'full_benchmark.json').read_text()) for folder in folders]
    metadata=[json.loads((folder/'chronos_metadata.json').read_text()) for folder in folders]
    for key in ['weights_sha256','dataset_sha256','reference_artifact_sha256','revision']:
        if len({m[key] for m in metadata})!=1:raise ValueError('entrées ou poids différents entre partitions')
    config=reports[0]['config'].copy()
    if any(r['config']['dataset_and_artifact_sha256']!=config['dataset_and_artifact_sha256'] for r in reports):raise ValueError('empreintes différentes')
    panel_path=ROOT/'data/benchmark'/('panel-'+config['dataset_and_artifact_sha256'][:12]+'.parquet')
    panel=pd.read_parquet(panel_path)
    predictions=pd.concat([pd.read_parquet(folder/'backtest_predictions.parquet') for folder in folders],ignore_index=True)
    if predictions.duplicated(KEYS).any() or len(predictions)!=1200*43*16:raise ValueError('partitionnement incomplet ou dupliqué')
    if len(panel)!=537600 or panel.groupby('horizon').size().to_dict()!={h:33600 for h in range(1,17)}:raise ValueError('panel complet invalide')
    metrics=score(panel,predictions)
    predictions.to_parquet(LOCAL/'backtest_predictions.parquet',index=False)
    config.update(series=1200,origins=43,rows_scored=537600,execution='six SageMaker CPU jobs, independently rescored locally')
    for name in ['origin_shard','origin_dates']:config.pop(name,None)
    config['origin_shards']=6
    report={'config':config,'metrics':metrics,'metadata':metadata[0],
        'total_inference_seconds':sum(r['total_inference_seconds'] for r in reports),
        'checkpoint_directory':str(LOCAL),'scope':'Full 1200-series synthetic catalog; frozen zero-shot model; no production promotion.',
        'independent_rescore':True,'reference_model_package':json.loads(EVIDENCE.with_name('chronos_full_aws_s0.json').read_text())['active_before']['model_package_arn']}
    # Contrôle croisé avec les sorties CPU du pilote, sans les réutiliser comme résultat complet.
    pilot=json.loads((ROOT/'reports/chronos2_benchmark.json').read_text())
    previous=pd.concat([pd.read_parquet(p) for p in (ROOT/pilot['checkpoint_directory']).glob('*.parquet')],ignore_index=True)
    pair=previous.merge(predictions,on=KEYS,suffixes=('_pilot','_aws'),validate='one_to_one')
    report['pilot_crosscheck']={'rows':len(pair),'max_point_difference':float(np.max(np.abs(pair.chronos_pilot-pair.chronos_aws))),
        'mean_point_difference':float(np.mean(np.abs(pair.chronos_pilot-pair.chronos_aws)))}
    body=session().client('s3').get_object(Bucket=SETTINGS.bucket,Key='deployment/active.json')['Body'].read()
    first=json.loads(EVIDENCE.with_name('chronos_full_aws_s0.json').read_text())
    report['active_manifest_unchanged']=hashlib.sha256(body).hexdigest()==first['active_before_sha256']
    (ROOT/'reports/chronos2_full_benchmark.json').write_text(json.dumps(report,indent=2)+'\n')
    print({name:values['mean_wape'] for name,values in metrics.items()},flush=True)
    return report


def validate_forecast(actual,expected):
    """Refuse perte, doublon, décalage ou divergence du vrai serving AWS."""
    keys=['store_id','item_id','origin_date','target_date','horizon']
    for frame in [actual,expected]:
        if len(frame)!=19200 or frame.duplicated(keys).any():raise ValueError('lot courant incomplet ou dupliqué')
        for name in ['origin_date','target_date']:frame[name]=pd.to_datetime(frame[name])
        if frame[['store_id','item_id']].drop_duplicates().shape[0]!=1200 or set(frame.horizon)!=set(range(1,17)):
            raise ValueError('catalogue ou horizons incomplets')
        if not (frame.target_date==frame.origin_date+pd.to_timedelta(frame.horizon,unit='D')).all():raise ValueError('date cible incohérente')
        if not np.isfinite(frame[['forecast_units','p10','p90']]).all().all() or (frame[['forecast_units','p10','p90']]<0).any().any():raise ValueError('valeurs invalides')
        if not (frame.p10.le(frame.forecast_units+1e-5)&frame.p90.ge(frame.forecast_units-1e-5)).all():raise ValueError('quantiles incohérents')
    paired=actual.merge(expected,on=keys,suffixes=('_batch','_reference'),validate='one_to_one')
    if len(paired)!=19200:raise ValueError('identifiants différents')
    differences={}
    for column in ['forecast_units','p10','p90']:
        a,b=paired[column+'_batch'].to_numpy(),paired[column+'_reference'].to_numpy()
        if not np.allclose(a,b,rtol=1e-3,atol=1e-3):raise ValueError('serving différent du candidat évalué')
        differences[column]=float(np.max(np.abs(a-b)))
    return differences


def collect_transform():
    """Vérifie le Batch Transform réel puis conserve le candidat hors du préfixe actif."""
    file=ROOT/'docs/evidence/chronos_transform_aws.json';evidence=json.loads(file.read_text())
    s=session();sm=s.client('sagemaker');s3=s.client('s3');description=sm.describe_transform_job(TransformJobName=evidence['job'])
    evidence.update(status=description['TransformJobStatus'],failure_reason=description.get('FailureReason',''))
    file.write_text(json.dumps(evidence,indent=2)+'\n')
    if evidence['status']!='Completed':raise ValueError('Batch Transform non terminé avec succès')
    bucket,prefix=evidence['output_uri'].removeprefix('s3://').split('/',1)
    lines=s3.get_object(Bucket=bucket,Key=prefix+'input.jsonl.out')['Body'].read().decode().splitlines()
    actual=pd.DataFrame([record for line in lines if line.strip() for record in json.loads(line)])
    expected=pd.read_parquet(LOCAL/'s0/reference_forecast.parquet')
    differences=validate_forecast(actual,expected)
    actual['model_version']='chronos2-29ec3766';actual['model_job']=evidence['job']
    destination=LOCAL/'candidate_forecast.parquet';actual.to_parquet(destination,index=False)
    key=prefix+'candidate_forecast.parquet';s3.upload_file(str(destination),bucket,key)
    first=json.loads(EVIDENCE.with_name('chronos_full_aws_s0.json').read_text())
    active=s3.get_object(Bucket=SETTINGS.bucket,Key='deployment/active.json')['Body'].read()
    evidence.update(rows=len(actual),series=1200,horizons=16,verified=True,max_absolute_difference=differences,
        forecast_uri=f's3://{bucket}/{key}',active_manifest_unchanged=hashlib.sha256(active).hexdigest()==first['active_before_sha256'])
    file.write_text(json.dumps(evidence,indent=2)+'\n');print('Verified real AWS Batch Transform:',len(actual),'rows',flush=True)
    return evidence


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['collect','combine','transform','run'])
    parser.add_argument('--shard',type=int,default=0)
    args=parser.parse_args(argv)
    if args.action=='collect':collect(args.shard)
    elif args.action=='combine':combine()
    elif args.action=='transform':collect_transform()
    else:run_validation()


def run_validation():
    """Surveille les jobs de cette expérience, puis vérifie le serving réel."""
    import time
    from datetime import datetime, timezone
    from infra.sagemaker.chronos_benchmark import start_transform
    sm=session().client('sagemaker')
    files=[EVIDENCE.with_name(f'chronos_full_aws_s{i}.json') for i in range(6)]
    proofs=[json.loads(f.read_text()) for f in files]
    if len({p['run_id'] for p in proofs})!=1:
        raise ValueError('les six jobs doivent appartenir à la même exécution')
    launched=False;combined=False
    deadline=time.monotonic()+7200
    while time.monotonic()<deadline:
        statuses=[]
        for p in proofs:
            d=sm.describe_training_job(TrainingJobName=p['job']);statuses.append(d['TrainingJobStatus'])
        print('Training shards:',statuses,flush=True)
        if any(status in ['Failed','Stopped'] for status in statuses):
            for p,status in zip(proofs,statuses):
                if status=='InProgress':sm.stop_training_job(TrainingJobName=p['job'])
            if launched and sm.describe_transform_job(TransformJobName=job)['TransformJobStatus']=='InProgress':
                sm.stop_transform_job(TransformJobName=job)
            raise RuntimeError('partition échouée : autres jobs de cette expérience arrêtés')
        if statuses[0]=='Completed' and not launched:
            collect(0);job=start_transform();launched=True
        if all(status=='Completed' for status in statuses) and not combined:
            combine();combined=True
        if launched:
            d=sm.describe_transform_job(TransformJobName=job)
            print('Batch Transform:',d['TransformJobStatus'],flush=True)
            if d['TransformJobStatus']=='InProgress' and (datetime.now(timezone.utc)-d['CreationTime']).total_seconds()>1800:
                sm.stop_transform_job(TransformJobName=job)
                raise TimeoutError('Batch Transform du benchmark arrêté après 30 minutes')
            if d['TransformJobStatus']=='Failed':
                collect_transform()
            if d['TransformJobStatus']=='Completed' and combined:
                result=collect_transform()
                if not result.get('model_resource_deleted'):
                    try:
                        sm.delete_model(ModelName=result['model_name'])
                    except sm.exceptions.ClientError as exc:
                        if exc.response['Error']['Code']!='ValidationException':raise
                result['model_resource_deleted']=True
                (ROOT/'docs/evidence/chronos_transform_aws.json').write_text(json.dumps(result,indent=2)+'\n')
                print('Full AWS benchmark and real Batch Transform verified.',flush=True)
                return
        time.sleep(20)
    if launched and sm.describe_transform_job(TransformJobName=job)['TransformJobStatus']=='InProgress':
        sm.stop_transform_job(TransformJobName=job)
    raise TimeoutError('validation Chronos expirée ; le job de transform restant est arrêté')


if __name__=='__main__':
    main()
