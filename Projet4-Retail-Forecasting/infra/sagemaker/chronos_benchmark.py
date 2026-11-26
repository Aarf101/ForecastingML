"""Lance le benchmark complet et prépare la validation Batch Transform de Chronos."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import boto3

from retail_forecast.config import ROOT, SETTINGS

GPU_IMAGE='763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.6.0-gpu-py312'
CPU_TRAIN_IMAGE='763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.6.0-cpu-py312'
CPU_IMAGE='763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.6.0-cpu-py312'
REQUIREMENTS='''chronos-forecasting==2.3.2
transformers==5.24.1
pandas==3.0.5
lightgbm==4.7.0
pyarrow>=15,<27
python-dotenv>=1.0
'''
EVIDENCE=ROOT/'docs/evidence/chronos_full_aws.json'


def session():
    return boto3.Session(profile_name=SETTINGS.profile or None,region_name=SETTINGS.region)


def source_bundle():
    """Liste blanche de code ; les credentials et les données restent exclus."""
    buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w:gz') as tar:
        for path in sorted((ROOT/'src/retail_forecast').rglob('*.py')):
            tar.add(path,arcname=str(Path('retail_forecast')/path.relative_to(ROOT/'src/retail_forecast')))
        for name in ['chronos_entry.py','chronos_inference.py']:
            tar.add(Path(__file__).with_name(name),arcname=name)
        data=REQUIREMENTS.encode();info=tarfile.TarInfo('requirements.txt');info.size=len(data)
        tar.addfile(info,io.BytesIO(data))
    return buffer.getvalue()


def start(instance_type="ml.m5.xlarge", origin_shard=0, origin_shards=1, run_id=None):
    """Un job de calcul, limité à une heure ; aucun manifeste actif n'est écrit."""
    s=session();s3=s.client('s3');sm=s.client('sagemaker')
    active_body=s3.get_object(Bucket=SETTINGS.bucket,Key='deployment/active.json')['Body'].read()
    active=json.loads(active_body)
    run_id=run_id or datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
    device='cuda' if '.g' in instance_type else 'cpu'
    image=GPU_IMAGE if device=='cuda' else CPU_TRAIN_IMAGE
    threads=8 if '2xlarge' in instance_type else 16 if '4xlarge' in instance_type else 4 if 'xlarge' in instance_type else 2
    evidence_path=EVIDENCE if origin_shards==1 else EVIDENCE.with_name(f'chronos_full_aws_s{origin_shard}.json')
    name='retail-chronos-full-'+run_id+f'-s{origin_shard}';prefix='benchmarks/chronos/full/'+run_id+f'/s{origin_shard}'
    payload=source_bundle();code_sha=hashlib.sha256(payload).hexdigest();key='code/chronos/'+code_sha+'/source.tar.gz'
    s3.put_object(Bucket=SETTINGS.bucket,Key=key,Body=payload)
    data=ROOT/'data/raw/sales.parquet';s3.upload_file(str(data),SETTINGS.bucket,prefix+'/input/sales.parquet')
    request={'TrainingJobName':name,'RoleArn':SETTINGS.role_arn,
        'AlgorithmSpecification':{'TrainingImage':image,'TrainingInputMode':'File'},
        'InputDataConfig':[{'ChannelName':channel,'DataSource':{'S3DataSource':{'S3DataType':'S3Prefix','S3Uri':uri,'S3DataDistributionType':'FullyReplicated'}}}
            for channel,uri in [('sales',f's3://{SETTINGS.bucket}/{prefix}/input/sales.parquet'),('reference',active['model_data_url'])]],
        'OutputDataConfig':{'S3OutputPath':f's3://{SETTINGS.bucket}/{prefix}/artifacts'},
        'ResourceConfig':{'InstanceType':instance_type,'InstanceCount':1,'VolumeSizeInGB':50},
        'StoppingCondition':{'MaxRuntimeInSeconds':3600},
        'CheckpointConfig':{'S3Uri':f's3://{SETTINGS.bucket}/{prefix}/checkpoints','LocalPath':'/opt/ml/checkpoints'},
        'HyperParameters':{'sagemaker_program':'chronos_entry.py','sagemaker_submit_directory':f's3://{SETTINGS.bucket}/{key}',
            'sagemaker_container_log_level':'20','sagemaker_region':SETTINGS.region,'run-id':run_id,
            'device':device,'threads':str(threads),'origin-shard':str(origin_shard),'origin-shards':str(origin_shards)},
        'Tags':[{'Key':'project','Value':'retail-forecast'},{'Key':'purpose','Value':'chronos-full-validation'}]}
    response=sm.create_training_job(**request)
    evidence={'run_id':run_id,'job':name,'job_arn':response['TrainingJobArn'],'prefix':prefix,'status':'InProgress',
        'active_before_sha256':hashlib.sha256(active_body).hexdigest(),'active_before':active,
        'code_sha256':code_sha,'dataset_sha256':hashlib.file_digest(data.open('rb'),'sha256').hexdigest(),
        'image':image,'instance_type':instance_type,'origin_shard':origin_shard,'origin_shards':origin_shards,'max_runtime_seconds':3600,
        'purpose':'Frozen zero-shot evaluation, not fine-tuning','source_uri':f's3://{SETTINGS.bucket}/{key}'}
    evidence_path.write_text(json.dumps(evidence,indent=2)+'\n');print(name,flush=True)
    return name


def status(job=None,wait=False,evidence_path=EVIDENCE):
    """Décrit uniquement le job du benchmark et conserve ses identifiants de sortie."""
    evidence=json.loads(evidence_path.read_text());job=job or evidence['job'];sm=session().client('sagemaker')
    if wait:
        sm.get_waiter('training_job_completed_or_stopped').wait(TrainingJobName=job,WaiterConfig={'Delay':30,'MaxAttempts':130})
    description=sm.describe_training_job(TrainingJobName=job)
    evidence.update(status=description['TrainingJobStatus'],failure_reason=description.get('FailureReason',''))
    if 'ModelArtifacts' in description:
        evidence['model_artifact']=description['ModelArtifacts']['S3ModelArtifacts']
    for key in ['TrainingTimeInSeconds','BillableTimeInSeconds']:
        if key in description:evidence[key]=description[key]
    evidence_path.write_text(json.dumps(evidence,indent=2)+'\n');print(evidence['status'],evidence.get('failure_reason',''),flush=True)
    if wait and evidence['status']!='Completed':raise RuntimeError('benchmark AWS non terminé avec succès')
    return description


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['start','status','transform'])
    parser.add_argument('--wait',action='store_true',help='attendre le benchmark ; pour le serving utiliser chronos_validate run')
    parser.add_argument('--instance',default='ml.m5.xlarge')
    parser.add_argument('--origin-shard',type=int,default=0)
    parser.add_argument('--origin-shards',type=int,default=1)
    parser.add_argument('--run-id')
    args=parser.parse_args(argv)
    if args.action=='transform':
        start_transform()
        return
    path=EVIDENCE if args.origin_shards==1 else EVIDENCE.with_name(f'chronos_full_aws_s{args.origin_shard}.json')
    if args.action=='status' and not path.exists() and args.origin_shards==1:
        for proof in sorted(EVIDENCE.parent.glob('chronos_full_aws_s*.json')):
            print(proof.stem,flush=True)
            status(wait=args.wait,evidence_path=proof)
        return
    job=start(args.instance,args.origin_shard,args.origin_shards,args.run_id) if args.action=='start' else None
    status(job,args.wait,path)


def start_transform():
    """Valide le vrai conteneur CPU sans endpoint permanent ni publication active."""
    evidence=json.loads(EVIDENCE.with_name('chronos_full_aws_s0.json').read_text())
    sm=session().client('sagemaker');description=sm.describe_training_job(TrainingJobName=evidence['job'])
    if description['TrainingJobStatus']!='Completed':
        raise ValueError('attendre la préparation de l’artefact candidat')
    model_uri=description['ModelArtifacts']['S3ModelArtifacts']
    proof_file=ROOT/'docs/evidence/chronos_transform_aws.json'
    if proof_file.exists():
        previous=json.loads(proof_file.read_text())
        if previous['model_artifact']==model_uri:
            existing=sm.describe_transform_job(TransformJobName=previous['job'])
            if existing['TransformJobStatus'] in ['InProgress','Completed']:
                return previous['job']

    name='retail-chronos-transform-'+evidence['run_id']
    model_name=name+'-model'
    input_uri=evidence.get('batch_input_uri')
    if not input_uri:
        raise ValueError('extraire et vérifier les sorties avant de lancer le transform')
    try:
        existing=sm.describe_model(ModelName=model_name)['PrimaryContainer']
        if existing['ModelDataUrl']!=model_uri or existing['Image']!=CPU_IMAGE:
            raise ValueError('modèle existant incompatible avec le candidat évalué')
    except sm.exceptions.ClientError as exc:
        if exc.response['Error']['Code']!='ValidationException':raise
        sm.create_model(ModelName=model_name,ExecutionRoleArn=SETTINGS.role_arn,PrimaryContainer={
        'Image':CPU_IMAGE,'ModelDataUrl':model_uri,'Environment':{
            'SAGEMAKER_PROGRAM':'chronos_inference.py','SAGEMAKER_SUBMIT_DIRECTORY':'/opt/ml/model/code',
            'SAGEMAKER_MODEL_SERVER_WORKERS':'1','TS_DEFAULT_WORKERS_PER_MODEL':'1',
            'PIP_BREAK_SYSTEM_PACKAGES':'1'}})
    # Les sorties du job d'évaluation sont copiées explicitement, après vérification.
    output_uri=f"s3://{SETTINGS.bucket}/{evidence['prefix']}/batch-output/"
    sm.create_transform_job(TransformJobName=name,ModelName=model_name,BatchStrategy='SingleRecord',MaxConcurrentTransforms=1,MaxPayloadInMB=2,
        TransformInput={'DataSource':{'S3DataSource':{'S3DataType':'S3Prefix','S3Uri':input_uri}},'ContentType':'application/json','SplitType':'Line'},
        TransformOutput={'S3OutputPath':output_uri,'Accept':'application/json','AssembleWith':'Line'},
        TransformResources={'InstanceType':'ml.m5.large','InstanceCount':1},
        ModelClientConfig={'InvocationsTimeoutInSeconds':300,'InvocationsMaxRetries':1},
        Tags=[{'Key':'project','Value':'retail-forecast'},{'Key':'purpose','Value':'chronos-serving-validation'}])
    proof={'job':name,'model_name':model_name,'model_artifact':model_uri,'image':CPU_IMAGE,
           'input_uri':input_uri,'output_uri':output_uri,'status':'InProgress','active_published':False}
    (ROOT/'docs/evidence/chronos_transform_aws.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(name,flush=True)
    return name


if __name__=='__main__':
    main()
