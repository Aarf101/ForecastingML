"""Deux workflows Learner Lab : entraînement mensuel et prévision quotidienne."""

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path

from infra.sagemaker.launch import IMAGE
from retail_forecast.config import ROOT, SETTINGS
from retail_forecast.ingest.storage import s3_client


def bundle():
    """Liste blanche : aucun .env, identifiant AWS ou donnée brute."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w:gz') as archive:
        paths = [ROOT / 'pyproject.toml', ROOT / 'README.md',
                 ROOT / 'infra/sagemaker/requirements.txt']
        for folder in ['src/retail_forecast', 'infra', 'dashboards/streamlit']:
            paths.extend(sorted((ROOT / folder).rglob('*.py')))
        for path in paths:
            archive.add(path, arcname=str(path.relative_to(ROOT)))
    return buffer.getvalue()


def definition(mode, bundle_uri, bootstrap_uri):
    """Le job Processing attend les jobs imbriqués et échoue avec eux."""
    processing = {
        'Type': 'Task', 'Resource': 'arn:aws:states:::sagemaker:createProcessingJob.sync',
        'TimeoutSeconds': 10800,
        'Parameters': {
            'ProcessingJobName.$': "States.Format('retail-lifecycle-{}', States.UUID())",
            'RoleArn': SETTINGS.role_arn,
            'AppSpecification': {'ImageUri': IMAGE,
                'ContainerEntrypoint': ['python3', '/opt/ml/processing/input/script/bootstrap.py'],
                'ContainerArguments': [mode]},
            'Environment': {'S3_BUCKET': SETTINGS.bucket, 'AWS_REGION': SETTINGS.region,
                'SAGEMAKER_ROLE_ARN': SETTINGS.role_arn,
                'WAPE_THRESHOLD': str(SETTINGS.wape_threshold), 'S3_ENDPOINT_URL': ''},
            'ProcessingResources': {'ClusterConfig': {'InstanceCount': 1,
                'InstanceType': 'ml.m5.large', 'VolumeSizeInGB': 30}},
            'StoppingCondition': {'MaxRuntimeInSeconds': 10500},
            'ProcessingInputs': [
                {'InputName': name, 'S3Input': {'S3Uri': uri,
                 'LocalPath': f'/opt/ml/processing/input/{name}', 'S3DataType': 'S3Prefix',
                 'S3InputMode': 'File', 'S3DataDistributionType': 'FullyReplicated'}}
                for name, uri in [('bundle', bundle_uri), ('script', bootstrap_uri)]],
        }, 'End': True,
    }
    workflow = {'StartAt': 'CheckETL', 'TimeoutSeconds': 12000, 'States': {
        'CheckETL': {'Type': 'Choice', 'Choices': [
            {'And': [{'Variable': '$.skip_etl', 'IsPresent': True},
                     {'Variable': '$.skip_etl', 'BooleanEquals': True}], 'Next': 'Lifecycle'}],
            'Default': 'RunETL'},
        'RunETL': {'Type': 'Task', 'Resource': 'arn:aws:states:::glue:startJobRun.sync',
            'Parameters': {'JobName': 'retail-etl-raw-to-curated',
                           'Arguments': {'--BUCKET': SETTINGS.bucket}},
            'ResultPath': None, 'Next': 'Lifecycle'},
        'Lifecycle': processing}}
    if mode == 'training':
        import copy
        recovery = copy.deepcopy(processing)
        spec = recovery['Parameters']['AppSpecification']
        del spec['ContainerArguments']
        spec['ContainerArguments.$'] = "States.Array('training', '--evaluated-execution', $.evaluated_execution)"
        workflow['StartAt'] = 'CheckResume'
        workflow['States']['CheckResume'] = {'Type': 'Choice', 'Choices': [
            {'Variable': '$.evaluated_execution', 'IsPresent': True, 'Next': 'RecoveryLifecycle'}],
            'Default': 'CheckETL'}
        workflow['States']['RecoveryLifecycle'] = recovery
    return workflow


def deploy():
    """Réutilise LabRole ; ne crée aucun rôle IAM ni job de calcul."""
    import boto3
    from infra.aws.orchestrate import _clients, deploy_glue_job, deploy_lambda

    if not SETTINGS.role_arn or not SETTINGS.bucket or SETTINGS.s3_endpoint_url:
        raise ValueError('configurer le bucket AWS, LabRole et S3_ENDPOINT_URL vide')
    s3 = s3_client()
    payload = bundle()
    prefix = f'code/lifecycle/{hashlib.sha256(payload).hexdigest()}'
    s3.put_object(Bucket=SETTINGS.bucket, Key=f'{prefix}/runtime.tar.gz', Body=payload)
    bootstrap = Path(__file__).parents[1] / 'sagemaker/lifecycle_bootstrap.py'
    s3.put_object(Bucket=SETTINGS.bucket, Key=f'{prefix}/bootstrap.py', Body=bootstrap.read_bytes())
    clients = _clients()
    deploy_glue_job(clients, SETTINGS.role_arn)
    deploy_lambda(clients, SETTINGS.role_arn)
    sfn = boto3.client('stepfunctions', region_name=SETTINGS.region)
    events = boto3.client('events', region_name=SETTINGS.region)
    try:
        events.disable_rule(Name='retail-forecast-daily')
    except events.exceptions.ResourceNotFoundException:
        pass
    account = boto3.client('sts', region_name=SETTINGS.region).get_caller_identity()['Account']
    for mode, schedule in [('training', 'cron(0 2 1 * ? *)'), ('inference', 'cron(0 4 * * ? *)')]:
        name = f'retail-{mode}'
        arn = f'arn:aws:states:{SETTINGS.region}:{account}:stateMachine:{name}'
        body = json.dumps(definition(mode, f's3://{SETTINGS.bucket}/{prefix}/runtime.tar.gz',
                                     f's3://{SETTINGS.bucket}/{prefix}/bootstrap.py'))
        try:
            sfn.describe_state_machine(stateMachineArn=arn)
        except sfn.exceptions.StateMachineDoesNotExist:
            sfn.create_state_machine(name=name, definition=body, roleArn=SETTINGS.role_arn,
                                     type='STANDARD')
        else:
            sfn.update_state_machine(stateMachineArn=arn, definition=body, roleArn=SETTINGS.role_arn)
        # Désactivée au déploiement : activer après la première validation manuelle.
        events.put_rule(Name=name, ScheduleExpression=schedule, State='DISABLED')
        response = events.put_targets(Rule=name, Targets=[{
            'Id': name, 'Arn': arn, 'RoleArn': SETTINGS.role_arn, 'Input': '{}'}])
        if response.get('FailedEntryCount'):
            raise RuntimeError(response)
        print(f'{name}: {arn} ; horaire UTC {schedule} (désactivé)')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['print', 'up', 'run', 'status'])
    parser.add_argument("--mode", choices=["training", "inference"], default="training")
    parser.add_argument("--wait", action="store_true", help="attendre le résultat du workflow")
    args = parser.parse_args(argv)
    if args.action == 'up':
        deploy()
    elif args.action in ['run', 'status']:
        import boto3
        session = boto3.Session(profile_name=SETTINGS.profile or None, region_name=SETTINGS.region)
        account = session.client('sts').get_caller_identity()['Account']
        client = session.client('stepfunctions')
        arn = f'arn:aws:states:{SETTINGS.region}:{account}:stateMachine:retail-{args.mode}'
        if args.action == 'run':
            execution = client.start_execution(stateMachineArn=arn, input='{}')['executionArn']
            print(execution, flush=True)
            if args.wait:
                import time
                deadline = time.monotonic() + 12100
                while True:
                    result = client.describe_execution(executionArn=execution)
                    if result['status'] != 'RUNNING':
                        if result['status'] != 'SUCCEEDED':
                            raise RuntimeError(f"{result['status']}: {result.get('cause', '')}")
                        print('SUCCEEDED')
                        break
                    if time.monotonic() > deadline:
                        raise TimeoutError('attente du workflow expirée')
                    time.sleep(20)
        else:
            print(json.dumps(client.list_executions(stateMachineArn=arn, maxResults=5), default=str, indent=2))
    else:
        for mode in ['training', 'inference']:
            print(json.dumps(definition(mode, 's3://BUCKET/runtime.tar.gz',
                                        's3://BUCKET/bootstrap.py'), indent=2))


if __name__ == '__main__':
    main()
