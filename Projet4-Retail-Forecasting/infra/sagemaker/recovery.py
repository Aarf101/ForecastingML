"""Reprend une inscription échouée après une évaluation cloud réussie."""

import hashlib
import json
import math

from retail_forecast.config import SETTINGS
from infra.sagemaker.serving import serving_artifact


def recover_registration(sm, s3, execution_arn):
    """Vérifie la provenance réelle avant de réutiliser un artefact évalué."""
    from infra.sagemaker.register import GROUP

    account = SETTINGS.role_arn.split(':')[4]
    prefix = f'arn:aws:sagemaker:{SETTINGS.region}:{account}:pipeline/retail-forecast-mlops/execution/'
    if not execution_arn.startswith(prefix):
        raise ValueError('exécution étrangère au projet')
    execution = sm.describe_pipeline_execution(PipelineExecutionArn=execution_arn)
    if execution['PipelineExecutionStatus'] != 'Failed':
        raise ValueError('reprise réservée à une inscription échouée')
    steps = {}
    for page in sm.get_paginator('list_pipeline_execution_steps').paginate(PipelineExecutionArn=execution_arn):
        for step in page['PipelineExecutionSteps']:
            if step['StepName'] in steps:
                raise ValueError('étape ambiguë')
            steps[step['StepName']] = step
    required = ['PreprocessFeatures', 'TrainLightGBM', 'EvaluateModel', 'GateWAPE']
    if any(steps.get(name, {}).get('StepStatus') != 'Succeeded' for name in required):
        raise ValueError('évaluation indépendante incomplète')
    if steps['GateWAPE'].get('Metadata', {}).get('Condition', {}).get('Outcome') != 'True':
        raise ValueError('gate indépendant refusé')
    failures = [name for name, step in steps.items() if step['StepStatus'] != 'Succeeded']
    if failures != ['RegisterApproved-RegisterModel'] or steps[failures[0]]['StepStatus'] != 'Failed':
        raise ValueError('échec autre que inscription')
    train_name = steps['TrainLightGBM']['Metadata']['TrainingJob']['Arn'].rsplit('/', 1)[1]
    train = sm.describe_training_job(TrainingJobName=train_name)
    evaluation = sm.describe_processing_job(ProcessingJobName=
        steps['EvaluateModel']['Metadata']['ProcessingJob']['Arn'].rsplit('/', 1)[1])
    if train['TrainingJobStatus'] != 'Completed' or evaluation['ProcessingJobStatus'] != 'Completed':
        raise ValueError('jobs cloud incomplets')
    model_uri = train['ModelArtifacts']['S3ModelArtifacts']
    inputs = {item['S3Input']['LocalPath'].rsplit('/', 1)[1]: item['S3Input']['S3Uri'] for item in evaluation['ProcessingInputs']}
    if inputs.get('model') != model_uri:
        raise ValueError('artefact évalué différent de artefact entraîné')
    preprocess = sm.describe_processing_job(ProcessingJobName=
        steps['PreprocessFeatures']['Metadata']['ProcessingJob']['Arn'].rsplit('/', 1)[1])
    features = [item['S3Output']['S3Uri'] for item in preprocess['ProcessingOutputConfig']['Outputs']
                if item['OutputName'] == 'features']
    if len(features) != 1 or inputs.get('curated') != features[0]:
        raise ValueError('features évaluées étrangères à exécution')
    outputs = evaluation['ProcessingOutputConfig']['Outputs']
    paths = [item['S3Output']['S3Uri'] for item in outputs if item['OutputName'] == 'evaluation']
    if len(paths) != 1:
        raise ValueError('rapport indépendant ambigu')
    report_uri = paths[0].rstrip('/') + '/evaluation.json'
    bucket, key = report_uri.removeprefix('s3://').split('/', 1)
    report = json.loads(s3.get_object(Bucket=bucket, Key=key)['Body'].read())
    values = report.get('by_horizon', {})
    if report.get('independent') is not True or report.get('source') != 'recompute' or set(values) != set(map(str, range(1, 17))):
        raise ValueError('rapport indépendant incomplet')
    scores = [float(row['wape']) for row in values.values()]
    mean = float(report['regression_metrics']['test_wape']['value'])
    if (any(not math.isfinite(value) or value < 0 for value in [*scores, mean])
            or abs(sum(scores) / 16 - mean) > 0.000002
            or mean > SETTINGS.wape_threshold):
        raise ValueError('qualité indépendante insuffisante')
    serving_uri = serving_artifact(s3, model_uri)
    # Le registre peut refuser un même token après résolution interne de version.
    # Recherche explicite par provenance, sans sélectionner le dernier Approved.
    matches = []
    for page in sm.get_paginator('list_model_packages').paginate(ModelPackageGroupName=GROUP):
        for item in page['ModelPackageSummaryList']:
            detail = sm.describe_model_package(ModelPackageName=item['ModelPackageArn'])
            if (detail.get('CustomerMetadataProperties', {}).get('source_execution') == execution_arn
                    and detail.get('CustomerMetadataProperties', {}).get('serving_layout') == 'code_v1'):
                matches.append(detail)
    if matches:
        if len(matches) != 1:
            raise ValueError('plusieurs inscriptions pour la même reprise')
        package = matches[0]
        if (package['ModelApprovalStatus'] != 'Approved'
                or package['ModelPackageStatus'] != 'Completed'
                or package['InferenceSpecification']['Containers'][0]['ModelDataUrl'] != serving_uri
                or package['CustomerMetadataProperties'].get('evaluation_uri') != report_uri):
            raise ValueError('inscription existante révoquée ou incompatible')
        return package['ModelPackageArn']
    response = sm.create_model_package(
        ClientToken=hashlib.sha256((execution_arn + serving_uri).encode()).hexdigest()[:32],
        ModelPackageGroupName=GROUP,
        ModelPackageDescription='Reprise inscription après évaluation indépendante réussie',
        ModelApprovalStatus='Approved',
        InferenceSpecification={
            'Containers': [{'Image': train['AlgorithmSpecification']['TrainingImage'], 'ModelDataUrl': serving_uri}],
            'SupportedTransformInstanceTypes': ['ml.m5.large'],
            'SupportedRealtimeInferenceInstanceTypes': ['ml.m5.large'],
            'SupportedContentTypes': ['application/json'],
            'SupportedResponseMIMETypes': ['application/json']},
        ModelMetrics={'ModelQuality': {'Statistics': {'ContentType': 'application/json', 'S3Uri': report_uri}}},
        CustomerMetadataProperties={'serving_layout': 'code_v1', 'evaluated_model_uri': model_uri,
            'source_execution': execution_arn, 'evaluation_uri': report_uri,
            'mean_wape': str(mean), 'gate': f'mean_wape_le_{SETTINGS.wape_threshold}',
            'horizons': '-'.join(map(str, range(1, 17)))})
    return response['ModelPackageArn']
