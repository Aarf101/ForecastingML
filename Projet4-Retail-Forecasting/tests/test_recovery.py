"""Reprise interdite sans provenance et évaluation indépendantes valides."""

import io
import json
from unittest.mock import Mock

import pytest

from infra.sagemaker.recovery import recover_registration
from retail_forecast.config import SETTINGS


@pytest.fixture
def recovery(monkeypatch):
    monkeypatch.setattr("infra.sagemaker.recovery.serving_artifact", lambda s3, uri: uri)
    account = SETTINGS.role_arn.split(':')[4] if SETTINGS.role_arn else '123'
    arn = f'arn:aws:sagemaker:{SETTINGS.region}:{account}:pipeline/retail-forecast-mlops/execution/test'
    sm, s3 = Mock(), Mock()
    sm.create_model_package.return_value = {'ModelPackageArn': 'arn/package'}
    sm.describe_pipeline_execution.return_value = {'PipelineExecutionStatus': 'Failed'}
    steps = [
        {'StepName': 'PreprocessFeatures', 'StepStatus': 'Succeeded', 'Metadata': {'ProcessingJob': {'Arn': 'arn/pre'}}},
        {'StepName': 'TrainLightGBM', 'StepStatus': 'Succeeded', 'Metadata': {'TrainingJob': {'Arn': 'arn/train'}}},
        {'StepName': 'EvaluateModel', 'StepStatus': 'Succeeded', 'Metadata': {'ProcessingJob': {'Arn': 'arn/eval'}}},
        {'StepName': 'GateWAPE', 'StepStatus': 'Succeeded', 'Metadata': {'Condition': {'Outcome': 'True'}}},
        {'StepName': 'RegisterApproved-RegisterModel', 'StepStatus': 'Failed', 'Metadata': {}}]
    sm.get_paginator.side_effect = lambda name: type('Pages', (), {'paginate': lambda self, **kw: ([{'PipelineExecutionSteps': steps}] if name == 'list_pipeline_execution_steps' else [{'ModelPackageSummaryList': []}])})()
    sm.describe_training_job.return_value = {'TrainingJobStatus': 'Completed',
        'ModelArtifacts': {'S3ModelArtifacts': 's3://b/model'}, 'AlgorithmSpecification': {'TrainingImage': 'image'}}
    evaluation = {'ProcessingJobStatus': 'Completed', 'ProcessingInputs': [
        {'S3Input': {'LocalPath': '/input/model', 'S3Uri': 's3://b/model'}},
        {'S3Input': {'LocalPath': '/input/curated', 'S3Uri': 's3://b/features'}}],
        'ProcessingOutputConfig': {'Outputs': [{'OutputName': 'evaluation', 'S3Output': {'S3Uri': 's3://b/report'}}]}}
    sm.describe_processing_job.side_effect = lambda ProcessingJobName: evaluation if ProcessingJobName == 'eval' else {
        'ProcessingOutputConfig': {'Outputs': [{'OutputName': 'features', 'S3Output': {'S3Uri': 's3://b/features'}}]}}
    report = {'independent': True, 'source': 'recompute', 'by_horizon': {
        str(h): {'wape': 0.4} for h in range(1, 17)}, 'regression_metrics': {'test_wape': {'value': .4}}}
    s3.get_object.side_effect = lambda **kwargs: {'Body': io.BytesIO(json.dumps(report).encode())}
    return sm, s3, arn, steps, evaluation, report


def test_recovery_registers_exact_evaluated_artifact(recovery, monkeypatch):
    sm, s3, arn, *_ = recovery
    if not SETTINGS.role_arn:
        monkeypatch.setattr('infra.sagemaker.recovery.SETTINGS', type('Settings', (), {
            'role_arn': 'arn:aws:iam::123:role/LabRole', 'region': SETTINGS.region, 'wape_threshold': .45})())
    recover_registration(sm, s3, arn)
    request = sm.create_model_package.call_args.kwargs
    assert request['InferenceSpecification']['Containers'][0]['ModelDataUrl'] == 's3://b/model'
    assert request['ModelApprovalStatus'] == 'Approved'
    assert len(request['ClientToken']) <= 36
    assert request['ModelMetrics']['ModelQuality']['Statistics']['S3Uri'] == 's3://b/report/evaluation.json'


@pytest.mark.parametrize('fault', ['false_gate', 'missing_eval', 'foreign_model', 'dependent', 'poor_quality', 'foreign_features'])
def test_recovery_rejects_invalid_proof(recovery, fault, monkeypatch):
    sm, s3, arn, steps, evaluation, report = recovery
    if not SETTINGS.role_arn:
        monkeypatch.setattr('infra.sagemaker.recovery.SETTINGS', type('Settings', (), {
            'role_arn': 'arn:aws:iam::123:role/LabRole', 'region': SETTINGS.region, 'wape_threshold': .45})())
    if fault == 'false_gate': steps[3]['Metadata']['Condition']['Outcome'] = 'False'
    if fault == 'missing_eval': steps[2]['StepStatus'] = 'Failed'
    if fault == 'foreign_model': evaluation['ProcessingInputs'][0]['S3Input']['S3Uri'] = 's3://b/other'
    if fault == 'foreign_features': evaluation['ProcessingInputs'][1]['S3Input']['S3Uri'] = 's3://b/other'
    if fault == 'dependent': report['independent'] = False
    if fault == 'poor_quality': report['regression_metrics']['test_wape']['value'] = .9
    with pytest.raises(ValueError): recover_registration(sm, s3, arn)
    sm.create_model_package.assert_not_called()


@pytest.mark.parametrize('approval', ['Approved', 'Rejected'])
def test_recovery_reuses_provenance_and_respects_revocation(recovery, monkeypatch, approval):
    sm, s3, arn, steps, *_ = recovery
    if not SETTINGS.role_arn:
        monkeypatch.setattr('infra.sagemaker.recovery.SETTINGS', type('Settings', (), {
            'role_arn': 'arn:aws:iam::123:role/LabRole', 'region': SETTINGS.region, 'wape_threshold': .45})())
    sm.get_paginator.side_effect = lambda name: type('Pages', (), {'paginate': lambda self, **kw: (
        [{'PipelineExecutionSteps': steps}] if name == 'list_pipeline_execution_steps' else
        [{'ModelPackageSummaryList': [{'ModelPackageArn': 'arn/existing'}]}])})()
    sm.describe_model_package.return_value = {
        'ModelPackageArn': 'arn/existing', 'ModelApprovalStatus': approval, 'ModelPackageStatus': 'Completed',
        'InferenceSpecification': {'Containers': [{'ModelDataUrl': 's3://b/model'}]},
        'CustomerMetadataProperties': {'serving_layout': 'code_v1', 'source_execution': arn, 'evaluation_uri': 's3://b/report/evaluation.json'}}
    if approval == 'Approved':
        assert recover_registration(sm, s3, arn) == 'arn/existing'
    else:
        with pytest.raises(ValueError, match='révoquée'): recover_registration(sm, s3, arn)
    sm.create_model_package.assert_not_called()
