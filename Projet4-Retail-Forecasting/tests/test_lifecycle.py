"""Contrat de promotion : aucun effet actif avant réussite complète."""

import io
import json
from unittest.mock import Mock

import pytest
from botocore.exceptions import ClientError

from infra.sagemaker import lifecycle
from infra.aws import workflows


def test_publish_uses_conditional_write_and_preserves_previous():
    s3 = Mock()
    lifecycle.publish(s3, {'run_id': 'new'}, {'run_id': 'old'}, 'etag')
    kwargs = s3.put_object.call_args.kwargs
    assert kwargs['IfMatch'] == 'etag'
    assert json.loads(kwargs['Body'])['previous']['run_id'] == 'old'
    lifecycle.publish(s3, {'run_id': 'first'}, None, None)
    assert s3.put_object.call_args.kwargs['IfNoneMatch'] == '*'


def test_failure_does_not_publish(monkeypatch):
    from infra.sagemaker import launch
    monkeypatch.setattr(launch, '_sm_client', Mock())
    monkeypatch.setattr(lifecycle, 's3_client', Mock())
    monkeypatch.setattr(lifecycle, 'read_active', lambda *a, **kw: ({'model_package_arn': 'old'}, 'e'))
    def fail(*args):
        raise RuntimeError('batch failed')
    monkeypatch.setattr(lifecycle, 'run_forecast', fail)
    publish = Mock()
    monkeypatch.setattr(lifecycle, 'publish', publish)
    with pytest.raises(RuntimeError):
        lifecycle.run('inference')
    publish.assert_not_called()


def test_read_active_never_swallows_access_denied():
    s3 = Mock()
    s3.get_object.side_effect = ClientError({'Error': {'Code': 'AccessDenied'}}, 'GetObject')
    with pytest.raises(ClientError):
        lifecycle.read_active(s3, optional=True)


def test_workflows_use_unique_names_and_managed_wait():
    for mode in ['training', 'inference']:
        state = workflows.definition(mode, 's3://b/bundle', 's3://b/bootstrap')['States']['Lifecycle']
        assert state['Resource'].endswith('createProcessingJob.sync')
        assert 'States.UUID()' in state['Parameters']['ProcessingJobName.$']
        assert state['Parameters']['AppSpecification']['ContainerArguments'] == [mode]


def test_bundle_excludes_secrets():
    import tarfile
    with tarfile.open(fileobj=io.BytesIO(workflows.bundle())) as archive:
        names = archive.getnames()
    assert 'infra/sagemaker/lifecycle.py' in names
    assert 'infra/sagemaker/requirements.txt' in names
    assert 'src/retail_forecast/config.py' in names
    assert not any('.env' in n or '.aws' in n or '__pycache__' in n for n in names)


def test_notebook_startup_is_valid_shell(tmp_path):
    import subprocess
    from infra.aws.dashboard_host import startup
    script = tmp_path / 'start.sh'
    script.write_text(startup('s3://test/code/runtime.tar.gz'))
    subprocess.run(['bash', '-n', str(script)], check=True)


def test_notebook_deployment_matches_aws_api(monkeypatch):
    import botocore.session
    from botocore.validate import validate_parameters
    from infra.aws import dashboard_host as host

    sm = Mock()
    sm.describe_notebook_instance.return_value = {'NotebookInstanceStatus': 'Stopped'}
    service = botocore.session.get_session().get_service_model('sagemaker')
    sm.update_notebook_instance.side_effect = lambda **kwargs: validate_parameters(
        kwargs, service.operation_model('UpdateNotebookInstance').input_shape
    )
    session = Mock()
    session.client.return_value = sm
    monkeypatch.setattr(host.boto3, 'Session', lambda **kwargs: session)
    monkeypatch.setattr(host, 's3_client', Mock())
    monkeypatch.setattr(host, 'bundle', lambda: b'package')
    host.deploy('retail-forecast-nb')
    sm.start_notebook_instance.assert_called_once_with(NotebookInstanceName='retail-forecast-nb')


def test_training_code_uri_never_changes_under_an_existing_job(monkeypatch):
    from infra.sagemaker import launch
    objects = {}
    s3 = Mock()
    s3.put_object.side_effect = lambda **kw: objects.update({kw['Key']: kw['Body']})
    monkeypatch.setattr(launch, 'build_tarball', lambda: b'version1')
    first = launch.upload_code(s3, 'bucket')
    monkeypatch.setattr(launch, 'build_tarball', lambda: b'version2')
    second = launch.upload_code(s3, 'bucket')
    assert first != second
    assert objects[first.removeprefix('s3://bucket/')] == b'version1'


@pytest.mark.parametrize('status', ['SUCCEEDED', 'FAILED'])
def test_waited_workflow_reports_the_cloud_result(monkeypatch, status):
    import boto3
    session = Mock()
    session.client.return_value.get_caller_identity.return_value = {'Account': '123'}
    session.client.return_value.start_execution.return_value = {'executionArn': 'arn:run'}
    session.client.return_value.describe_execution.return_value = {'status': status, 'cause': 'gate'}
    monkeypatch.setattr(boto3, 'Session', lambda **kwargs: session)
    if status == 'FAILED':
        with pytest.raises(RuntimeError, match='FAILED'):
            workflows.main(['run', '--mode', 'training', '--wait'])
    else:
        workflows.main(['run', '--mode', 'training', '--wait'])


def test_successful_pipeline_without_true_gate_cannot_publish(monkeypatch):
    from infra.sagemaker import launch, pipeline, register
    sm = Mock()
    sm.describe_pipeline_execution.return_value = {'PipelineExecutionStatus': 'Succeeded'}
    sm.get_paginator.return_value.paginate.return_value = [{'PipelineExecutionSteps': [
        {'StepName': 'GateWAPE', 'StepStatus': 'Succeeded', 'Metadata': {'Condition': {'Outcome': 'False'}}},
        {'StepName': 'RegisterApproved-RegisterModel', 'StepStatus': 'Succeeded',
         'Metadata': {'RegisterModel': {'Arn': 'arn/candidate'}}}]}]
    monkeypatch.setattr(launch, '_sm_client', lambda: sm)
    monkeypatch.setattr(register, 'ensure_group', Mock())
    monkeypatch.setattr(pipeline, 'build_pipeline', Mock())
    monkeypatch.setattr(lifecycle, 's3_client', Mock())
    monkeypatch.setattr(lifecycle, 'read_active', lambda *a, **kw: (None, None))
    forecast, publish = Mock(), Mock()
    monkeypatch.setattr(lifecycle, 'run_forecast', forecast)
    monkeypatch.setattr(lifecycle, 'publish', publish)
    with pytest.raises(RuntimeError, match='gate indépendant'):
        lifecycle.run('training')
    forecast.assert_not_called()
    publish.assert_not_called()


def test_existing_model_cannot_silently_serve_an_old_artifact():
    from infra.sagemaker.batch_transform import ensure_model
    sm = Mock()
    sm.describe_model.return_value = {'PrimaryContainer': {'Image': 'image', 'ModelDataUrl': 'old'}}
    with pytest.raises(ValueError, match='incompatible'):
        ensure_model(sm, 'name', 'image', 'new')
    sm.create_model.assert_not_called()


def test_transform_timeout_stops_its_own_job():
    from infra.sagemaker.batch_transform import run_transform
    from botocore.exceptions import WaiterError
    sm = Mock()
    sm.get_waiter.return_value.wait.side_effect = WaiterError(name='transform', reason='timeout', last_response={})
    sm.describe_transform_job.return_value = {'TransformJobStatus': 'InProgress'}
    with pytest.raises(WaiterError):
        run_transform(sm, 'own-job', 'model', 's3://b/input')
    sm.stop_transform_job.assert_called_once_with(TransformJobName='own-job')


def test_model_creation_points_framework_at_packaged_handler(monkeypatch):
    from infra.sagemaker.batch_transform import ensure_model
    monkeypatch.setattr('infra.sagemaker.batch_transform._role', lambda: 'arn:aws:iam::123456789012:role/test')
    sm = Mock()
    sm.describe_model.side_effect = ClientError({'Error': {'Code': 'ValidationException'}}, 'DescribeModel')
    ensure_model(sm, 'name', 'image', 's3://b/model')
    env = sm.create_model.call_args.kwargs['PrimaryContainer']['Environment']
    assert env['SAGEMAKER_PROGRAM'] == 'inference.py'
    assert env['SAGEMAKER_SUBMIT_DIRECTORY'] == '/opt/ml/model/code'


def test_model_with_wrong_handler_directory_cannot_be_reused():
    from infra.sagemaker.batch_transform import ensure_model
    sm = Mock()
    sm.describe_model.return_value = {'PrimaryContainer': {
        'Image': 'image', 'ModelDataUrl': 's3://b/model', 'Environment': {'SAGEMAKER_PROGRAM': 'inference.py'}}}
    with pytest.raises(ValueError): ensure_model(sm, 'name', 'image', 's3://b/model')


def test_serverless_installs_dependencies_on_importable_writable_path(monkeypatch):
    """Le site utilisateur non importé ne doit pas masquer le handler."""
    from infra.sagemaker.batch_transform import ensure_model
    monkeypatch.setattr('infra.sagemaker.batch_transform._role', lambda: 'arn:aws:iam::123456789012:role/test')
    sm = Mock()
    sm.describe_model.side_effect = ClientError({'Error': {'Code': 'ValidationException'}}, 'DescribeModel')
    ensure_model(sm, 'serverless', 'image', 's3://b/model', serverless=True)
    env = sm.create_model.call_args.kwargs['PrimaryContainer']['Environment']
    assert env['PIP_TARGET'] == '/opt/ml/model/code'
    assert env['PIP_NO_DEPS'] == '1'
    assert env['PIP_USER'] == '0'
    assert env['PIP_TARGET'] in env['PYTHONPATH'].split(':')
    assert '/opt/ml/model/code' in env['PYTHONPATH'].split(':')


def test_endpoint_cleanup_waits_before_removing_dependencies(monkeypatch):
    """Un endpoint en suppression conserve ses dépendances jusqu'à sa disparition."""
    from infra.sagemaker import deploy
    sm = Mock()
    monkeypatch.setattr(deploy, '_sm', lambda: sm)
    events = []
    sm.delete_endpoint.side_effect = lambda **kw: events.append('endpoint')
    sm.get_waiter.return_value.wait.side_effect = lambda **kw: events.append('wait')
    sm.delete_endpoint_config.side_effect = lambda **kw: events.append('config')
    sm.delete_model.side_effect = lambda **kw: events.append('model')
    assert deploy.down() == 0
    assert events == ['endpoint', 'wait', 'config', 'model']
