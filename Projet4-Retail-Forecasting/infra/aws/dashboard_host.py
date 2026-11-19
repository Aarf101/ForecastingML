"""Héberge Streamlit sur le notebook existant du Learner Lab."""

import argparse
import base64
import hashlib
import shlex

import boto3

from infra.aws.workflows import bundle
from retail_forecast.config import SETTINGS
from retail_forecast.ingest.storage import s3_client


def startup(uri):
    """Le notebook reçoit le code par son rôle IAM, jamais des identifiants."""
    command = (
        f'env S3_BUCKET={shlex.quote(SETTINGS.bucket)} '
        f'AWS_REGION={shlex.quote(SETTINGS.region)} S3_ENDPOINT_URL= '
        '"$PY" -m streamlit run /home/ec2-user/SageMaker/retail-current/dashboards/streamlit/app.py '
        '--server.address 0.0.0.0 --server.port 8501 --server.headless true'
    )
    return f'''#!/bin/bash
set -euo pipefail
aws s3 cp {shlex.quote(uri)} /tmp/retail-dashboard.tar.gz
mkdir -p /home/ec2-user/SageMaker/retail-current
tar -xzf /tmp/retail-dashboard.tar.gz -C /home/ec2-user/SageMaker/retail-current
chown -R ec2-user:ec2-user /home/ec2-user/SageMaker/retail-current
cat > /home/ec2-user/SageMaker/start-retail.sh <<'SCRIPT'
#!/bin/bash
set -euo pipefail
PY=""
for candidate in /home/ec2-user/anaconda3/envs/*/bin/python; do
    if "$candidate" -c 'import sys; assert sys.version_info >= (3,10)' 2>/dev/null; then
        PY="$candidate"
        break
    fi
done
test -n "$PY"
"$PY" -m pip install -q -e '/home/ec2-user/SageMaker/retail-current[ui]' boto3
cd /home/ec2-user/SageMaker/retail-current
exec {command}
SCRIPT
chown ec2-user:ec2-user /home/ec2-user/SageMaker/start-retail.sh
chmod +x /home/ec2-user/SageMaker/start-retail.sh
nohup runuser -l ec2-user -c /home/ec2-user/SageMaker/start-retail.sh > /home/ec2-user/SageMaker/retail-dashboard.log 2>&1 &
'''


def deploy(notebook):
    """Ajoute un démarrage automatique sans écraser un lifecycle tiers."""
    session = boto3.Session(profile_name=SETTINGS.profile or None, region_name=SETTINGS.region)
    sm = session.client('sagemaker')
    detail = sm.describe_notebook_instance(NotebookInstanceName=notebook)
    name = 'retail-dashboard-start'
    running = detail['NotebookInstanceStatus'] == 'InService'
    if detail['NotebookInstanceStatus'] not in ['Stopped', 'InService']:
        raise ValueError('attendre la fin du démarrage du notebook')
    if detail.get('NotebookInstanceLifecycleConfigName') not in (None, name):
        raise ValueError('un lifecycle tiers est présent : ne pas le remplacer')
    payload = bundle()
    key = f'code/dashboard/{hashlib.sha256(payload).hexdigest()}/runtime.tar.gz'
    s3_client().put_object(Bucket=SETTINGS.bucket, Key=key, Body=payload)
    script = startup(f's3://{SETTINGS.bucket}/{key}')
    content = [{'Content': base64.b64encode(script.encode()).decode()}]
    try:
        sm.describe_notebook_instance_lifecycle_config(NotebookInstanceLifecycleConfigName=name)
    except sm.exceptions.ClientError as exc:
        if exc.response['Error']['Code'] != 'ValidationException':
            raise
        sm.create_notebook_instance_lifecycle_config(NotebookInstanceLifecycleConfigName=name, OnStart=content)
    else:
        sm.update_notebook_instance_lifecycle_config(NotebookInstanceLifecycleConfigName=name, OnStart=content)
    if running:
        # Même lifecycle : actualise le code sans interrompre le notebook.
        if detail.get('NotebookInstanceLifecycleConfigName') != name:
            raise ValueError('le notebook actif ne possède pas le lifecycle attendu')
        import requests
        from urllib.parse import urlsplit
        from retail_forecast.config import ROOT
        http = requests.Session()
        url = sm.create_presigned_notebook_instance_url(NotebookInstanceName=notebook)['AuthorizedUrl']
        response = http.get(url, timeout=30)
        parts = urlsplit(response.url)
        base = f'{parts.scheme}://{parts.netloc}'
        for cookie in http.cookies:
            if cookie.name == '_xsrf':
                http.headers['X-XSRFToken'] = cookie.value
        for relative in ['dashboards/streamlit/app.py', 'src/retail_forecast/agent/brief.py']:
            result = http.put(base + '/api/contents/retail-current/' + relative,
                json={'type': 'file', 'format': 'base64',
                      'content': base64.b64encode((ROOT / relative).read_bytes()).decode()}, timeout=30)
            result.raise_for_status()
        print('Dashboard actualisé sur le notebook actif.')
        return
    sm.update_notebook_instance(NotebookInstanceName=notebook, LifecycleConfigName=name)
    sm.get_waiter('notebook_instance_stopped').wait(NotebookInstanceName=notebook)
    sm.start_notebook_instance(NotebookInstanceName=notebook)
    print(f'{notebook}: démarrage demandé ; ouvrir /proxy/8501/ dans Jupyter')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--notebook', default='retail-forecast-nb')
    args = parser.parse_args(argv)
    deploy(args.notebook)


if __name__ == '__main__':
    main()
