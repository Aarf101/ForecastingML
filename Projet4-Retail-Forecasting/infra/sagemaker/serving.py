"""Adapte le rangement du handler sans modifier les fichiers évalués."""

import gzip
import hashlib
import io
import tarfile

from retail_forecast.config import SETTINGS


def serving_artifact(s3, model_uri):
    """Conserve chaque membre original et ajoute le répertoire standard code/."""
    bucket, key = model_uri.removeprefix('s3://').split('/', 1)
    body = s3.get_object(Bucket=bucket, Key=key)['Body'].read()
    with tarfile.open(fileobj=io.BytesIO(body), mode='r:gz') as source:
        members = {member.name.removeprefix('./'): member for member in source.getmembers() if member.isfile()}
        if 'code/inference.py' in members and 'code/requirements.txt' in members:
            return model_uri
        if 'inference.py' not in members or 'requirements.txt' not in members:
            raise ValueError('handler ou dépendances absents de artefact évalué')
        output = io.BytesIO()
        with tarfile.open(fileobj=output, mode='w') as dest:
            for member in source.getmembers():
                dest.addfile(member, source.extractfile(member) if member.isfile() else None)
            for name in ['inference.py', 'requirements.txt']:
                target = 'code/' + name
                if target in members:
                    continue
                content = source.extractfile(members[name]).read()
                info = tarfile.TarInfo(target)
                info.size = len(content)
                info.mode = 0o644
                dest.addfile(info, io.BytesIO(content))
    payload = gzip.compress(output.getvalue(), mtime=0)
    target = f'models/serving/{hashlib.sha256(payload).hexdigest()}/model.tar.gz'
    s3.put_object(Bucket=SETTINGS.bucket, Key=target, Body=payload)
    return f's3://{SETTINGS.bucket}/{target}'
