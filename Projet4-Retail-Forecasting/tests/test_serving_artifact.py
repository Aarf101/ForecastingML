"""Le ré-empaquetage de serving ne modifie aucun membre évalué."""

import io
import tarfile
from unittest.mock import Mock

from infra.sagemaker.serving import serving_artifact


def test_serving_layout_preserves_evaluated_files():
    files = {'model_1.txt': b'weights', 'metadata.json': b'{}',
             'inference.py': b'handler', 'requirements.txt': b'lightgbm==4.5.0'}
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w:gz') as archive:
        for name, data in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    s3 = Mock()
    s3.get_object.side_effect = lambda **kwargs: {'Body': io.BytesIO(buffer.getvalue())}
    uri = serving_artifact(s3, 's3://bucket/evaluated.tar.gz')
    payload = s3.put_object.call_args.kwargs['Body']
    with tarfile.open(fileobj=io.BytesIO(payload), mode='r:gz') as archive:
        result = {m.name: archive.extractfile(m).read() for m in archive if m.isfile()}
    assert all(result[name] == data for name, data in files.items())
    assert result['code/inference.py'] == files['inference.py']
    assert result['code/requirements.txt'] == files['requirements.txt']
    assert serving_artifact(s3, 's3://bucket/evaluated.tar.gz') == uri
