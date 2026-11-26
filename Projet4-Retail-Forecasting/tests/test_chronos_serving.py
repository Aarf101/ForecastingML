"""Contrat d'inférence Chronos, testé sans poids ni service externe."""

import io
import json
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from infra.sagemaker import chronos_inference as serving


def sales_frame():
    dates=pd.date_range('2024-01-01',periods=300)
    return pd.DataFrame({'date':list(dates)*2,'store_id':'S01','item_id':['I001']*300+['I002']*300,'units':3})


def test_chronos_requests_keep_complete_histories_and_reject_missing_dates():
    sales=sales_frame();requests=serving.requests_from_sales(sales,chunk_size=1)
    assert len(requests)==2
    assert all(len(r['series'][0]['target'])==256 for r in requests)
    assert all(set(r['series'][0])=={'series_id','origin_date','target'} for r in requests)
    serving.input_fn(json.dumps(requests[0]),'application/json')
    with pytest.raises(ValueError,match='incomplet'):
        serving.requests_from_sales(sales.drop(index=290))


@pytest.mark.parametrize('change',[lambda s:s.update(target=[3.]*255),lambda s:s.update(target=[float('nan')]*256),lambda s:s.update(stockout=1)])
def test_chronos_input_rejects_invalid_or_leaking_context(change):
    request=serving.requests_from_sales(sales_frame())[0]
    change(request['series'][0])
    with pytest.raises(ValueError):serving.input_fn(json.dumps(request),'application/json')


def test_chronos_handler_preserves_series_horizons_and_never_exposes_targets():
    request=serving.requests_from_sales(sales_frame())[0]
    class FakePipeline:
        def predict_df(self,history,**kwargs):
            future=kwargs['future_df']
            assert history.date.max()<future.date.min()
            assert 'target' not in future
            assert not {'stockout','stock_on_hand','promo','unit_price'}.intersection(history.columns)
            return future.assign(predictions=4.,**{'0.1':2.,'0.9':8.})
    result=serving.predict_fn(serving.input_fn(json.dumps(request),'application/json'),{'pipeline':FakePipeline()})
    assert len(result)==32
    assert {r['horizon'] for r in result}==set(range(1,17))
    assert all(r['p10']<=r['forecast_units']<=r['p90'] for r in result)
    assert len(json.loads(serving.output_fn(result,'application/json')))==32


def test_chronos_job_is_bounded_and_does_not_publish_active_manifest(tmp_path,monkeypatch):
    from infra.sagemaker import chronos_benchmark as launcher
    active={'model_data_url':'s3://bucket/model.tar.gz','model_package_arn':'package-v7'}
    s3=Mock();s3.get_object.return_value={'Body':io.BytesIO(json.dumps(active).encode())}
    sm=Mock();sm.create_training_job.return_value={'TrainingJobArn':'arn:job'}
    session=Mock();session.client.side_effect=lambda name:{'s3':s3,'sagemaker':sm}[name]
    monkeypatch.setattr(launcher,'session',lambda:session)
    monkeypatch.setattr(launcher,'source_bundle',lambda:b'code')
    monkeypatch.setattr(launcher,'ROOT',tmp_path)
    monkeypatch.setattr(launcher,'EVIDENCE',tmp_path/'evidence.json')
    raw=tmp_path/'data/raw';raw.mkdir(parents=True);(raw/'sales.parquet').write_bytes(b'data')
    launcher.start()
    request=sm.create_training_job.call_args.kwargs
    assert request['StoppingCondition']['MaxRuntimeInSeconds']==3600
    assert request['ResourceConfig']['InstanceCount']==1
    assert all(call.kwargs['Key']!='deployment/active.json' for call in s3.put_object.call_args_list)


def test_full_catalog_serving_comparison_rejects_drift_and_missing_pairs():
    from infra.sagemaker.chronos_validate import validate_forecast
    frame=pd.MultiIndex.from_product([range(1200),range(1,17)],names=['series','horizon']).to_frame(index=False)
    frame['store_id']=frame.series.map(lambda s:f'S{s//60+1:02d}')
    frame['item_id']=frame.series.map(lambda s:f'I{s%60+1:03d}')
    frame['origin_date']=pd.Timestamp('2024-12-30')
    frame['target_date']=frame.origin_date+pd.to_timedelta(frame.horizon,unit='D')
    frame=frame.assign(forecast_units=4.,p10=2.,p90=8.)
    assert validate_forecast(frame.copy(),frame.copy())=={'forecast_units':0.,'p10':0.,'p90':0.}
    changed=frame.copy();changed.loc[0,'forecast_units']=5.
    with pytest.raises(ValueError,match='différent'):
        validate_forecast(changed,frame.copy())
    with pytest.raises(ValueError,match='incomplet'):
        validate_forecast(frame.iloc[:-1].copy(),frame.copy())


def test_validation_stops_only_its_own_timed_out_transform(tmp_path,monkeypatch):
    from datetime import datetime, timedelta, timezone
    from infra.sagemaker import chronos_validate as validation, chronos_benchmark as launcher
    monkeypatch.setattr(validation,'EVIDENCE',tmp_path/'full.json')
    for i in range(6):
        (tmp_path/f'chronos_full_aws_s{i}.json').write_text(json.dumps({'run_id':'run','job':f'own-{i}'}))
    sm=Mock();sm.describe_training_job.return_value={'TrainingJobStatus':'Completed'}
    sm.describe_transform_job.return_value={'TransformJobStatus':'InProgress','CreationTime':datetime.now(timezone.utc)-timedelta(minutes=31)}
    session=Mock();session.client.return_value=sm
    monkeypatch.setattr(validation,'session',lambda:session)
    monkeypatch.setattr(validation,'collect',lambda i:None)
    monkeypatch.setattr(validation,'combine',lambda:None)
    monkeypatch.setattr(launcher,'start_transform',lambda:'own-transform')
    with pytest.raises(TimeoutError,match='30 minutes'):
        validation.run_validation()
    sm.stop_transform_job.assert_called_once_with(TransformJobName='own-transform')
    sm.stop_training_job.assert_not_called()
