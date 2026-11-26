"""Contrat de comparaison : mêmes cibles et aucune observation future fournie."""

import pandas as pd
import pytest

from retail_forecast.evaluate.benchmark_chronos import context_frames, score
from retail_forecast.features.build import TARGET


def test_context_does_not_expose_future_sales_prices_or_promotions():
    dates = pd.date_range('2024-01-01', periods=40)
    sales = pd.DataFrame({'series_id':['S01/I001']*40, 'date':dates, 'units':range(40),
                          'promo':[0]*20+[1]*20, 'unit_price':[10]*20+[99]*20,
                          'is_holiday':0, 'lead_time_days':3, 'stock_on_hand':999})
    origin = dates[19]
    context, future = context_frames(sales, origin, 10, 'retail')
    assert len(context) == 10
    assert context.date.max() == origin
    assert context.unit_price.eq(10).all()
    assert context.promo.eq(0).all()
    assert len(future) == 16 and future.date.min() > origin
    assert set(future.columns) == {'series_id','date','dow','month','day','is_holiday'}
    assert 'stock_on_hand' not in context


def test_comparison_requires_every_matched_target():
    panel = pd.DataFrame({'store_id':['S01'], 'item_id':['I001'], 'date':[pd.Timestamp('2024-01-01')],
                          'horizon':[1], TARGET:[5.], 'seasonal_naive':[4.],
                          'lightgbm':[5.], 'lightgbm_p10':[2.], 'lightgbm_p90':[7.]})
    pred = panel[['store_id','item_id','date','horizon']].assign(chronos=4.,chronos_p10=1.,chronos_p90=6.)
    result = score(panel, pred)
    assert result['lightgbm']['mean_wape'] == 0
    assert result['chronos']['mean_wape'] == pytest.approx(.2)
    with pytest.raises(ValueError, match='manquantes'):
        score(panel, pred.iloc[:0])


def test_series_sample_is_balanced_and_uses_only_identifiers():
    from retail_forecast.evaluate.benchmark_chronos import select_series
    sales = pd.MultiIndex.from_product([['S01','S02','S03'],['I001','I002','I003','I004']], names=['store_id','item_id']).to_frame(index=False)
    chosen = select_series(sales, 6)
    assert len(chosen) == 6
    assert all(len(pair) == 2 for pair in chosen)
    assert pd.Series([s for s,i in chosen]).value_counts().to_dict() == {'S01':2,'S02':2,'S03':2}
    assert select_series(sales.iloc[::-1], 6) == chosen
