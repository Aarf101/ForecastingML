"""Génère la synthèse et le graphique depuis un benchmark Chronos terminé."""

import argparse
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=Path('reports/chronos2_benchmark.json'))
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--doc', type=Path, default=Path('docs/chronos_benchmark.md'))
    parser.add_argument('--figure', type=Path, default=Path('reports/chronos2_by_horizon.png'))
    args = parser.parse_args(argv)
    import json,hashlib
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from retail_forecast.evaluate.benchmark_chronos import select_series
    p=args.report;r=json.loads(p.read_text());c=r['config'];m=r['metrics']
    sales=pd.read_parquet('data/raw/sales.parquet',columns=['store_id','item_id']);pairs=select_series(sales,c['series']);sample=pd.DataFrame(pairs,columns=sales.columns)
    r['sample']={'stores':int(sample.store_id.nunique()),'items':int(sample.item_id.nunique()),'series_ids':['/'.join(t) for t in pairs]}
    r['reference_model_package']=json.loads(Path('data/benchmark/active.json').read_text())['model_package_arn']
    r['dataset_sha256']=hashlib.file_digest(open('data/raw/sales.parquet','rb'),'sha256').hexdigest()
    r['reference_artifact_sha256']=hashlib.file_digest(open('data/benchmark/active-model.tar.gz','rb'),'sha256').hexdigest()
    r['chronos_weights_sha256']=hashlib.file_digest(open(args.weights,'rb'),'sha256').hexdigest()
    panel=pd.read_parquet('data/benchmark/panel-'+c['dataset_and_artifact_sha256'][:12]+'.parquet')
    idx=pd.MultiIndex.from_tuples(pairs,names=['store_id','item_id'])
    panel=panel[panel.set_index(['store_id','item_id']).index.isin(idx)].copy()
    pred=pd.concat([pd.read_parquet(f) for f in Path(r['checkpoint_directory']).glob('*.parquet')],ignore_index=True)
    joined=panel.merge(pred,on=['store_id','item_id','date','horizon'],validate='one_to_one')
    joined['error_lgb']=(joined.y-joined.lightgbm).abs()
    joined['error_chronos']=(joined.y-joined.chronos).abs()
    g=joined.groupby(['store_id','item_id','horizon'])[['y','error_lgb','error_chronos']].sum()
    arrays=[g[col].unstack('horizon').to_numpy() for col in ['y','error_lgb','error_chronos']]
    import numpy as np
    rng=np.random.default_rng(20261126);gaps=[]
    for _ in range(20):
     ids=rng.integers(0,len(pairs),size=(100,len(pairs)))
     denom=arrays[0][ids].sum(axis=1)
     gaps.extend(100*((arrays[2][ids].sum(axis=1)/denom).mean(axis=1)-(arrays[1][ids].sum(axis=1)/denom).mean(axis=1)))
    r['bootstrap_wape_gap_pp_95']=list(map(float,np.quantile(gaps,[.025,.975])))
    r['bootstrap_method']='2000 paired resamples of whole series; preserves horizons and dates, does not capture all cross-series dependence.'
    r['relative_wape_change_vs_lightgbm']=(m['chronos']['mean_wape']/m['lightgbm']['mean_wape']-1)
    from importlib.metadata import version
    local_versions={name:version(name) for name in ['chronos-forecasting','torch','transformers','lightgbm','pandas']}
    r['report_builder_versions']=local_versions
    r['versions']=r.get('metadata',{}).get('versions',local_versions)
    r['scope']=f"Matched comparison on {c['series']}/1200 synthetic series; not a real-sales benchmark."
    p.write_text(json.dumps(r,indent=2)+'\n')
    fig,axes=plt.subplots(2,1,figsize=(9,7),sharex=True,layout='constrained')
    for name,label,color in [('lightgbm','LightGBM v7 (moyenne)','#2166ac'),('chronos','Chronos-2 (médiane)','#d6604d'),('seasonal_naive','Baseline saisonnière','#777777')]:
     d=m[name]['by_horizon'];hs=sorted(map(int,d));axes[0].plot(hs,[100*d[str(h)]['wape'] for h in hs],marker='o',markersize=3,label=label,color=color)
     if name!='seasonal_naive':axes[1].plot(hs,[100*d[str(h)]['coverage_80'] for h in hs],marker='o',markersize=3,label=label,color=color)
    axes[0].set(ylabel='WAPE (%) — plus bas = mieux',title=f"{c['series']} séries appariées · 28 origines par horizon · données synthétiques")
    axes[1].axhline(80,color='#777777',linestyle='--',label='80 % nominal');axes[1].set(xlabel='Horizon (jours)',ylabel='Couverture P10–P90 (%)',xticks=range(1,17))
    for ax in axes:ax.grid(alpha=.2);ax.legend(frameon=False)
    fig.savefig(args.figure,dpi=160);plt.close(fig)
    text='''\n## Résultat exécuté\n\n| Mesure | LightGBM v7 | Chronos-2 |\n|---|---:|---:|\n'''
    for label,key,fmt in [('WAPE moyen','mean_wape','.2%'),('RMSLE moyen','mean_rmsle','.4f'),('Couverture P10/P90','mean_coverage_80','.2%')]:
     text+=f"| {label} | {m['lightgbm'][key]:{fmt}} | {m['chronos'][key]:{fmt}} |\n"
    text+=f"\nBaseline saisonnière sur ces mêmes observations : **{m['seasonal_naive']['mean_wape']:.2%}** de WAPE.\n"
    delta=r['relative_wape_change_vs_lightgbm'];text+=f"Écart relatif de WAPE de Chronos-2 face à LightGBM : **{delta:+.2%}** (un signe négatif signifie moins d’erreur). Le modèle actif n’a pas été remplacé. Le résultat reste propre au jeu synthétique ; la publication active est un parcours distinct.\n"
    text+=f"\nLes {c['series']} séries couvrent {r['sample']['stores']} magasins et {r['sample']['items']} articles distincts.\nTemps cumulé d’inférence Chronos sur les 43 origines : {r['total_inference_seconds']/60:.1f} minutes,\nsur {c['device']}, avec {c['threads']} threads par processus. Il s’agit du temps cumulé, pas du délai d’une seule prévision de 16 jours.\n"
    text+=f"\nÉcart de WAPE, intervalle bootstrap descriptif à 95 % : {r['bootstrap_wape_gap_pp_95'][0]:+.2f} à {r['bootstrap_wape_gap_pp_95'][1]:+.2f} points. Rééchantillonnage de séries entières ; les corrélations entre séries ne sont pas toutes représentées.\n"
    text+=f'\n![Comparaison par horizon](../reports/{args.figure.name})\n\nPreuve détaillée : [JSON du benchmark](../reports/{args.report.name}).\n'
    doc=args.doc;original=doc.read_text().split('\n## Résultat exécuté')[0];doc.write_text(original.rstrip()+'\n'+text)
    print(text)


if __name__ == "__main__":
    main()
