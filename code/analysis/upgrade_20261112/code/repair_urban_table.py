"""Rebuild population-weighted urban share using the original spatial inputs."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio import features

BASE = Path('__PROJECT_ROOT__')
SD = Path('__BRAZIL_SOURCE_ROOT__')
OUT = BASE / 'upgrade_20261112/derived/urban_repair'
OUT.mkdir(parents=True, exist_ok=False)
G = BASE / 'work/grid'
meta = json.loads((G / 'meta.json').read_text())
nr, nc = meta['nrows'], meta['ncols']
pixel = meta['dx']
polygons = gpd.read_file(SD / 'river/data/official/ibge_malha/extracted/BR_Municipios_2022.shp')
code = next(c for c in polygons.columns if c.upper().startswith('CD_MUN'))
polygons['m6'] = polygons[code].astype(str).str[:6].astype(np.int64)
minx, miny, maxx, maxy = polygons.total_bounds
c0, c1 = int(np.floor((minx + 180) / pixel)), int(np.ceil((maxx + 180) / pixel))
r0, r1 = int(np.floor((85 - maxy) / pixel)), int(np.ceil((85 - miny) / pixel))
transform = rasterio.transform.from_origin(-180 + c0 * pixel, 85 - r0 * pixel, pixel, pixel)
municipalities = features.rasterize(zip(polygons.geometry, polygons.m6),
    out_shape=(r1-r0,c1-c0), transform=transform, fill=0, dtype='int64')
valid = municipalities.ravel() > 0
codes, inverse = np.unique(municipalities.ravel()[valid], return_inverse=True)
population = np.memmap(G/'pop2025.f32', dtype='float32', mode='r', shape=(nr,nc))[r0:r1,c0:c1].ravel()[valid]
urban = np.memmap(G/'urban.u8', dtype='uint8', mode='r', shape=(nr,nc))[r0:r1,c0:c1].ravel()[valid]
assert set(np.unique(urban)).issubset({0,1})
assert np.isfinite(population).all() and (population >= 0).all()
pop = np.bincount(inverse, weights=population, minlength=len(codes))
urban_pop = np.bincount(inverse, weights=population * urban, minlength=len(codes))
urban_cells = np.bincount(inverse, weights=urban, minlength=len(codes))
aggregates = pd.DataFrame({'m6':codes,'pop_recomputed':pop,'urban_population':urban_pop,'urban_cell_count':urban_cells})
old = pd.read_csv(BASE/'work/flow_holdout.csv')
old = old.drop(columns=[c for c in old if c.startswith('Unnamed:')])
old = old.rename(columns={'urban_share':'legacy_urban_share'})
d = old.merge(aggregates, on='m6', how='left', validate='one_to_one')
assert d.urban_population.notna().all()
assert np.allclose(d['pop'], d.pop_recomputed, rtol=1e-10, atol=1e-6)
assert np.allclose(d.legacy_urban_share, d.urban_cell_count/d['pop'], rtol=1e-8)
cent = pd.read_csv(SD/'01_data/interim/municipios_master.csv', dtype={'codigo_ibge':str})
cent['m6'] = pd.to_numeric(cent.codigo_ibge.str[:6], errors='raise')
cent = cent[['m6','latitude','longitude']].dropna().drop_duplicates('m6')
d = d.merge(cent,on='m6',how='left',validate='one_to_one')
excluded = d[d[['latitude','longitude']].isna().any(axis=1)]
excluded[['m6']].to_csv(OUT/'excluded_coordinates.csv',index=False)
d = d.dropna(subset=['latitude','longitude']).copy()
d['urban_share'] = d.urban_population/d['pop']
d.to_csv(OUT/'municipalities_corrected.csv',index=False)
report = {'original_rows':len(old),'analysis_rows':len(d),'excluded_coordinates':len(excluded),
          'population_matches_original':True,'legacy_variable_matches_urban_cells_per_person':True,
          'corrected_urban_share_min':float(d.urban_share.min()),'corrected_urban_share_max':float(d.urban_share.max()),
          'urban_definition':'GHS_SMOD2020 classes 21,22,23,30; population weights GHSL2025; original rasterization',
          'source_script_sha256':hashlib.sha256((BASE/'src/27_flow_holdout.py').read_bytes()).hexdigest(),
          'corrected_table_sha256':hashlib.sha256((OUT/'municipalities_corrected.csv').read_bytes()).hexdigest()}
(OUT/'audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
