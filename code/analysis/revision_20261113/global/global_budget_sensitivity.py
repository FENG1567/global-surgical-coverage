"""Paired conditional budget scenarios and composition-strengthening experiments."""
from pathlib import Path
import hashlib
import json
import subprocess
import time

import numpy as np
import pandas as pd

BASE = Path('__PROJECT_ROOT__')
GRID = BASE / 'work/grid'
OUT = BASE / 'revision_20261113/global'
NR, NC = 17400, 43200
SEED = 20261113
THRESHOLDS = (30, 120)


def prepare_inputs():
    hospitals = pd.read_parquet(BASE / 'data/osm/osm_hospitals_all.parquet').reset_index(drop=True)
    hospitals['facility_idx'] = hospitals.index
    hospitals['row'] = np.floor((85 - hospitals.lat.to_numpy()) * 120).astype('int64')
    hospitals['col'] = np.floor((hospitals.lon.to_numpy() + 180) * 120).astype('int64')
    assert hospitals.iso3.notna().all()
    assert hospitals.row.between(0, NR - 1).all()
    assert hospitals.col.between(0, NC - 1).all()
    population = np.memmap(GRID / 'pop2025.f32', dtype='float32', mode='r', shape=(NR, NC))
    local = np.zeros(len(hospitals))
    valid = hospitals.row.between(1, NR - 2) & hospitals.col.between(1, NC - 2)
    rows = hospitals.loc[valid, 'row'].to_numpy()
    cols = hospitals.loc[valid, 'col'].to_numpy()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            local[valid] += population[rows + dr, cols + dc]
    hospitals['local_pop'] = local
    hospitals['high_local_pop'] = local > hospitals.groupby('iso3').local_pop.transform('median')
    hospitals['cell1'] = (np.floor(85 - hospitals.lat) * 1000 + np.floor(hospitals.lon + 180)).astype('int64')
    capability = pd.read_csv(BASE / 'outputs/hospital_capability.csv').set_index('iso3')
    provenance = []
    for iso, country in hospitals.groupby('iso3', sort=True):
        matched = iso in capability.index
        row = capability.loc[iso].to_dict() if matched else {}
        central = float(row.get('p_c', .80))
        lower = float(row.get('p_c_lo', central))
        upper = float(row.get('p_c_hi', central))
        lower = lower if np.isfinite(lower) else central
        upper = upper if np.isfinite(upper) else central
        assert 0 <= lower <= central <= upper <= 1
        provenance.append({'iso3': iso, 'n_candidates': len(country), 'capability_lookup_matched': matched,
                           'p_c': central, 'p_c_lo': lower, 'p_c_hi': upper,
                           'p_c_source': row.get('p_c_source', 'fallback_0.80'),
                           'survey_year': row.get('survey_year', ''), 'basis': row.get('basis', ''),
                           'n_survey_hospitals': row.get('n_survey_hospitals', ''),
                           'n_osm_in_source': row.get('n_osm_hospitals', ''), 'source': row.get('source', '')})
    return hospitals, pd.DataFrame(provenance)


def paired_orders(hospitals):
    rng = np.random.default_rng(SEED)
    orders = {}
    for iso, country in hospitals.groupby('iso3', sort=True):
        ids = country.index.to_numpy()
        random_order = rng.permutation(ids)
        blocks = []
        for cell in rng.permutation(country.cell1.unique()):
            blocks.append(rng.permutation(country.index[country.cell1.eq(cell)].to_numpy()))
        cluster_order = np.concatenate(blocks)
        orders[iso] = {'random': random_order, 'cluster': cluster_order}
    return orders


def select(hospitals, provenance, orders, budget, allocation, exact_high_counts=None):
    selected = []
    checks = []
    probability_column = {'lower': 'p_c_lo', 'central': 'p_c', 'upper': 'p_c_hi'}[budget]
    lookup = provenance.set_index('iso3')
    high = hospitals.high_local_pop.to_numpy()
    for iso, country in hospitals.groupby('iso3', sort=True):
        probability = float(lookup.loc[iso, probability_column])
        count = max(1, int(round(len(country) * probability)))
        order = orders[iso][allocation]
        if exact_high_counts is None:
            chosen = order[:count]
        else:
            high_count = exact_high_counts[iso]
            chosen = np.concatenate([order[high[order]][:high_count],
                                     order[~high[order]][:count - high_count]])
        assert len(chosen) == count and len(np.unique(chosen)) == count
        n_high = int(high[country.index].sum())
        observed_high = int(high[chosen].sum())
        checks.append({'iso3': iso, 'n_candidates': len(country), 'p': probability, 'k_c': count,
                       'n_selected': len(chosen), 'high_selected': observed_high,
                       'actual_country_q': observed_high / count,
                       'minimum_feasible_high_count': max(0, count - (len(country) - n_high)),
                       'maximum_feasible_high_count': min(count, n_high),
                       'exact_high_target': exact_high_counts[iso] if exact_high_counts else np.nan})
        selected.append(chosen)
    return np.concatenate(selected), pd.DataFrame(checks)


def coverage(travel_path, codes, provenance):
    population = np.memmap(GRID / 'pop2025.f32', dtype='float32', mode='r', shape=(NR, NC))
    countries = np.memmap(GRID / 'country.i16', dtype='int16', mode='r', shape=(NR, NC))
    urban = np.memmap(GRID / 'urban.u8', dtype='uint8', mode='r', shape=(NR, NC))
    rural = np.memmap(GRID / 'rural.u8', dtype='uint8', mode='r', shape=(NR, NC))
    travel = np.memmap(travel_path, dtype='float32', mode='r', shape=(NR, NC))
    size = int(codes.idx.max()) + 1
    denominator = np.zeros(size)
    reached = {threshold: np.zeros(size) for threshold in THRESHOLDS}
    strata = {name: {'population': 0., '30': 0., '120': 0.} for name in ['all', 'urban', 'rural']}
    for start in range(0, NR, 150):
        country = countries[start:start + 150].ravel()
        pop = population[start:start + 150].ravel()
        tt = travel[start:start + 150].ravel()
        valid = (country >= 0) & (pop > 0)
        weights = pop[valid].astype('float64')
        ids = country[valid]
        denominator += np.bincount(ids, weights=weights, minlength=size)
        for threshold in THRESHOLDS:
            reached[threshold] += np.bincount(ids, weights=weights * (tt[valid] <= threshold), minlength=size)
        for name, is_member in [('all', np.ones(len(pop), dtype=bool)),
                                ('urban', urban[start:start + 150].ravel().astype(bool)),
                                ('rural', rural[start:start + 150].ravel().astype(bool))]:
            mask = valid & is_member
            strata[name]['population'] += float(pop[mask].sum(dtype='float64'))
            for threshold in THRESHOLDS:
                strata[name][str(threshold)] += float(pop[mask & (tt <= threshold)].sum(dtype='float64'))
    country_rows = codes.copy()
    indices = country_rows.idx.astype(int).to_numpy()
    country_rows['population'] = denominator[indices]
    for threshold in THRESHOLDS:
        country_rows[f'coverage_{threshold}'] = np.divide(reached[threshold][indices], denominator[indices],
                                                        out=np.full(len(indices), np.nan), where=denominator[indices] > 0)
    country_rows = country_rows.merge(provenance[['iso3', 'p_c_source', 'capability_lookup_matched']], on='iso3', how='left')
    summary = []
    for name, values in strata.items():
        for threshold in THRESHOLDS:
            summary.append({'population_group': name, 'threshold_minutes': threshold,
                            'population': values['population'],
                            'coverage_percent': values[str(threshold)] / values['population'] * 100})
    # This subgroup uses the common global facility assignments, including neighboring-country sources.
    measured = country_rows.p_c_source.eq('measured')
    for label, mask in [('measured_ratio_countries', measured), ('prior_or_unmatched_countries', ~measured)]:
        weights = country_rows.loc[mask, 'population']
        for threshold in THRESHOLDS:
            values = country_rows.loc[mask, f'coverage_{threshold}']
            summary.append({'population_group': label, 'threshold_minutes': threshold,
                            'population': float(weights.sum()),
                            'coverage_percent': float((weights * values).sum() / weights.sum() * 100)})
    return country_rows, pd.DataFrame(summary)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    hospitals, provenance = prepare_inputs()
    provenance.to_csv(OUT / 'country_budget_provenance.csv', index=False)
    print('INPUTS', len(hospitals), 'countries', len(provenance), 'budget_sources', provenance.p_c_source.value_counts().to_dict(), flush=True)
    orders = paired_orders(hospitals)
    central_ids, central_checks = select(hospitals, provenance, orders, 'central', 'random')
    exact_high = dict(zip(central_checks.iso3, central_checks.high_selected.astype(int)))
    cases = [(budget, allocation, False) for budget in ['lower', 'central', 'upper'] for allocation in ['random', 'cluster']]
    cases.append(('central', 'cluster', True))
    codes = pd.read_csv(GRID / 'countries.csv')
    all_summary, all_checks, all_countries = [], [], []
    for budget, allocation, country_exact in cases:
        case = f'{budget}_{allocation}' + ('_country_exact_q' if country_exact else '')
        folder = OUT / case
        folder.mkdir(exist_ok=True)
        ids, checks = select(hospitals, provenance, orders, budget, allocation, exact_high if country_exact else None)
        selected = hospitals.loc[ids, ['facility_idx', 'iso3', 'lat', 'lon', 'local_pop', 'high_local_pop']]
        selected.to_csv(folder / 'selection.csv', index=False)
        np.save(folder / 'selected_ids.npy', ids)
        checks.insert(0, 'scenario', case)
        checks.to_csv(folder / 'country_budget_and_composition_checks.csv', index=False)
        source_path = folder / 'source.u8'
        travel_path = folder / 'travel.f32'
        travel_log = folder / 'travel.log'
        complete = (travel_path.exists() and travel_path.stat().st_size == NR * NC * 4
                    and travel_log.exists() and 'done.' in travel_log.read_text())
        if not complete:
            source = np.memmap(source_path, dtype='uint8', mode='w+', shape=(NR, NC))
            source[:] = 0
            source[hospitals.loc[ids, 'row'].to_numpy(), hospitals.loc[ids, 'col'].to_numpy()] = 1
            source.flush()
            del source
            print('RUN_START', case, 'selected', len(ids), 'q', float(selected.high_local_pop.mean()), flush=True)
            with travel_log.open('w') as log:
                subprocess.run([str(BASE / 'src/ttdijkstra'), str(GRID / 'friction.f32'), str(source_path),
                                str(travel_path), str(NR), str(NC), '120', '0'], stdout=log, stderr=log, check=True)
        country_rows, summary = coverage(travel_path, codes, provenance)
        country_rows.insert(0, 'scenario', case)
        country_rows.to_csv(folder / 'country_coverage.csv', index=False)
        summary.insert(0, 'scenario', case)
        summary['budget'] = budget
        summary['allocation'] = allocation
        summary['selected_facilities'] = len(ids)
        summary['global_q'] = float(selected.high_local_pop.mean())
        summary['global_q_in_illustrative_interval'] = .45 <= selected.high_local_pop.mean() <= .68
        summary['minimum_feasible_global_q'] = checks.minimum_feasible_high_count.sum() / len(ids)
        summary['maximum_feasible_global_q'] = checks.maximum_feasible_high_count.sum() / len(ids)
        summary['country_exact_composition'] = country_exact
        summary.to_csv(folder / 'summary.csv', index=False)
        all_summary.append(summary)
        all_checks.append(checks)
        all_countries.append(country_rows)
        pd.concat(all_summary, ignore_index=True).to_csv(OUT / 'global_sensitivity_summary.csv', index=False)
        pd.concat(all_checks, ignore_index=True).to_csv(OUT / 'all_country_quota_checks.csv', index=False)
        pd.concat(all_countries, ignore_index=True).to_csv(OUT / 'all_country_coverages.csv', index=False)
        print('RUN_DONE', case, summary[summary.population_group.eq('all')][['threshold_minutes', 'coverage_percent']].to_dict('records'), flush=True)
    pd.DataFrame([{'file': 'hospital_capability.csv', 'sha256': hashlib.sha256((BASE / 'outputs/hospital_capability.csv').read_bytes()).hexdigest()},
                  {'file': 'global_budget_sensitivity.py', 'sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}]).to_csv(OUT / 'input_script_hashes.csv', index=False)
    (OUT / 'COMPLETE.json').write_text(json.dumps({'complete': True, 'scenarios': len(cases), 'seed': SEED,
                                                   'budget_interpretation': 'source-specified sensitivity bounds, not probability intervals',
                                                   'global_scope': 'country-group coverage uses globally assigned facilities, allowing cross-border sources'}, indent=2))
    print('GLOBAL_SENSITIVITY_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
