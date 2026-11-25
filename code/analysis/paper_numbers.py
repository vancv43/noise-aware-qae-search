# Noise-aware architecture and hyperparameter search for quantum autoencoders.
# Anonymized code for peer review; author information will be added after the review.
"""Derive every number quoted in main.tex from saved result files (no new runs)."""
import json, glob, collections
import numpy as np
from scipy.stats import spearmanr
R = 'results'
out = {}
ana = json.load(open(f'{R}/search_v2/analysis.json'))
fin = json.load(open(f'{R}/final_v2/final_report.json'))
for n in ['4', '6']:
    recs = []
    for f in sorted(glob.glob(f'{R}/search_v2/*/out/*/evals_n{n}_*.jsonl')):  # fixed order: first record of a phenotype is kept
        arm = f.split('evals_n')[1].split('_s')[0].split('_', 1)[1]
        for line in open(f):
            r = json.loads(line)
            if r.get('status') != 'ok': continue
            r['arm'] = arm; recs.append(r)
    # unique phenotypes (first record of each phenotype key) for the mechanism analysis
    uniq = {}
    for r in recs:
        p = r['phenotype']; key = (p['topology'], p['depth'], p['rotset'], p['k'], round(p['lr'], 4))
        uniq.setdefault(key, r)
    U = list(uniq.values())
    f2 = np.array([r['f2'] for r in U]); fi = np.array([r['val_fid_ideal'] for r in U]); fn = np.array([r['val_fid_noisy'] for r in U])
    gap = fi - fn
    rho, p = spearmanr(f2, gap)
    floor = {'4': 0.487, '6': 0.345}[n]
    floor_exact = {'4': 0.48739587980277166, '6': 0.3454279253984468}[n]   # mean validation fidelity of the noisy input
    table_bands = collections.OrderedDict()   # CNOT bands of the band table in the Results
    for a, b in [(0, 0), (1, 12), (13, 24), (25, 40), (41, 72), (73, 180)]:
        m = (f2 >= a) & (f2 <= b)
        if m.sum(): table_bands[f'{a}-{b}'] = dict(n=int(m.sum()), ideal=round(float(fi[m].mean()), 3), noisy=round(float(fn[m].mean()), 3),
                                                  above_floor_pct=round(100 * float((fn[m] > floor_exact).mean()), 1),
                                                  best_noisy=round(float(fn[m].max()), 3))
    bins = collections.OrderedDict()
    edges = [0, 1, 13, 25, 49, 10**6]
    labels = ['0', '1-12', '13-24', '25-48', '>48']
    for a, b, lab in zip(edges[:-1], edges[1:], labels):
        m = (f2 >= a) & (f2 < b)
        if m.sum(): bins[lab] = dict(n=int(m.sum()), ideal=float(fi[m].mean()), noisy=float(fn[m].mean()), gap=float(gap[m].mean()),
                                    best_ideal=float(fi[m].max()), best_noisy=float(fn[m].max()))
    out[n] = dict(n_records=len(recs), n_unique=len(U), spearman_gap_f2=[float(rho), float(p)], bins=bins, table_bands=table_bands,
                  frac_noisy_below_floor=float((fn < floor).mean()), best_ideal=float(fi.max()), best_noisy=float(fn.max()),
                  f2_of_best_ideal=int(f2[fi.argmax()]), f2_of_best_noisy=int(f2[fn.argmax()]),
                  kendall=ana['per_n'][n]['kendall_ideal_vs_noisy'], friedman=ana['per_n'][n]['friedman_hv'],
                  hv={a: dict(median=v['median'], iqr=v['iqr']) for a, v in ana['per_n'][n]['hv'].items()},
                  igd={a: v['median'] for a, v in ana['per_n'][n]['igd_plus'].items()},
                  wil_hv=ana['per_n'][n]['wilcoxon_hv_namopso_vs'], wil_igd=ana['per_n'][n]['wilcoxon_igd_namopso_vs'],
                  test_mean=fin['per_n'][n]['mean'], test_sd=fin['per_n'][n]['sd'], floor_test=fin['per_n'][n]['floor'],
                  sel_f2=fin['per_n'][n]['f2'], above_floor=fin['per_n'][n]['above_floor_main'],
                  wil_test=fin['per_n'][n]['wilcoxon_main_namopso_vs'], gap_sel=fin['per_n'][n]['ideal_vs_main_gap'])
json.dump(out, open(f'{R}/paper_numbers.json', 'w'), indent=1)
for n in out:
    o = out[n]; print(n, o['n_records'], o['n_unique'], 'rho', o['spearman_gap_f2'], 'below floor', round(o['frac_noisy_below_floor'], 3),
                      'best ideal', round(o['best_ideal'], 3), o['f2_of_best_ideal'], 'best noisy', round(o['best_noisy'], 3), o['f2_of_best_noisy'])
    for k, v in o['bins'].items(): print('  ', k, {a: round(b, 3) if isinstance(b, float) else b for a, b in v.items()})
