"""Checks protect study identity, exact values, provenance, and portable assets."""
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
D = json.loads((HERE/'data/evidence.json').read_text())


def rows(path):
    with (ROOT/path).open() as f:
        return list(csv.DictReader(f))


def value(study, label, group='all'):
    return next(r['value'] for r in D['studies'][study]['rows'] if r['label']==label and r['group']==group)


class EvidenceTests(unittest.TestCase):
    def test_source_hashes_match_owning_files(self):
        for source in D['sources'].values():
            self.assertEqual(hashlib.sha256((ROOT/source['path']).read_bytes()).hexdigest(),source['sha256'])

    def test_k_sweep_uses_natural_only_arm_c(self):
        source=rows('results/targeted_study/e3-20261107-all-seeds/k_curve.csv')
        for r in D['studies']['k']['rows']:
            expected=next(s for s in source if s['arm']=='C' and int(s['k1'])==r['k'])
            self.assertEqual(r['value'],float(expected['ndcg10']))
        self.assertEqual(value('k','200'),0.008571615205226335)
        self.assertEqual(value('k','500'),0.00792289049450521)

    def test_retraining_four_conditions_are_validation(self):
        self.assertTrue(D['studies']['retrain']['context'].startswith('Validation'))
        expected=[0.008140742517442344,0.008810533700347121,0.010386864192191135,0.011018932991806694]
        labels=['P1_ranker','P1ranker_on_P3','P3_sizematched_ranker','P3_ranker']
        for label, n in zip(labels,expected):self.assertEqual(value('retrain',label),n)
        self.assertAlmostEqual(value('retrain','P3_ranker')/value('retrain','P1_ranker')-1,.353553789,places=6)

    def test_timing_snapshots_match_test_source(self):
        for timing,system in [('T0','S1'),('T1','S2'),('T2','S3')]:
            for group in ['all','new','warm']:
                self.assertEqual(value('timing',timing,group),value('final',system,group))
        self.assertAlmostEqual(value('timing','T2')/value('timing','T0')-1,.213692,places=5)

    def test_frozen_test_study_keeps_october_controls_and_macro_recall(self):
        study = D['studies']['frozen']
        self.assertTrue(study['context'].startswith('Test'))
        self.assertAlmostEqual(value('frozen','armC_mlp/tt200'), .005177249166269799, places=14)
        self.assertAlmostEqual(value('frozen','din/tt200'), .0062381689201561, places=14)
        self.assertNotEqual(value('frozen','armC_mlp/tt200'), value('ranker','mlp_tt'))
        fresh = next(r for r in study['rows'] if r['label']=='armC_mlp/fresh' and r['group']=='all')
        self.assertAlmostEqual(fresh['recall'], .06989198097803684, places=14)
        self.assertFalse(any('recall' in r for r in study['rows'] if r['group']!='all'))
        contrasts = {(c['reference'],c['candidate'],c['group']):c for c in study['contrasts']}
        mlp = contrasts['armC_mlp/tt200','armC_mlp/fresh','all']
        din = contrasts['din/tt200','din/fresh','all']
        self.assertLess(mlp['ci_low'],0); self.assertGreater(mlp['ci_high'],0)
        self.assertLess(din['ci_high'],0)

    def test_corrected_matching_report_has_provenance_and_recomputed_intervals(self):
        report = ROOT/'results/test_final_ranker/20261108-normalization-corrected/matching_report'
        manifest = json.loads((report/'provenance.json').read_text())
        self.assertEqual(manifest['events'],297390)
        self.assertEqual(manifest['users'],112081)
        self.assertFalse(manifest['model_execution'])
        for path, digest in manifest['outputs'].items():
            self.assertEqual(hashlib.sha256((report/path).read_bytes()).hexdigest(),digest)
        self.assertEqual(hashlib.sha256((ROOT/'scripts/report_corrected_ranker.py').read_bytes()).hexdigest(),manifest['source_sha256'])
        self.assertAlmostEqual(value('ranker','din_rrf'),.008268564730483458,places=14)
        self.assertAlmostEqual(value('ranker','mlp_rrf'),.007312312571406011,places=14)
        self.assertTrue(any(c['reference']=='mlp_rrf' and c['candidate']=='din_rrf' for c in D['studies']['ranker']['contrasts']))

    def test_final_and_original_baselines_are_separate(self):
        self.assertAlmostEqual(value('final','S8')/value('final','S1')-1,.6119897,places=5)
        self.assertAlmostEqual(value('final','S8')/value('ranker','mlp_tt'),1.9984,places=3)
        self.assertFalse(any(r['group']=='warm' for r in D['studies']['ranker']['rows']))

    def test_mixed_composition_is_fixed_200_item_budget(self):
        source=json.loads((ROOT/'results/analysis/20261106-pools/pool_report.json').read_text())
        mixing=D['studies']['retrieval']['mixing']
        self.assertEqual(mixing['composition'],source['P3_val_composition_mean_per_list'])
        self.assertAlmostEqual(sum(mixing['composition'].values()),200,places=10)
        self.assertLess(mixing['composition']['both_sources'],mixing['overlap'])

    def test_expansion_accounting_matches_measured_k_differences(self):
        for r in D['studies']['k']['expansion']:
            self.assertAlmostEqual(r['gained']+r['competition'],value('k',str(r['high']))-value('k',str(r['low'])),places=12)

    def test_confidence_intervals_are_valid_paired_differences(self):
        for key,s in D['studies'].items():
            for c in s.get('contrasts',[]):
                self.assertLessEqual(c['ci_low'],c['diff']);self.assertLessEqual(c['diff'],c['ci_high'])
                self.assertAlmostEqual(value(key,c['candidate'],c['group'])-value(key,c['reference'],c['group']),c['diff'],places=12)

    def test_aggregate_only_and_finite_values(self):
        prohibited={'user','user_id','inst','asin','token','checkpoint','password','private_key'}
        for s in D['studies'].values():
            for r in s['rows']:
                self.assertFalse(prohibited.intersection(r))
                self.assertTrue(math.isfinite(r['value']));self.assertGreaterEqual(r['value'],0)
        self.assertNotIn('/Users/',(HERE/'data/evidence.json').read_text())

    def test_csv_and_inline_bundle_match_json(self):
        js=(HERE/'data/evidence.js').read_text()
        self.assertEqual(json.loads(js.removeprefix('window.POSTER_DATA = ').rstrip(';\n')),D)
        for key,s in D['studies'].items():
            with (HERE/'data'/f'{key}.csv').open() as f:
                exported=list(csv.DictReader(f))
            self.assertEqual(len(exported),len(s['rows']))
            for a,b in zip(exported,s['rows']):self.assertEqual(float(a['value']),b['value'])

    def test_poster_bytes_match_manifest(self):
        self.assertEqual(hashlib.sha256((HERE/D['poster']['path']).read_bytes()).hexdigest(),D['poster']['sha256'])

    def test_assets_and_anchor_targets_exist(self):
        html=(HERE/'index.html').read_text()
        ids=set(re.findall(r'\bid="([^"]+)"',html))
        for target in re.findall(r'href="#([^"]+)"',html):
            if not target.startswith('explorer'):self.assertIn(target,ids)
        for path in re.findall(r'(?:src|href)="([^"#]+)"',html):
            if not re.match(r'\w+:',path):self.assertTrue((HERE/path).exists(),path)
        self.assertNotRegex(html,r'(?:src|href)="https?://')


if __name__=='__main__':unittest.main()
