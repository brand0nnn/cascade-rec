"""Prerequisite inventories must disclose missing inputs without running or rewriting a study."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from cascade.reproduction import MANIFESTS, ROOT, digest, input_paths, inspect_study


def manifest(record):
    return {'schema':1, 'study':'synthetic', 'inputs':{'files':[record]},
            'phases':{'inspect':{'groups':['files'], 'entrypoint':'do not execute'}}}


def test_explicit_seed_inventory_does_not_hide_missing_matches(tmp_path):
    record = {'path':'runs/seed{seed}', 'files':['model.pt','config.json'], 'expand':{'seed':[0,1,2]}}
    (tmp_path/'runs/seed0').mkdir(parents=True)
    (tmp_path/'runs/seed0/model.pt').write_bytes(b'synthetic state')
    result = inspect_study(manifest(record),'inspect',root=tmp_path)
    assert len(result['inputs']) == 6
    assert result['status'] == 'not_ready' and len(result['missing']) == 5
    assert 'runs/seed2/model.pt' in result['missing']
    assert result['outputs_written'] is False and result['models_executed'] is False


def test_recorded_identity_changes_and_optional_hashes_are_explicit(tmp_path):
    source=tmp_path/'input.json';source.write_bytes(b'original')
    spec=manifest({'path':'input.json','sha256':digest(source)})
    assert inspect_study(spec,'inspect',root=tmp_path)['status'] == 'ready'
    source.write_bytes(b'changed')
    result=inspect_study(spec,'inspect',root=tmp_path)
    assert result['status'] == 'not_ready' and result['changed'] == ['input.json']
    result=inspect_study(manifest({'path':'input.json'}),'inspect',root=tmp_path)
    assert result['inputs'][0]['sha256'] is None and result['inputs'][0]['identity_matches'] is None
    result=inspect_study(manifest({'path':'input.json'}),'inspect',root=tmp_path,hash_files=True)
    assert result['inputs'][0]['sha256'] == digest(source)
    assert source.read_bytes() == b'changed'


def test_fresh_output_check_never_creates_or_overwrites(tmp_path):
    source=tmp_path/'input.json';source.write_bytes(b'input')
    spec=manifest({'path':'input.json'})
    output=Path('new-parent/new-run')
    result=inspect_study(spec,'inspect',root=tmp_path,fresh_output=output)
    assert result['status'] == 'ready' and result['fresh_output']['available']
    assert sorted(p.name for p in tmp_path.iterdir()) == ['input.json']
    occupied=tmp_path/output;occupied.mkdir(parents=True)
    sentinel=occupied/'model.pt';sentinel.write_bytes(b'existing evidence')
    result=inspect_study(spec,'inspect',root=tmp_path,fresh_output=output)
    assert result['status'] == 'not_ready' and not result['fresh_output']['available']
    assert sentinel.read_bytes() == b'existing evidence'


@pytest.mark.parametrize('path',['../outside/file','/absolute/file','.'])
def test_manifest_paths_stay_repository_relative(path):
    with pytest.raises(ValueError,match='repository-relative'):
        input_paths({'path':path})


def test_one_checksum_cannot_stand_for_multiple_checkpoints(tmp_path):
    record={'path':'seed{seed}.pt','expand':{'seed':[0,1]},'sha256':'wrong'}
    with pytest.raises(ValueError,match='exactly one file'):
        inspect_study(manifest(record),'inspect',root=tmp_path)


@pytest.mark.parametrize('study', sorted(p.stem for p in MANIFESTS.glob('*.json')))
def test_poster_aggregate_inventories_match_recorded_files(study):
    spec=json.loads((MANIFESTS/(study+'.json')).read_text())
    report=inspect_study(spec,'inspect')
    assert report['status'] == 'ready', report
    assert all(row['identity_matches'] is True for row in report['inputs'])
    for phase in spec['phases'].values():
        assert phase['groups']
        for group in phase['groups']:
            assert spec['inputs'][group]
            for record in spec['inputs'][group]:assert input_paths(record)


def test_r1_keeps_october_and_november_checkpoint_families_separate():
    r1=json.loads((MANIFESTS/'r1.json').read_text())
    rq3=json.loads((MANIFESTS/'rq3.json').read_text())
    paths=[p for record in r1['inputs']['october_rankers'] for p in input_paths(record)]
    assert len([p for p in paths if p.endswith('/model.pt')]) == 6
    assert not any('hybrid_study' in p for p in paths)
    assert any('20261028-din-natural' in p for p in paths)
    assert all('hybrid_study' in p for record in rq3['inputs']['runs'] for p in input_paths(record))


def test_cli_missing_bulk_inputs_exit_two_and_do_not_create_outputs(tmp_path):
    result=subprocess.run([sys.executable,'-m','cascade.reproduction','--study','r1','--phase','din-val',
                           '--root',str(tmp_path),'--fresh-output','new/run','--require-ready'],
                          cwd=ROOT,capture_output=True,text=True)
    assert result.returncode == 2, result.stderr
    report=json.loads(result.stdout)
    assert report['missing'] and report['outputs_written'] is False
    assert report['prediction_arrays_loaded'] is False and not list(tmp_path.iterdir())
