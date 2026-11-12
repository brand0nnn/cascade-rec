"""Compatibility with source frozen before the 12 November structural cleanup.

Only synthetic arrays/checkpoints execute here. No historical Test file is read,
and no experiment or saved report is run in its original output directory.
"""
import ast
from dataclasses import asdict
import hashlib
import importlib
import importlib.util
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from cascade.models.ranker_mlp import MLPRanker, RankerFeatureSpec
from cascade.pipelines import frozen_ranker_followup as followup
from cascade.pipelines import graph_report_support as support
from cascade.pipelines.frozen_followup import reporting, runtime
from cascade.pipelines.matching_recipe import matching_recipe
from cascade.pipelines.ranker_training import create_ranker

FIXTURE = Path(__file__).parent / 'fixtures/refactor_20261112'
MANIFEST = json.loads((FIXTURE/'manifest.json').read_text())
LEGACY_SOURCE = (FIXTURE/'frozen_ranker_followup.py').read_text()
LEGACY_FUNCTIONS = {n.name: n for n in ast.parse(LEGACY_SOURCE).body if isinstance(n, ast.FunctionDef)}


def legacy_module(name):
    spec = importlib.util.spec_from_file_location('legacy_'+name, FIXTURE/(name+'.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reference_sources_are_frozen():
    for row in MANIFEST['sources'].values():
        assert hashlib.sha256((FIXTURE/row['snapshot']).read_bytes()).hexdigest() == row['sha256']


class RemoveContext(ast.NodeTransformer):
    """Undo only the explicit context plumbing to compare the original calculations/control flow."""
    def visit_Attribute(self, node):
        if isinstance(node.value, ast.Name) and node.value.id == 'ctx':
            name = node.attr.upper()
            assert name in MANIFEST['r1_context_constants']
            return ast.Name(id=name, ctx=node.ctx)
        return self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id in MANIFEST['r1_context_functions']:
            assert isinstance(node.args[0], ast.Name) and node.args[0].id == 'ctx'
            node.args.pop(0)
        return self.generic_visit(node)


@pytest.mark.parametrize('name', MANIFEST['r1_modules'])
def test_r1_operations_preserve_original_syntax_and_legacy_signature(name):
    module = importlib.import_module('cascade.pipelines.frozen_followup.'+MANIFEST['r1_modules'][name])
    node = ast.parse(inspect.getsource(getattr(module, name))).body[0]
    if name in MANIFEST['r1_context_functions']:
        assert node.args.args.pop(0).arg == 'ctx'
    actual = RemoveContext().visit(node)
    assert ast.dump(actual, include_attributes=False) == ast.dump(LEGACY_FUNCTIONS[name], include_attributes=False)
    legacy = legacy_module('frozen_ranker_followup')
    assert inspect.signature(getattr(followup, name)) == inspect.signature(getattr(legacy, name))


def test_r1_default_paths_devices_and_cli_dispatch_are_unchanged():
    legacy = legacy_module('frozen_ranker_followup')
    context = followup.study_context()
    for name in MANIFEST['r1_context_constants']:
        assert getattr(followup, name) == getattr(legacy, name) == getattr(context, name.lower())
    current = ast.parse(inspect.getsource(followup.main)).body[0]
    assert ast.dump(current, include_attributes=False) == ast.dump(LEGACY_FUNCTIONS['main'], include_attributes=False)


GRAPH_NAMES = {'_load_archive':'load_archive', '_relative':'relative_path', '_composition':'composition',
               '_cases':'write_cases', '_canonical':'canonical_value', 'feature_groups':'feature_groups',
               'validate_prediction':'validate_prediction', 'validate_selection':'validate_selection'}


@pytest.mark.parametrize('old,new', GRAPH_NAMES.items())
def test_promoted_report_helpers_preserve_original_syntax(old, new):
    original = {n.name:n for n in ast.parse((FIXTURE/'graph_latent_report.py').read_text()).body
                if isinstance(n, ast.FunctionDef)}[old]
    current = ast.parse(inspect.getsource(getattr(support, new))).body[0]
    current.name = old
    for node in ast.walk(current):
        if isinstance(node, ast.Name) and node.id in GRAPH_NAMES.values():
            node.id = next(k for k,v in GRAPH_NAMES.items() if v == node.id)
    assert ast.dump(current, include_attributes=False) == ast.dump(original, include_attributes=False)


@pytest.mark.parametrize('name', ['graph_family_report', 'ranking_objective_report'])
def test_report_clients_change_helper_imports_without_changing_calculations(name):
    original = {n.name:n for n in ast.parse((FIXTURE/(name+'.py')).read_text()).body
                if isinstance(n, ast.FunctionDef)}
    current_module = importlib.import_module('cascade.pipelines.'+name)
    class RestoreNames(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id in GRAPH_NAMES.values():
                node.id = next(k for k,v in GRAPH_NAMES.items() if v == node.id)
            return node
    for function, before in original.items():
        after = ast.parse(inspect.getsource(getattr(current_module, function))).body[0]
        after = RestoreNames().visit(after)
        assert ast.dump(after, include_attributes=False) == ast.dump(before, include_attributes=False), function


RECIPE_CASES = [(script, kind, extra, residual)
                for script in ('run_hybrid_study', 'score_test_final', 'audit_hybrid_normalization', 'score_test_rrfl')
                for kind in ('mlp', 'din') for extra in (0, 1) for residual in (False, True)
                if (script == 'run_hybrid_study' or extra == 0)
                and (script != 'score_test_rrfl' or (kind == 'din' and not residual))]


@pytest.mark.parametrize('script,kind,extra,residual', RECIPE_CASES)
def test_matching_recipe_preserves_rng_weights_checkpoint_and_logits(script, kind, extra, residual):
    tree = ast.parse((FIXTURE/(script+'.py')).read_text())
    assignments = {t.id:n.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
                   for t in n.targets if isinstance(t, ast.Name)}
    construction = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
                        and isinstance(n.func, ast.Name) and n.func.id == 'create_ranker')
    cardinalities = {name:8 for name in MLPRanker.categorical_names}
    namespace = {'RankerFeatureSpec':RankerFeatureSpec, 'builder':SimpleNamespace(item_cardinalities=cardinalities),
                 'kind':kind, 'extra':extra, 'arm':'din_rrf_res' if residual else 'din_rrf',
                 'name':'din_rrf_res' if residual else 'din_rrf'}
    evaluate = lambda node: eval(compile(ast.Expression(node), '<frozen-recipe>', 'eval'), namespace)
    spec_before = evaluate(assignments['spec'])
    cfg_node = construction.args[1]
    cfg_before = evaluate(assignments[cfg_node.id] if isinstance(cfg_node, ast.Name) else cfg_node)
    spec_after, cfg_after = matching_recipe(cardinalities, kind, extra_numeric=extra, residual_score=residual)
    assert asdict(spec_before) == asdict(spec_after)
    torch.manual_seed(197)
    before = create_ranker(spec_before, cfg_before).eval()
    rng_before = torch.random.get_rng_state()
    torch.manual_seed(197)
    after = create_ranker(spec_after, cfg_after).eval()
    assert torch.equal(rng_before, torch.random.get_rng_state())
    assert before.state_dict().keys() == after.state_dict().keys()
    for key, value in before.state_dict().items():
        assert torch.equal(value, after.state_dict()[key]), key
    after.load_state_dict(before.state_dict(), strict=True)
    batch = {name:torch.tensor([1,2,3]) for name in MLPRanker.categorical_names}
    batch.update(history_item_ids=torch.tensor([[0,1,2],[0,0,3],[0,0,0]]),
                 history_leaf_ids=torch.tensor([[0,1,2],[0,0,3],[0,0,0]]),
                 numeric=torch.arange(3*(4+extra), dtype=torch.float32).reshape(3,4+extra)/10,
                 item_text=torch.arange(3*384, dtype=torch.float32).reshape(3,384)/1000)
    with torch.no_grad():
        torch.testing.assert_close(before(batch), after(batch), rtol=0, atol=0)


def test_candidate_roundtrip_prefixes_and_overwrite_guards_match(tmp_path, monkeypatch):
    legacy = legacy_module('frozen_ranker_followup')
    monkeypatch.setattr(legacy, 'BULK', tmp_path/'before')
    monkeypatch.setattr(followup, 'BULK', tmp_path/'after')
    args = (np.array([4,5,6]), np.array([1,1,2]), np.array([[1,2,3],[2,4,5],[4,6,7]]),
            np.array([[.3,.2,.1],[.6,.4,.2],[.9,.6,.3]]), np.array([2,5,99]))
    before, after = legacy.as_lists(*args), followup.as_lists(*args)
    for key in before:
        np.testing.assert_array_equal(before[key], after[key]); assert before[key].dtype == after[key].dtype
    for runner, lists in ((legacy,before), (followup,after)):
        path = runner.save_lists('fresh','val',lists)
        original_bytes = path.read_bytes()
        with pytest.raises(ValueError, match='refusing to overwrite'):
            runner.save_lists('fresh','val',lists)
        assert path.read_bytes() == original_bytes
    for k in (None,1,2,3):
        a, b = legacy.load_lists('fresh','val',k), followup.load_lists('fresh','val',k)
        for key in a:np.testing.assert_array_equal(a[key],b[key])


def test_protocol_guard_matches_and_rejects_reopening(tmp_path, monkeypatch):
    legacy = legacy_module('frozen_ranker_followup')
    for runner in (legacy,followup):
        monkeypatch.setattr(runner,'BULK',tmp_path)
        monkeypatch.setattr(runner,'PROTOCOL',tmp_path/'protocol.md')
        with pytest.raises(ValueError,match='lock the written protocol'):
            runner.protocol_guard()
    protocol = tmp_path/'protocol.md';protocol.write_text('synthetic fixed protocol')
    (tmp_path/'protocol_lock.json').write_text(json.dumps({'protocol_sha256':followup.digest(protocol)}))
    legacy.protocol_guard();followup.protocol_guard()
    marker = tmp_path/'test_opened_din.json';marker.write_text('existing evidence')
    for runner in (legacy,followup):
        with pytest.raises(ValueError,match='Test already opened'):
            runner.record_test_open('din')
    assert marker.read_text() == 'existing evidence'
    protocol.write_text('changed')
    for runner in (legacy,followup):
        with pytest.raises(ValueError,match='locked Test protocol changed'):
            runner.protocol_guard()


def test_saved_rank_report_matches_original_bytes(tmp_path, monkeypatch, capsys):
    """Exercise the old/new complete Validation report on synthetic saved ranks, never model scoring."""
    from cascade.data.instances import ProcessedData
    legacy = legacy_module('frozen_ranker_followup')
    ids = np.arange(6);users = np.array([1,1,2,2,3,3]);targets = np.arange(1,7)
    frame = pd.DataFrame({'inst':ids,'user':users,'item':targets})
    data = SimpleNamespace(n_items=6,items=pd.DataFrame({'first_ts':[0,legacy.T1,0,legacy.T1,0,legacy.T1]}),
                           split_instances=lambda _:frame)
    monkeypatch.setattr(ProcessedData,'load',lambda *a,**k:data)
    monkeypatch.setattr(legacy,'BULK',tmp_path/'before-bulk');monkeypatch.setattr(followup,'BULK',tmp_path/'after-bulk')
    monkeypatch.setattr(legacy,'OUT',tmp_path/'before-report');monkeypatch.setattr(followup,'OUT',tmp_path/'after-report')
    monkeypatch.setattr(legacy,'git_hash',lambda:'synthetic-compatibility')
    monkeypatch.setattr(reporting,'git_hash',lambda:'synthetic-compatibility')
    for runner in (legacy,followup):
        directory=runner.BULK/'ranks/val';directory.mkdir(parents=True)
        # Use the same files in each directory.
        for ranker in ('armC_mlp','din'):
            for pool in ('tt','fresh','mixed'):
                seed = len(ranker)+len(pool)
                ranks=np.random.default_rng(seed).integers(1,30,size=(3,2 if pool=='tt' else 1,6)).astype(float)
                ranks[:,:,0]=np.inf
                np.savez(directory/(pool+'__'+ranker+'.npz'),inst=ids,user=users,
                         counts=np.array([200,500] if pool=='tt' else [200]),ranks=ranks)
        runner.phase_report()
    for filename in ('levels.csv','contrasts.csv','git_hash.txt'):
        assert (legacy.OUT/filename).read_bytes() == (followup.OUT/filename).read_bytes()
