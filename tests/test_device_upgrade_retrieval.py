import numpy as np

from cascade.analysis.device_upgrade_retrieval import KEEP, assemble, current_lines, is_newer, item_line_gen, ranks_in_d


def test_current_lines_and_newer():
    cur = current_lines(frozenset({'iphone|xr', 'iphone|11'}))
    assert cur == {'iphone': 11.0}
    assert is_newer(frozenset({'iphone|13 pro'}), cur)
    assert not is_newer(frozenset({'iphone|11 pro'}), cur)          # same generation is not newer
    assert not is_newer(frozenset({'pixel|6'}), cur)                 # other family
    many = frozenset({'iphone|13', 'iphone|12', 'iphone|8', 'iphone|7'})
    assert not is_newer(many, cur) and is_newer(many, cur, None)      # > 3 keys excluded only with the cap
    assert current_lines(frozenset({'galaxy_a|a52'})) == {}
    assert item_line_gen(many) == {}


def test_assemble_and_ranks():
    p1 = np.arange(1, 201)
    app, inter = assemble(p1, [500, 501], list(p1[KEEP:]) + [700, 701])
    assert len(app) == len(set(app)) == 200 and app[:KEEP] == list(range(1, 181))
    assert app[KEEP:KEEP + 3] == [500, 501, 181]
    assert inter[9] == 500 and inter[19] == 501 and inter[29] == 181 and inter[:9] == list(range(1, 10))
    assert sorted(inter) == sorted(app)
    assert ranks_in_d(5.0, 4, app, 5) == (5.0, 5.0)
    assert ranks_in_d(10.0, 9, app, 10) == (10.0, 11.0)
    assert ranks_in_d(180.0, 179, app, 180) == (180.0, 199.0)
    assert ranks_in_d(np.inf, -1, app, 501) == (182.0, 20.0)
    r = ranks_in_d(200.0, 199, app, 200)
    assert r == (np.inf, np.inf)                                      # P1 rank 200 pushed out
    assert inter.index(181) + 1 == ranks_in_d(181.0, 180, app, 181)[1]
