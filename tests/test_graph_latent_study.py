from types import SimpleNamespace

import numpy as np
import pandas as pd

from cascade.pipelines.graph_latent_study import negative_plan,split_cutoff


def test_inner_split_is_defined_on_all_queries_and_keeps_timestamp_ties():
    rows=pd.DataFrame({'ts':[9,1,8,8,8,2,3,4,5,6]})
    cutoff,prefix=split_cutoff(rows)
    assert cutoff==8
    assert prefix.tolist()==[False,True,False,False,False,True,True,True,True,True]
    assert (rows.ts[prefix]<cutoff).all() and (rows.ts[~prefix]>=cutoff).all()


def test_negative_plans_pair_rows_columns_and_keep_positive_in_original_list():
    ids=np.tile(np.arange(1,201),(4,1));targets=np.array([7,13,21,29])
    archive={'cand':ids,'target':targets}
    rows,cols=negative_plan(archive,np.array([0,2,3]),seed=1,epochs=2)
    again=negative_plan(archive,np.array([0,2,3]),seed=1,epochs=2)
    assert np.array_equal(rows,again[0]) and np.array_equal(cols,again[1])
    for order,columns in zip(rows,cols):
        actual=np.take_along_axis(ids[order],columns,axis=1)
        assert np.array_equal(actual[:,0],targets[order])
        assert not (actual[:,1:]==targets[order,None]).any()
        assert all(len(set(c))==51 for c in columns)
        assert set(order)=={0,2,3}
