from types import SimpleNamespace

import numpy as np
import pytest

from cascade.pipelines.local_evidence_export import check_candidates,ranks_and_top10


def test_logit_export_retains_pessimistic_ties_and_never_credits_absent_target():
    ids=np.array([[1,2,3,0],[1,2,3,0]])
    scores=np.array([[2.,2.,1.,-np.inf],[2.,2.,1.,-np.inf]])
    ranks,top=ranks_and_top10(scores,ids,np.array([1,4]))
    assert ranks[0]==2 and np.isinf(ranks[1])
    assert top[0].tolist()==[1,2,3,0]


def test_candidate_audit_rejects_future_history_duplicates_and_allows_padding():
    batch=SimpleNamespace(ts=np.array([10]),mask_rows=np.array([0]),mask_items=np.array([1]))
    first=np.array([999,1,1,11])
    check_candidates(np.array([[2,0,0]]),batch,first)
    with pytest.raises(ValueError,match='unavailable'):check_candidates(np.array([[2,3]]),batch,first)
    with pytest.raises(ValueError,match='history'):check_candidates(np.array([[1,2]]),batch,first)
    with pytest.raises(ValueError,match='duplicate'):check_candidates(np.array([[2,2]]),batch,first)


def test_failure_funnel_counts_rescues_demotions_and_retained_hits():
    from cascade.pipelines.local_plan_completion import failure_flags
    base=np.array([np.inf,1,20,1,20]);rank=np.array([np.inf,20,1,1,20])
    flags=failure_flags(base,rank)
    assert [flags[k].sum() for k in ('retrieval_miss','ranking_miss','hit','rescue','demotion','retained_hit')]==[1,2,2,1,1,1]
    with pytest.raises(ValueError,match='retrieval miss'):failure_flags(np.array([np.inf]),np.array([1]))


def test_month_history_novelty_partitions_cover_every_event():
    import pandas as pd
    from cascade.pipelines.local_plan_completion import month_groups
    instances=pd.DataFrame({'ts':pd.to_datetime(['2021-09-01','2021-09-02','2021-10-01','2021-10-02'],utc=True).astype('int64')//1000000,
        'item':[1,2,3,4],'hist_len':[1,3,6,11]})
    data=SimpleNamespace(items=pd.DataFrame({'item':[1,2,3,4],'first_ts':[1,10,1,10]}),split={'t1':10})
    groups=list(month_groups(data,instances))
    total=sum(mask.sum() for m,h,n,mask in groups if h=='all' and n=='all')
    hist=sum(mask.sum() for m,h,n,mask in groups if h!='all' and n=='all')
    newwarm=sum(mask.sum() for m,h,n,mask in groups if h=='all' and n!='all')
    assert total==hist==newwarm==4


def test_concentration_excludes_unknown_metadata_and_preserves_missing_support():
    from cascade.pipelines.local_completion_report import concentration
    hhi,dominant,unique=concentration(np.array([0,1,1,2]))
    assert hhi==pytest.approx(5/9) and dominant==pytest.approx(2/3) and unique==2
    hhi,dominant,unique=concentration(np.array([0,0]))
    assert np.isnan(hhi) and np.isnan(dominant) and unique==0


def test_monthly_introductions_clip_partial_val_windows_and_exclude_test():
    import pandas as pd
    from cascade.pipelines.local_completion_report import temporal_population
    def ms(s):return pd.Timestamp(s,tz='UTC').value//1000000
    data=SimpleNamespace(items=pd.DataFrame({'item':[1,2,3,4],
        'first_ts':[ms('2021-09-01'),ms('2021-09-18'),ms('2021-10-12'),ms('2021-10-20')]}),
        split={'t1':ms('2021-09-15'),'t2':ms('2021-10-15')})
    instances=pd.DataFrame({'ts':[ms('2021-09-20'),ms('2021-10-14')],'item':[2,3],'user':[1,2]})
    result=temporal_population(data,instances,'sample')
    assert result.catalogue_introductions_during_observed_val_window.tolist()==[1,1]
    assert result.partial_month.all() and result.events.sum()==2
