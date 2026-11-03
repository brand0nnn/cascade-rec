"""Frozen pre-T0 PCA/text and BPR factors for the local ranking study.

These APIs fit representations without calling validation or Test. BPR exports
its final fixed-epoch state; PCA fits only the pre-T0 catalogue, then applies the
same inductive transform to later products. Existing outputs are never reused.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
from pathlib import Path
import shutil
import time

import numpy as np
from sklearn.decomposition import PCA
import torch
from torch import nn
from torch.nn import functional as F

from cascade.eval.saved_results import digest, write_json
from cascade.utils.config import REPO_ROOT

DIMENSION = 64


def _array_hash(array):
    value=np.ascontiguousarray(array)
    h=hashlib.sha256()
    h.update(str(value.dtype).encode());h.update(str(value.shape).encode());h.update(value.tobytes())
    return h.hexdigest()


def _data_hashes(data):
    result={}
    if not hasattr(data,'dir'):return result
    for name in ('items.parquet','interactions.parquet','sequences.npz','split.json'):
        path=Path(data.dir)/name
        if path.is_file():
            try:key=str(path.resolve().relative_to(REPO_ROOT.resolve()))
            except ValueError:key=str(path.resolve())
            result[key]=digest(path)
    return result


def _fresh_output(out):
    out=Path(out)
    out.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(Path(__file__),out/'source.py')
    return out


def transform_pca(text,mean,components,valid_text):
    """Apply a frozen PCA state; missing/padding vectors remain exactly zero."""
    text=np.asarray(text,dtype=np.float32)
    valid=np.asarray(valid_text,dtype=bool).copy()
    if text.ndim!=2 or valid.shape!=(len(text),):raise ValueError('PCA text/mask shapes differ')
    if not np.isfinite(text).all():raise ValueError('nonfinite text vectors')
    valid[0]=False
    vectors=np.zeros((len(text),len(components)),np.float32)
    vectors[valid]=(text[valid]-mean)@components.T
    norm=np.linalg.norm(vectors,axis=1,keepdims=True)
    known=valid&(norm[:,0]>1e-12)
    vectors=np.divide(vectors,np.maximum(norm,1e-12)).astype(np.float32)
    vectors[~known]=0
    return vectors,known


def prepare_pca(data,text,out):
    """Save PCA64 fitted on valid pre-T0 item text, without whitening or labels."""
    out=Path(out)
    if out.exists():raise FileExistsError(out)
    text=np.asarray(text,dtype=np.float32)
    if text.ndim!=2 or text.shape[0]!=data.n_items+1 or text.shape[1]<DIMENSION:
        raise ValueError('PCA needs an (N+1,D>=64) text table')
    if not np.isfinite(text).all():raise ValueError('nonfinite text vectors')
    items=data.items.sort_values('item')
    if not np.array_equal(items.item.to_numpy(),np.arange(1,data.n_items+1)):
        raise ValueError('metadata must cover every processed item ID exactly once')
    valid=np.linalg.norm(text,axis=1)>1e-12;valid[0]=False
    valid[items.item.to_numpy()] &= items.text.fillna('').astype(str).str.strip().ne('').to_numpy()
    cutoff=int(data.split['t_ranker_train'])
    fit_ids=items.item.to_numpy()[(items.first_ts.to_numpy()<cutoff)&valid[items.item.to_numpy()]]
    if len(fit_ids)<DIMENSION:raise ValueError('PCA64 needs at least 64 valid pre-T0 products')
    before=_data_hashes(data);text_hash=_array_hash(text);source=digest(Path(__file__))
    out=_fresh_output(out)
    write_json(out/'provenance.json',{'status':'running','validation_used':False,'test_predictions_loaded':False})
    started=time.perf_counter()
    pca=PCA(n_components=DIMENSION,svd_solver='full',whiten=False)
    pca.fit(text[fit_ids])
    mean=np.asarray(pca.mean_,np.float32);components=np.asarray(pca.components_,np.float32)
    vectors,known=transform_pca(text,mean,components,valid)
    np.save(out/'pca_vectors.npy',vectors);np.save(out/'pca_known.npy',known)
    np.savez_compressed(out/'pca_state.npz',mean=mean,components=components,
        explained_variance=pca.explained_variance_,explained_variance_ratio=pca.explained_variance_ratio_,
        fit_item_ids=fit_ids,valid_text=valid)
    with np.load(out/'pca_state.npz') as state:
        restored,restored_known=transform_pca(text,state['mean'],state['components'],state['valid_text'])
    if not np.array_equal(vectors,restored) or not np.array_equal(known,restored_known):raise ValueError('PCA state replay differs')
    if before!=_data_hashes(data) or text_hash!=_array_hash(text) or source!=digest(Path(__file__)):
        raise ValueError('PCA inputs/source changed during fitting')
    proof={'status':'complete','algorithm':'PCA','dimension':DIMENSION,'device':'cpu','svd_solver':'full','whiten':False,
        'fit_strict_before_ms':cutoff,'fit_items':len(fit_ids),'known_output_items':int(known.sum()),
        'fit_item_ids_sha256':_array_hash(fit_ids),'text_sha256':text_hash,'input_sha256':before,'source_sha256':source,
        'fit_and_export_seconds':time.perf_counter()-started,'reload_vectors_exact':True,
        'validation_calls':0,'validation_used':False,'checkpoint_selection':'none','test_predictions_loaded':False,
        'versions':{name:importlib.metadata.version(name) for name in ('numpy','scipy','scikit-learn')},
        'output_sha256':{name:digest(out/name) for name in ('pca_vectors.npy','pca_known.npy','pca_state.npz')}}
    write_json(out/'provenance.json',proof)
    print(f'PCA64 complete: {len(fit_ids)} pre-T0 products -> {out}',flush=True)
    return {'vectors_path':out/'pca_vectors.npy','known_path':out/'pca_known.npy','state_path':out/'pca_state.npz','provenance':proof}


class PreT0NegativeSampler:
    """Uniform allowed items with rejection of every training-user positive."""
    def __init__(self,users,items,allowed_items,seed):
        users=np.asarray(users,dtype=np.int64);items=np.asarray(items,dtype=np.int64)
        self.allowed_items=np.unique(np.asarray(allowed_items,dtype=np.int64))
        if users.shape!=items.shape or users.ndim!=1 or not len(users):raise ValueError('invalid training pairs')
        if not len(self.allowed_items) or (self.allowed_items<=0).any() or (users<=0).any():raise ValueError('invalid negative pool/users')
        if not np.isin(items,self.allowed_items).all():raise ValueError('positive escaped pre-T0 pool')
        self.used={}
        for user,item in zip(users,items):self.used.setdefault(int(user),set()).add(int(item))
        allowed=set(self.allowed_items.tolist())
        if any(allowed<=values for values in self.used.values()):raise ValueError('training user exhausts negative pool')
        self.rng=np.random.default_rng(seed);self.returned=0;self.first_negatives=[];self.stream=hashlib.sha256()

    def sample(self,users):
        users=np.asarray(users,dtype=np.int64)
        if users.ndim!=1 or any(int(u) not in self.used for u in users):raise ValueError('unknown training user')
        result=self.allowed_items[self.rng.integers(len(self.allowed_items),size=len(users))]
        bad=np.asarray([int(item) in self.used[int(user)] for user,item in zip(users,result)],bool)
        while bad.any():
            result[bad]=self.allowed_items[self.rng.integers(len(self.allowed_items),size=int(bad.sum()))]
            bad=np.asarray([int(item) in self.used[int(user)] for user,item in zip(users,result)],bool)
        if not np.isin(result,self.allowed_items).all():raise ValueError('negative escaped pre-T0 pool')
        self.returned+=len(result)
        self.first_negatives.extend(result[:max(0,1024-len(self.first_negatives))].tolist())
        self.stream.update(result.astype('<i8',copy=False).tobytes())
        return result


class BPRFactors(nn.Module):
    """Dense pairwise MF with zero unknown rows and direct processed-ID indexing."""
    def __init__(self,n_users,n_items,dim=DIMENSION,seed=0,known_users=None,known_items=None):
        super().__init__()
        user_mask=np.ones(n_users+1,bool) if known_users is None else np.asarray(known_users,bool).copy()
        item_mask=np.ones(n_items+1,bool) if known_items is None else np.asarray(known_items,bool).copy()
        if user_mask.shape!=(n_users+1,) or item_mask.shape!=(n_items+1,):raise ValueError('factor masks have wrong shapes')
        user_mask[0]=False;item_mask[0]=False
        # Preserve the caller's ordinary RNG and avoid future vocabulary size
        # changing the random values or scale of the fitted rows.
        with torch.random.fork_rng(devices=[]):
            self.user_embedding=nn.Embedding(n_users+1,dim,padding_idx=0,device='cpu',dtype=torch.float32)
            self.item_embedding=nn.Embedding(n_items+1,dim,padding_idx=0,device='cpu',dtype=torch.float32)
            generator=torch.Generator(device='cpu').manual_seed(seed)
            with torch.no_grad():
                for layer,mask in ((self.user_embedding,user_mask),(self.item_embedding,item_mask)):
                    layer.weight.zero_();ids=torch.as_tensor(np.flatnonzero(mask),dtype=torch.long)
                    values=torch.empty((len(ids),dim),dtype=torch.float32)
                    nn.init.xavier_normal_(values,generator=generator)
                    layer.weight.index_copy_(0,ids,values)

    def forward(self,users,items):
        return (self.user_embedding(users)*self.item_embedding(items)).sum(-1)

    def loss(self,users,positive,negative):
        return F.softplus(self(users,negative)-self(users,positive)).mean()


def _unit_item_factors(model,known):
    values=F.normalize(model.item_embedding.weight.detach().cpu(),dim=-1,eps=1e-12).numpy().copy()
    values[~known]=0;values[0]=0
    return values.astype(np.float32,copy=False)


def train_bpr(data,out,seed,device,smoke=False):
    """Train final-epoch BPR64 solely on pre-T0 reviews, including first events."""
    out=Path(out)
    if out.exists():raise FileExistsError(out)
    if not isinstance(seed,int) or seed<0:raise ValueError('seed must be a nonnegative integer')
    cutoff=int(data.split['t_ranker_train'])
    prior=data.interactions(['retriever_train']).sort_values(['user','ts','item'],kind='stable').reset_index(drop=True)
    if not len(prior) or not (prior.ts.to_numpy()<cutoff).all():raise ValueError('BPR requires strictly pre-T0 interactions')
    users=prior.user.to_numpy(dtype=np.int64);items=prior.item.to_numpy(dtype=np.int64)
    if (users<=0).any() or (users>data.n_users).any() or (items<=0).any() or (items>data.n_items).any():
        raise ValueError('training IDs out of range')
    known=np.zeros(data.n_items+1,bool);known[np.unique(items)]=True
    known_users=np.zeros(data.n_users+1,bool);known_users[np.unique(users)]=True
    negative_seed=seed+2000003;order_seed=seed+1000003
    sampler=PreT0NegativeSampler(users,items,np.flatnonzero(known),negative_seed)
    order_rng=np.random.default_rng(order_seed)
    epochs=1 if smoke else 300;batch_size=4096;device=torch.device(device)
    before=_data_hashes(data);source=digest(Path(__file__))
    out=_fresh_output(out)
    config={'dimension':DIMENSION,'seed':seed,'device':str(device),'technical_smoke':smoke,
        'epochs':epochs,'batch_size':batch_size,'learning_rate':.001,'weight_decay':0.,
        'objective':'mean softplus(negative_score-positive_score)','negative_sample_num':1,
        'negative_pool':'pre-T0 products excluding every user positive','negative_rng_seed':negative_seed,'order_rng_seed':order_seed,
        'initialisation':'Xavier-normal known-user/item matrices; unknown and padding rows zero',
        'fit_strict_before_ms':cutoff,'interaction_count':len(prior),'includes_first_user_interactions':True,
        'known_items':int(known.sum()),'known_users':int(known_users.sum()),'checkpoint_selection':'none; final fixed epoch',
        'id_mapping':'processed integer IDs directly; 0 is padding','validation_used':False,'test_predictions_loaded':False}
    write_json(out/'config.json',config);write_json(out/'metrics.json',{'status':'running',**config})
    model=BPRFactors(data.n_users,data.n_items,seed=seed,known_users=known_users,known_items=known).to(device)
    optimiser=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=0.)
    initial_hash=_array_hash(model.item_embedding.weight.detach().cpu().numpy())
    started=time.perf_counter();final_loss=None
    with (out/'train_log.csv').open('w',newline='') as log:
        writer=csv.DictWriter(log,fieldnames=['epoch','loss','epoch_seconds']);writer.writeheader()
        for epoch in range(epochs):
            epoch_started=time.perf_counter();total=0.;order=order_rng.permutation(len(prior))
            for start in range(0,len(prior),batch_size):
                idx=order[start:start+batch_size];negative=sampler.sample(users[idx])
                u=torch.as_tensor(users[idx],dtype=torch.long,device=device)
                p=torch.as_tensor(items[idx],dtype=torch.long,device=device)
                n=torch.as_tensor(negative,dtype=torch.long,device=device)
                loss=model.loss(u,p,n)
                if not torch.isfinite(loss):raise ValueError('nonfinite BPR loss')
                optimiser.zero_grad();loss.backward();optimiser.step()
                total+=float(loss.detach().cpu())*len(idx)
            final_loss=total/len(prior)
            writer.writerow({'epoch':epoch+1,'loss':final_loss,'epoch_seconds':time.perf_counter()-epoch_started});log.flush()
            if epoch==0 or (epoch+1)%25==0 or epoch+1==epochs:
                print(f'BPR seed {seed}: epoch {epoch+1}/{epochs}, loss={final_loss:.6f}',flush=True)
    seconds=time.perf_counter()-started
    del optimiser
    model=model.cpu().eval()
    state={key:value.detach().cpu() for key,value in model.state_dict().items()}
    torch.save(state,out/'model.pth');np.save(out/'bpr_known.npy',known);np.save(out/'bpr_known_users.npy',known_users)
    vectors=_unit_item_factors(model,known);np.save(out/'bpr_vectors.npy',vectors)
    restored=BPRFactors(data.n_users,data.n_items,seed=seed,known_users=known_users,known_items=known)
    restored.load_state_dict(torch.load(out/'model.pth',map_location='cpu',weights_only=True),strict=True);restored.eval()
    if not all(torch.equal(value,restored.state_dict()[key]) for key,value in state.items()):raise ValueError('portable BPR weights differ')
    query_users=torch.from_numpy(users[:128]);query_items=torch.from_numpy(items[:128])
    with torch.no_grad():
        if not torch.equal(model(query_users,query_items),restored(query_users,query_items)):raise ValueError('BPR score reload differs')
    if not np.array_equal(vectors,_unit_item_factors(restored,known)):raise ValueError('BPR vector reload differs')
    if np.any(state['item_embedding.weight'].numpy()[~known]) or np.any(state['user_embedding.weight'].numpy()[~known_users]):
        raise ValueError('unknown BPR rows changed')
    if sampler.returned!=epochs*len(prior):raise ValueError('BPR did not visit every training event per epoch')
    if before!=_data_hashes(data) or source!=digest(Path(__file__)):raise ValueError('BPR source/inputs changed during training')
    write_json(out/'sampler_audit.json',{'strict_before_ms':cutoff,'allowed_item_ids':np.flatnonzero(known),
        'allowed_items':int(known.sum()),'negatives_returned':sampler.returned,'first_negative_ids':sampler.first_negatives,
        'negative_stream_sha256':sampler.stream.hexdigest(),'future_negatives':0,'user_positive_negatives':0})
    metrics={'status':'complete',**config,'train_seconds':seconds,'final_loss':final_loss,
        'validation_calls':0,'validation_used':False,'test_predictions_loaded':False,
        'initial_item_factors_sha256':initial_hash,'reload_max_weight_difference':0.,'reload_scores_exact':True,'reload_vectors_exact':True}
    write_json(out/'metrics.json',metrics)
    proof={'status':'complete','input_sha256':before,'source_sha256':source,
        'training_pairs_sha256':_array_hash(np.stack([users,items,prior.ts.to_numpy(dtype=np.int64)],axis=1)),
        'initial_item_factors_sha256':initial_hash,'fit_strict_before_ms':cutoff,'validation_calls':0,
        'validation_used':False,'test_predictions_loaded':False,'checkpoint_selection':'none; final fixed epoch',
        'torch_version':torch.__version__,'numpy_version':np.__version__,'device':str(device),
        'output_sha256':{name:digest(out/name) for name in ('model.pth','bpr_vectors.npy','bpr_known.npy','bpr_known_users.npy','config.json','metrics.json','sampler_audit.json','train_log.csv')}}
    write_json(out/'provenance.json',proof)
    print(f'BPR factors complete: seed {seed}, {epochs} epochs, {seconds:.2f}s -> {out}',flush=True)
    return {'vectors_path':out/'bpr_vectors.npy','known_path':out/'bpr_known.npy','model_path':out/'model.pth','metrics':metrics,'provenance':proof}
