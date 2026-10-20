"""Corrected-core-only Morgan encoding; original parser/algebra/sketch contract."""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pandas as pd
import study as S

_ENGINE=None

def loaded_core_is_correct(directory):
    directory=Path(directory).resolve()
    for name,m in tuple(sys.modules.items()):
        if name=='synrfp' or name.startswith('synrfp.'):
            filename=getattr(m,'__file__',None)
            S.need(filename and Path(filename).resolve().is_relative_to(directory),'Mixed or unpinned SynRFP module: '+name)

def engine(root=S.ROOT):
    global _ENGINE
    directory=S.CORRECTED.resolve();loaded_core_is_correct(directory)
    if _ENGINE is None:
        S.core_ready(root)
        sys.path.insert(0,str(directory))
        from synrfp.tokenizers.morgan import MorganTokenizer
        S.need(MorganTokenizer.PROFILE==S.PROFILE,'Wrong imported Morgan semantics')
        loaded_core_is_correct(directory)
        E=S.module('morgan43_frozen_algebra',S.REPO/'Experiments/20_focused_factorial/encode_factorial.py')
        loaded_core_is_correct(directory)
        tokenizer=MorganTokenizer(E.NODE_ATTRS,E.EDGE_ATTRS)
        _ENGINE=E,tokenizer
    loaded_core_is_correct(directory)
    return _ENGINE

def encode_reactions(strings,valid,root=S.ROOT):
    E,tokenizer=engine(root)
    S.need(len(strings)==len(valid),'Validity/source length mismatch')
    configs={k:v for k,v in E.VARIANTS.items() if v['tokenizer']=='morgan'}
    arrays={name:np.zeros((len(strings),256),np.uint8) for name in configs}
    for i,(rxn,ok) in enumerate(zip(strings,valid)):
        if not ok:continue
        # Old refusals remain zero. New defects on previously valid rows raise.
        reaction=E._reaction(rxn,'classification')
        for radius in (2,3):
            left=tokenizer.tokens_graph(reaction.reactant,radius);right=tokenizer.tokens_graph(reaction.product,radius)
            for sketch in E.SKETCHES:
                for mode,value in E.encode_tokens(left,right,sketch).items():
                    S.need(value.shape==(256,) and value.dtype==np.uint8,'Wrong corrected feature shape')
                    arrays[E.variant_name('morgan',radius,sketch,mode)][i]=value
    return arrays

def _piece(arg):
    strings,valid,root=arg;return encode_reactions(strings,valid,root)

def feature_ready(root,plan,source,cache=None,deep=False):
    cache=cache or S.HashCache();spec=plan['sources'][source];path=Path(spec['feature_receipt'])
    if not path.exists():return None
    r=S.read(path);S.need(r['phase']=='complete' and r['plan_sha256']==S.sha(Path(root)/'plan.json')
        and r['tokenizer_profile']==S.PROFILE and r['n']==spec['n'],'Changed corrected feature receipt')
    expected={Path(p).name for p in spec['features'].values()}|{Path(p).with_suffix('.json').name for p in spec['features'].values()}
    S.need(set(r['files'])==expected,'Incomplete corrected feature catalog')
    for name,h in r['files'].items():cache.verify(path.parent/name,h)
    for config,filename in spec['features'].items():
        fp=Path(filename);m=S.read(fp.with_suffix('.json'))
        S.need(m['spec']==plan['configs'][config] and m['tokenizer_profile']==S.PROFILE and m['plan_sha256']==r['plan_sha256']
            and m['source_table_sha256']==plan['pinned_files'][spec['table']] and m['rowcount']==spec['n']
            and m['feature_sha256']==r['files'][fp.name],'Changed corrected feature metadata')
        if deep:
            f=pd.read_csv(spec['table'],dtype={'row_id':str})
            with np.load(spec['validity_source'],allow_pickle=False) as z:valid=z['ok']
            with np.load(fp,allow_pickle=False) as z:
                S.need(set(z.files)=={'X','ok','n_bits','source_positions','row_ids'} and z['X'].shape==(len(f),256)
                    and z['X'].dtype==np.uint8 and int(z['n_bits'])==2048 and np.array_equal(z['ok'],valid)
                    and np.array_equal(z['source_positions'],f.source_position.to_numpy(dtype=np.int64))
                    and np.array_equal(z['row_ids'].astype(str),f.row_id.to_numpy(dtype=str))
                    and not np.any(z['X'][~valid]),'Corrected row/validity identity mismatch')
    return r

def disk_guard(root):
    for path in (Path.home(),Path(root)/'artifacts'):
        S.need(shutil.disk_usage(path.resolve()).free>=40*1024**3,'Below40GiB storage reserve')

def compact_staging(root,plan,source):
    """Remove only new, disposable chunk matrices after canonical deep closure."""
    root=Path(root);spec=plan['sources'][source]
    receipt=feature_ready(root,plan,source,deep=True)
    S.need(receipt is not None,'Cannot compact an unclosed canonical feature pool')
    directory=root/'artifacts/encoding_journals'/source;catalog=directory/'CHUNKS_SEALED.json'
    if not catalog.exists():return # no historical staging, e.g. fixture
    S.need(not directory.is_symlink(),'Refuse symlinked scratch directory')
    ledger=S.read(catalog);plan_sha=S.sha(root/'plan.json')
    S.need(ledger['plan_sha256']==plan_sha and ledger['source']==source and ledger['tokenizer_profile']==S.PROFILE,'Wrong staging owner')
    canonical=Path(spec['feature_receipt']);marker=directory/'COMPACTED.json'
    if marker.exists():
        old=S.read(marker);S.need(old['canonical_receipt_sha256']==S.sha(canonical) and old['historical_staging_catalog_sha256']==S.sha(catalog),'Compaction closure changed')
    anchor=(root/'artifacts/encoding_journals').resolve()
    deleted=0
    for name,info in ledger['chunks'].items():
        S.need(Path(name).name==name and name.endswith('.npz'),'Invalid disposable staging path')
        path=directory/name;metadata=path.with_suffix('.json')
        S.need(not path.is_symlink() and not metadata.is_symlink() and path.parent.resolve().is_relative_to(anchor),'Unsafe staging path')
        S.need(S.sha(metadata)==info['receipt_sha256'],'Historical staging receipt changed')
        if path.exists():
            S.need(path.is_file() and S.sha(path)==info['sha256'],'Disposable staging changed')
            path.unlink();deleted+=1
    if not marker.exists():S.write(marker,dict(schema='morgan43-owned-staging-compacted-v1',phase='complete',source=source,
        plan_sha256=plan_sha,canonical_receipt_sha256=S.sha(canonical),historical_staging_catalog_sha256=S.sha(catalog),
        chunks_compacted=len(ledger['chunks']),only_new_disposable_chunk_matrices_removed=True,finished_utc=S.now()))
    return deleted

def encode_source(root,plan,source,threads=2,chunk_size=128):
    root=Path(root);spec=plan['sources'][source]
    if feature_ready(root,plan,source):compact_staging(root,plan,source);return True
    disk_guard(root);engine(root)
    frame=pd.read_csv(spec['table'],dtype={'row_id':str});S.need(len(frame)==spec['n'],'Changed source rowcount')
    with np.load(spec['validity_source'],allow_pickle=False) as z:
        valid=z['ok'];S.need(np.array_equal(z['source_positions'],frame.source_position.to_numpy())
            and np.array_equal(z['row_ids'].astype(str),frame.row_id.to_numpy(dtype=str)),'Original validity identity mismatch')
    strings=frame.rxn.astype(str).tolist();parent=Path(spec['feature_receipt']).parent;parent.mkdir(parents=True,exist_ok=True)
    journal=root/'artifacts/encoding_journals'/source;journal.mkdir(parents=True,exist_ok=True)
    pending=[];pieces=[];plan_sha=S.sha(root/'plan.json')
    for start in range(0,len(frame),chunk_size):
        end=min(start+chunk_size,len(frame));p=journal/(str(start)+'.npz');receipt=p.with_suffix('.json')
        identity=dict(plan_sha256=plan_sha,source=source,start=start,end=end,tokenizer_profile=S.PROFILE)
        if receipt.exists():
            r=S.read(receipt);S.need(all(r.get(k)==v for k,v in identity.items()) and S.sha(p)==r['sha256'],'Changed encoding chunk')
        else:pending.append((strings[start:end],valid[start:end],str(root)));pieces.append((p,receipt,identity))
    start_time=time.monotonic()
    with ProcessPoolExecutor(max_workers=threads) as pool:
        for (p,receipt,identity),values in zip(pieces,pool.map(_piece,pending,chunksize=1)):
            S.npz(p,**values);S.write(receipt,{**identity,'sha256':S.sha(p)})
            S.write(root/('encode_'+source+'_status.json'),dict(phase='encoding',source=source,rows_encoded=identity['end'],expected=len(frame),updated_utc=S.now()))
    # A historical staging ledger is separate from retained canonical outputs.
    chunks={}
    for start in range(0,len(frame),chunk_size):
        p=journal/(str(start)+'.npz');chunks[p.name]=dict(sha256=S.sha(p),receipt_sha256=S.sha(p.with_suffix('.json')))
    sealed=dict(plan_sha256=plan_sha,source=source,tokenizer_profile=S.PROFILE,chunks=chunks)
    if (journal/'CHUNKS_SEALED.json').exists():S.need(S.read(journal/'CHUNKS_SEALED.json')==sealed,'Staging catalog changed')
    else:S.write(journal/'CHUNKS_SEALED.json',sealed)
    for config in plan['configs']:
        disk_guard(root);fp=Path(spec['features'][config]);meta=fp.with_suffix('.json')
        if meta.exists():
            m=S.read(meta);S.need(m['plan_sha256']==plan_sha and m['tokenizer_profile']==S.PROFILE and m['spec']==plan['configs'][config]
                and m['feature_sha256']==S.sha(fp),'Changed partial feature export');continue
        scratch=journal/(config+'.scratch.npy');X=np.lib.format.open_memmap(scratch,mode='w+',dtype=np.uint8,shape=(len(frame),256))
        for start in range(0,len(frame),chunk_size):
            p=journal/(str(start)+'.npz');r=S.read(p.with_suffix('.json'));S.need(S.sha(p)==r['sha256'],'Encoding chunk changed during assembly')
            with np.load(p,allow_pickle=False) as z:X[start:r['end']]=z[config]
        X.flush();S.npz(fp,X=X,ok=valid,n_bits=np.int64(2048),source_positions=frame.source_position.to_numpy(dtype=np.int64),row_ids=frame.row_id.to_numpy(dtype=str))
        del X;scratch.unlink() # disposable new scratch only; never a model/cache
        S.write(meta,dict(spec=plan['configs'][config],tokenizer_profile=S.PROFILE,plan_sha256=plan_sha,
            source_table_sha256=plan['pinned_files'][spec['table']],rowcount=len(frame),feature_sha256=S.sha(fp),
            original_validity_source_sha256=spec['validity_source_sha256'],invalid_zero_rows=int((~valid).sum())))
    files={p.name:S.sha(p) for c in plan['configs'] for p in (Path(spec['features'][c]),Path(spec['features'][c]).with_suffix('.json'))}
    S.write(spec['feature_receipt'],dict(phase='complete',schema='morgan43-features-v1',plan_sha256=plan_sha,tokenizer_profile=S.PROFILE,
        source=source,n=len(frame),valid=int(valid.sum()),invalid_zero_rows=int((~valid).sum()),files=files,elapsed_seconds=time.monotonic()-start_time,finished_utc=S.now()))
    feature_ready(root,plan,source,deep=True);compact_staging(root,plan,source);return True
