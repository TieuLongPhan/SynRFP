from pathlib import Path
import csv,json,re,hashlib,statistics,collections,math
base=Path('Experiments/72_completed_references/publication_metadata/20261123');out=base/'barrier_statistical_inputs';out.mkdir(exist_ok=True)
fpsource=Path('Experiments/65_default_optimal/publication_metadata/default_optimal_v1/barrier/cells.csv')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
fp=[r for r in csv.DictReader(fpsource.open())if re.fullmatch(r'cv_r[0-4]_f[0-4]',r['split'])and r.get('fraction')in ('','1','1.0')];assert len(fp)==1575
cells=[];verified=0
for r in fp:
 p=Path(r['source_result_path'])if r['source_result_path']and r['source_result_sha256']else fpsource
 h=sha(p)
 if p!=fpsource:
  assert h==r['source_result_sha256'],p
  d=json.loads(p.read_text())
  for m in ['mae','rmse','r2']:assert math.isclose(float(r[m]),float(d[m]),rel_tol=1e-12,abs_tol=1e-12),(p,m)
  verified+=1
 match=re.fullmatch('cv_r([0-4])_f([0-4])',r['split'])
 cells.append(dict(dataset=r['dataset'],target=r['target']or('G_act'if r['dataset']=='cyclo'else'ea'),method=r['method'],configuration=r['configuration'],split=r['split'],repeat=int(match[1]),fold=int(match[2]),n_train=int(r['ntrain']),n_validation=int(r['nval']),n_test=int(r['ntest']),mae=float(r['mae']),rmse=float(r['rmse']),r2=float(r['r2']),units=r['units'],source_result_path=str(p),source_result_sha256=h,source_kind='native_result_receipt'if p!=fpsource else'published_per_fit_scalar_table',split_sha256=r['split_sha256'],rows_sha256=r['rows_sha256'],selection_interpretation=r['selection_interpretation']))
ref40=list(csv.DictReader((base/'dmpnn/all_matched_cells.csv').open()));ref14=list(csv.DictReader((base/'dmpnn/primary14_random_5x5_cells.csv').open()))
for ref in [r for r in ref40 if r['target']=='ea'and r['dataset']in{r['dataset']for r in fp}and re.fullmatch('cv_r[0-4]_f[0-4]',r['split'])]+ref14:
 ds=ref['dataset'];anchor=next(r for r in cells if r['dataset']==ds and r['method']=='drfp'and r['split']==ref['split'])
 p=Path(ref['result_path']);assert sha(p)==ref['result_sha256'];d=json.loads(p.read_text())
 for m in ['mae','rmse','r2']:assert math.isclose(float(ref[m]),d[m],rel_tol=1e-12,abs_tol=1e-12)
 vals=[int(ref.get(k)or ref[other])for k,other in [('n_train','ntrain'),('n_val','nval'),('n_test','ntest')]]
 assert vals==[anchor['n_train'],anchor['n_validation'],anchor['n_test']],(ds,ref['split'])
 cells.append(dict(dataset=ds,target=anchor['target'],method='dmpnn',configuration='chemprop_v2_default'if ds in ['cyclo','gdb7']else'mapped_cgr_REAC_DIFF',split=ref['split'],repeat=anchor['repeat'],fold=anchor['fold'],n_train=vals[0],n_validation=vals[1],n_test=vals[2],mae=d['mae'],rmse=d['rmse'],r2=d['r2'],units=anchor['units'],source_result_path=str(p),source_result_sha256=ref['result_sha256'],source_kind='native_result_receipt',split_sha256=ref.get('split_sha256')or d.get('splits_json_sha256',''),rows_sha256=ref.get('rows_sha256',''),selection_interpretation='fixed_reference_recipe'))
assert len(cells)==1800
assert len({(r['dataset'],r['method'],r['repeat'],r['fold'])for r in cells})==1800
cells.sort(key=lambda r:(r['dataset'],r['method'],r['repeat'],r['fold']))
g=collections.defaultdict(list)
for r in cells:g[r['dataset'],r['method'],r['repeat']].append(r)
repeats=[]
for (ds,m,rep),rs in sorted(g.items()):
 assert len(rs)==5
 repeats.append(dict(dataset=ds,method=m,repeat=rep,n_folds=5,**{metric:statistics.mean(r[metric]for r in rs)for metric in ['mae','rmse','r2']}))
wide=[]
for ds,rep in sorted({(r['dataset'],r['repeat'])for r in repeats}):
 row={'dataset':ds,'repeat':rep}
 for r in repeats:
  if(r['dataset'],r['repeat'])==(ds,rep):
   for metric in ['mae','rmse','r2']:row[r['method']+'__'+metric]=r[metric]
 assert len(row)==26;wide.append(row)
for name,rows in [('per_fit_scores.csv',cells),('per_repeat_mean_scores.csv',repeats),('paired_repeat_scores_wide.csv',wide)]:
 with(out/name).open('w')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
(out/'AUDIT.json').write_text(json.dumps({'phase':'complete','per_fit_scores':len(cells),'repeat_mean_rows':len(repeats),'paired_dataset_repeat_rows':len(wide),'datasets':sorted({r['dataset']for r in cells}),'methods':sorted({r['method']for r in cells}),'FP_native_receipt_hashes_and_metrics_verified':verified,'FP_scores_from_published_table_without_individual_receipt_pointer':1575-verified,'DMPNN_native_receipt_hashes_and_metrics_verified':225,'same_train_validation_test_counts_per_DMPNN_DRFP_pair':True,'FP_source_csv_sha256':sha(fpsource),'outputs_sha256':{str(p.relative_to(base)):sha(p)for p in out.glob('*.csv')},'statistics_not_run':True,'new_model_fits':0,'prediction_arrays_published':False},indent=2)+'\n')
print('Exported',len(cells),'per-fit scores;',len(repeats),'repeat means;',len(wide),'paired dataset-repeat rows;FP nativeverified',verified)
