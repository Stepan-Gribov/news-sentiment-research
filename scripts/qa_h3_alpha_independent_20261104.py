"""External audit: no production kernel imports, no source/output mutation.
Raw endpoint arithmetic and news joins -> independent grouped NTS -> direct
past-window OLS and forward legs -> full-dummy OLS + fund CR1 and t(G-1).
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import pathlib,json,hashlib,sys,datetime as dt,traceback,time
import numpy as np,pandas as pd,duckdb
from scipy import linalg,stats
BASE=pathlib.Path('./analysis_outputs/h3_alpha_20261104')
RUN=pathlib.Path(sys.argv[1]) if len(sys.argv)>1 else BASE/'outputs/run_bf23bf409aff708f'
OUT=BASE/'independent_qa'/RUN.name;OUT.mkdir(parents=True,exist_ok=True)
START=time.monotonic();checks=[];warnings=[];details={}
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def clean(v):
 if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [clean(x) for x in v]
 if isinstance(v,(float,np.floating)):return float(v) if np.isfinite(v) else None
 if isinstance(v,(np.integer,)):return int(v)
 if isinstance(v,(np.bool_,)):return bool(v)
 if isinstance(v,(dt.datetime,dt.date,pd.Timestamp,pathlib.Path)):return str(v)
 return v
def save(name,v):
 p=OUT/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(clean(v),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');t.replace(p)
def stage(s,**kw):
 save('checkpoint.json',dict(stage=s,pid=os.getpid(),run=RUN.name,utc=dt.datetime.now(dt.timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-START,**kw));print(s,kw,flush=True)
def check(name,ok,**evidence):
 checks.append(dict(check=name,status='PASS' if bool(ok) else 'FAIL',**evidence))
def close(name,a,b,rtol=1e-8,atol=1e-11,**kw):
 a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float);same=a.shape==b.shape and np.allclose(a,b,rtol=rtol,atol=atol,equal_nan=True)
 err=float(np.nanmax(abs(a-b))) if a.shape==b.shape and np.isfinite(a-b).any() else 0.
 check(name,same,max_absolute_error=err,rtol=rtol,atol=atol,**kw)
def q(p):return "'"+str(p).replace('\\','/').replace("'","''")+"'"
def digest(vals):return hashlib.sha256('\n'.join(map(str,vals)).encode()).hexdigest()
def verify_records(records,prefix):
 for name,r in records.items():
  p=pathlib.Path(r['path']);actual=sha(p);check(prefix+':'+name,actual==r['sha256'],sha256=actual)

def main():
 stage('PINNING_INPUTS_AND_SEALED_RUN')
 manifest=json.loads((RUN/'manifest.json').read_text(encoding='utf-8'));im=json.loads((RUN/'input_manifest.json').read_text(encoding='utf-8'));inputs=im['inputs']
 verify_records(inputs,'run_input');verify_records(manifest['files'],'run_output')
 before_manifest=sha(RUN/'manifest.json');details['run']=dict(path=str(RUN),manifest_sha256=before_manifest,input_manifest_sha256=sha(RUN/'input_manifest.json'),version=im['version'],fund_release=str(pathlib.Path(inputs['fund_returns']['path']).parent),spec_sha256=inputs['spec']['sha256'])
 ntsroot=pathlib.Path(inputs['nts_primary']['path']).parent;ntsmanifest=json.loads((ntsroot/'manifest.json').read_text(encoding='utf-8'));verify_records(ntsmanifest['inputs'],'NTS_input');verify_records(ntsmanifest['files'],'NTS_output');ni=ntsmanifest['inputs']
 spec=json.loads(pathlib.Path(inputs['spec']['path']).read_text(encoding='utf-8'));handoff=json.loads(pathlib.Path(inputs['handoff']['path']).read_text(encoding='utf-8'));actions=json.loads(pathlib.Path(ni['actions']['path']).read_text(encoding='utf-8'))
 c=duckdb.connect(config={'threads':2,'memory_limit':'2GB','temp_directory':str(OUT/'spill')})
 def view(name,path):c.execute('CREATE TEMP VIEW '+name+' AS FROM read_parquet('+q(path)+')')
 def ck(name,sql):
  n=int(c.execute(sql).fetchone()[0]);check(name,n==0,violations=n)
 view('u',ntsroot/'quantity_news_contributions.parquet');view('n',inputs['nts_primary']['path']);view('ledger',ni['portfolio_units']['path']);view('target',ni['quantity_panel']['path']);view('news',ni['current_net']['path']);view('lags',ni['lag_map']['path']);view('lw',ni['lag_windows']['path']);view('cal',ni['full_report_calendar']['path']);view('selected',ntsroot/'selected_origins.parquet');view('maturities',ni['scheduled_events']['path'])
 stage('RAW_ENDPOINTS_NEWS_AND_INDEPENDENT_NTS')
 ck('unique_corporate_rows','SELECT count(*) FROM (SELECT fund_id,interval_id,security_id,count(*) n FROM u GROUP BY ALL HAVING n>1)')
 ck('unique_source_report_security','SELECT count(*) FROM (SELECT report_id,security_id,count(*) n FROM ledger WHERE is_reported_position GROUP BY ALL HAVING n>1)')
 ck('latest_origin_no_fallback','SELECT count(*) FROM selected s WHERE s.end_date<>(SELECT max(c.end_date) FROM cal c WHERE c.fund_id=s.fund_id AND strftime(c.end_date,\'%Y-%m\')=strftime(s.end_date,\'%Y-%m\'))')
 ck('selected_one_fund_month','SELECT count(*) FROM (SELECT fund_id,strftime(end_date,\'%Y-%m\') AS m,count(*) n FROM selected GROUP BY ALL HAVING n>1)')
 # Independently verify every copied endpoint against the original versioned ledger.
 for suffix,report,full in [('0','opening_report_id','opening_full_portfolio'),('1','closing_report_id','closing_full_portfolio')]:
  ck('raw_endpoint_Q_'+suffix,"SELECT count(*) FROM u LEFT JOIN ledger p ON p.report_id=u."+report+" AND p.security_id=u.security_id AND p.is_reported_position WHERE (p.report_id IS NOT NULL AND (u.Q"+suffix+" IS DISTINCT FROM p.quantity_est)) OR (p.report_id IS NULL AND ((u.Q"+suffix+" IS NOT NULL AND NOT (u."+full+" AND u.Q"+suffix+"=0))))")
  for field,outfield in [('unit_price_rub','P'),('quote_date','quote'),('quote_age_days','age')]:
   ck('raw_endpoint_'+outfield+suffix,'SELECT count(*) FROM u JOIN ledger p ON p.report_id=u.'+report+' AND p.security_id=u.security_id AND p.is_reported_position WHERE u.'+outfield+suffix+' IS DISTINCT FROM p.'+field)
 # Full corporate-union denominator, independent of inherited/news inclusion.
 c.execute('''CREATE TEMP TABLE endpoint_keys AS SELECT interval_id,security_id FROM (SELECT s.interval_id,p.security_id FROM selected s JOIN ledger p ON p.report_id=s.opening_report_id WHERE p.is_reported_position UNION SELECT s.interval_id,p.security_id FROM selected s JOIN ledger p ON p.report_id=s.closing_report_id WHERE p.is_reported_position)''')
 c.execute('''CREATE TEMP TABLE expected_corporate AS SELECT k.interval_id,k.security_id FROM endpoint_keys k JOIN selected s USING(interval_id) LEFT JOIN ledger a ON a.report_id=s.opening_report_id AND a.security_id=k.security_id AND a.is_reported_position LEFT JOIN ledger b ON b.report_id=s.closing_report_id AND b.security_id=k.security_id AND b.is_reported_position WHERE coalesce(a.asset_class,b.asset_class) IN ('equity','bond') AND coalesce(coalesce(a.market_type,b.market_type),'unknown') NOT IN ('ofz_bond','subfederal_bond','municipal_bond','state_bond','ifi_bond','etf_ppif','public_ppif')''')
 ck('all_corporate_union_keys',"SELECT count(*) FROM ((SELECT interval_id,security_id FROM expected_corporate EXCEPT SELECT interval_id,security_id FROM u) UNION ALL (SELECT interval_id,security_id FROM u EXCEPT SELECT interval_id,security_id FROM expected_corporate))")
 ck('inherited_current_news_exact','SELECT count(*) FROM u JOIN target t USING(row_id) JOIN news w ON t.company_period_id=w.company_period_id WHERE u.inherited_target AND (u.net_sentiment IS DISTINCT FROM w.net_sentiment OR u.start_date<>w.period_start OR u.end_date<>w.period_end OR u.company_id<>w.company_id)')
 ck('lag_news_exact_report_chain','SELECT count(*) FROM u JOIN lags l USING(row_id) WHERE u.inherited_target AND (u.net_sentiment_lag1 IS DISTINCT FROM l.net_sentiment_lag1 OR u.lag_end_date IS DISTINCT FROM l.lag_end_date OR (l.lag_interval_id IS NOT NULL AND (l.lag_end_date<>u.start_date OR l.lag_closing_report_id<>u.opening_report_id)))')
 ck('lag_matches_original_company_window','SELECT count(*) FROM u JOIN lags l USING(row_id) JOIN lw w ON w.company_period_id=l.lag_company_period_id WHERE u.inherited_target AND (w.company_id<>u.company_id OR w.period_start<>u.lag_start_date OR w.period_end<>u.lag_end_date OR w.net_sentiment IS DISTINCT FROM u.net_sentiment_lag1)')
 # Independently derive endpoint admissibility and quoted-unit conversion.
 valid=[]
 for s in ['0','1']:
  full='opening_full_portfolio' if s=='0' else 'closing_full_portfolio';d='start_date' if s=='0' else 'end_date'
  expr=f"(Q{s} IS NOT NULL AND isfinite(Q{s}) AND Q{s}>=0 AND ((Q{s}>0 AND P{s}>0 AND isfinite(P{s}) AND age{s} BETWEEN 0 AND 7 AND quote{s}<={d} AND upper(currency{s}) IN ('RUB','RUR','SUR') AND isfinite(V{s}) AND abs(V{s}-Q{s}*P{s})<=1e-7+1e-10*abs(V{s})) OR (Q{s}=0 AND ((NOT observed{s} AND {full}) OR (observed{s} AND V{s}=0)))))"
  valid.append('coalesce('+expr+',false)')
 ratio='1.0';ambig='false'
 for event in actions['events']:
  condition=f"isin={q(event['isin'])} AND quote0<DATE {q(event['post_quote_from'])} AND quote1>=DATE {q(event['post_quote_from'])} AND Q0>0 AND Q1>0"
  ratio+='*(CASE WHEN '+condition+' THEN '+str(event['closing_to_opening_quantity_ratio'])+' ELSE 1 END)'
  if event.get('event_date'):ambig+=' OR (isin='+q(event['isin'])+' AND ((start_date>=DATE '+q(event['event_date'])+' AND start_date<DATE '+q(event['post_quote_from'])+') OR (end_date>=DATE '+q(event['event_date'])+' AND end_date<DATE '+q(event['post_quote_from'])+')))'
 c.execute('CREATE TEMP TABLE rederived AS SELECT *,('+valid[0]+') v0,('+valid[1]+') v1,('+ratio+') ratio,('+ambig+') transition FROM u')
 c.execute('''CREATE TEMP TABLE calc AS SELECT *,CASE WHEN v0 AND v1 AND NOT coalesce(identity_conflict,false) AND NOT coalesce(class_conflict,false) AND NOT coalesce(transition,false) THEN CASE WHEN Q0>0 AND (Q1>0 OR (Q1=0 AND closing_full_portfolio)) THEN (Q1*ratio-Q0)*P0 WHEN Q0=0 AND Q1>0 AND opening_full_portfolio THEN Q1*P1 WHEN Q0=0 AND Q1=0 THEN 0 END END calc_BQ,
 EXISTS(SELECT 1 FROM maturities m WHERE m.security_id=rederived.security_id AND m.scheduled_maturity_date>rederived.start_date AND m.scheduled_maturity_date<=rederived.end_date) calc_maturity FROM rederived''')
 ck('endpoint_admissibility0','SELECT count(*) FROM calc WHERE v0 IS DISTINCT FROM endpoint_valid0');ck('endpoint_admissibility1','SELECT count(*) FROM calc WHERE v1 IS DISTINCT FROM endpoint_valid1')
 ck('security_split_once','SELECT count(*) FROM calc WHERE abs(ratio-closing_to_opening_quantity_ratio)>1e-12 OR abs(Q1_opening_units-Q1*ratio)>1e-6+1e-11*abs(Q1*ratio)')
 ck('BQ_all_rows_reconstruction','SELECT count(*) FROM calc WHERE (calc_BQ IS NULL)<>(B_Q IS NULL) OR abs(calc_BQ-B_Q)>1e-5+1e-11*abs(calc_BQ)')
 ck('maturities_exclusion','SELECT count(*) FROM calc WHERE calc_maturity IS DISTINCT FROM scheduled_redemption_in_interval')
 e="coalesce(inherited_target,false) AND isfinite(calc_BQ) AND company_id IS NOT NULL AND isfinite(net_sentiment) AND isfinite(net_sentiment_lag1) AND abs(net_sentiment)<=1+1e-12 AND abs(net_sentiment_lag1)<=1+1e-12 AND coalesce(eligible_primary,false) AND coalesce(lag_eligible_primary,false) AND NOT coalesce(issuer_identity_conflict,false) AND NOT coalesce(asset_class_conflict,false) AND NOT calc_maturity"
 c.execute('CREATE TEMP TABLE ecalc AS SELECT *,coalesce(('+e+'),false) ec FROM calc')
 ck('common_E_exact','SELECT count(*) FROM ecalc WHERE ec IS DISTINCT FROM NTS_common_E')
 c.execute('''CREATE TEMP TABLE aggregates AS SELECT fund_id,interval_id,sum(abs(calc_BQ)) gross_u,sum(abs(calc_BQ)) FILTER(WHERE ec) gross_e,count(*) FILTER(WHERE ec) ec_count,
 sum(calc_BQ*net_sentiment) FILTER(WHERE ec) num0,sum(abs(calc_BQ*net_sentiment)) FILTER(WHERE ec) den0,
 sum(calc_BQ*net_sentiment_lag1) FILTER(WHERE ec) numL,sum(abs(calc_BQ*net_sentiment_lag1)) FILTER(WHERE ec) denL,
 sum(abs(calc_BQ)) FILTER(WHERE ec AND net_sentiment<>0) cover0,sum(abs(calc_BQ)) FILTER(WHERE ec AND net_sentiment_lag1<>0) coverL,
 count(DISTINCT company_id) FILTER(WHERE ec AND abs(calc_BQ*net_sentiment)>0) companies0,count(DISTINCT company_id) FILTER(WHERE ec AND abs(calc_BQ*net_sentiment_lag1)>0) companiesL,
 sha256(coalesce(string_agg(row_id,chr(10) ORDER BY row_id) FILTER(WHERE ec),'')) ehash FROM ecalc GROUP BY fund_id,interval_id''')
 c.execute('''CREATE TEMP TABLE comparison AS SELECT n.*,a.gross_u,a.gross_e,a.ec_count,coalesce(a.num0,0) num0,coalesce(a.den0,0) den0,coalesce(a.numL,0) numL,coalesce(a.denL,0) denL,coalesce(a.cover0,0)/a.gross_u cov0,coalesce(a.coverL,0)/a.gross_u covL,a.companies0,a.companiesL,a.ehash
 FROM n LEFT JOIN aggregates a USING(fund_id,interval_id)''')
 for actual,expected in [('gross_B_Q_all_measurable_corporate','gross_u'),('gross_B_Q_E','coalesce(gross_e,0)'),('current_numerator','num0'),('current_denominator','den0'),('lagnews_numerator','numL'),('lagnews_denominator','denL')]:
  ck('NTS_aggregate_'+actual,'SELECT count(*) FROM comparison WHERE corporate_positions_total IS NOT NULL AND (('+actual+' IS NULL)<>(('+expected+') IS NULL) OR abs('+actual+'-('+expected+'))>1e-5+1e-11*abs('+expected+'))')
 for label,num,den,cov,companies in [('current','num0','den0','cov0','companies0'),('lagnews','numL','denL','covL','companiesL')]:
  expected='CASE WHEN '+den+'>1e-12*greatest(1,coalesce(gross_e,0)) AND '+companies+'>=2 AND '+cov+'>=.5 THEN '+num+'/'+den+' END'
  ck('NTS_score_'+label,'SELECT count(*) FROM comparison WHERE (NTS_Q_'+label+' IS NULL)<>(('+expected+') IS NULL) OR abs(NTS_Q_'+label+'-('+expected+'))>1e-10')
  ck('NTS_coverage_'+label,'SELECT count(*) FROM comparison WHERE isfinite('+cov+') AND abs(news_trade_coverage_'+label+'-'+cov+')>1e-10')
 ck('NTS_common_set_hash','SELECT count(*) FROM comparison WHERE corporate_positions_total IS NOT NULL AND common_E_hash<>ehash')
 examples=[]
 for label,cond in [('equity',"asset_class='equity' AND Q0>0 AND Q1>0"),('bond',"asset_class='bond' AND Q0>0 AND Q1>0"),('entry','confirmed_entry'),('exit','confirmed_exit'),('split','known_split_adjusted')]:
  rows=c.execute('SELECT fund_id,interval_id,security_id,asset_class,Q0,Q1,P0,P1,V0,V1,observed0,observed1,opening_full_portfolio,closing_full_portfolio,ratio,calc_BQ,B_Q,net_sentiment,net_sentiment_lag1,ec,raw_current.current_numerator,raw_current.current_denominator,raw_current.news_trade_coverage_current FROM ecalc e JOIN n raw_current USING(fund_id,interval_id) WHERE '+cond+' AND calc_BQ IS NOT NULL ORDER BY e.fund_id,e.interval_id,e.security_id LIMIT 1').fetchdf().to_dict('records')
  examples.append(dict(kind=label,rows=rows))
 details['quantity_examples']=examples;details['NTS_manifest_sha256']=sha(ntsroot/'manifest.json')
 stage('RETURN_ADAPTER_AND_RAW_FACTORS')
 raw=pd.read_parquet(inputs['fund_returns']['path']);adapt=pd.read_parquet(RUN/'monthly_returns_adapter.parquet');cross=pd.read_parquet(inputs['crosswalk']['path']);cash=pd.read_parquet(inputs['cash_events']['path'])
 raw=raw.sort_values(['fund_id','month']).reset_index(drop=True);adapt=adapt.sort_values(['fund_id','month']).reset_index(drop=True)
 check('return_keys_unchanged',raw[['fund_id','month']].equals(adapt[['fund_id','month']]))
 for col in ['unit_value','previous_unit_nav','fund_return']:close('return_adapter_preserves_'+col,raw[col],adapt[col],rtol=0,atol=0)
 dates0=pd.to_datetime(raw.start_valuation_date);dates1=pd.to_datetime(raw.end_valuation_date);period=pd.PeriodIndex(raw.month,freq='M')
 identity=raw.identity_verified.eq(True)&dates0.ge(pd.to_datetime(raw.identity_valid_from))&dates1.le(pd.to_datetime(raw.identity_valid_to))
 adjacency=dates0.dt.to_period('M').astype(str).eq((period-1).astype(str))&dates1.dt.to_period('M').astype(str).eq(period.astype(str))
 knowncash=(raw['known_distribution_date_unresolved'].eq(True) if handoff.get('known_cash_exclusion_scope')=='source_flagged_record_payment_months' else raw.fund_id.astype(str).isin(set(cash.fund_id.astype(str))))
 splitbad=np.zeros(len(raw),dtype=bool)
 for ev in handoff.get('known_fund_splits',[]):
  if ev.get('source_already_backadjusted'):check('fund_split_no_second_adjustment_'+str(ev['fund_id']),ev.get('apply_additional_split_adjustment') is False)
  else:splitbad|=(raw.fund_id.astype(str).eq(str(ev['fund_id']))&dates0.lt(pd.Timestamp(ev['event_date']))&dates1.ge(pd.Timestamp(ev['event_date']))).to_numpy()
 eligible=(raw.return_verified.eq(True)&identity&adjacency&raw.unit_value.gt(0)&raw.previous_unit_nav.gt(0)&raw.currency.fillna('').str.upper().isin(['RUB','RUR','SUR'])&np.isfinite(raw.fund_return)&~knowncash&~splitbad).to_numpy(bool)
 check('diagnostic_return_adapter_exact',np.array_equal(eligible,adapt.diagnostic_eligible.to_numpy(bool)))
 treligible=eligible&raw.total_return_verified.eq(True).to_numpy(bool)&raw.distribution_verified.eq(True).to_numpy(bool)&raw.split_verified.eq(True).to_numpy(bool)
 check('TR_eligibility_no_proxy_fallback',np.array_equal(treligible,adapt.TR_eligible.to_numpy(bool)))
 f=pd.read_parquet(RUN/'monthly_factors_adapter.parquet');original_f=pd.read_parquet(inputs['factors']['path'])
 for new,old in [('risk_free','rf_m'),('equity_return','MCFTR_return_m'),('bond_return','RGBITR_return_m')]:close('factor_adapter_'+new,f[new],original_f[old],rtol=0,atol=0)
 factor_root=pathlib.Path(inputs['factors']['path']).parent
 for series,col in [('MCFTR','equity_return'),('RGBITR','bond_return')]:
  daily=pd.read_parquet(factor_root/'normalized'/f'{series}_daily.parquet');daily['m']=pd.to_datetime(daily.date).dt.to_period('M');ends=daily.sort_values('date').groupby('m').tail(1).set_index('m').index_value
  expected=np.array([ends.get(pd.Period(m,freq='M'),np.nan)/ends.get(pd.Period(m,freq='M')-1,np.nan)-1 for m in f.month]);close('raw_total_return_factor_'+series,expected,f[col],rtol=1e-11,atol=1e-12)
 rf=pd.read_parquet(factor_root/'normalized/ruonia_index_daily.parquet');details['RF_source_columns']=list(rf)
 datecol=next(x for x in ['date','Date','dt'] if x in rf);valuecol=next(x for x in ['index_value','ruonia_index','value','Value'] if x in rf)
 ri=rf.set_index(pd.to_datetime(rf[datecol]).dt.normalize())[valuecol]
 expected=[]
 for month in f.month:
  m=pd.Period(month,freq='M');a=ri.get(m.start_time.normalize(),np.nan);b=ri.get((m+1).start_time.normalize(),np.nan);expected.append(b/a-1)
 close('raw_RUONIA_shift1_endpoints',expected,f.risk_free,rtol=1e-11,atol=1e-12)
 check('RF_shift1_explicit',f.shift_days.eq(1).all())
 stage('INDEPENDENT_PAST_FITS_AND_FUTURE_LEGS')
 alpha_path=RUN/'alpha/UNIT_VALUE_KNOWN_ADJUSTMENTS/HISTORY60_MIN36/alpha.parquet';alpha=pd.read_parquet(alpha_path);origin=pd.read_parquet(RUN/'origin_eligibility.parquet')
 r=adapt.copy();r['eligible_external']=eligible;r=r.merge(f[['month','risk_free','equity_return','bond_return','factors_verified']],on='month',validate='many_to_one');r['period']=pd.PeriodIndex(r.month,freq='M')
 r['all_ok']=r.eligible_external& r.factors_verified.eq(True)&np.isfinite(r[['fund_return','risk_free','equity_return','bond_return']]).all(axis=1)
 groups={str(fund):g.sort_values('period').set_index('period',drop=False) for fund,g in r.groupby('fund_id',sort=False)}
 training_paths=list((alpha_path.parent).glob('chunk_*.training.parquet'));training=pd.concat([pd.read_parquet(p) for p in training_paths],ignore_index=True);trainmap={tid:g for tid,g in training.groupby('training_id',sort=False)}
 expected_alphas={};beta_cache={};origin_examples=[];maxerr=0.;nrep=0
 wanted=alpha.loc[alpha.past_alpha.notna()].copy()
 for key,g_a in wanted.groupby(['fund_id','interval_id'],sort=False):
  a=g_a.iloc[0];g=groups[str(a.fund_id)];d=pd.Timestamp(a.origin_date);m=d.to_period('M');last=m if m.end_time.normalize()<=d else m-1;first=last-59
  t=g.loc[g.period.between(first,last)&g.all_ok&pd.to_datetime(g.end_valuation_date).le(d)&pd.to_datetime(g.start_valuation_date).le(d)]
  recent=int(t.period.ge(last-11).sum());X=np.column_stack([np.ones(len(t)),t.equity_return-t.risk_free,t.bond_return-t.risk_free]);y=(t.fund_return-t.risk_free).to_numpy(float)
  b=linalg.lstsq(X,y,lapack_driver='gelsy')[0];standard=np.column_stack([np.ones(len(t)),(X[:,1:]-X[:,1:].mean(0))/X[:,1:].std(0,ddof=1)]);condition=float(np.linalg.cond(standard))
  tid=a.training_id;record=trainmap.get(tid)
  check('training_members:'+str(tid),record is not None and tuple(record.month)==tuple(t.month),fund_id=a.fund_id,interval_id=a.interval_id,n_past=len(t))
  if record is not None:
   close('training_y:'+str(tid),record.fund_excess,y,rtol=1e-12,atol=1e-13)
   close('training_X:'+str(tid),record[['x_equity_return','x_bond_return']],X[:,1:],rtol=1e-12,atol=1e-13)
  close('past_beta:'+str(tid),[a.past_alpha,a.beta_equity_return,a.beta_bond_return],b,rtol=1e-8,atol=1e-11)
  check('history_gate:'+str(tid),len(t)>=36 and recent>=9 and condition<=10000 and a.n_past==len(t) and a.n_past_recent==recent and str(last)==a.training_last_allowed and str(first)==a.training_first_allowed)
  beta_cache[key]=b
  for row in g_a.itertuples(index=False):
   months=pd.period_range(m+1,m+row.horizon,freq='M');legs=g.reindex(months);full=bool(legs.all_ok.fillna(False).all());anchor=legs.start_valuation_date.iloc[0];anchorok=pd.notna(anchor) and pd.Timestamp(anchor)>=d
   outcome=float(np.mean((legs.fund_return-legs.risk_free).to_numpy(float)-np.column_stack([legs.equity_return-legs.risk_free,legs.bond_return-legs.risk_free])@b[1:])) if full and anchorok else np.nan
   expected_alphas[(str(row.fund_id),str(row.interval_id),int(row.horizon))]=outcome
   if pd.notna(row.alpha):
    nrep+=1;maxerr=max(maxerr,abs(outcome-row.alpha));check('future_alpha:'+row.training_id+':h'+str(row.horizon),bool(full and anchorok and np.isclose(outcome,row.alpha,rtol=1e-9,atol=1e-11)),difference=outcome-row.alpha)
   else:check('future_NA:'+row.training_id+':h'+str(row.horizon),not(full and anchorok),status_saved=row.status)
 details['alpha_reconstruction']=dict(unique_past_fits=len(beta_cache),forward_outcomes=nrep,max_absolute_difference=maxerr)
 check('all_numeric_alpha_reconstructed',nrep==int(alpha.alpha.notna().sum()))
 stage('FULL_DUMMY_OLS_AND_FUND_CR1')
 models=json.loads((RUN/'all_model_details.json').read_text(encoding='utf-8'));regrep=[]
 for model in models:
  if model.get('status')!='ESTIMATED' or model['sensitivity']!='PRIMARY' or model['horizon'] not in [1,3] or model['branch']!='UNIT_VALUE_KNOWN_ADJUSTMENTS':continue
  model_id=model['model_id'];dpath=RUN/'regression'/model['branch']/'PRIMARY'/model_id;sample=pd.read_parquet(dpath/'used_sample.parquet').sort_values(['fund_id','end_date','interval_id']).reset_index(drop=True)
  pool=alpha.merge(origin,on=['fund_id','interval_id'],validate='many_to_one',suffixes=('','_origin'));mask=pool.horizon.eq(model['horizon'])&pool.universe_eligible.eq(True)&pool.nts_pair_eligible.eq(True)&np.isfinite(pool[['alpha','past_alpha','NTS_Q_current','NTS_Q_lagnews']]).all(axis=1)
  if model['group']!='pooled':mask&=pool.fund_focus.eq(model['group'])
  expected=pool.loc[mask];sizes=expected.groupby('fund_id').interval_id.transform('size');expected=expected.loc[sizes.ge(2)].sort_values(['fund_id','end_date','interval_id']).reset_index(drop=True)
  check('exact_model_rows:'+model_id,np.array_equal(sample[['fund_id','interval_id']].to_numpy(),expected[['fund_id','interval_id']].to_numpy()))
  check('ordered_sample_hash:'+model_id,digest(sample.fund_id.astype(str)+'\t'+sample.interval_id.astype(str))==model['sample_hash'])
  close('row_alpha_inputs:'+model_id,sample.alpha,[expected_alphas[(str(x.fund_id),str(x.interval_id),int(x.horizon))] for x in sample.itertuples(index=False)],rtol=1e-9,atol=1e-11)
  terms=['NTS_Q_current']+(['NTS_Q_lagnews'] if model['distributed'] else [])+['past_alpha'];n=len(sample);codes,uniques=pd.factorize(sample.fund_id,sort=True);G=len(uniques)
  X=sample[terms].to_numpy(float);Y=sample.alpha.to_numpy(float);sizes=np.bincount(codes);means=np.vstack([X[codes==j].mean(0) for j in range(G)]);WX=X-means[codes];absorbed=np.max(abs(WX),axis=0)<=1e-12*np.maximum(1,np.max(abs(X),axis=0))
  terms=[x for x,drop in zip(terms,absorbed) if not drop];X=sample[terms].to_numpy(float);k=len(terms);D=np.column_stack([np.eye(G)[codes],X]);coef=linalg.lstsq(D,Y,lapack_driver='gelsy')[0];res=Y-D@coef
  bread=linalg.inv(D.T@D);scores=np.vstack([D[codes==j].T@res[codes==j] for j in range(G)]);correction=G/(G-1)*(n-1)/(n-G-k);covD=correction*bread@scores.T@scores@bread
  T=np.zeros((k+1,G+k));T[0,:G]=sizes/n;T[1:,G:]=np.eye(k);beta=T@coef;cov=T@covD@T.T;names=['const']+terms
  reported=pd.DataFrame(model['coefficients']).set_index('term').loc[names];native_cov=pd.read_csv(dpath/'full_covariance.csv',index_col=0,encoding='utf-8-sig').loc[names,names]
  close('LSDV_coefficients:'+model_id,beta,reported.beta,rtol=1e-7,atol=1e-10)
  close('LSDV_CR1_full_covariance:'+model_id,cov,native_cov,rtol=1e-6,atol=1e-11)
  check('rank_df_clusters:'+model_id,model['K_full']==G+k and model['residual_df']==n-G-k and model['reference_df']==G-1 and model['clusters']==['fund_id'] and model['fund_FE'] and not model['time_FE'])
  qcrit=stats.t.ppf(.975,G-1);se=np.sqrt(np.diag(cov));tt=beta/se;pp=2*stats.t.sf(abs(tt),G-1)
  for col,values in [('SE',se),('t',tt),('p',pp),('CI_low',beta-qcrit*se),('CI_high',beta+qcrit*se)]:close('inference_'+col+':'+model_id,values,reported[col],rtol=1e-6,atol=1e-9)
  WY=Y-pd.Series(Y).groupby(codes).transform('mean').to_numpy();r2=1-np.dot(res,res)/np.dot(WY,WY);close('within_R2:'+model_id,[r2],[model['within_R2']],rtol=1e-8,atol=1e-10)
  tests={x['test']:x for x in model['target_tests']}
  if model['distributed']:
   block=cov[1:3,1:3];b=beta[1:3];variance=block.sum();b_sum=b.sum();se_sum=np.sqrt(variance);p_sum=2*stats.t.sf(abs(b_sum/se_sum),G-1);joint=b@linalg.solve(block,b)/2;p_joint=stats.f.sf(joint,2,G-1)
   close('sum_beta_SE_p:'+model_id,[b_sum,se_sum,p_sum],[tests['current+lagnews=0'][x] for x in ['beta','SE','p']],rtol=1e-7,atol=1e-10)
   close('joint_F_p:'+model_id,[joint,p_joint],[tests['joint_current_lagnews=0'][x] for x in ['F','p']],rtol=1e-7,atol=1e-10)
  regrep.append(dict(model_id=model_id,n=n,funds=G,rank=G+k,sample_hash=model['sample_hash'],max_coef_difference=float(np.max(abs(beta-reported.beta.to_numpy()))),max_cov_difference=float(np.max(abs(cov-native_cov.to_numpy()))),independent_full_dummy_fit=True))
  # Complete row-by-row source and forward-leg example for each fixed group/horizon.
  x=sample.iloc[0];origin_examples.append(dict(group=model['group'],horizon=model['horizon'],variant=model['variant'],fund_id=x.fund_id,interval_id=x.interval_id,origin_date=x.end_date,training_beta=beta_cache[(str(x.fund_id),str(x.interval_id))].tolist(),alpha_independent=expected_alphas[(str(x.fund_id),str(x.interval_id),int(x.horizon))],alpha_saved=x.alpha))
  if model['variant']=='CURRENT_COMMON':
   other=next(z for z in models if z['branch']==model['branch'] and z['sensitivity']==model['sensitivity'] and z['group']==model['group'] and z['horizon']==model['horizon'] and z['variant']=='CURRENT_PLUS_LAGNEWS')
   check('current_vs_lag_common_sample:'+model_id,model['sample_hash']==other['sample_hash'] and model['nobs']==other['nobs'])
 details['regression_replicas']=regrep;details['origin_examples']=origin_examples
 for group in ['pooled','equity','bond','mixed']:
  for h in [1,3]:check('required_replica_coverage:'+group+':h'+str(h),any(x['model_id'].endswith(group+'__h'+str(h)+'__CURRENT_PLUS_LAGNEWS') for x in regrep))
 tr_alpha=pd.read_parquet(RUN/'alpha/TR_VERIFIED/HISTORY60_MIN36/alpha.parquet');summary=json.loads((RUN/'summary.json').read_text(encoding='utf-8'))
 check('TR_NOT_ESTIMABLE_honest', (summary['primary_TR_H3_status']=='NOT_ESTIMABLE')==(not tr_alpha.alpha.notna().any()))
 check('source_run_unchanged',sha(RUN/'manifest.json')==before_manifest)
 verify_records(manifest['files'],'unchanged_run_output')
 details['collector_status']={}
 for n in ['fund_history/checkpoint.json','fund_history/official_monthly_returns/checkpoint.json','fund_history/collection_completed.json']:
  if (BASE/n).exists():details['collector_status'][n]=json.loads((BASE/n).read_text(encoding='utf-8'))
 result=dict(status='PASS' if all(x['status']=='PASS' for x in checks) else 'FAIL',audit_type='EXTERNAL_INDEPENDENT',production_kernels_imported=False,own_existing_internal_QA_not_counted=True,run=details['run'],checks_pass=sum(x['status']=='PASS' for x in checks),checks_fail=sum(x['status']=='FAIL' for x in checks),checks=checks,details=details,warnings=warnings,source_model_limits=summary['limitations'],completed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),runtime_seconds=time.monotonic()-START)
 save('independent_qa.json',result);stage('COMPLETED',status=result['status'],checks_pass=result['checks_pass'],checks_fail=result['checks_fail']);print(json.dumps(dict(status=result['status'],pass_count=result['checks_pass'],fail_count=result['checks_fail'],run=RUN.name,report=str(OUT/'independent_qa.json'))))
if __name__=='__main__':
 try:main()
 except Exception as e:save('execution_failure.json',dict(error=repr(e),traceback=traceback.format_exc(),checks=checks,details=details));stage('QA_EXECUTION_FAILED',error=repr(e));raise
