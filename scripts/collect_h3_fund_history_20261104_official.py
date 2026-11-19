"""Bounded public official-source corroboration; writes only new H3 data."""
import pathlib,json,hashlib,time,datetime as dt
import requests,pandas as pd,duckdb
R=pathlib.Path('./analysis_outputs/h3_alpha_20261104/fund_history');O=R/'official_sources';O.mkdir(exist_ok=True)
urls={
 'arsagera_api_documentation.pdf':'https://arsagera.ru/files/arsagera_metriki_fonda.pdf',
 'arsagera_return_method.html':'https://arsagera.ru/kuda_i_kak_investirovat/klyuchevye_metodiki_upravleniya_kapitalom/poryadok_opredeleniya_dohodnosti_paevogo_investicionnogo_fonda/',
 'arsagera_faq.html':'https://arsagera.ru/otvety_na_voprosy/',
 'arsagera_unit_page.html':'https://arsagera.ru/products/opifa_arsagera_-_fond_akcij/stoimost_paya_i_scht/',
 'wim_split_chart.html':'https://www.wimsavings.ru/ru/pif/chart/',
 'wim_fund_parameters.html':'https://www.wimsavings.ru/ru/pif/parametres/basic-parameters/'
}
funds={'fa':('403','0363-75409054'),'f4si':('404','0364-75409132'),'f64':('481','0439-75408664'),'fo':('3435','2721')}
for code in funds:urls['arsagera_api_'+code+'.json']='https://arsagera.ru/api/v1/funds/'+code+'/fund-metrics/'
manifest=[];errors=[]
for n,url in urls.items():
 p=O/n
 try:
  if not p.exists():
   for attempt in range(3):
    try:
     time.sleep(1.1);x=requests.get(url,timeout=(10,35),headers={'User-Agent':'Academic-fund-history-research/1.0'});x.raise_for_status();p.write_bytes(x.content);break
    except requests.exceptions.SSLError:raise
    except Exception:
     if attempt==2:raise
     time.sleep(2**attempt)
  manifest.append(dict(file=str(p),url=url,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),observed_utc=dt.datetime.now(dt.timezone.utc).isoformat()))
 except Exception as e:errors.append(dict(url=url,error=repr(e)))
rows=[]
for code,(fund,reg) in funds.items():
 p=O/('arsagera_api_'+code+'.json')
 if not p.exists():continue
 try:
  data=json.loads(p.read_text(encoding='utf-8'));data=data['data']
  for i,v in enumerate(data):rows.append(dict(fund_id=fund,registration_number=reg,series_id='arsagera:'+code,valuation_date=v['date'],unit_value=float(v['nav_per_share']),NAV_RUB=float(v['total_net_assets']),currency='RUB',source_path=str(p),json_pointer='/data/'+str(i),source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),source_url=urls[p.name],distribution_status='not_verified',split_status='not_verified'))
 except Exception as e:errors.append(dict(file=str(p),error=repr(e)))
if rows:
 frame=pd.DataFrame(rows);frame.valuation_date=pd.to_datetime(frame.valuation_date);frame.to_parquet(O/'arsagera_official_unit_nav.parquet',index=False,compression='zstd')
 c=duckdb.connect();c.register('official',frame);prior=str(R/'v1_existing_cache/unit_values_daily.parquet')
 parity=c.execute('SELECT o.fund_id,count(*) n_compared,count(*) FILTER(WHERE abs(o.unit_value-p.unit_nav)<=.011) n_agree,min(o.valuation_date) first_date,max(o.valuation_date) last_date,max(abs(o.unit_value-p.unit_nav)) max_difference FROM official o JOIN read_parquet(?) p ON o.fund_id=p.fund_id AND o.valuation_date=p.value_date GROUP BY 1 ORDER BY 1',[prior]).fetchdf().to_dict('records')
else:parity=[]
summary=dict(status='COLLECTED_OFFICIAL_UNIT_VALUES' if rows else 'NO_OFFICIAL_UNIT_VALUES_COLLECTED',official_rows=len(rows),series=len({x['fund_id'] for x in rows}),files=manifest,errors=errors,Investfunds_comparison=parity,not_asserted_total_return=True)
(O/'official_collection_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf-8');print(json.dumps(summary,ensure_ascii=False,default=str))
