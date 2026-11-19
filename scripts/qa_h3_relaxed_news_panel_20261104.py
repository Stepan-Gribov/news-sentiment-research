"""Independent read-only source/zero/mask QA of the separately sealed H3 panel.
No production module imports, alternative panel or regressions. All SQL objects
are temporary audit objects. Production files and old sealed sources are read.
"""
from pathlib import Path
import datetime as dt
import hashlib
import json
import os
import time
import traceback
import argparse
import duckdb

BASE=Path('./analysis_outputs')
parser=argparse.ArgumentParser()
parser.add_argument('--panel-root',default=str(BASE/'h3_relaxed_news_20261104/panels/5adc30709c6f4cc2'))
parser.add_argument('--manifest-sha',default='b0ce21d89b36d62a7c01d6cb4c2c8e3dfc5620322063597d411d7af6907389cf')
args=parser.parse_args()
PANEL=Path(args.panel_root)
OUT=BASE/('h3_alpha_20261104/qa/h3_relaxed_news_panel_'+PANEL.name+'_20261104')
OUT.mkdir(parents=True,exist_ok=True)
SPEC_SHA='36a64070c5dfcba749d7baa1f15b639f9801a39fcac12142e372a94a2554699f'
MANIFEST_SHA=args.manifest_sha
PRIOR={'panels/nts_ee5a8bf9b450137a/manifest.json':'598e05ce14d7b2942fdc3056633a882feb51c3883aebc9c81e1c4f322db80695','outputs/run_d0440fad412ee5e9/manifest.json':'d61349f8275762eee3c224a8118d69eec81fb08107d48a6fc00f63b324a09628','outputs/provider_TR4_b00ab703609fb58f/manifest.json':'af25eb9dc57288ef54d2b0839477f0ced6b9539708bd3a88a8cbd2257b8e98c3'}
START=time.monotonic();checks=[];details={}


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4194304),b''):h.update(b)
    return h.hexdigest()


def save(name,value):
    p=OUT/name;t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str,allow_nan=False)+'\n',encoding='utf-8');t.replace(p)


def stage(name,**kw):
    save('checkpoint.json',dict(stage=name,pid=os.getpid(),elapsed_seconds=time.monotonic()-START,**kw));print(name,kw,flush=True)


def ck(name,condition,**kw):checks.append(dict(check=name,status='PASS' if bool(condition) else 'FAIL',**kw))


def q(p):return "'"+str(p).replace('\\','/').replace("'","''")+"'"


def records(m,prefix):
    for group in ['inputs','files']:
        for name,r in m[group].items():
            p=Path(r['path']);ck(prefix+':'+group+':'+name,p.is_file() and p.stat().st_size==r['bytes'] and sha(p)==r['sha256'])


def main():
    stage('VERIFY_SEALED_PANEL_AND_FROZEN_SOURCES')
    m=json.loads((PANEL/'manifest.json').read_text(encoding='utf-8'))
    ck('pinned_sealed_panel',sha(PANEL/'manifest.json')==MANIFEST_SHA and m['status']=='RELAXED_PANEL_SEALED')
    ck('frozen_V3_spec',m['spec_sha256']==SPEC_SHA and m['inputs']['spec']['sha256']==SPEC_SHA)
    records(m,'before')
    for rel,expected in PRIOR.items():ck('old_manifest_before:'+rel,sha(BASE/'h3_alpha_20261104'/rel)==expected)
    c=duckdb.connect(config={'threads':2,'memory_limit':'2GB','temp_directory':str(OUT/'spill')})
    def v(name,path):c.execute('CREATE TEMP VIEW '+name+' AS FROM read_parquet('+q(path)+')')
    def sqlcheck(name,sql):
        count=int(c.execute(sql).fetchone()[0]);ck(name,count==0,violations=count)
    def different(a,b,rtol='1e-10',atol='1e-9'):
        return '(coalesce(isfinite('+a+'),false)<>coalesce(isfinite('+b+'),false) OR (coalesce(isfinite('+a+'),false) AND abs('+a+'-'+b+')>'+atol+'+'+rtol+'*abs('+b+')))'
    for name,key in [('src','contributions'),('decisions','decisions'),('original_origins','origins'),('net','net'),('lag','lag'),('cal','calendar'),('events','events'),('registry','registry'),('approved','approved_scope')]:v(name,m['inputs'][key]['path'])
    for name,file in [('n','relaxed_security_window_ledger.parquet'),('scores','relaxed_fund_period_scores.parquet'),('w','company_window_status_ledger.parquet'),('members','reconstructed_window_members.parquet')]:v(name,PANEL/file)
    for name,keys in [('src','fund_id,interval_id,security_id'),('n','fund_id,interval_id,security_id'),('decisions','fund_id,interval_id,security_id'),('scores','fund_id,interval_id'),('w','company_period_id'),('members','company_period_id,pair_id')]:
        sqlcheck('unique:'+name,'SELECT count(*) FROM (SELECT '+keys+',count(*) k FROM '+name+' GROUP BY ALL HAVING k>1)')
    sqlcheck('all655542_source_positions_exact','SELECT count(*) FROM ((SELECT fund_id,interval_id,security_id FROM src EXCEPT ALL SELECT fund_id,interval_id,security_id FROM n) UNION ALL (SELECT fund_id,interval_id,security_id FROM n EXCEPT ALL SELECT fund_id,interval_id,security_id FROM src))')
    ck('position_count',c.execute('SELECT count(*) FROM n').fetchone()[0]==655542)
    aliases={x:'legacy_'+x for x in ['eligible_for_score','NTS_common_E','eligible_primary','lag_eligible_primary']}
    fields=[r[0] for r in c.execute('DESCRIBE src').fetchall()]
    sqlcheck('every_frozen_quantity_news_source_field_preserved','SELECT count(*) FROM src p JOIN n USING(fund_id,interval_id,security_id) WHERE '+' OR '.join('p."'+f+'" IS DISTINCT FROM n."'+aliases.get(f,f)+'"' for f in fields))
    sqlcheck('source_position_id_preserved','SELECT count(*) FROM src p JOIN n USING(fund_id,interval_id,security_id) WHERE n.position_id IS DISTINCT FROM p.row_id')
    sqlcheck('approved_news_universe_exact_frozen_registry','SELECT count(*) FROM ((SELECT DISTINCT company_id FROM approved EXCEPT SELECT DISTINCT company_id FROM registry) UNION ALL (SELECT DISTINCT company_id FROM registry EXCEPT SELECT DISTINCT company_id FROM approved))')
    sqlcheck('predecessor_unique','SELECT count(*) FROM (SELECT fund_id,closing_report_id,count(*) k FROM cal GROUP BY ALL HAVING k>1)')
    threshold=json.loads(Path(m['inputs']['threshold']['path']).read_text(encoding='utf-8'))['cutoff_percent']
    ck('unchanged_fixed_P99_threshold',abs(float(threshold)-322.7771642245575)<=1e-10)
    c.execute("""CREATE TEMP TABLE masks AS SELECT d.fund_id,d.interval_id,d.security_id,d.row_id,
        coalesce(array_to_string(list_filter(string_split(coalesce(exclusion_reason_augmented,''),';'),x->x<>'' AND x NOT IN ('sentiment_not_primary','missing_net_sentiment','missing_control_log_news_count_interval')),';'),'') qa_nonnews,
        coalesce(array_to_string(list_filter(string_split(coalesce(exclusion_reason_augmented,''),';'),x->x IN ('sentiment_not_primary','missing_net_sentiment','missing_control_log_news_count_interval')),';'),'') qa_removed_news,
        coalesce(trimmed_upper_p99,false) OR coalesce((quantity_open>0 AND quantity_close>0 AND quantity_change_pct>?),false) qa_P99
        FROM decisions d""",[threshold])
    c.execute("""CREATE TEMP TABLE e0 AS SELECT p.fund_id,p.interval_id,p.security_id,p.row_id position_id,p.company_id,p.B_Q,p.inherited_target,
        p.start_date,p.end_date,p.opening_report_id,p.scheduled_redemption_in_interval,
        d.row_id qa_candidate_row_id,d.qa_nonnews,d.qa_removed_news,d.qa_P99,
        coalesce(d.qa_nonnews='',false) AND NOT coalesce(d.qa_P99,false) qa_scope,
        coalesce(isfinite(p.B_Q),false) AND NOT coalesce(p.scheduled_redemption_in_interval,false) qa_action,
        p.company_id IS NOT NULL AND p.company_id IN (SELECT company_id FROM registry) qa_mapped,
        prev.interval_id qa_previous_interval_id,prev.start_date qa_previous_start,prev.end_date qa_previous_end,
        prev.interval_id IS NOT NULL AND prev.end_date=p.start_date qa_previous_valid
        FROM src p LEFT JOIN masks d USING(fund_id,interval_id,security_id)
        LEFT JOIN cal prev ON prev.fund_id=p.fund_id AND prev.closing_report_id=p.opening_report_id""")
    sqlcheck('all_union_positions_have_original_decisions','SELECT count(*) FROM e0 WHERE qa_candidate_row_id IS NULL')
    sqlcheck('nonnews_reason_and_P99_masks','SELECT count(*) FROM e0 e JOIN n USING(fund_id,interval_id,security_id) WHERE n.inherited_nonnews_reason IS DISTINCT FROM e.qa_nonnews OR n.removed_legacy_news_reasons IS DISTINCT FROM e.qa_removed_news OR n.inherited_fixed_P99_excluded IS DISTINCT FROM e.qa_P99 OR n.quantity_in_scope IS DISTINCT FROM e.qa_scope OR n.quantity_action_eligible IS DISTINCT FROM e.qa_action OR n.mapping_in_scope IS DISTINCT FROM e.qa_mapped')
    sqlcheck('news_only_restoration','SELECT count(*) FROM e0 e JOIN n USING(fund_id,interval_id,security_id) WHERE n.restored_news_only IS DISTINCT FROM (e.qa_scope AND NOT coalesce(e.inherited_target,false) AND e.qa_removed_news<>\'\')')
    sqlcheck('real_previous_window_chain','SELECT count(*) FROM e0 e JOIN n USING(fund_id,interval_id,security_id) WHERE n.previous_start IS DISTINCT FROM e.qa_previous_start OR n.previous_end IS DISTINCT FROM e.qa_previous_end OR n.previous_window_valid IS DISTINCT FROM e.qa_previous_valid')
    sqlcheck('physically_eligible_current_and_lag_window_validity_without_paired_gate','SELECT count(*) FROM e0 e JOIN n USING(fund_id,interval_id,security_id) WHERE e.qa_scope AND e.qa_action AND (n.current_window_valid IS DISTINCT FROM e.qa_mapped OR n.lagnews_window_valid IS DISTINCT FROM (e.qa_mapped AND e.qa_previous_valid))')
    details['auditor_scope_correction']='Initial audit compared window flags on physically excluded positions. Their false flags do not affect E or scores. The corrected check applies to physical candidates; independent E/products/scores remain checked on every row. Prior audit artifact is retained.'
    def wid(start,end):return "sha256('actual_report_interval'||chr(9)||company_id||chr(9)||strftime("+start+",'%Y-%m-%d')||chr(9)||strftime("+end+",'%Y-%m-%d'))"
    c.execute('CREATE TEMP TABLE e1 AS SELECT *,CASE WHEN qa_scope AND qa_action AND qa_mapped THEN '+wid('start_date','end_date')+' END qa_current_id,CASE WHEN qa_scope AND qa_action AND qa_mapped AND qa_previous_valid THEN '+wid('qa_previous_start','qa_previous_end')+' END qa_lag_id FROM e0')
    sqlcheck('exact_window_identifiers','SELECT count(*) FROM e1 e JOIN n USING(fund_id,interval_id,security_id) WHERE n.current_window_id IS DISTINCT FROM e.qa_current_id OR n.lagnews_window_id IS DISTINCT FROM e.qa_lag_id')
    stage('RAW_EXISTING_AND_213_RECONSTRUCTED_WINDOWS')
    cols='company_period_id,company_id,period_start,period_end,net_sentiment,n_candidates,n_valid,n_relevant,n_errors_or_unattempted,n_undated_company_candidates,eligible_primary,state,sample_hash'
    c.execute("CREATE TEMP TABLE old_windows AS SELECT * EXCLUDE(priority) FROM (SELECT "+cols+",1 priority FROM net WHERE period_kind='actual_report_interval' UNION ALL SELECT "+cols+",2 priority FROM lag) QUALIFY row_number() OVER(PARTITION BY company_period_id ORDER BY priority)=1")
    sqlcheck('existing_window_source_fields_exact','SELECT count(*) FROM old_windows o JOIN w USING(company_period_id) WHERE '+' OR '.join('o.'+f+' IS DISTINCT FROM w.'+f for f in cols.split(',') if f!='company_period_id'))
    sqlcheck('all_old_exact_windows_present','SELECT count(*) FROM old_windows o LEFT JOIN w USING(company_period_id) WHERE w.company_period_id IS NULL')
    c.execute('CREATE TEMP TABLE needed AS SELECT qa_current_id company_period_id,company_id,start_date period_start,end_date period_end FROM e1 WHERE qa_current_id IS NOT NULL UNION SELECT qa_lag_id,company_id,qa_previous_start,qa_previous_end FROM e1 WHERE qa_lag_id IS NOT NULL')
    c.execute('CREATE TEMP TABLE missing AS SELECT k.* FROM needed k LEFT JOIN old_windows o USING(company_period_id) WHERE o.company_period_id IS NULL')
    ck('needed_and_missing_window_counts',c.execute('SELECT count(*) FROM needed').fetchone()[0]==20675 and c.execute('SELECT count(*) FROM missing').fetchone()[0]==213)
    sqlcheck('exact_new_window_keys','SELECT count(*) FROM ((SELECT company_period_id FROM missing EXCEPT SELECT company_period_id FROM w WHERE window_source=\'RECONSTRUCTED_MISSING_EXACT_WINDOW\') UNION ALL (SELECT company_period_id FROM w WHERE window_source=\'RECONSTRUCTED_MISSING_EXACT_WINDOW\' EXCEPT SELECT company_period_id FROM missing))')
    sqlcheck('canonical_events_only','SELECT count(*) FROM events WHERE NOT coalesce(is_representative,false)')
    c.execute('CREATE TEMP TABLE raw_members AS SELECT k.company_period_id,e.pair_id,e.article_id,e.company_id,e.event_date,e.valid,e.attempted,e.relevance,e.sentiment_label FROM missing k JOIN events e ON e.company_id=k.company_id AND e.event_date>k.period_start AND e.event_date<=k.period_end')
    sqlcheck('every_reconstructed_member_from_exact_frozen_window','SELECT count(*) FROM ((SELECT * FROM raw_members EXCEPT ALL SELECT * FROM members) UNION ALL (SELECT * FROM members EXCEPT ALL SELECT * FROM raw_members))')
    c.execute("""CREATE TEMP TABLE rebuilt AS SELECT k.company_period_id,count(e.pair_id) n_candidates,
        count(e.pair_id) FILTER(WHERE e.valid) n_valid,count(e.pair_id) FILTER(WHERE e.valid AND e.relevance='relevant') n_relevant,
        count(e.pair_id) FILTER(WHERE e.valid AND e.relevance='relevant' AND e.sentiment_label='positive') n_positive,
        count(e.pair_id) FILTER(WHERE e.valid AND e.relevance='relevant' AND e.sentiment_label='negative') n_negative,
        sha256(string_agg(e.pair_id,chr(10) ORDER BY e.pair_id)) sample_hash
        FROM missing k LEFT JOIN raw_members e USING(company_period_id) GROUP BY k.company_period_id""")
    for field in ['n_candidates','n_valid','n_relevant','n_positive','n_negative','sample_hash']:
        sqlcheck('rebuilt_raw_'+field,'SELECT count(*) FROM rebuilt r JOIN w USING(company_period_id) WHERE r.'+field+' IS DISTINCT FROM w.'+field)
    sqlcheck('rebuilt_errors_and_Net','SELECT count(*) FROM rebuilt r JOIN w USING(company_period_id) WHERE w.n_errors_or_unattempted IS DISTINCT FROM (r.n_candidates-r.n_valid) OR w.eligible_primary IS DISTINCT FROM (r.n_candidates>0 AND r.n_candidates=r.n_valid AND r.n_relevant>0) OR '+different('w.net_sentiment',"CASE WHEN r.n_candidates>0 AND r.n_candidates=r.n_valid AND r.n_relevant>0 THEN (r.n_positive-r.n_negative)::DOUBLE/r.n_relevant END",'1e-12','1e-13'))
    sqlcheck('rebuilt_undated_flags_from_raw_events',"SELECT count(*) FROM w LEFT JOIN (SELECT company_id,count(*) k FROM events WHERE event_date IS NULL GROUP BY company_id) e USING(company_id) WHERE w.window_source='RECONSTRUCTED_MISSING_EXACT_WINDOW' AND w.n_undated_company_candidates IS DISTINCT FROM coalesce(e.k,0)")
    first,last=c.execute('SELECT min(event_date),max(event_date) FROM events').fetchone()
    c.execute('CREATE TEMP TABLE wx0 AS SELECT w.*, company_id IS NOT NULL AND company_id IN (SELECT company_id FROM registry) qa_mapping,period_start>=? AND period_end<=? qa_window FROM w',[first,last])
    c.execute("""CREATE TEMP TABLE wx AS SELECT *,CASE WHEN company_id IS NOT NULL AND qa_mapping AND qa_window AND n_errors_or_unattempted=0 AND n_valid=n_candidates
        THEN CASE WHEN n_relevant>0 AND isfinite(net_sentiment) AND abs(net_sentiment)<=1+1e-12 THEN net_sentiment
        WHEN n_relevant=0 AND n_undated_company_candidates=0 THEN 0. END END qa_value FROM wx0""")
    sqlcheck('independent_window_scope_dates_and_known_value','SELECT count(*) FROM wx WHERE mapping_in_scope IS DISTINCT FROM qa_mapping OR window_valid IS DISTINCT FROM qa_window OR news_known IS DISTINCT FROM coalesce(isfinite(qa_value),false) OR '+different('value','qa_value','1e-12','1e-13'))
    sqlcheck('source_present_window_records','SELECT count(*) FROM wx WHERE source_present IS DISTINCT FROM true')
    sqlcheck('independent_known_window_status',"SELECT count(*) FROM wx WHERE isfinite(qa_value) AND news_status IS DISTINCT FROM CASE WHEN n_relevant>0 THEN 'OBSERVED_SENTIMENT' WHEN n_candidates=0 THEN 'ZERO_NO_CANDIDATES_IN_FROZEN_CORPUS' ELSE 'ZERO_ALL_CANDIDATES_IRRELEVANT' END")
    sqlcheck('unknown_company_never_known_value','SELECT count(*) FROM wx WHERE company_id IS NULL AND (isfinite(value) OR news_known)')
    sqlcheck('no_false_absence_with_undated_or_errors',"SELECT count(*) FROM wx WHERE news_status IN ('ZERO_NO_CANDIDATES_IN_FROZEN_CORPUS','ZERO_ALL_CANDIDATES_IRRELEVANT') AND (n_undated_company_candidates<>0 OR n_errors_or_unattempted<>0 OR n_relevant<>0 OR NOT qa_mapping OR NOT qa_window)")
    sqlcheck('old_eligible_dated_Net_preserved_with_undated', 'SELECT count(*) FROM wx WHERE eligible_primary AND isfinite(net_sentiment) AND qa_mapping AND qa_window AND '+different('value','net_sentiment','0','0'))
    c.execute('CREATE TEMP TABLE ex AS SELECT e.*,cw.qa_value qa_current_value,lw.qa_value qa_lag_value,cw.company_period_id IS NOT NULL qa_current_present,lw.company_period_id IS NOT NULL qa_lag_present,cw.news_status qa_current_status,lw.news_status qa_lag_status,cw.n_undated_company_candidates qa_current_undated,lw.n_undated_company_candidates qa_lag_undated FROM e1 e LEFT JOIN wx cw ON cw.company_period_id=e.qa_current_id LEFT JOIN wx lw ON lw.company_period_id=e.qa_lag_id')
    stage('INDEPENDENT_POSITION_PRODUCTS_AND_ALL_INTERVAL_AGGREGATES')
    for label,prefix,valid in [('current','current','qa_mapped'),('lagnews','lag','qa_mapped AND qa_previous_valid')]:
        value='qa_'+prefix+'_value';present='qa_'+prefix+'_present';status='qa_'+prefix+'_status'
        sqlcheck(label+':copied_source_and_unknown_sentiment','SELECT count(*) FROM ex e JOIN n USING(fund_id,interval_id,security_id) WHERE n.'+label+'_source_present IS DISTINCT FROM e.'+present+' OR n.'+label+'_news_known IS DISTINCT FROM coalesce(isfinite(e.'+value+'),false) OR n.'+label+'_news_status IS DISTINCT FROM e.'+status+' OR n.'+label+'_undated_audit_count IS DISTINCT FROM e.qa_'+prefix+'_undated OR '+different('n.'+label+'_news_value','e.'+value,'0','0'))
        eligible='qa_scope AND qa_action AND ('+valid+') AND ((coalesce(isfinite('+value+'),false) AND '+present+') OR B_Q=0)'
        product='CASE WHEN '+eligible+' THEN CASE WHEN B_Q=0 THEN 0. ELSE B_Q*'+value+' END END'
        weight='abs('+product+')'
        c.execute('CREATE TEMP TABLE exp_'+label+' AS SELECT *,coalesce('+eligible+',false) qa_E,'+product+' qa_product,'+weight+' qa_weight FROM ex')
        sqlcheck(label+':independent_E_not_old_paired_mask','SELECT count(*) FROM exp_'+label+' e JOIN n USING(fund_id,interval_id,security_id) WHERE n.'+label+'_usable_E IS DISTINCT FROM e.qa_E OR n.'+label+'_unknown_contribution IS DISTINCT FROM (e.qa_scope AND NOT e.qa_E)')
        sqlcheck(label+':zero_movement_unknown_news_flag','SELECT count(*) FROM exp_'+label+' e JOIN n USING(fund_id,interval_id,security_id) WHERE n.'+label+'_zero_product_from_measured_no_movement_unknown_news IS DISTINCT FROM (e.qa_E AND e.B_Q=0 AND NOT coalesce(isfinite(e.'+value+'),false))')
        sqlcheck(label+':products_and_weights_keep_unknown_NA','SELECT count(*) FROM exp_'+label+' e JOIN n USING(fund_id,interval_id,security_id) WHERE '+different('n.'+label+'_product','e.qa_product','1e-12','1e-10')+' OR '+different('n.'+label+'_weight','e.qa_weight','1e-12','1e-10'))
        c.execute('''CREATE TEMP TABLE agg_'''+label+''' AS SELECT fund_id,interval_id,count(*) positions,
            sum(abs(B_Q)) FILTER(WHERE isfinite(B_Q)) gross_all,
            count(*) FILTER(WHERE qa_E) observed,
            coalesce(sum(qa_product),0) numerator,coalesce(sum(qa_weight),0) denominator,
            coalesce(sum(abs(B_Q)) FILTER(WHERE qa_E),0) gross_E,
            count(DISTINCT company_id) FILTER(WHERE qa_E AND qa_weight>0) companies,
            coalesce(sum(abs(B_Q)) FILTER(WHERE qa_E AND coalesce(isfinite('''+value+'''),false) AND '''+value+'''<>0),0) covered,
            count(*) FILTER(WHERE qa_scope AND NOT qa_E) unknown_count,
            count(*) FILTER(WHERE qa_E AND B_Q<>0) moving,
            count(*) FILTER(WHERE qa_E AND '''+status+''' IN ('ZERO_NO_CANDIDATES_IN_FROZEN_CORPUS','ZERO_ALL_CANDIDATES_IRRELEVANT')) absence,
            count(*) FILTER(WHERE qa_E AND '''+status+'''='OBSERVED_SENTIMENT' AND '''+value+'''=0) balanced,
            count(*) FILTER(WHERE qa_E AND B_Q=0 AND NOT coalesce(isfinite('''+value+'''),false)) movement_zero_unknown_news,
            sha256(coalesce(string_agg(position_id,chr(10) ORDER BY position_id) FILTER(WHERE qa_E),'')) E_hash
            FROM exp_'''+label+' GROUP BY fund_id,interval_id')
        c.execute('CREATE TEMP VIEW cmp_'+label+' AS SELECT s.*,a.*,CASE WHEN positions IS NOT NULL THEN 1e-12*greatest(1.,gross_E) END qa_epsilon,CASE WHEN observed>0 THEN CASE WHEN denominator<=1e-12*greatest(1.,gross_E) THEN 0. ELSE numerator/denominator END END qa_score FROM scores s LEFT JOIN agg_'+label+' a USING(fund_id,interval_id)')
        sqlcheck(label+':every_score_NA_or_observed_zero_or_ratio','SELECT count(*) FROM cmp_'+label+' WHERE '+different('NTS_Q_'+label+'_relaxed','qa_score','1e-10','1e-12')+' OR '+label+'_eligible IS DISTINCT FROM coalesce(isfinite(qa_score),false)')
        for native,expected in [('numerator','numerator'),('denominator','denominator'),('epsilon','qa_epsilon'),('usable_positions','observed'),('gross_B_Q_E','gross_E'),('company_count','companies'),('coverage','covered/nullif(gross_all,0)'),('unknown_contribution_count','unknown_count')]:
            sqlcheck(label+':aggregate_'+native,'SELECT count(*) FROM cmp_'+label+' WHERE '+different(label+'_'+native,expected,'1e-10','1e-7' if native in ['numerator','denominator','gross_B_Q_E'] else '1e-12'))
        sqlcheck(label+':exact_E_hash','SELECT count(*) FROM cmp_'+label+' WHERE '+label+'_E_hash IS DISTINCT FROM E_hash')
        sqlcheck(label+':score_status_semantics','SELECT count(*) FROM cmp_'+label+" WHERE "+label+"_status IS DISTINCT FROM CASE WHEN positions IS NULL THEN 'NA_NO_APPLICABLE_SCOPE' WHEN observed=0 THEN 'NA_NO_USABLE_OBSERVED_SUBPORTFOLIO' WHEN denominator<=qa_epsilon THEN 'OBSERVED_SUBPORTFOLIO_ZERO' WHEN numerator=0 THEN 'OBSERVED_OFFSETTING_ALIGNMENT' ELSE 'OBSERVED_RATIO' END")
        for native,condition in [('zero_no_measured_movement','moving=0'),('zero_has_confirmed_news_absence','absence>0'),('zero_has_observed_neutral_balanced_signal','balanced>0'),('zero_has_unknown_news_no_movement','movement_zero_unknown_news>0')]:
            sqlcheck(label+':'+native,'SELECT count(*) FROM cmp_'+label+' WHERE positions IS NOT NULL AND '+label+'_'+native+' IS DISTINCT FROM (observed>0 AND denominator<=qa_epsilon AND '+condition+')')
        details[label+'_cases']=c.execute('SELECT count(*) intervals,count(*) FILTER(WHERE '+label+"_eligible) eligible,count(*) FILTER(WHERE "+label+"_status='OBSERVED_SUBPORTFOLIO_ZERO') observed_subset_zero,count(*) FILTER(WHERE "+label+"_status='OBSERVED_SUBPORTFOLIO_ZERO' AND unknown_count>0) supported_zero_with_unknown_rows,count(*) FILTER(WHERE observed=0) empty_E,count(*) FILTER(WHERE companies=1 AND denominator>qa_epsilon) one_company_ratio,count(*) FILTER(WHERE covered/nullif(gross_all,0)<.5 AND observed>0) eligible_below_old50pct,count(*) FILTER(WHERE movement_zero_unknown_news>0) intervals_with_zero_movement_unknown_news FROM cmp_"+label).fetchdf().to_dict('records')[0]
    sqlcheck('full_original_origin_keys_before_news_gate','SELECT count(*) FROM ((SELECT fund_id,interval_id FROM scores EXCEPT ALL SELECT fund_id,interval_id FROM original_origins) UNION ALL (SELECT fund_id,interval_id FROM original_origins EXCEPT ALL SELECT fund_id,interval_id FROM scores))')
    origin_fields=['opening_report_id','closing_report_id','start_date','end_date','origin_month','fund_focus','fund_focus_source','fund_focus_evidence','index_status','exclude_index_fund','universe_eligible','identity_origin_verified','row_id','explicit_incompatible_mandate']
    sqlcheck('old_universe_origin_identity_and_classification_preserved','SELECT count(*) FROM scores s JOIN original_origins o USING(fund_id,interval_id) WHERE '+' OR '.join('s.'+x+' IS DISTINCT FROM o.'+x for x in origin_fields))
    details['independent_current_lag_availability']=c.execute('SELECT count(*) FILTER(WHERE current_eligible AND NOT lagnews_eligible) current_only_known,count(*) FILTER(WHERE lagnews_eligible AND NOT current_eligible) lag_only_known,count(*) FILTER(WHERE current_eligible AND lagnews_eligible) both_known FROM scores').fetchdf().to_dict('records')[0]
    details['rows_checked']=dict(positions=c.execute('SELECT count(*) FROM n').fetchone()[0],origins=c.execute('SELECT count(*) FROM scores').fetchone()[0],windows=c.execute('SELECT count(*) FROM w').fetchone()[0],reconstructed_windows=c.execute('SELECT count(*) FROM missing').fetchone()[0],reconstructed_members=c.execute('SELECT count(*) FROM members').fetchone()[0])
    details['missing_identity_disclosure']=c.execute('SELECT count(*) FILTER(WHERE company_id IS NULL) unknown_company_rows,count(*) FILTER(WHERE qa_scope AND company_id IS NULL) unknown_company_in_quantity_scope FROM e0').fetchdf().to_dict('records')[0]
    details['prior_manifest_sha256']=PRIOR
    stage('FINAL_IMMUTABILITY_CHECKS')
    records(m,'after')
    ck('panel_manifest_unchanged',sha(PANEL/'manifest.json')==MANIFEST_SHA)
    for rel,expected in PRIOR.items():ck('old_manifest_after:'+rel,sha(BASE/'h3_alpha_20261104'/rel)==expected)


try:
    main()
    report=dict(status='PASS' if all(x['status']=='PASS' for x in checks) else 'FAIL',audit_type='EXTERNAL_INDEPENDENT_SEALED_PANEL_ONLY',panel=str(PANEL),panel_manifest_sha256=MANIFEST_SHA,spec_sha256=SPEC_SHA,audit_code_sha256=sha(__file__),production_module_imports=False,alternative_production_panel_created=False,regressions_run=0,new_coefficients_inspected=False,checks_pass=sum(x['status']=='PASS' for x in checks),checks_fail=sum(x['status']=='FAIL' for x in checks),checks=checks,details=details,completed_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),runtime_seconds=time.monotonic()-START,limitations=['This verifies the sealed score panel, not expanded alpha or the forthcoming 48 model cells.','Frozen B_Q is checked unchanged against the separately audited v3 quantity ledger; this bounded follow-up does not rerun raw portfolio price reconstruction.','Unknown company rows retain missing identity and are excluded; the producer outside_news_universe_count combines outside/unknown mapping, so it is not evidence that every excluded company is known outside the registry.','New/source input hashes and old manifest pins are checked before/after; full old-output rehash belongs to the final numeric audit.'])
    save('independent_panel_qa.json',report)
    save('summary.json',{k:v for k,v in report.items() if k!='checks'})
    stage('COMPLETED',status=report['status'],checks_pass=report['checks_pass'],checks_fail=report['checks_fail'])
    if report['checks_fail']:
        print('FAILURES',json.dumps([x for x in checks if x['status']=='FAIL'],ensure_ascii=False));raise SystemExit(1)
except Exception:
    save('execution_failure.json',dict(traceback=traceback.format_exc(),checks=checks,details=details));stage('EXECUTION_FAILED');raise
