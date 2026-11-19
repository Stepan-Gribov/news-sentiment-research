"""Content-addressed preparation provenance and exact W1/R1 classification binding."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from title_name_regressions_core import IntegrityError,sha256_file


PREPARATION_SCHEMA=2
APPROVED_RESULT_MODEL="gemma4_e4b_q4qat"
APPROVED_PROMPT_VERSION="target-financial-sentiment-ru-v1-chat"
PREPARATION_MODULES=("run_title_name_regressions.py","title_name_regressions_core.py",
    "title_name_regressions_data.py","title_name_regressions_news.py","title_name_regressions_integrity.py",
    "title_name_regressions_crosswalk.py")
PREPARED_FILES=("catalog_sources.json","news_candidates.parquet","event_cluster_audit.parquet",
    "event_representatives.parquet","holdings_inventory.parquet","report_quality.parquet",
    "all_true_intervals.parquet","issuer_security_conflicts.csv","company_month_states.parquet",
    "news_state_audit.csv","w2_edge_selection_audit.parquet","w2_input_metadata.json",
    "scheduled_corporate_events.parquet","temporal_exclusion_audit.csv")


def content_fingerprint(value):
    raw=json.dumps(value,ensure_ascii=True,sort_keys=True,separators=(",",":"),allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def file_snapshot(path,*,optional=False):
    path=Path(path)
    if not path.is_file():
        if optional:return dict(path=str(path),exists=False)
        raise IntegrityError("Required provenance file is absent: "+str(path))
    before=path.stat()
    digest=sha256_file(path)
    after=path.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
        raise IntegrityError("Input changed while it was fingerprinted: "+str(path))
    return dict(path=str(path),exists=True,bytes=before.st_size,mtime_ns=before.st_mtime_ns,sha256=digest)


def preparation_identity(spec_path,audit_path,source_paths,catalog_description,llm_binding,code_directory):
    dependencies={name:sha256_file(Path(code_directory)/name) for name in PREPARATION_MODULES}
    sources=[file_snapshot(p) for p in source_paths]
    db_path=Path(llm_binding["database_path"])
    sources.append(file_snapshot(str(db_path)+"-wal",optional=True))
    payload=dict(schema=PREPARATION_SCHEMA,spec_sha256=sha256_file(spec_path),audit_sha256=sha256_file(audit_path),
        preparation_code_hashes=dependencies,catalog_fingerprint=content_fingerprint(catalog_description),
        sources=sources,llm_binding=llm_binding)
    payload["fingerprint"]=content_fingerprint(payload)
    return payload


def seal_preparation(root,identity):
    root=Path(root)
    return dict(identity=identity,derived_output_hashes={name:sha256_file(root/name) for name in PREPARED_FILES})


def validate_preparation(root,state,current_identity):
    """No legacy cache is silently upgraded or stamped with newer code hashes."""
    provenance=state.get("preparation_provenance")
    if not provenance or provenance.get("identity",{}).get("schema")!=PREPARATION_SCHEMA:
        raise IntegrityError("Legacy preparation has no verified transformation provenance; use a fresh versioned output directory")
    if provenance["identity"]!=current_identity:
        raise IntegrityError("Preparation code/specification/audit/catalog/source fingerprint changed; use a fresh versioned output directory")
    expected=provenance.get("derived_output_hashes",{})
    if set(expected)!=set(PREPARED_FILES):
        raise IntegrityError("Preparation output fingerprint set is incomplete")
    for name,digest in expected.items():
        if not (Path(root)/name).is_file() or sha256_file(Path(root)/name)!=digest:
            raise IntegrityError("Prepared output changed or disappeared: "+name)
    return True


def verified_llm_binding(spec,audit,input_manifest,run_manifest,*,input_manifest_path,run_manifest_path):
    if not audit.get("audit_execution_complete") or audit.get("status")!="AUDIT_COMPLETE" or audit.get("qa_refinement_in_progress"):
        raise IntegrityError("A completed, final W1 audit is required for preparation")
    if audit.get("llm",{}).get("status")!="READY" or audit["llm"].get("integrity_failures"):
        raise IntegrityError("The exact audited LLM input did not pass W1")
    observed=spec["observed_input_audit"]
    audited=audit["llm"]
    input_digest=observed["input_sha256"]
    if not (input_manifest.get("input_sha256")==run_manifest.get("input_sha256")==audited["input"]["sha256"]==input_digest):
        raise IntegrityError("Frozen R1, W1 and actual input/run manifests refer to different input contents")
    if input_manifest.get("total")!=observed["mapping_pairs"]:
        raise IntegrityError("Input count differs from frozen R1")
    expected_run=audited["run_manifest.json"]["content"]
    for key in ["prompt_version","prompt_sha256"]:
        if run_manifest.get(key)!=observed[key] or run_manifest.get(key)!=expected_run.get(key):
            raise IntegrityError("Actual LLM prompt identity is not the frozen R1/W1 identity: "+key)
    if run_manifest.get("prompt_version")!=APPROVED_PROMPT_VERSION:
        raise IntegrityError("Unsupported prompt alias")
    model_name=str(run_manifest.get("model",{}).get("path","")).replace("\\","/").rsplit("/",1)[-1]
    if model_name!=observed["model_file"] or run_manifest.get("model")!=expected_run.get("model"):
        raise IntegrityError("Actual LLM model metadata differs from frozen R1/W1")
    for actual_path,key in [(input_manifest_path,"input_manifest.json"),(run_manifest_path,"run_manifest.json")]:
        if sha256_file(actual_path)!=audited[key]["file"]["sha256"]:
            raise IntegrityError("Actual LLM manifest bytes differ from the audited file: "+key)
    logical=audited.get("sqlite",{}).get("logical_results_sha256")
    if not isinstance(logical,str) or len(logical)!=64:
        raise IntegrityError("W1 lacks a WAL-aware logical classification digest")
    return dict(input_sha256=input_digest,logical_results_sha256=logical,
        prompt_version=observed["prompt_version"],prompt_sha256=observed["prompt_sha256"],
        model_identifier=APPROVED_RESULT_MODEL,model_manifest=run_manifest["model"],
        run_manifest_sha256=audited["run_manifest.json"]["file"]["sha256"],
        input_manifest_sha256=audited["input_manifest.json"]["file"]["sha256"],
        database_path=spec["resources"]["llm_database"],
        model_weights_content_hash_verified=False)
