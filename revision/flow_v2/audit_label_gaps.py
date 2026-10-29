"""Diagnose absent original ground truth without assigning inferred attack labels."""
import json
import duckdb
import pandas as pd
from common import HERE,digest,log,save
from label_flows import canonical_keys


def audit():
    manifest=json.loads((HERE/'manifests/common_dataset.json').read_text())
    config=json.loads((HERE/'config.json').read_text())
    unknown=pd.read_parquet(HERE/'prepared/unresolved_labels.parquet')
    unknown['key']=canonical_keys(unknown)
    connection=duckdb.connect();connection.execute('SET threads=4')
    connection.register('unknown_keys',unknown[['key']].drop_duplicates())
    labels=list(map(str,(HERE/'label_index').glob('*.parquet')))
    if manifest.get('official_event_sources_sha256'):
        path=HERE/'manifests/official_event_sources.json'
        if digest(path)!=manifest['official_event_sources_sha256']:
            raise AssertionError('Official source manifest changed after label preparation.')
        official=json.loads(path.read_text())
        labels.extend(str(HERE/row['index_path']) for row in official['files'])
    support=connection.execute('''SELECT l.key,count(*) AS original_records_any_time,
        min(l.start_s) AS earliest_original_start_s,max(l.end_s) AS latest_original_end_s
        FROM read_parquet(?) l SEMI JOIN unknown_keys u ON l.key=u.key GROUP BY l.key''',[labels]).df()
    connection.close()
    details=unknown[['flow_uid','capture_id','role','protocol_number','label_match_status','key']].merge(
        support,on='key',how='left',validate='many_to_one')
    details['original_records_any_time']=details.original_records_any_time.fillna(0).astype(int)
    details['key_present_any_time']=details.original_records_any_time.gt(0)
    details.drop(columns=['key']).to_parquet(HERE/'results/unresolved_label_support.parquet',index=False)
    counts=details.groupby(['capture_id','role','label_match_status','key_present_any_time']).size().reset_index(name='rows')
    counts.to_csv(HERE/'results/unresolved_label_support_summary.csv',index=False)
    coverage=pd.read_csv(HERE/'results/ton_label_coverage.csv')
    coverage['unresolved']=coverage.total-coverage.labeled
    roles=coverage.groupby('role')[['total','labeled','unresolved']].sum()
    roles['coverage']=roles.labeled/roles.total
    roles.to_csv(HERE/'results/ton_label_coverage_by_role.csv')
    largest=coverage.sort_values('unresolved',ascending=False).iloc[0]
    if len(unknown)!=manifest['unknown_labels_excluded']:
        raise AssertionError('Unresolved record count changed.')
    measured=float(coverage.labeled.sum()/coverage.total.sum())
    if abs(measured-manifest['ton_label_coverage'])>1e-12:
        raise AssertionError('Label coverage cannot be independently reproduced.')
    absent=int((~details.key_present_any_time).sum())
    result=dict(required_coverage=config['required_ton_label_coverage'],observed_coverage=measured,
        annotation_tables_checked=len(labels),official_point_tables_included=bool(manifest.get('official_event_sources_sha256')),
        primary_quality_gate_passed=manifest['quality_gate_passed'],unknown_records=len(unknown),
        unknown_without_same_tuple_at_any_original_time=absent,
        unknown_with_tuple_but_no_complete_consistent_time_support=len(unknown)-absent,
        largest_gap_capture=largest.capture_id,largest_gap_records=int(largest.unresolved),
        largest_gap_coverage=float(largest.coverage),
        largest_gap_share_of_unknown=float(largest.unresolved/len(unknown)),
        prepared_data_manifest_sha256=digest(HERE/'manifests/common_dataset.json'),
        labels_inferred_or_modified=False,threshold_relaxed=False,capture_roles_changed=False,
        common_models_trained=any((HERE/'models/common').rglob('frozen.json')),
        cause_limit='Absent tuple support establishes unavailable matching table annotations; it does not establish which dataset release, extraction filtering or capture provenance caused the absence.',
        official_source='https://research.unsw.edu.au/projects/toniot-datasets',
        missing_resource='Official SecurityEvents_GroundTruth_datasets and capture-to-event/clock mapping, or an independently validated complete matching annotation release.',
        prohibited_workarounds=['filename-derived labels','nearest-record labels','unverified attacker-IP rules',
            'relaxing the frozen coverage threshold','dropping the low-coverage final capture'],
        next_action='Obtain authoritative matching ground truth; validate its mapping without inspecting common model performance; rerun the unchanged quality gate before common fitting.',
        not_a_classifier_performance_measure=True,code_sha256=digest(__file__))
    save(HERE/'results/label_gap_audit.json',result)
    log(f'LABEL GAP AUDIT: {len(unknown):,} unresolved; {absent:,} lack the same tuple in all original tables at any timestamp')
    log('COMMON EXPERIMENT REMAINS STOPPED: no inferred labels, no lowered coverage threshold, no removed final capture')


if __name__=='__main__':
    audit()
