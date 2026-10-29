"""Distinguish implementation, executed evidence, blocked gates and deferred hardware."""
from datetime import datetime,timezone
import json
import pandas as pd
from common import HERE,ROOT,digest,log,save
from experiment import lock_protocol


def read(name):
    path=HERE/'results'/name
    return json.loads(path.read_text()) if path.exists() else None


def run():
    protocol=lock_protocol()
    common_manifest=HERE/'manifests/common_dataset.json'
    data=json.loads(common_manifest.read_text())
    extraction=read('extraction.json');raw=read('source_integrity.json')
    steps=[]
    def add(name,status,evidence,note=''):
        steps.append(dict(step=name,status=status,evidence=evidence,note=note))
    add('Pinned real raw data','verified' if raw['complete'] and len(raw['files'])==67 else 'incomplete',
        'results/source_integrity.json','Mirror object integrity is not independent official-release authenticity.')
    add('Chronology-safe common extraction','verified' if extraction['complete'] and len(extraction['captures'])==44 else 'incomplete',
        'results/extraction.json','Complete packet records only; incomplete source tails retained in originals and reported.')
    add('Physical features and cross-role observational overlap','verified' if read('prepared_audit.json') else 'not_executed',
        'results/prepared_audit.json','Observational identity is not proof of packet-byte identity or independent acquisition sessions.')
    add('Author-hosted event annotation import','verified' if read('official_event_integrity.json') else 'not_executed',
        'manifests/official_event_sources.json','Point timestamps do not acquire fabricated durations or implicit benign labels.')
    add('Common label quality gate','passed' if data['quality_gate_passed'] else 'failed',
        'manifests/common_dataset.json',f"Observed coverage {data['ton_label_coverage']:.6%}; required 95%; not classifier accuracy.")
    for track in ['legacy','common']:
        for title,file in [('Five-seed source-only cost experiment','result_audit_'+track+'.json'),
            ('Matched-cost teacher/supervised/KD/INT8 controls',track+'_ablation_audit.json'),
            ('Matched-backend float/INT8 CPU controls',track+'_matched_backend_cpu_audit.json'),
            ('External SHAP and descriptive PDP',track+'_xai_audit.json')]:
            result=read(file)
            add(track+': '+title,'verified' if result else 'implemented_not_executed','results/'+file,
                'Historical diagnostic, not a fresh common-flow test.' if track=='legacy' else 'No target metrics select source policies.')
    add('Common matched-source LR/RF controls','verified' if read('common_baseline_audit.json') else 'implemented_not_executed',
        'results/common_baseline_audit.json')
    for name in ['final_windows_tests','final_linux_feature_tests','final_linux_chronology_tests','official_event_tests']:
        result=read('step_'+name+'.json')
        add(name,'verified' if result and result['returncode']==0 else 'not_verified','results/step_'+name+'.json')
    add('Real Raspberry Pi/Jetson tests','deferred_by_user','', 'Not performed; desktop timings are not embedded-device evidence.')
    add('New experiment integration into manuscript/reviewer responses','not_completed','',
        'Original submission and previous working revision retained. New experiment outputs are not silently substituted for submitted results.')
    original_pdf=ROOT/'output/pdf/Manuscript.pdf'
    originals=(digest(original_pdf)=='3513f7e5d42b58ca97b64728f3cb720b02341769c6f0621bb9e285a50cc1d9dc') if original_pdf.exists() else None
    if originals is False:
        raise AssertionError('Original submitted PDF changed.')
    common_done=all(read(name) for name in ['result_audit_common.json','common_ablation_audit.json',
        'common_matched_backend_cpu_audit.json','common_xai_audit.json','common_baseline_audit.json'])
    payload=dict(checked_utc=datetime.now(timezone.utc).isoformat(),steps=steps,
        common_experiment_fully_verified=bool(common_done and data['quality_gate_passed']),
        all_revision_requests_completed=False,hardware_completed=False,submitted_original_unchanged=originals,
        immutable_training_protocol_sha256=protocol,common_data_manifest_sha256=digest(common_manifest),
        raw_flow_observations=sum(r['n_flows_observed'] for r in extraction['captures']),
        retained_hash_sample_flows=sum(r['n_flows_saved'] for r in extraction['captures']),
        note='A successful audit command means the status is checked, not that pending scientific work is complete.')
    save(HERE/'results/final_execution_checklist.json',payload)
    lines=['REVIZYON: GERCEK UYGULAMA VE KONTROL DURUMU',payload['checked_utc'],'',
        'Bu dosya calisma kontroludur; dergiye gonderilecek ZIP icin degildir.',
        'Eski tablo/proxy deneyi ile yeni ham-akis deneyi birbirine karistirilmamistir.',
        'Raspberry Pi / Jetson testi yapilmamistir. Tum revizyonlar tamamlandi denmemektedir.','']
    for item in steps:
        lines.append(item['status'].upper()+' | '+item['step'])
        if item['note']:
            lines.append('  '+item['note'])
        if item['evidence']:
            lines.append('  Kanit: '+item['evidence'])
    legacy=HERE/'results/distilled_external_summary.csv'
    if legacy.exists():
        lines.extend(['','DISTILLED STUDENT: TUM ONCEDEN BELIRLENMIS ISLETIM NOKTALARI',pd.read_csv(legacy).to_string(index=False)])
    cpu=HERE/'results/legacy_matched_backend_cpu.csv'
    if cpu.exists():
        table=pd.read_csv(cpu)
        lines.extend(['','LEGACY: AYNI TORCHSCRIPT ALTYAPISI, CPU1, BATCH1',
            table.groupby('family')[['model_ms_median','state_dict_bytes','exported_script_bytes']].mean().to_string(),
            'Boyut azalmasi, hizlanma veya dusuk model-ozel RAM kullanimi kaniti degildir. RSS tum surece aittir.'])
    lines.extend(['','SONRAKI ADIM',
        'Common-flow kalite esigi gecmediyse egitim acilmaz; etiketler tahmin edilmez ve dusuk kapsama sahip final kaydi silinmez.',
        'Tum bilimsel sonuclar dogrulandiktan sonra makale ve her hakemin ayri yaniti kanitlarla guncellenmelidir.'])
    (HERE/'results/UYGULAMA_KONTROLU.txt').write_text('\n'.join(lines),encoding='utf-8')
    log('CHECKLIST WRITTEN: executed/verified, failed and pending stages are distinguished; not a claim all revisions are finished')


if __name__=='__main__':
    run()
