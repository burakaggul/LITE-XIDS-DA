"""Resume only independently matching certificate inputs and completed captures."""
import json
import pandas as pd
from common import HERE,digest,log,save
import packet_label_witness as witness


def run():
    directory=HERE/'cache/packet_witness_v3'
    manifest=json.loads((directory/'manifest.json').read_text());input_sha=digest(directory/'manifest.json')
    code_sha=digest(witness.__file__);records=[]
    for row in manifest['captures']:
        output=HERE/'results/packet_witness_v3'/(row['key']+'.csv');metadata=output.with_suffix('.json')
        chronology=json.loads((HERE/'cache/chronological/ton_iot'/(row['key']+'.json')).read_text())
        valid=False
        if output.exists() and metadata.exists():
            old=json.loads(metadata.read_text())
            valid=old['code_sha256']==code_sha and old['input_manifest_sha256']==input_sha and old['normalized_source_sha256']==chronology['normalized_sha256']
            if valid:
                actual=pd.read_csv(output);expected=pd.read_csv(directory/(row['key']+'_flows.csv.gz')).drop_duplicates('flow_uid')
                valid=len(actual)==old['candidates'] and set(actual.flow_uid)==set(expected.flow_uid) and int(actual.certified.sum())==old['certified']
        if not valid:
            witness.run_capture(row['capture_id'])
        else:
            log('PACKET WITNESS COMPLETED CAPTURE VERIFIED '+row['capture_id'])
        info=json.loads(metadata.read_text())
        records.append(dict(capture_id=row['capture_id'],csv=str(output.relative_to(HERE)).replace('\\','/'),
            csv_sha256=digest(output),metadata=str(metadata.relative_to(HERE)).replace('\\','/'),
            metadata_sha256=digest(metadata),candidates=info['candidates'],certified=info['certified']))
        save(HERE/'results/all_packet_witnesses_v3.json',dict(captures=records,complete=len(records)==len(manifest['captures']),
            input_manifest_sha256=input_sha,source_code_sha256=code_sha))
    log(f'ALL PACKET WITNESSES VERIFIED: {sum(r["certified"] for r in records):,}/{sum(r["candidates"] for r in records):,} supported flows; rejected records remain unknown')


if __name__=='__main__':
    run()
