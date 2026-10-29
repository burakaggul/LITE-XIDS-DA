from __future__ import annotations

import json
import hashlib
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from protocol import audit_projection, hashes, binary_labels

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from litexids.data.semantic_mapping import build_semantic_features


def family(text):
    s = str(text).lower()
    if s in ['0', 'normal', 'benign', 'benigntraffic'] or 'benign' in s:
        return 'benign'
    if any(k in s for k in ['ddos', 'dos', 'flood']):
        return 'flooding'
    if any(k in s for k in ['recon', 'scan']):
        return 'reconnaissance'
    if any(k in s for k in ['password', 'brute', 'dictionary']):
        return 'brute_force'
    if any(k in s for k in ['spoof', 'mitm']):
        return 'spoofing_mitm'
    if any(k in s for k in ['xss', 'injection', 'web', 'upload', 'browser', 'sql']):
        return 'web_injection'
    if 'mirai' in s:
        return 'mirai'
    if 'backdoor' in s:
        return 'backdoor'
    if 'malformed' in s:
        return 'mqtt_malformed'
    return s


def file_digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def load_pool(name):
    directory = ROOT/'data'/'raw'/('cic_iomt_2024_original_schema' if name=='cic_iomt_2024' else name)
    suffix = '*.parquet' if name == 'cic_iot_2023' else '*.csv'
    frames = []
    for path in sorted(directory.rglob(suffix)):
        d = pd.read_parquet(path) if path.suffix == '.parquet' else pd.read_csv(path, low_memory=False)
        d['_uid'] = [f'{path.name}:{i}' for i in range(len(d))]
        d['_capture'] = path.name
        if name == 'cic_iomt_2024':
            # The source release supplies capture-category files, not per-row verified labels.
            d['_attack'] = path.name.replace('_train.pcap.csv', '').replace('_test.pcap.csv','')
            d['label'] = int(not path.name.startswith('Benign'))
        else:
            d['_attack'] = d['Label_orig'] if name == 'cic_iot_2023' else d['type']
        frames.append(d)
    if not frames:
        raise FileNotFoundError(directory)
    return pd.concat(frames, ignore_index=True)


def flowmeter_mapping(d):
    z = pd.DataFrame(index=d.index)
    z['duration'] = d['Flow Duration']/1e6
    z['protocol_number'] = d['Protocol']
    z['packet_total'] = d['Total Fwd Packet']+d['Total Bwd packets']
    z['bytes_total'] = d['Total Length of Fwd Packet']+d['Total Length of Bwd Packet']
    z['bytes_mean'] = d['Packet Length Mean']
    z['bytes_min'] = d['Packet Length Min']
    z['bytes_max'] = d['Packet Length Max']
    z['rate'] = d['Flow Bytes/s']
    for col, value in [('tcp', 6), ('udp', 17), ('icmp', 1)]:
        z[col] = (d['Protocol'] == value).astype(float)
    z['iat_proxy'] = d['Flow IAT Mean']/1e6
    return z.replace([np.inf, -np.inf], np.nan)


def prepare(name, pool, n, seed):
    d = pool.sample(n=min(n, len(pool)), random_state=seed).copy()
    y = binary_labels(d['label'])
    metadata = {'label', 'Label', 'Label_orig', 'attack_class', 'type', 'Flow ID',
        'Src IP', 'Dst IP', 'Src Port', 'Dst Port', 'Timestamp',
        'src_ip', 'dst_ip', 'src_port', 'dst_port', '_uid', '_capture', '_attack'}
    predictors = d.drop(columns=[c for c in metadata if c in d])
    native = predictors.copy()
    if name=='ton_iot':
        native=native.drop(columns=['proto'])
        protocol=d['proto'].astype(str).str.lower()
        native['native_protocol_code']=protocol.map({'tcp':6.,'udp':17.,'icmp':1.}).fillna(0.)
        for value in ['tcp','udp','icmp']:
            native['native_protocol_'+value]=protocol.eq(value).astype(float)
    native = native.replace([np.inf, -np.inf], np.nan)
    original_columns = [c for c in d.columns if not c.startswith('_')]
    # Identity fields are excluded from models, not from raw-record identity checks.
    identity_columns = original_columns+(['_capture'] if name=='cic_iomt_2024' else [])
    keep = ~d[identity_columns].duplicated().to_numpy()
    label_fields = {'label', 'Label', 'Label_orig', 'attack_class', 'type'}
    raw_predictors = d[[c for c in original_columns if c not in label_fields]]
    if name=='cic_iomt_2024':
        raw_predictors=raw_predictors.assign(capture=d['_capture'])
    d, native, y = d.iloc[keep].reset_index(drop=True), native.iloc[keep].reset_index(drop=True), y[keep]
    mapped = build_semantic_features(d,'label',name).drop(columns='label')
    audit = audit_projection(raw_predictors.iloc[keep].reset_index(drop=True), mapped, y)
    proxy_counts = pd.crosstab(hashes(mapped), y)
    audit['deterministic_proxy_accuracy_ceiling'] = float(proxy_counts.max(axis=1).sum()/len(y))
    audit.update(dataset=name, seed=seed, sampled=len(keep), removed_exact_native_label_duplicates=int((~keep).sum()),
        benign=int((y == 0).sum()), attack=int((y == 1).sum()))
    return dict(name=name, native=native, x=mapped, y=y, uid=d['_uid'].to_numpy(),
        family=d['_attack'].map(family).to_numpy(), attack=d['_attack'].to_numpy(),
        device=(d['src_ip'].astype(str).to_numpy() if name == 'ton_iot' else
            d['Src IP'].astype(str).to_numpy() if 'Src IP' in d else None),
        capture=d['_capture'].to_numpy() if name=='cic_iomt_2024' else None,
        timestamp=None, audit=audit)


MAPPING = [
    ('duration', 'flow_duration', 'duration', 'unavailable -> 0', 'CIC units not established; missing in IoMT'),
    ('protocol_number', 'Protocol Type (window aggregate)', 'proto -> IANA number', 'Protocol Type', 'window aggregate versus flow code'),
    ('packet_total', 'Number (window aggregate)', 'src_pkts + dst_pkts', 'Number', 'different aggregation units/windows'),
    ('bytes_total', 'Tot size', 'src_bytes + dst_bytes', 'Tot size', 'CIC/IoMT size statistic is not a flow byte total'),
    ('bytes_mean', 'AVG', '(src_bytes+dst_bytes)/(src_pkts+dst_pkts)', 'AVG', 'window versus flow mean; byte definitions differ'),
    ('bytes_min', 'Min', 'min(src_bytes, dst_bytes)', 'Min', 'TON directional total, not packet minimum'),
    ('bytes_max', 'Max', 'max(src_bytes, dst_bytes)', 'Max', 'TON directional total, not packet maximum'),
    ('rate', 'Rate', '(src_bytes+dst_bytes)/duration', 'Rate', 'packet/window rate versus byte rate'),
    ('tcp', 'TCP (window proportion)', '1[proto=tcp]', 'TCP', 'window proportion versus flow indicator'),
    ('udp', 'UDP (window proportion)', '1[proto=udp]', 'UDP', 'window proportion versus flow indicator'),
    ('icmp', 'ICMP (window proportion)', '1[proto=icmp]', 'ICMP', 'window proportion versus flow indicator'),
    ('iat_proxy', 'IAT (original units)', 'duration/(src_pkts+dst_pkts)', 'IAT (original units)', 'units/statistic not harmonized; omitted in matched-window control'),
]


def write_manifest():
    files = []
    for directory in ['cic_iot_2023', 'ton_iot', 'cic_iomt_2024_original_schema', 'ul_ece_2025']:
        for p in sorted((ROOT/'data'/'raw'/directory).rglob('*')):
            if p.is_file():
                files.append(dict(path=str(p.relative_to(ROOT)), bytes=p.stat().st_size,
                    sha256=file_digest(p)))
    (ROOT/'revision'/'results'/'data_manifest.json').write_text(json.dumps(files, indent=2), encoding='utf-8')
    pd.DataFrame(MAPPING, columns=['mapped_feature','CIC_IoT_2023','TON_IoT',
        'CICIoMT2024_packet_window','comparability_warning']).to_csv(ROOT/'revision'/'results'/'feature_mapping.csv', index=False)
