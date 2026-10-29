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
    directory = ROOT/'data'/'raw'/name
    suffix = '*.parquet' if name == 'cic_iot_2023' else '*.csv'
    frames = []
    for path in sorted(directory.glob(suffix)):
        d = pd.read_parquet(path) if path.suffix == '.parquet' else pd.read_csv(path, low_memory=False)
        d['_uid'] = [f'{path.name}:{i}' for i in range(len(d))]
        d['_capture'] = path.name
        if name == 'cic_iomt_2024':
            # This mirror has CICFlowMeter rather than the original CIC packet-window schema.
            d['_attack'] = path.name.replace('_train.pcap_Flow.csv', '')
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
    raw_predictors = d.drop(columns=[c for c in metadata if c in d])
    native = raw_predictors.select_dtypes(include='number').astype(float)
    native = native.replace([np.inf, -np.inf], np.nan)
    # Full native predictor and label duplicates only; mapping collisions are never removed.
    keep = ~pd.DataFrame({'key': hashes(raw_predictors), 'y': y}).duplicated().to_numpy()
    d, native, y = d.iloc[keep].reset_index(drop=True), native.iloc[keep].reset_index(drop=True), y[keep]
    mapped = flowmeter_mapping(d) if name == 'cic_iomt_2024' else build_semantic_features(d,
        'label', name).drop(columns='label')
    audit = audit_projection(raw_predictors.iloc[keep].reset_index(drop=True), mapped, y)
    audit.update(dataset=name, seed=seed, sampled=len(keep), removed_exact_native_label_duplicates=int((~keep).sum()),
        benign=int((y == 0).sum()), attack=int((y == 1).sum()))
    return dict(name=name, native=native, x=mapped, y=y, uid=d['_uid'].to_numpy(),
        family=d['_attack'].map(family).to_numpy(), attack=d['_attack'].to_numpy(),
        device=(d['src_ip'].astype(str).to_numpy() if name == 'ton_iot' else
            d['Src IP'].astype(str).to_numpy() if name == 'cic_iomt_2024' else None),
        timestamp=(pd.to_datetime(d['Timestamp'], format='%d/%m/%Y %I:%M:%S %p', errors='coerce')
            if name == 'cic_iomt_2024' else None), audit=audit)


MAPPING = [
    ('duration', 'flow_duration', 'duration', 'Flow Duration / 1e6', 'seconds not established for CIC; proxy'),
    ('protocol_number', 'Protocol Type (window aggregate)', 'proto -> IANA number', 'Protocol', 'window mean versus flow code; not exact'),
    ('packet_total', 'Number (window aggregate)', 'src_pkts + dst_pkts', 'Total Fwd Packet + Total Bwd packets', 'window versus bidirectional flow'),
    ('bytes_total', 'Tot size', 'src_bytes + dst_bytes', 'Total Length of Fwd Packet + Total Length of Bwd Packet', 'CIC Tot size is not a flow byte total'),
    ('bytes_mean', 'AVG', '(src_bytes+dst_bytes)/(src_pkts+dst_pkts)', 'Packet Length Mean', 'packet/window/flow and payload definitions differ'),
    ('bytes_min', 'Min', 'min(src_bytes, dst_bytes)', 'Packet Length Min', 'TON is directional total, not packet minimum'),
    ('bytes_max', 'Max', 'max(src_bytes, dst_bytes)', 'Packet Length Max', 'TON is directional total, not packet maximum'),
    ('rate', 'Rate', '(src_bytes+dst_bytes)/duration', 'Flow Bytes/s', 'CIC rate and TON byte rate not dimensionally equivalent'),
    ('tcp', 'TCP (window proportion)', '1[proto=tcp]', '1[Protocol=6]', 'proportion versus flow indicator'),
    ('udp', 'UDP (window proportion)', '1[proto=udp]', '1[Protocol=17]', 'proportion versus flow indicator'),
    ('icmp', 'ICMP (window proportion)', '1[proto=icmp]', '1[Protocol=1]', 'proportion versus flow indicator'),
    ('iat_proxy', 'IAT (original units)', 'duration/(src_pkts+dst_pkts)', 'Flow IAT Mean / 1e6', 'not the same statistic; CIC units not harmonized'),
]


def write_manifest():
    files = []
    for directory in ['cic_iot_2023', 'ton_iot', 'cic_iomt_2024', 'ul_ece_2025']:
        for p in sorted((ROOT/'data'/'raw'/directory).rglob('*')):
            if p.is_file():
                files.append(dict(path=str(p.relative_to(ROOT)), bytes=p.stat().st_size,
                    sha256=file_digest(p)))
    (ROOT/'revision'/'results'/'data_manifest.json').write_text(json.dumps(files, indent=2), encoding='utf-8')
    pd.DataFrame(MAPPING, columns=['mapped_feature','CIC_IoT_2023','TON_IoT',
        'CICIoMT2024_flow_mirror','comparability_warning']).to_csv(ROOT/'revision'/'results'/'feature_mapping.csv', index=False)
