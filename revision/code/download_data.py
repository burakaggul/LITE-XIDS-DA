"""Download public sources only. Dataset files are not redistributed in the code repository."""
from pathlib import Path
import hashlib
import json
import urllib.request
import zipfile

ROOT=Path(__file__).resolve().parents[2]
SOURCES={
    'cic_iot_2023':('lacg030175/CIC-IoT-2023-neto-subsample',[
        'data/train-00000-of-00001.parquet','data/test-00000-of-00001.parquet']),
    'ton_iot':('codymlewis/TON_IoT_network',['train_test_network.csv']),
    'cic_iomt_2024':('somnath0100/CICIoMT2024Small',[
        'CSVS/Benign_train.pcap_Flow.csv','CSVS/ARP_Spoofing_train.pcap_Flow.csv',
        'CSVS/MQTT-Malformed_Data_train.pcap_Flow.csv','CSVS/Recon-VulScan_train.pcap_Flow.csv',
        'CSVS/TCP_IP-DDoS-UDP1_train.pcap_Flow.csv','CSVS/TCP_IP-DoS-UDP1_train.pcap_Flow.csv'])}


def main():
    records=[]
    for name,(repository,files) in SOURCES.items():
        info=json.load(urllib.request.urlopen('https://huggingface.co/api/datasets/'+repository))
        commit=info['sha']
        for file in files:
            url=f'https://huggingface.co/datasets/{repository}/resolve/{commit}/{file}'
            out=ROOT/'data'/'raw'/name/Path(file).name
            out.parent.mkdir(parents=True,exist_ok=True)
            if not out.exists():
                urllib.request.urlretrieve(url,out)
            records.append(dict(dataset=name,url=url,path=str(out.relative_to(ROOT)),
                sha256=hashlib.sha256(out.read_bytes()).hexdigest()))
    out=ROOT/'data'/'raw'/'ul_ece_2025'
    out.mkdir(parents=True,exist_ok=True)
    archive=out/'dataset.zip'
    url='https://zenodo.org/api/records/15305814/files/mirzaakhi/UL-ECE-DDoS-H-IoT-Datasets2025-v1.0.0.zip/content'
    if not archive.exists():
        urllib.request.urlretrieve(url,archive)
    if hashlib.md5(archive.read_bytes()).hexdigest()!='c830a2eb46bd16faa3a0dd4e88e38f9a':
        raise ValueError('Zenodo checksum mismatch')
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            target=(out/member.filename).resolve()
            if not target.is_relative_to(out.resolve()):
                raise ValueError('Unsafe archive member')
        z.extractall(out)
    records.append(dict(dataset='UL-ECE-DDoS-H-IoT2025',url=url,
        sha256=hashlib.sha256(archive.read_bytes()).hexdigest()))
    (ROOT/'revision'/'results').mkdir(parents=True,exist_ok=True)
    (ROOT/'revision'/'results'/'download_provenance.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    print('Public data downloaded; verify source descriptions and licenses before reuse.')


if __name__=='__main__':
    main()
