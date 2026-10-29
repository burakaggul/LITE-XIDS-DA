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
}
REVISIONS={'cic_iot_2023':'049cd63fb1ea66e6d6e285b638ebaf145c6de562',
    'ton_iot':'b953d9388d2db3bc392ea7150476e4f219a2a1cb'}
EXPECTED={'train-00000-of-00001.parquet':'bcc477f125aa7cecbeba50cd3b9641e20dcf521e882c437a704f73c30c96b353',
    'test-00000-of-00001.parquet':'16eb6b12e6443a270a09a90c08adf90279d0bcb374f2346f143ac069df99ac6b',
    'train_test_network.csv':'26ddc513552de36de6428b2e578efaed2b57504c716dfba847cc0109a64e1974'}


def main():
    records=[]
    for name,(repository,files) in SOURCES.items():
        commit=REVISIONS[name]
        for file in files:
            url=f'https://huggingface.co/datasets/{repository}/resolve/{commit}/{file}'
            out=ROOT/'data'/'raw'/name/Path(file).name
            out.parent.mkdir(parents=True,exist_ok=True)
            if not out.exists():
                urllib.request.urlretrieve(url,out)
            if hashlib.sha256(out.read_bytes()).hexdigest()!=EXPECTED[out.name]:
                raise ValueError('Pinned source hash mismatch: '+out.name)
            records.append(dict(dataset=name,url=url,path=str(out.relative_to(ROOT)),
                sha256=hashlib.sha256(out.read_bytes()).hexdigest()))
    iomtdir=ROOT/'data'/'raw'/'cic_iomt_2024_original_schema'
    iomtdir.mkdir(parents=True,exist_ok=True)
    iomtarchive=ROOT/'revision'/'inputs'/'CICIoMT2024_original_schema.zip'
    iomtarchive.parent.mkdir(parents=True,exist_ok=True)
    iomturl='https://www.kaggle.com/api/v1/datasets/download/outstandard09/ciciomt2024?datasetVersionNumber=1'
    if not iomtarchive.exists():
        urllib.request.urlretrieve(iomturl,iomtarchive)
    if hashlib.sha256(iomtarchive.read_bytes()).hexdigest()!='7903996f5861881000df917ee77814f27ed4330610b487a93f1b4adce03b72bb':
        raise ValueError('CICIoMT2024 archive hash mismatch')
    with zipfile.ZipFile(iomtarchive) as z:
        for member in z.infolist():
            if not (iomtdir/member.filename).resolve().is_relative_to(iomtdir.resolve()):
                raise ValueError('Unsafe CICIoMT archive member')
        if not any(iomtdir.rglob('*.csv')):
            z.extractall(iomtdir)
    records.append(dict(dataset='CICIoMT2024',url=iomturl,
        sha256=hashlib.sha256(iomtarchive.read_bytes()).hexdigest(),
        labels='capture categories encoded by published filenames, not independently verified row labels'))
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
