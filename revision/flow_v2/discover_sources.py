"""Discover actual public PCAP objects; do not equate a dataset title with raw data."""
from urllib.parse import quote
import json
import requests
from common import HERE,log,save

SESSION=requests.Session()
SESSION.headers['User-Agent']='LITE-XIDS-data-provenance/2.0'


def hf_tree(repository):
    info=SESSION.get('https://huggingface.co/api/datasets/'+repository,timeout=30)
    info.raise_for_status()
    commit=info.json()['sha']
    url='https://huggingface.co/api/datasets/'+repository+'/tree/'+commit+'?recursive=true&limit=1000'
    entries=[]
    while url:
        response=SESSION.get(url,timeout=60)
        response.raise_for_status()
        entries.extend(response.json())
        url=response.links.get('next',{}).get('url')
    pcaps=[dict(repository=repository,revision=commit,path=r['path'],bytes=r.get('size'),
        sha256=r.get('lfs',{}).get('oid'),url='https://huggingface.co/datasets/'+repository+
        '/resolve/'+commit+'/'+quote(r['path'],safe='/')) for r in entries
        if r['type']=='file' and r['path'].lower().endswith(('.pcap','.pcapng','.pcap.gz'))]
    save(HERE/'manifests'/('hf_'+repository.replace('/','_')+'.json'),dict(repository=repository,revision=commit,pcaps=pcaps))
    log(f'DISCOVER HF {repository}: {len(pcaps)} PCAP objects, {sum(r["bytes"] or 0 for r in pcaps)/2**30:.2f} GiB')
    return pcaps


def main():
    log('STAGE discovery START: old study preserved; looking for raw captures and independent label metadata')
    result={}
    for name,url in [
        ('cic_official_form','https://cicresearch.ca/IOTDataset/CIC_IOT_Dataset2023/'),
        ('ton_official','https://research.unsw.edu.au/projects/toniot-datasets')]:
        try:
            response=SESSION.get(url,timeout=30)
            result[name]=dict(status=response.status_code,url=url,content_bytes=len(response.content))
            save(HERE/'cache'/(name+'.json'),dict(**result[name],text=response.text))
            log(f'PROBE {name}: HTTP {response.status_code}')
        except requests.RequestException as error:
            result[name]=dict(error=type(error).__name__)
            log(f'PROBE {name}: {type(error).__name__}')
    for repository in ['somnath0100/CICIoT2023Small','codymlewis/TON_IoT_network']:
        try:
            result[repository]=dict(pcaps=len(hf_tree(repository)))
        except (requests.RequestException,KeyError) as error:
            result[repository]=dict(error=str(error))
            log(f'PROBE {repository}: {type(error).__name__}')
    search=SESSION.get('https://huggingface.co/api/datasets',params={'search':'ton_iot','limit':100},timeout=30)
    search.raise_for_status()
    result['ton_hf_repositories']=[row['id'] for row in search.json()]
    save(HERE/'manifests/discovery.json',result)
    log('STAGE discovery FINISHED: manifests contain exact raw objects, not assumed row labels')


if __name__=='__main__':
    main()
