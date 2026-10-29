"""Source-only operating points and cost selection; no external metrics accepted."""
import numpy as np


def source_point(labels,scores,fpr_limit):
    y=np.asarray(labels,dtype=int);s=np.asarray(scores,dtype=float)
    if len(y)!=len(s) or not np.isfinite(s).all() or set(np.unique(y))!={0,1}:
        raise ValueError('Finite source scores and both source classes required.')
    if not 0<=fpr_limit<=1:
        raise ValueError('FPR limit must be in [0,1].')
    n0=int((y==0).sum());n1=int((y==1).sum())
    order=np.argsort(-s,kind='stable');ys=y[order];ss=s[order]
    boundaries=np.r_[np.flatnonzero(ss[:-1]!=ss[1:]),len(ss)-1]
    fp=np.cumsum(ys==0)[boundaries];tp=np.cumsum(ys==1)[boundaries]
    thresholds=np.r_[np.nextafter(ss[0],np.inf),ss[boundaries]]
    fp=np.r_[0,fp];tp=np.r_[0,tp]
    allowed=np.flatnonzero(fp<=int(np.floor(fpr_limit*n0+1e-12)))
    best=max(allowed,key=lambda i:(int(tp[i]),-int(fp[i]),float(thresholds[i])))
    return dict(threshold=float(thresholds[best]),source_recall=float(tp[best]/n1),
        source_fpr=float(fp[best]/n0),source_fp=int(fp[best]),source_tp=int(tp[best]),
        source_benign=n0,source_attack=n1,fpr_limit=float(fpr_limit))


def select_cost(candidates):
    permitted={'candidate_id','benign_cost','threshold','source_recall','source_fpr','source_fp',
        'source_tp','source_benign','source_attack','fpr_limit'}
    if not candidates or any(set(c)-permitted for c in candidates):
        raise ValueError('Cost selection accepts source-calibration evidence only.')
    if any(c['source_fpr']>c['fpr_limit']+1e-12 for c in candidates):
        raise ValueError('A candidate violates the source FPR constraint.')
    return max(candidates,key=lambda c:(c['source_recall'],-c['source_fpr'],-c['benign_cost'],
        tuple(-ord(x) for x in c['candidate_id'])))
