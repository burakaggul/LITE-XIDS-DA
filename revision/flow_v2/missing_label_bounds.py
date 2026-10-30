"""Sharp metric bounds over all assignments of unknown binary ground truth."""
import numpy as np


def confusion_metrics(tn,fp,fn,tp):
    ratio=lambda a,b:a/b if b else np.nan
    return dict(fpr=ratio(fp,tn+fp),recall=ratio(tp,tp+fn),precision=ratio(tp,tp+fp),
        balanced_accuracy=(ratio(tn,tn+fp)+ratio(tp,tp+fn))/2,
        macro_f1=(ratio(2*tn,2*tn+fp+fn)+ratio(2*tp,2*tp+fp+fn))/2,
        accuracy=ratio(tn+tp,tn+fp+fn+tp))


def metric_bounds(tn,fp,fn,tp,unknown_positive,unknown_negative):
    values=[tn,fp,fn,tp,unknown_positive,unknown_negative]
    if any(int(v)!=v or v<0 for v in values):
        raise ValueError('Nonnegative integer confusion/prediction counts are required.')
    if tn+fp==0 or tp+fn==0:
        raise ValueError('Both known true classes are required for these metric bounds.')
    u1,u0=int(unknown_positive),int(unknown_negative)
    correct=confusion_metrics(tn+u0,fp,fn,tp+u1)
    incorrect=confusion_metrics(tn,fp+u1,fn+u0,tp)
    result={}
    for name in correct:
        a,b=correct[name],incorrect[name]
        result[name+'_lower']=min(a,b)
        result[name+'_upper']=max(a,b)
    result.update(known_rows=int(tn+fp+fn+tp),unknown_rows=u0+u1,unknown_alarms=u1,
        unknown_no_alarms=u0,full_sample_rows=int(sum(values)),
        missing_labels_imputed=False,
        interpretation='Bounds permit every binary assignment of the unobserved labels, including nonrandom missingness. They are not confidence intervals and no assignments enter training or selection.')
    return result
