"""Held-out end-to-end dice counts, face accuracy, tray accuracy, and abstention."""
import argparse
import json
from pathlib import Path
from training.common import safe_image, read_records


def iou(a,b):
    w=max(0,min(a['x']+a['w'],b['x']+b['w'])-max(a['x'],b['x']))
    h=max(0,min(a['y']+a['h'],b['y']+b['h'])-max(a['y'],b['y']))
    intersection=w*h
    return intersection/(a['w']*a['h']+b['w']*b['h']-intersection)


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepared',required=True)
    p.add_argument('--detector',required=True);p.add_argument('--classifier',required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--threshold',type=float,default=.8)
    p.add_argument('--allow-smoke',action='store_true');p.add_argument('--output',required=True)
    a=p.parse_args();root=Path(a.prepared)
    from training.predict import Pipeline
    pipeline=Pipeline(a.detector,a.classifier,a.device,a.threshold,a.allow_smoke)
    report=json.loads((root/'report.json').read_text())
    frames=[r for r in read_records(root/'records.snapshot.jsonl')
            if report['split_by_record'][r['id']]=='test' and r['id'] not in report['excluded']]
    stats=dict(frames=0,ground_truth_dice=0,predicted_dice=0,matched_dice=0,trusted_faces=0,
               correct_faces=0,accepted_faces=0,correct_accepted_faces=0,
               fully_trusted_trays=0,exact_trays=0,accepted_exact_trays=0)
    details=[]
    for r in frames:
        pred=pipeline.predict(safe_image(r['image']));ps=pred['dice'];used=set();correct=0
        stats['frames']+=1;stats['ground_truth_dice']+=len(r['dice']);stats['predicted_dice']+=len(ps)
        for d in r['dice']:
            options=[(iou(d['box'],x['box']),i) for i,x in enumerate(ps) if i not in used and x['type']==d['type']]
            overlap,index=max(options,default=(0,-1))
            match=overlap>=.5
            if match:used.add(index);stats['matched_dice']+=1
            if d.get('value_trusted') and not d.get('unreadable'):
                stats['trusted_faces']+=1
                ok=match and ps[index]['value']==d['value']
                stats['correct_faces']+=int(ok);correct+=int(ok)
                if match and ps[index]['accepted']:
                    stats['accepted_faces']+=1;stats['correct_accepted_faces']+=int(ok)
        trusted=bool(r['dice']) and all(d.get('value_trusted') and not d.get('unreadable') for d in r['dice'])
        exact=trusted and correct==len(r['dice'])==len(ps)
        if trusted:
            stats['fully_trusted_trays']+=1;stats['exact_trays']+=int(exact)
            stats['accepted_exact_trays']+=int(exact and pred['confidence_gate_passed'])
        details.append(dict(id=r['id'],trusted=trusted,exact=exact,prediction=pred))
    def ratio(a,b):return stats[a]/stats[b] if stats[b] else None
    metrics=dict(type_iou50_recall=ratio('matched_dice','ground_truth_dice'),
        face_accuracy_including_missed_dice=ratio('correct_faces','trusted_faces'),
        accepted_face_precision=ratio('correct_accepted_faces','accepted_faces'),
        face_acceptance_coverage=ratio('accepted_faces','trusted_faces'),
        exact_tray_accuracy=ratio('exact_trays','fully_trusted_trays'))
    result=dict(counts=stats,metrics=metrics,details=details,threshold=a.threshold,
                smoke=a.allow_smoke,note='Group-held-out test. Matching is type-aware greedy IoU >= 0.5; faces require trusted labels.')
    Path(a.output).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(counts=stats,metrics=metrics),indent=2))


if __name__=='__main__':main()
