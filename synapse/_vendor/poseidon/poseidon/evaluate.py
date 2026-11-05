"""Frozen held-out protocol. Report controls alongside the learned results."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import random
from fractions import Fraction
import time

def evaluate(checkpoint=None, output="runs/evaluation.json", episodes=24):
    import torch
    from .core import CoreRuntime, SCENE_LABELS, active_core_path, hash_features
    from .calibrate import expected_calibration_error
    from .media import generate_scene_examples, render_scene
    from .world import benchmark
    from .reasoning import solve
    torch.set_num_threads(2)
    checkpoint = Path(checkpoint) if checkpoint else active_core_path(".")
    core = CoreRuntime(checkpoint)
    scenes = generate_scene_examples(2048, seed=771122, split="test")
    with torch.inference_mode():
        inputs = hash_features([s["prompt"] for s in scenes], core.model.config.hash_buckets)
        predictions=[]
        for start in range(0,len(inputs),128):
            predictions.append(core.model(inputs[start:start+128]))
        correct={name:0 for name in SCENE_LABELS}
        exact=0
        examples=[]
        for index,item in enumerate(scenes):
            result=predictions[index//128]
            guessed={name:int(result["scene"][name][index%128].argmax()) for name in SCENE_LABELS}
            hits={name:guessed[name]==item["labels"][name] for name in SCENE_LABELS}
            for name in correct:
                correct[name]+=hits[name]
            exact+=all(hits.values())
            if len(examples)<12 or (not all(hits.values()) and len(examples)<32):
                examples.append({"prompt":item["prompt"],"expected":item["labels"],"predicted":guessed,"correct":all(hits.values())})
        calibration={}
        for j,name in enumerate(SCENE_LABELS):
            logits=torch.cat([p["scene"][name] for p in predictions])
            labels=torch.tensor([s["labels"][name] for s in scenes])
            calibration[name]={"temperature":core.temperatures.get(name,1.0),
                               "test_ece_raw":expected_calibration_error(logits.softmax(-1),labels),
                               "test_ece_served":expected_calibration_error(core.calibrated(name,logits).softmax(-1),labels),
                               "mean_served_confidence":float(core.calibrated(name,logits).softmax(-1).max(-1).values.mean())}
    print(json.dumps({"event":"scene_evaluated","exact":exact,"total":len(scenes)}),flush=True)
    seeds=list(range(91000001,91000001+episodes))
    worlds={}
    for scarcity in (1.0,2.0):
        worlds[str(scarcity)]=benchmark(core,seeds=seeds,max_steps=256,scarcity=scarcity)
        print(json.dumps({"event":"survival_evaluated","scarcity":scarcity,"summary":worlds[str(scarcity)]["summary"]}),flush=True)
    rng=random.Random(334455)
    arithmetic=[]
    for _ in range(100):
        a,b,c=[rng.randint(-100,100) for i in range(3)]
        expression=f"({a}+{b})*{c}"
        expected=str((a+b)*c)
        arithmetic.append({"expression":expression,"expected":expected,"actual":solve(expression)["answer"]})
    for _ in range(100):
        a=rng.choice([v for v in range(-12,13) if v]);x=rng.randint(-100,100);b=rng.randint(-100,100)
        expression=f"{a}*x+({b})={a*x+b}"
        arithmetic.append({"expression":expression,"expected":str(x),"actual":solve(expression)["answer"]})
    path=Path(output);path.parent.mkdir(parents=True,exist_ok=True)
    media=[]
    for kind,prompt in [("image","Create two small cyan spheres with orbit motion."),("video","Make three medium purple cubes with bounce motion."),("mesh","Build one large yellow pyramid with still motion.")]:
        prediction=core.scene(prompt)
        artifact=render_scene(prediction,path.parent.parent/"outputs/examples",kind,seed=42)
        media.append({"prompt":prompt,"prediction":prediction,"artifact":artifact})
    result={"schema":"poseidon-evaluation-v1","checkpoint_sha256":hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),"checkpoint":str(Path(checkpoint).resolve()),"core":core.diagnostics(),"scene_test":{"examples":len(scenes),"semantic_groups":len({s['group'] for s in scenes}),"split":"held-out semantic groups via media generator test split","exact_accuracy":exact/len(scenes),"attribute_accuracy":{name:value/len(scenes) for name,value in correct.items()},"calibration":calibration,"examples_and_failures":examples},"survival":worlds,"math_tool":{"correct":sum(r['expected']==r['actual'] for r in arithmetic),"total":len(arithmetic),"neural_reasoning_score":False,"backend":"exact rational arithmetic and linear solver","rows":arithmetic},"media_examples":media,"limits":["Synthetic compositional scene accuracy, not arbitrary text-to-image evaluation.","Behavior cloning from an encoded heuristic; survival is limited to this synthetic environment.","Math tool results are not unaided neural reasoning.","External memory is explicit retrieval, not proven neural memory transfer.","No claim of improvement over previous Supermix models without matched evaluation."]}
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(result,indent=2),encoding='utf-8');temporary.replace(path)
    print(json.dumps({"output":str(path),"scene_exact_accuracy":result['scene_test']['exact_accuracy'],"math_correct":result['math_tool']['correct']}),flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',default=None,help='defaults to runs/active_core.json target, else runs/tidal/core.pt');p.add_argument('--output',default='runs/evaluation.json');p.add_argument('--episodes',type=int,default=24);a=p.parse_args()
    if not 1<=a.episodes<=1000:p.error('episodes must be 1..1000')
    evaluate(a.checkpoint,a.output,a.episodes)
