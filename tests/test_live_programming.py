import numpy as np
from synapse.agent import IntegratedAgent
from synapse.config import SynapseConfig
from synapse.patches import make_patch
from synapse.stateio import digest


def test_live_programming_screen_uses_actual_worlds_and_restores_rejection():
    from synapse.live_programming import screen_live_patch
    agent=IntegratedAgent(SynapseConfig(nodes_per_source=16))
    value=float(agent.motor.weight[0,0].detach())
    packet=make_patch(agent,[{'tensor':'motor.weight','indices':[[0,0]],'values':[value+.01]}])
    before=digest(agent.export_state())
    result=screen_live_patch(agent,packet,task='forage',steps=8,seeds=[19],min_gain=100)
    assert result['accepted'] is False
    assert result['world_steps']==32
    assert len(result['paired_returns']['baseline'])==1
    assert digest(agent.export_state())==before


def test_live_rsi_receipt_is_real_and_bounded():
    from synapse.live_programming import run_live_rsi
    agent=IntegratedAgent(SynapseConfig(nodes_per_source=16))
    result=run_live_rsi(agent,task='push',steps=8,candidates=1,seed=7,min_gain=100)
    assert result['candidates_evaluated']==1
    assert result['accepted_count']==0
    assert result['evidence_kind']=='measured_world_development_screen'
    assert result['screens'][0]['world_steps']==32
