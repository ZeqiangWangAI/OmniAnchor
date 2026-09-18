import pytest

from scripts.summarize_campaign_cost import aggregate


def test_allocation_summary_counts_failed_and_running_without_steps():
    rows=[dict(JobIDRaw='1',State='FAILED',allocated_gpu_hours_observed=.5,allocated_cpu_hours_observed=2),
          dict(JobIDRaw='2',State='RUNNING',allocated_gpu_hours_observed=1,allocated_cpu_hours_observed=4)]
    result=aggregate(rows)
    assert result['FAILED']['gpu_hours']==.5 and result['RUNNING']['gpu_hours']==1
    with pytest.raises(ValueError,match='Duplicate'):
        aggregate(rows+[rows[0]])
    with pytest.raises(ValueError,match='Job-step'):
        aggregate([{**rows[0],'JobIDRaw':'1.batch'}])
