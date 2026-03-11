import os
import pytest
import numpy as np

from openreward.environments import ToolOutput
from analysis import DSBenchAnalysis, AnswerParams as AnalysisAnswerParams
from modeling import DSBenchModeling, AnswerParams as ModelingAnswerParams

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENREWARD_API_KEY = os.getenv("OPENREWARD_API_KEY")

ANALYSIS_TASKS = DSBenchAnalysis.list_tasks("test")
ANALYSIS_EXAMPLE_TASK = ANALYSIS_TASKS[0]

MODELING_TASKS = DSBenchModeling.list_tasks("test")
MODELING_EXAMPLE_TASK = MODELING_TASKS[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("task", ANALYSIS_TASKS)
async def test_analysis_gold(task):
    env = DSBenchAnalysis(task_spec=task, secrets={"openai_api_key": OPENAI_API_KEY})

    answer = env.validated.answer
    result: ToolOutput = await env.answer(AnalysisAnswerParams(answer=str(answer)))
    assert result.reward == 1.0


@pytest.mark.asyncio
@pytest.mark.parametrize("task", ANALYSIS_TASKS)
async def test_analysis_xfail(task):
    env = DSBenchAnalysis(task_spec=task, secrets={"openai_api_key": OPENAI_API_KEY})

    incorrect_answer = 123456789
    result: ToolOutput = await env.answer(AnalysisAnswerParams(answer=incorrect_answer))
    assert result.reward == 0.0


@pytest.mark.asyncio
@pytest.mark.parametrize("task", MODELING_TASKS)
async def test_modeling_gold(task):
    env = DSBenchModeling(task_spec=task, secrets={"api_key": OPENREWARD_API_KEY})
    try:
        await env.setup()

        result: ToolOutput = await env.answer(ModelingAnswerParams(path_to_submission="/private/answers.csv"))
        assert result.reward is not None
        assert np.isclose(result.reward, 1.0), f"Expected reward to be 1.0, got {result.reward}, full result: {result}"
    finally:
        await env.teardown()


@pytest.mark.asyncio
@pytest.mark.parametrize("task", MODELING_TASKS)
async def test_modeling_xfail(task):
    env = DSBenchModeling(task_spec=task, secrets={"api_key": OPENREWARD_API_KEY})
    try:
        await env.setup()

        # Submit the sample submission - should get a lower reward than gold
        result: ToolOutput = await env.answer(ModelingAnswerParams(path_to_submission="/workdir/sample_submission.csv"))
        assert result.reward is not None and result.reward < 1.0
    finally:
        await env.teardown()
