import traceback
from pathlib import Path
import json
from typing import Any

from pydantic import BaseModel
import pandas as pd
import openai

from openreward.environments import Environment, tool, JSONObject, ToolOutput, TextBlock, Split


def load_jsonl(path: Path) -> list[JSONObject]:
    return [json.loads(line) for line in open(path)]

DATA_METADATA = load_jsonl(Path(__file__).parent / "data_analysis_metadata.jsonl")

import os

if os.path.exists('/orwd_data'):
    DATA_DIR = Path("/orwd_data") / "analysis_data"
else:
    DATA_DIR = Path(__file__).parent / "analysis_data"

_CACHED_TASKS: list[JSONObject] | None = None
_EXCEL_CACHE: dict[tuple, str] = {}


def _parse_excel_files(excel_paths: list[Path]) -> str:
    """Parse Excel files and return formatted content string. Cached across sessions."""
    cache_key = tuple(str(p) for p in sorted(excel_paths))
    if cache_key in _EXCEL_CACHE:
        return _EXCEL_CACHE[cache_key]
    content = ""
    for excel_path in excel_paths:
        xls = pd.ExcelFile(excel_path)
        sheets = {sheet_name: xls.parse(sheet_name) for sheet_name in xls.sheet_names}
        combined_text = ""
        for sheet_name, df in sheets.items():
            assert isinstance(df, pd.DataFrame)
            sheet_text = df.to_string(index=False)
            combined_text += f"Sheet name: {sheet_name}\n{sheet_text}\n\n"
        content += f"The excel file {excel_path.name} is: " + combined_text
    _EXCEL_CACHE[cache_key] = content
    return content


class TaskSpec(BaseModel):
    id: str
    introduction: str
    question: str
    answer: str | int | dict[str, Any]
    image_paths: list[Path]
    excel_paths: list[Path]


class AnswerParams(BaseModel):
    answer: str | int


class DSBenchAnalysis(Environment):
    def __init__(self, task_spec: JSONObject, secrets: dict[str, str] = {}) -> None:
        super().__init__(task_spec)
        self.validated = TaskSpec.model_validate(task_spec)

        api_key = secrets.get("openai_api_key")
        if not api_key:
            raise ValueError("OpenAI API key must be provided via secrets parameter")
        self.client = openai.AsyncClient(api_key=api_key)

        # Pre-parse Excel content (cached across sessions sharing same files)
        self._excel_content = _parse_excel_files(self.validated.excel_paths) if self.validated.excel_paths else ""

    async def get_prompt(self) -> list[TextBlock]:
        prompt = """You are a data analyst. I will give you a background introduction and data analysis question. You must answer the question."""
        if self._excel_content:
            prompt += f"The workbook is detailed as follows. {self._excel_content} \n"
        prompt += f"The introduction is detailed as follows. \n {self.validated.introduction} \n"
        prompt += f"The questions are detailed as follows. \n {self.validated.question}"
        return [TextBlock(text=prompt)]

    @tool
    async def answer(self, params: AnswerParams) -> ToolOutput:
        """Submit your final answer to the data analysis question."""
        try:
            prompt = (
                f"Please judge whether the generated answer is right or wrong. We require that the correct answer "
                f"to the prediction gives a clear answer, not just a calculation process or a disassembly of ideas. "
                f"The question is {self.validated.question}. The true answer is \n {self.validated.answer}. \n The predicted answer is \n {params.answer}.\n "
                f"If the predicted answer is right, please output True. Otherwise output Flase. "
                f"Don't output any other text content. You only can output True or False."
            )
            response = await self.client.chat.completions.create(
                model="gpt-5.4",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ],
                max_completion_tokens=256,
            )
            content = response.choices[0].message.content
            assert content is not None
            correct = content.lower() == "true"
            return ToolOutput(
                metadata={"correct": correct, "given_answer": params.answer, "correct_answer": self.validated.answer, "judge_response": content},
                blocks=[TextBlock(text="Correct!" if correct else "Incorrect.")],
                reward=1.0 if correct else 0.0,
                finished=True,
            )
        except Exception:
            return ToolOutput(
                metadata={"error": traceback.format_exc()},
                blocks=[TextBlock(text="Error during grading.")],
                reward=0.0,
                finished=True,
            )

    @classmethod
    def list_tasks(cls, split: str) -> list[JSONObject]:
        global _CACHED_TASKS
        if split != "test":
            raise ValueError(f"Unknown split: {split}")
        if _CACHED_TASKS is not None:
            return _CACHED_TASKS

        tasks = []
        for task_family in DATA_METADATA:
            task_dir = DATA_DIR / str(task_family["id"])
            introduction_file = task_dir / "introduction.txt"
            with open(introduction_file, "r") as f:
                introduction = f.read()

            assert isinstance(task_family["questions"], list)
            assert isinstance(task_family["answers"], list)
            assert len(task_family["questions"]) == len(task_family["answers"])

            # Glob once per task family (shared across all questions)
            image_files: list[Path] = sorted(list(task_dir.glob("*.png")) + list(task_dir.glob("*.jpg")))
            excel_files: list[Path] = sorted(list(task_dir.glob("*.xlsx")) + list(task_dir.glob("*.xlsb")) + list(task_dir.glob("*.xlsm")))
            excel_files = [file for file in excel_files if "answer" not in file.name.lower()]

            for question_id, answer in zip(task_family["questions"], task_family["answers"]):
                question_file = task_dir / f"{question_id}.txt"
                with open(question_file, "r") as f:
                    question = f.read()

                tasks.append(TaskSpec(
                    id=f"analysis_{task_family['id']}_{question_id}",
                    introduction=introduction,
                    question=question,
                    answer=answer,
                    image_paths=image_files,
                    excel_paths=excel_files,
                ).model_dump())
        _CACHED_TASKS = tasks
        return tasks

    @classmethod
    def list_splits(cls) -> list[Split]:
        return [Split(name="test", type="test")]
