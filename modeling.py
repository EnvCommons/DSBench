import base64
from shlex import quote
import traceback
from pathlib import Path
import json

from pydantic import BaseModel

from openreward.environments import Environment, tool, JSONObject, ToolOutput, TextBlock, Split
from openreward import AsyncOpenReward, SandboxBucketConfig, SandboxSettings


def load_jsonl(path: Path) -> list[JSONObject]:
    return [json.loads(line) for line in open(path)]

DATA_METADATA = load_jsonl(Path(__file__).parent / "data_modeling_metadata.jsonl")

import os

if os.path.exists('/orwd_data'):
    DATA_DIR = Path("/orwd_data") / "modeling_data"
else:
    DATA_DIR = Path(__file__).parent / "modeling_data"

MOUNT_DATA_DIR = "/dsbench_data"

TASKS_IDS_TO_IGNORE = [
    "data-science-london-scikit-learn",
    "predict-who-is-more-influential-in-a-social-network",
    "amazon-employee-access-challenge",
    "playground-series-s3e26",
    "bioresponse",
    "20-newsgroups-ciphertext-challenge",
    "digit-recognizer",
    "reducing-commercial-aviation-fatalities",
    "otto-group-product-classification-challenge",
    "ciphertext-challenge-iii",
    "tabular-playground-series-oct-2021",
    "tabular-playground-series-may-2021",
    "kaggle-llm-science-exam",
    "chaii-hindi-and-tamil-question-answering",
    "tabular-playground-series-jun-2021",
    "contradictory-my-dear-watson",
    "llm-prompt-recovery",
    "covid19-global-forecasting-week-5",
]


class TaskSpec(BaseModel):
    id: str
    instructions: str
    local_path_to_evaluation_file: Path
    baseline_score: float
    ground_truth_score: float
    max_response_length: int | None = None


class AnswerParams(BaseModel):
    path_to_submission: str | int


class BashParams(BaseModel, extra="forbid"):
    command: str


class ViewParams(BaseModel, extra="forbid"):
    path: str
    start: int | None = None  # 1-indexed inclusive
    end: int | None = None    # 1-indexed inclusive


class StrReplaceParams(BaseModel, extra="forbid"):
    path: str
    old_str: str
    new_str: str


class CreateParams(BaseModel, extra="forbid"):
    path: str
    content: str


class InsertParams(BaseModel, extra="forbid"):
    path: str
    start: int  # 1-indexed line number to insert before
    content: str


class DSBenchModeling(Environment):
    def __init__(self, task_spec: JSONObject, secrets: dict[str, str] = {}) -> None:
        super().__init__(task_spec)
        self.validated = TaskSpec.model_validate(task_spec)

        api_key = secrets.get("api_key")
        if not api_key:
            raise ValueError("OpenReward API key must be provided via secrets parameter")

        self.sandbox_settings = SandboxSettings(
            environment="GeneralReasoning/DSBench",
            image="us-central1-docker.pkg.dev/indigo-idea-457514-b5/environments/dsbench-agent:latest",
            machine_size="1:2",
            block_network=False,
            bucket_config=SandboxBucketConfig(
                mount_path=MOUNT_DATA_DIR,
                read_only=True,
                only_dir="dsbench",
            )
        )
        or_client = AsyncOpenReward(api_key=api_key)
        self.sandbox = or_client.sandbox(self.sandbox_settings)

    async def setup(self) -> None:
        await self.sandbox.start()

        # copy dataset from bucket to /workdir
        await self.sandbox.run(f"cp {MOUNT_DATA_DIR}/{self.validated.id}.zip /tmp")
        await self.sandbox.run(f"unzip -o /tmp/{self.validated.id}.zip -d /workdir")
        await self.sandbox.run(f"rm /tmp/{self.validated.id}.zip")

        # copy answer file
        await self.sandbox.run("mkdir -p /private && chmod 777 /private")
        await self.sandbox.run(f"cp {MOUNT_DATA_DIR}/{self.validated.id}_answer.csv /private/answers.csv")

    async def teardown(self) -> None:
        await self.sandbox.stop()

    async def get_prompt(self) -> list[TextBlock]:
        prompt = """You are a data scientist. I have a data modeling task. You must give me the predicted results as a CSV file as detailed in the following content. Don't ask me any questions. I provide you with three files. One is training data, one is test data. There is also a sample file for submission."""
        prompt += f"\n{self.validated.instructions}"
        return [TextBlock(text=prompt)]

    @tool
    async def answer(self, params: AnswerParams) -> ToolOutput:
        """Submit your predicted results CSV file for evaluation."""
        try:
            await self.sandbox.upload(str(self.validated.local_path_to_evaluation_file), "/tmp/evaluate.py")

            task_id = self.validated.id
            await self.sandbox.run(f"mkdir -p /tmp/results/{task_id}")
            cmd = (
                f"source /workdir/.venv/bin/activate && "
                f"python /tmp/evaluate.py "
                f"--path /tmp/results "
                f"--name {task_id} "
                f"--answer_file /private/answers.csv "
                f"--predict_file {params.path_to_submission}"
            )
            output, _ = await self.sandbox.run(cmd)

            result = await self.sandbox.download(f"/tmp/results/{task_id}/result.txt")
            score = result.decode("utf-8").strip()

            lower_is_better = self.validated.ground_truth_score < self.validated.baseline_score
            if score == "nan":
                reward = 0.0
            else:
                if lower_is_better:
                    reward = max(0, (self.validated.baseline_score - eval(score)) / (self.validated.baseline_score - self.validated.ground_truth_score))
                else:
                    reward = max(0, (eval(score) - self.validated.baseline_score) / (self.validated.ground_truth_score - self.validated.baseline_score))

            return ToolOutput(
                metadata={"evaluation_script_output": output, "score": score, "reward": reward},
                blocks=[TextBlock(text=f"Evaluation complete. Score: {score}, Reward: {reward:.4f}")],
                reward=reward,
                finished=True,
            )
        except Exception:
            return ToolOutput(
                metadata={"error": traceback.format_exc()},
                blocks=[TextBlock(text="Error during evaluation.")],
                reward=0.0,
                finished=True,
            )

    @classmethod
    def list_tasks(cls, split: str) -> list[JSONObject]:
        if split != "test":
            raise ValueError(f"Unknown split: {split}")

        tasks = []
        task_files = sorted([i for i in (DATA_DIR / "task").iterdir() if i.is_file() and i.suffix == ".txt"])
        for task_file in task_files:
            task_id = task_file.stem.split(".")[0]
            if task_id in TASKS_IDS_TO_IGNORE:
                continue

            with open(task_file, "r") as f:
                instructions = f.read()

            local_path_to_evaluation_file = DATA_DIR / "evaluation" / f"{task_id}_eval.py"
            assert local_path_to_evaluation_file.exists(), f"Evaluation file {local_path_to_evaluation_file} does not exist"

            baseline_score_file = DATA_DIR / "save_performance" / "baseline" / task_id / "result.txt"
            with open(baseline_score_file, "r") as f:
                baseline_score = float(f.read().strip())
            ground_truth_score_file = DATA_DIR / "save_performance" / "GT" / task_id / "result.txt"
            with open(ground_truth_score_file, "r") as f:
                ground_truth_score = float(f.read().strip())
            tasks.append(TaskSpec(
                id=task_id,
                instructions=instructions,
                local_path_to_evaluation_file=local_path_to_evaluation_file,
                baseline_score=baseline_score,
                ground_truth_score=ground_truth_score,
            ).model_dump())
        return tasks

    @classmethod
    def list_splits(cls) -> list[Split]:
        return [Split(name="test", type="test")]

    @tool
    async def bash(self, params: BashParams) -> ToolOutput:
        """Execute a bash command."""
        output, code = await self.sandbox.run(f"source /workdir/.venv/bin/activate && {params.command}", timeout=600)
        max_len = self.validated.max_response_length

        if isinstance(max_len, int) and len(output) > max_len:
            output = f"...(truncated)\n{output[-max_len:]}"

        return ToolOutput(
            metadata={"output": output, "exit_code": code},
            blocks=[TextBlock(text=f"{output}\n\n(exit {code})")],
            reward=0.0,
            finished=False,
        )

    # ---------- Text Editor tools (bash-only implementations) ----------

    @tool
    async def view(self, params: ViewParams) -> ToolOutput:
        """View file contents. Optionally specify a 1-indexed [start, end] line range."""
        p = quote(params.path)
        if params.start is not None or params.end is not None:
            start = params.start if params.start is not None else 1
            end = params.end if params.end is not None else '$'
            cmd = f"sed -n '{start},{end}p' {p}"
        else:
            cmd = f"cat {p}"
        output, code = await self.sandbox.run(cmd)
        max_len = self.validated.max_response_length
        if isinstance(max_len, int) and len(output) > max_len:
            output = f"...(truncated)\n{output[-max_len:]}"
        return ToolOutput(
            metadata={"content": output, "exit_code": code, "path": params.path},
            blocks=[TextBlock(text=output)],
            reward=0.0,
            finished=False,
        )

    @tool
    async def str_replace(self, params: StrReplaceParams) -> ToolOutput:
        """Replace all occurrences of old_str with new_str in the given file. Use this tool to edit files."""
        path = params.path
        suffix = Path(path).suffix
        backup = f"{path}_old{suffix}"

        py = (
            "from pathlib import Path\n"
            f"p = Path({json.dumps(path)})\n"
            f"old = {json.dumps(params.old_str)}\n"
            f"new = {json.dumps(params.new_str)}\n"
            "text = p.read_text()\n"
            "p.write_text(text.replace(old, new))\n"
        )

        cmd = (
            f"set -e\n"
            f"cp {quote(path)} {quote(backup)}\n"
            f"python3 - << 'PY'\n{py}PY\n"
            f"git diff --no-index {quote(backup)} {quote(path)} || true"
        )

        output, exit_code = await self.sandbox.run(cmd)
        max_len = self.validated.max_response_length
        if isinstance(max_len, int) and len(output) > max_len:
            output = f"...(truncated)\n{output[-max_len:]}"
        return ToolOutput(
            metadata={"diff": output, "exit_code": exit_code, "backup_path": backup, "path": path},
            blocks=[TextBlock(text=output)],
            reward=0.0,
            finished=False,
        )

    @tool
    async def insert(self, params: InsertParams) -> ToolOutput:
        """Insert content at the given 1-indexed line number. Use this tool to edit files."""
        path = params.path
        suffix = Path(path).suffix
        backup = f"{path}_old{suffix}"

        py = (
            "from pathlib import Path\n"
            "import sys\n"
            f"p = Path({json.dumps(path)})\n"
            f"start = int({json.dumps(params.start)})\n"
            f"content = {json.dumps(params.content)}\n"
            "if not p.exists():\n"
            "    p.parent.mkdir(parents=True, exist_ok=True)\n"
            "    p.write_text('')\n"
            "text = p.read_text()\n"
            "lines = text.splitlines(keepends=True)\n"
            "idx = max(0, min(start - 1, len(lines)))\n"
            "new_text = ''.join(lines[:idx]) + content + ''.join(lines[idx:])\n"
            "p.write_text(new_text)\n"
        )

        cmd = (
            f"set -e\n"
            f"if [ -f {quote(path)} ]; then cp {quote(path)} {quote(backup)}; "
            f"else mkdir -p $(dirname {quote(path)}); : > {quote(path)}; cp {quote(path)} {quote(backup)}; fi\n"
            f"python3 - << 'PY'\n{py}PY\n"
            f"git diff --no-index {quote(backup)} {quote(path)} || true"
        )

        output, _ = await self.sandbox.run(cmd)
        max_len = self.validated.max_response_length
        if isinstance(max_len, int) and len(output) > max_len:
            output = f"...(truncated)\n{output[-max_len:]}"
        return ToolOutput(
            metadata={"diff": output, "exit_code": 0, "backup_path": backup, "path": path, "start": params.start},
            blocks=[TextBlock(text=output)],
            reward=0.0,
            finished=False,
        )

    @tool
    async def create(self, params: CreateParams) -> ToolOutput:
        """Create a file with the given content."""
        path = params.path
        path_q = quote(path)
        b64 = base64.b64encode(params.content.encode()).decode()
        cmd = (
            f"set -e; "
            f"mkdir -p $(dirname {path_q}); "
            f"printf '%s' {quote(b64)} | base64 -d > {path_q}; "
            f"printf 'Created {path} (%s bytes)\\n' $(wc -c < {path_q})"
        )
        output, code = await self.sandbox.run(cmd)
        msg = output.strip()
        return ToolOutput(
            metadata={"message": msg, "path": path, "bytes": len(params.content), "exit_code": code},
            blocks=[TextBlock(text=msg)],
            reward=0.0,
            finished=False,
        )
