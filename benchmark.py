import asyncio
from agents.basicagent.sample import sample
from agents.backends.utils import EnvironmentConfig, ExecutionConstraints, ModelConfig


async def main():
   await sample(
        environment_config=EnvironmentConfig(
            environment="dsbenchmodeling",
            split="all",
            host="http://0.0.0.0:2020",
            max_tasks=1,
        ),
        model_config=ModelConfig(
            model="anthropic/claude-sonnet-4.5",
            backend_name="openrouter",
            max_output_tokens=16_000,
            max_context_window=128_000,
        ),
        local_log_dir="logs",
    )

if __name__ == "__main__":
    asyncio.run(main())