# DSBench

[![OpenReward Environment](https://img.shields.io/badge/%E2%AD%90%20OpenReward-Environment-f7e6cc)](https://openreward.ai/GeneralReasoning/DSBench) [![Hugging Face Dataset](https://img.shields.io/badge/Hugging%20Face-Dataset-orange)](https://huggingface.co/datasets/liqiang888/DSBench)

## Description

DSBench is an environment for evaluating data science agents on realistic data analysis and data modeling tasks. It is based on the [DSBench benchmark](https://arxiv.org/abs/2409.07703) (ICLR 2025), which collects tasks from ModelOff financial modeling competitions and Kaggle machine learning challenges.

The environment provides two variants:

- **DSBenchAnalysis**: Single-turn data analysis questions over Excel workbooks sourced from ModelOff competitions (2012-2017). The agent is given the workbook data, background context, and a question, and must submit a final answer.
- **DSBenchModeling**: Multi-step machine learning tasks sourced from Kaggle competitions. The agent is given a sandbox with training data, test data, and a sample submission file, and must produce a predictions CSV.

## Capabilities

- Financial data analysis and reasoning over complex Excel workbooks
- End-to-end machine learning modeling (data exploration, feature engineering, model training, prediction)
- Code execution in isolated sandboxes
- Evaluation against Kaggle competition metrics

## Compute Requirements

- **Analysis**: No sandbox required (single-turn evaluation)
- **Modeling**: Agents are given a sandbox with 2 CPUs and 2GB of RAM

## License

[MIT](https://github.com/LiqiangJing/DSBench/blob/main/LICENSE).

## Tasks

All tasks are in a single `test` split:

- **Analysis**: 466 data analysis questions across 38 task families from ModelOff competitions
- **Modeling**: 74 active machine learning tasks from Kaggle competitions (18 excluded due to data issues)

## Reward Structure

**Analysis**: Binary reward (0 or 1) determined by an LLM judge comparing the agent's answer against the ground truth.

**Modeling**: Continuous reward normalized between baseline and ground truth performance:

$$reward = \frac{score - baseline}{ground\_truth - baseline}$$

For metrics where lower is better (e.g., RMSE), the formula is inverted. Rewards are clamped to [0, 1].

## Data

- **Analysis data**: Excel workbooks and task descriptions from [ModelOff/Eloquence](https://www.eloquens.com/) financial modeling competitions
- **Modeling data**: Kaggle competition datasets, stored in cloud storage and mounted into sandboxes

## Tools

**Analysis variant** (1 tool):
- `answer` - Submit a final answer for grading

**Modeling variant** (6 tools):
- `bash` - Execute bash commands in the sandbox
- `view` - View file contents with optional line ranges
- `str_replace` - Replace text in files
- `insert` - Insert content at a line number
- `create` - Create a file with content
- `answer` - Submit a predictions CSV for evaluation

## Other Environment Requirements

- **Analysis**: Requires `openai_api_key` secret for LLM-based grading
- **Modeling**: Requires `api_key` secret (OpenReward API key) for sandbox provisioning

## Citations

```bibtex
@inproceedings{jing2025dsbench,
  title={DSBench: How Far Are Data Science Agents from Becoming Data Science Experts?},
  author={Liqiang Jing and Zhehui Huang and Xiaoyang Wang and Wenlin Yao and Wenhao Yu and Kaixin Ma and Hongming Zhang and Xinya Du and Dong Yu},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=DSsSPr0RZJ}
}
```
