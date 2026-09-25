# Agentic AI for Lung Cancer Diagnostic Workflow Evaluation

This repository implements a graph-based agentic framework for simulating and evaluating lung cancer diagnostic workflows.

The system uses a Large Language Model as an orchestrator that selects and executes predefined clinical tools. Agent trajectories are compared against reference tool sequences to evaluate whether the model follows the expected diagnostic workflow.

The framework is designed to assess **LLM orchestration behaviour independently from the predictive performance of individual clinical models**. Clinical tools therefore return deterministic or predefined outputs rather than executing real diagnostic algorithms.

## Overview

The pipeline contains three main components:

1. **Agent orchestration**
2. **Simulated clinical tools**
3. **Trajectory evaluation**

The diagnostic agent is implemented with **LangGraph** and uses tool calling through **LangChain**.

Given a set of available clinical files, such as CT, PET, biopsy slides, or reports, the agent determines which tools should be executed and in which order.

The resulting sequence of tool calls is then compared with a predefined ground-truth clinical workflow.

## Workflow

The agent receives queries containing the clinical information available for a patient.

The agent can iteratively call tools until no additional tool execution is requested.


## Output

Each successful evaluation is stored in a JSONL file containing:

```json
{
  "ID": 1,
  "task": "diagnosis",
  "model": "model_name",
  "reasoning_effort": "none",
  "web_search": "no",
  "query": "...",
  "tool_calls": [],
  "obtained_files": [],
  "content": "..."
}
```

Aggregated metrics are stored in a JSON file containing:

```json
{
  "task": "diagnosis",
  "model": "...",
  "reasoning_effort": "none",
  "tool_accuracy": 0.0,
  "tool_position_accuracy": 0.0,
  "avg_levenshtein_distance": 0.0,
  "avg_missed_tools": 0.0,
  "avg_hallucinated_tools": 0.0,
  "time": "...",
  "usage_and_costs": {
    "total_input_tokens": 0,
    "total_output_tokens": 0
  }
}
```

Queries that cannot be completed after the configured retry attempts are stored separately.

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd <repository-name>
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it.

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

Install the required packages.


## API Configuration

API credentials are loaded from a local `.env` file.

Example:

```env
LLM_API_KEY_1=your_api_key
LLM_API_KEY_2=your_second_api_key
```

Do not commit `.env` files containing credentials.

Add the following to `.gitignore`:

```gitignore
.env
.venv/
__pycache__/
*.pyc
*_evaluation/
```

The current model configuration is created through the OpenAI-compatible interface exposed by Ollama:

```python
ChatOpenAI(
    model="deepseek-v3.2:cloud",
    base_url="https://ollama.com/v1",
    ...
)
```

Model configuration should be adapted according to the inference provider being evaluated.

## Running the Evaluation

Ensure that the following files are available in the expected working directory:

```text
diagnosis_dataset.xlsx
gt_tools_diagnosis.xlsx
.env
```

Then run the evaluation script:

```bash
python <evaluation_script>.py
```

For every query, the pipeline:

```text
1. creates the agent input;
2. executes the LangGraph workflow;
3. records all tool calls;
4. extracts generated intermediate files;
5. compares the generated trajectory with the reference trajectory;
6. accumulates evaluation metrics;
7. records token usage;
8. saves the final outputs.
```

## Reproducibility

The evaluation separates two components that would otherwise be confounded:

```text
Clinical tool performance
            vs.
LLM orchestration performance
```

The tools use predefined behaviour so that errors in the final trajectory can be attributed primarily to the orchestration strategy rather than variability in underlying clinical prediction models.

This makes it possible to study whether an agent:

- selects the correct clinical tools sequence;
- executes them in the correct order;
- omits required operations;
- introduces unnecessary operations;
- deviates from the expected clinical workflow.


## Limitations

The current implementation is an experimental evaluation framework.

The clinical tools are simulated and must not be interpreted as validated diagnostic algorithms.

The framework therefore evaluates agent orchestration within a controlled simulated environment rather than real-world diagnostic accuracy.

## Intended Use

This repository is intended for research on:

```text
Agentic AI
Clinical workflow orchestration
Tool-use evaluation
LLM trajectory evaluation
Clinical pathway simulation
Lung cancer diagnostic workflows
```

It is not intended for clinical decision-making or patient care.

## Citation

If this repository accompanies a scientific publication, citation information should be added here once the corresponding publication is available.
