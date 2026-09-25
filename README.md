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

Example:

```text
Hi, I have these files available:
15_PET_MARIO_ROSSI,
15_TISSUE_SLIDE_MARIO_ROSSI

Could you help me with the diagnosis?
```

For this example, the expected diagnostic trajectory is:

```text
domain_translator
→ nodule_detector
→ nodule_segmentator
→ nodule_classifier
→ radiology_report_generator
→ mutation_predictor
→ cell_detector
→ cell_segmentator
→ cell_classifier
→ biopsy_report_generator
```

The agent can iteratively call tools until no additional tool execution is requested.

## Simulated Clinical Tools

The current implementation includes the following tools.

### Radiology

```text
domain_translator
nodule_detector
nodule_segmentator
nodule_classifier
radiology_report_generator
```

`domain_translator` simulates cross-modality synthesis between CT and PET.

`nodule_detector` identifies a simulated pulmonary nodule.

`nodule_segmentator` generates a simulated segmented nodule.

`nodule_classifier` assigns a predefined disease stage and lesion category according to the synthetic filename convention.

`radiology_report_generator` generates a simulated radiological report after image classification.

### Histopathology

```text
mutation_predictor
cell_detector
cell_segmentator
cell_classifier
biopsy_report_generator
```

`mutation_predictor` generates simulated biomarker information from an available biopsy slide.

`cell_detector` represents cell or nuclei detection.

`cell_segmentator` generates a simulated cellular segmentation.

`cell_classifier` assigns a predefined histological class.

`biopsy_report_generator` generates the corresponding pathology report.

These tools are intended to simulate the outputs of individual clinical modules. They are not medical diagnostic models.

## Agent Rules

The diagnostic agent follows a system prompt containing explicit procedural constraints.

Examples include:

- If only CT or PET is available, cross-modality translation must be performed first.
- Existing examinations or reports must not be regenerated.
- A radiology report can only be generated after CT classification.
- If a biopsy slide is available, mutation prediction precedes cellular analysis.
- A pathology report requires an available cell classification.
- The final model response provides the resulting clinical consideration after tool execution.

Few-shot examples are included in the prompt to demonstrate correct tool sequences.

## Dataset

The current implementation expects two Excel files:

```text
diagnosis_dataset.xlsx
gt_tools_diagnosis.xlsx
```

### `diagnosis_dataset.xlsx`

The script currently reads the following fields by column position:

```text
ID
PET
CT_SCAN
RADIOLOGY_REPORT
TISSUE_SLIDE
BIOPSY_REPORT
```

Available files are converted into natural-language queries for the agent.

### `gt_tools_diagnosis.xlsx`

This file contains the reference sequence of tool calls associated with each query.

Example:

```python
[
    "domain_translator",
    "nodule_detector",
    "nodule_segmentator",
    "nodule_classifier",
    "radiology_report_generator"
]
```

The ordering of the diagnostic queries and the reference trajectories must correspond across the two files.

## Evaluation Metrics

Agent-generated tool trajectories are compared with their reference sequences using several complementary metrics.

### Exact Sequence Match

Measures whether the complete predicted tool sequence exactly matches the reference sequence.

```text
1 = exact match
0 = otherwise
```

Implemented as:

```python
tool_match(predicted, ground_truth)
```

### Positional Tool Accuracy

Measures the proportion of reference positions containing the correct tool.

For reference sequence \(G\) and predicted sequence \(T\):

```text
correct positions / number of reference tools
```

Implemented as:

```python
correct_tool_pos(predicted, ground_truth)
```

### Levenshtein Distance

Measures the edit distance between the predicted and reference trajectories.

The permitted operations are:

```text
insertion
deletion
substitution
```

A lower value indicates a trajectory closer to the reference workflow.

Implemented as:

```python
levenshtein_distance(predicted, ground_truth)
```

### Missed Tools

Measures the proportion of required reference tool calls absent from the predicted trajectory.

Implemented as:

```python
missed_tools(predicted, ground_truth)
```

### Unnecessary Tool Calls

Counts tool calls produced by the agent that cannot be matched to the reference trajectory.

Implemented as:

```python
invented_tools(predicted, ground_truth)
```

### First Error Position

The repository also contains a function for identifying the first position at which the generated sequence differs from the reference sequence:

```python
first_E(predicted, ground_truth)
```

This metric is currently defined in the script but is not included in the final aggregated metrics.

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
