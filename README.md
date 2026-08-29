# Knowledge-Graph-Enhanced LLM Reasoning for Improved and Explainable Sexism Detection

MSc Artificial Intelligence Capstone Project — University of Galway

This project combines a fine-tuned RoBERTa classifier with a domain-specific knowledge graph and LLM-based reasoning to improve sexism detection on the [EDOS benchmark](https://arxiv.org/abs/2303.04222) (SemEval-2023 Task 10), while producing knowledge-graph-grounded natural language explanations for every prediction.

## Headline Result

| Model | Task A (Macro F1) | Task B (Macro F1) |
|---|---|---|
| Baseline 1 (RoBERTa) | 0.8306 | 0.6061 |
| Baseline 2 (RoBERTa + LLM refiner) | 0.8270 | 0.6030 |
| Proposed A (KG v1 clean + keyword retrieval) | 0.8296 | 0.5766 |
| **Proposed B+C (KG v2 + semantic retrieval)** | 0.8286 | **0.6285** |

Proposed B+C improves Task B Macro F1 by **+0.0224** over the RoBERTa baseline, with gains across all four categories, while generating a knowledge-graph-grounded explanation for every prediction. A follow-up experiment isolating knowledge graph quality from retrieval method shows the KG construction upgrade accounts for roughly **75%** of this gain, with semantic retrieval contributing the remainder.

Full methodology, results, and limitations are documented in the final report (`/report`).

## What This Project Does

Four models are evaluated on EDOS Task A (binary sexism detection) and Task B (four-category classification: threats, derogation, animosity, prejudiced discussions):

1. **Baseline 1** — RoBERTa fine-tuned with class-weighted loss to address category imbalance.
2. **Baseline 2** — RoBERTa's prediction reviewed and optionally revised by an LLM, with no external knowledge.
3. **Proposed A** — Adds a domain-specific knowledge graph (KG v1) retrieved via keyword/entity matching.
4. **Proposed B+C** — Uses a higher-quality, category-constrained knowledge graph (KG v2) retrieved via semantic similarity.

The knowledge graph is built from labeled sexist posts using a three-stage pipeline (entity extraction, LLM-based triple generation, schema normalization) into six fixed relations mapped directly to EDOS Task B categories:

| Relation | EDOS Category |
|---|---|
| `STEREOTYPED_AS` | Derogation |
| `FRAMED_AS_INFERIOR` | Derogation |
| `THREATENED_WITH` | Threats |
| `EXPRESSED_ANIMOSITY_TOWARDS` | Animosity |
| `IDEOLOGICALLY_DISCREDITED` | Prejudiced discussions |

For Task B, the relation of a retrieved triple carries category-level signal directly, so the LLM's output is both a classification and a semantically grounded explanation — not just a label with highlighted words.

## Repository Structure

```
.
├── data/
│   ├── kg/                          # Knowledge graph JSON files (triples + indexes)
│   │   ├── sexism_kg.json           # KG v1 (raw)
│   │   ├── sexism_kg_clean.json     # KG v1, cleaned
│   │   └── sexism_kg_v2.json        # KG v2 (category-constrained, used by Proposed B+C)
│   └── ...                          # EDOS dataset files
├── models/
│   ├── roberta_task_A/              # Fine-tuned RoBERTa checkpoint, Task A
│   └── roberta_task_B/              # Fine-tuned RoBERTa checkpoint, Task B
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline1_roberta.ipynb
│   ├── 03_baseline2_llm_refiner.ipynb
│   ├── 04_kg_construction.ipynb
│   ├── 04b_kg_construction_v2.ipynb
│   ├── 05_proposed_model_A.ipynb
│   └── 06_Proposed_Model_B.ipynb
├── results/                         # Checkpoints, confusion matrices, classification reports
├── src/                             # Shared helper modules
├── report/                          # Final IEEE-format report and figures
├── .env                             # API keys (not committed — see Setup)
└── .gitignore
```

## Setup

**Requirements:** Python 3.10+, a CUDA-capable or integrated GPU (project developed and evaluated on a single integrated GPU), and a Google Gemini API key.

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root with:

```
GEMINI_API_KEY=your_key_here
```

Notebooks 03, 04, 04b, 05, and 06 require this key for LLM-based classification, refinement, and knowledge graph construction.

## Reproducing the Results

Run notebooks in order — each stage depends on artifacts saved by the previous one:

1. **`01_data_exploration.ipynb`** — Loads the EDOS dataset, computes class distributions and text length statistics.
2. **`02_baseline1_roberta.ipynb`** — Fine-tunes RoBERTa for Task A and Task B with class-weighted loss. Saves checkpoints to `models/`.
3. **`03_baseline2_llm_refiner.ipynb`** — Runs the RoBERTa + LLM refiner pipeline (Baseline 2).
4. **`04_kg_construction.ipynb`** / **`04b_kg_construction_v2.ipynb`** — Builds KG v1 and KG v2 from the Task B training split. Saves to `data/kg/`.
5. **`05_proposed_model_A.ipynb`** — Runs Proposed A (KG v1 clean + keyword retrieval).
6. **`06_Proposed_Model_B.ipynb`** — Runs Proposed B+C (KG v2 + semantic retrieval), the explainability evaluation, and the KG-quality-vs-retrieval-method ablation.

All LLM-dependent pipelines checkpoint every 100 posts to `results/`, so interrupted runs resume automatically rather than restarting from scratch.

## Key Findings

- **Knowledge graph construction quality matters more than retrieval sophistication.** An unconstrained relation-generation prompt (KG v1) produced only 25.3–70.8% relation-category alignment; a category-constrained prompt (KG v2) raised this to 97.4–100%. A controlled ablation isolating this from the retrieval method shows the KG upgrade accounts for ~75% of the total Task B gain.
- **Task A shows a small, structural regression.** Because the knowledge graph is built exclusively from sexist posts, it contains no negative evidence — 100% of LLM overrides on Task A move from not-sexist to sexist, never the reverse. Confidence gating (LLM invoked only below 0.80 RoBERTa confidence) limits this to ~0.002–0.003 Macro F1.
- **Explanations are generally well-grounded, with one clear exception.** A manual evaluation of 50 explanations rates them consistently well on relevance, consistency, and usefulness — except for the animosity category, which scores roughly a full point lower on every metric, likely due to its more implicit, sarcastic language.
- **The model never admits a weak retrieval match.** Despite an explicit prompt instruction to do so, 0 of 45 valid explanations ever stated that a retrieved relation wasn't relevant — even in cases where the model's own wording hedged on the connection. This raises an open question about whether "grounded" explanations are always genuinely selective.

Full discussion in Sections VII and VIII of the report.

## Limitations

- No negative evidence in the knowledge graph (see Task A regression above).
- Explanation grounding selectivity is not independently verified.
- Single-rater manual evaluation, with no inter-rater agreement check.
- Compute-constrained: single `roberta-base` model on a single integrated GPU, no ensembling, continued pre-training, or data augmentation — unlike current state-of-the-art EDOS systems.
- Task C (11-way fine-grained classification) was not evaluated.

## Citation

If referencing this work, please cite the accompanying report:

```
A. Nair, "Knowledge-Graph-Enhanced LLM Reasoning for Improved and Explainable
Sexism Detection," MSc Capstone Project, University of Galway, 2026.
```

## Use of AI

Claude (Anthropic) was used as a productivity aid for implementation code — following an established structure, writing fallback and error-handling logic, and supporting debugging — and for drafting grammatically coherent, well-structured report text based on the author's own findings and direction. Experimental design, methodology, and result interpretation are the author's own. See Section X of the report for the full declaration.

## Acknowledgments

Supervised by Jamal Nasir, University of Galway. Built on the [EDOS dataset](https://arxiv.org/abs/2303.04222) (Kirk et al., SemEval-2023 Task 10).
