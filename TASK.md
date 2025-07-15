# TASK.md – Independent Single‑LLM Project

| # | Task | Owner / Agent | Input | Output | Acceptance | ETA |
|---|------|---------------|-------|--------|------------|-----|
| 1 | **Repo scaffold** | DevOps | – | Git skeleton, `requirements.txt` | `pip install` succeeds | D‑1 |
| 2 | **Download dataset** | Data | GitHub raw JSON | `data/finfact.json` | SHA‑256 verified | Day 1 |
| 3 | **EDA notebook** | Data | finfact.json | `notebooks/eda.ipynb` | Runs in Colab ≤ 2 min | Day 2 |
| 4 | **Prompt bank builder** | Prompt | train split | `prompts/*.txt` | 8 balanced examples | Day 3 |
| 5 | **Script A – Zero‑shot** | LLM | standard prompt | `preds_standard.csv` | – | Day 5 |
| 6 | **Script B – Zero‑shot CoT** | LLM | cot prompt | `preds_cot.csv` | – | Day 5 |
| 7 | **Script C – Few‑shot CoT** | LLM | few‑shot prompt | `preds_fewshot.csv` | – | Day 6 |
| 8 | **Evaluate metrics** | Eval | predictions + labels | `reports/metrics.md` | Accuracy + F1s table | Day 7 |
| 9 | **Unit tests & CI** | QA | source code | GitHub Actions pass | ≥ 80 % coverage | Day 8 |
|10 | **Docs & report** | Docs | metrics + planning | Updated docs | Render OK | Day 10 |

---

## Dependencies
2 → 3 → 4 → (5‑7) → 8 → 10

---

## Model Invocation Examples
```bash
python scripts/zeroshot_standard.py --model o4-mini --temp 0.3
python scripts/zeroshot_cot.py      --model o4-mini --temp 0.3
python scripts/fewshot_cot.py       --model o4-mini --temp 0.3
```
*Set `--model gpt-4.1` only if `o4-mini` is unavailable.*

---

### Exit Criteria
* Colab EDA independent of inference scripts  
* Each LLM script callable **stand‑alone** as shown above  
* Reproducible metrics within ±0.5 %  
* Total API spend ≤ USD 5

---

*Documents maintained by ML/DS expert (LLM specialist).*
