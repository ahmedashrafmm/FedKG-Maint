# FedKG-Maint: Privacy-Preserving Predictive Maintenance

Federated learning + maintenance knowledge graph + LLM explanations, designed to run on a **free Google Colab T4**.

Factories each have too few failure examples and cannot share raw data. FedKG-Maint lets them train one
shared RUL / failure predictor with federated learning, protects model updates with differential privacy,
and explains each prediction to maintenance engineers with an LLM grounded in a knowledge graph.

## Pipeline

```
 Factories (FD001-FD003 / Azure groups)                 Held-out factory (FD004)
   local data -> local normalisation -> local KG                 |
        | model updates only (clipped + noised)                  | evaluation only
        v                                                        |
   Aggregation server (FedAvg / FedProx) ----------------------->+
        |
   global RUL / failure model
        |
   prediction + retrieved KG facts -> LLM -> grounded explanation
```

## Datasets
- NASA C-MAPSS (FD001-FD004): https://data.nasa.gov/dataset/c-mapss-aircraft-engine-simulator-data
- Microsoft Azure Predictive Maintenance: https://www.kaggle.com/datasets/arnabbiswas1/microsoft-azurepredictive-maintenance

Place the C-MAPSS text files in `data/raw/CMAPSS/` (`train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt`, ...).

## Quick start (Colab)
```bash
!git clone https://github.com/<your-username>/FedKG-Maint.git
%cd FedKG-Maint
!pip install -r requirements.txt
!python scripts/train_federated.py --mode centralized
!python scripts/train_federated.py --mode local
!python scripts/train_federated.py --strategy fedavg
!python scripts/train_federated.py --strategy fedprox --dp
```

## Roadmap
- [x] Repo skeleton, C-MAPSS pipeline, LSTM, FedAvg/FedProx, simple DP, metrics
- [ ] Phase 1: centralized + local baselines, held-out factory results
- [ ] Phase 2: federated strategies, heterogeneity and client-count studies
- [ ] Phase 3: privacy trade-off (Opacus / accountant), leakage discussion
- [ ] Phase 4: knowledge graph + LLM explanations (with vs without grounding)
- [ ] Phase 5: Azure PdM failure prediction (F1, PR-AUC)
- [ ] Phase 6: ablations, multiple seeds, significance tests, final report

## Structure
```
configs/   experiment configs        src/fedkg/  library code
scripts/   training entry points     docs/       project proposal PDF
notebooks/ Colab notebooks           results/    metrics, plots
```

## License
MIT
