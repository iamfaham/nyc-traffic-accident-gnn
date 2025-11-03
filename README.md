# NYC Traffic Accident Prediction using Graph Neural Networks

Predicting traffic accidents at NYC intersections using spatiotemporal Graph Neural Networks with road network topology.

## 🎯 Results

- **F1 Score: 0.150** (best model)
- **Recall: 29.7%** (catches 30% of crashes)
- **AUC-ROC: 0.750**
- **Accuracy: 70.6%**

## 📊 Data

- **100,000 crash records** from NYC Open Data (Sept 2024 - Oct 2025)
- **4,618 intersections** across Manhattan, Brooklyn, Bronx
- **8,185 road connections** from OpenStreetMap
- **6.4M node-time pairs** after negative sampling

## 🏗️ Model Architecture

```

GraphSAGE (3 layers) with Temporal Features
Input (11 features) → 256 → 256 → 128 → Output (crash/no-crash)

```

## 📋 Features (11 Total)

**Static Road Features:**

- `degree`, `num_edges`, `highway_class`, `avg_lanes`, `avg_speed`

**Temporal Features:**

- `hour`, `day_of_week`, `is_weekend`, `month`

**Historical Features:**

- `past_30d_crashes`, `days_since_last`

## 🚀 Quick Start

### Installation

```

pip install -r requirements.txt

```

### Run Training

```

jupyter notebook nyc_traffic_accident_prediction.ipynb

```

### Make Predictions

```

import torch
from model import SpatiotemporalGNN

model = SpatiotemporalGNN(num_features=11, hidden_dim=256, output_dim=2)
model.load_state_dict(torch.load('best_model.pt'))

predictions = model(X_test, edge_index, node_indices)
crash_probs = torch.softmax(predictions, dim=1)[:, 1]

```

## 📁 Project Structure

```

.
├── notebooks/
│   └── nyc_traffic_accident_prediction.ipynb
├── src/
│   ├── data/
│   ├── model/
│   └── inference/
├── models/
│   └── best_model.pt
├── requirements.txt
└── README.md

```

## 🔬 Key Experiments

| Model                    | F1        | Precision | Recall    |
| ------------------------ | --------- | --------- | --------- |
| **Best (11 features)**   | **0.150** | **0.101** | **0.297** |
| 19 features + focal loss | 0.147     | 0.083     | 0.697     |
| Temporal attention       | 0.134     | 0.078     | 0.651     |

## ⚠️ Limitations

1. **Cold Start Problem** - Model relies on `past_30d_crashes`, cannot predict at new intersections
2. **No Real Traffic Data** - Uses temporal proxies (rush hour) instead of actual vehicle counts
3. **Class Imbalance** - Only 3.2% crashes in dataset, leads to lower precision (8.3%)

## 🔮 Future Work

- [ ] Reduce historical data dependency using temporal crash patterns
- [ ] Integrate real traffic volume data (Google Maps, NYC DOT APIs)
- [ ] Multi-task learning (crash occurrence + severity)
- [ ] Test on other cities for generalization

## 📚 Data Sources

- **Crashes**: [NYC Open Data Portal](https://data.cityofnewyork.us/resource/h9gi-nx95)
- **Road Network**: [OpenStreetMap](https://www.openstreetmap.org)
- **Speed Limits**: OSM tags

## 📊 Confusion Matrix (Test Set)

```


            Predicted Safe      Predicted Crash

Actual Safe      8695            3575  ← False positives
ActualCrash      126             283  ← True positives (catches 69%)

```

**Interpretation:** Model is optimized for **safety** (catches most crashes) rather than efficiency (low false alarm rate).
