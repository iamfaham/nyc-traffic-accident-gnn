# NYC Traffic Accident Prediction using Spatiotemporal Graph Neural Networks

A deep learning project that predicts traffic accidents in New York City using Graph Neural Networks (GNNs) with spatial and temporal features.

## Project Overview

This project leverages the NYC road network topology and historical crash data to predict collision likelihood at different intersections throughout the day. The model uses GraphSAGE (Graph Sample and Aggregate) architecture to capture both spatial relationships between road segments and temporal patterns in accident occurrence.

### Key Features

- **Spatiotemporal Prediction**: Combines road network structure with temporal patterns
- **Real-time Data**: Uses NYC Open Data collision records (100K+ recent crashes)
- **Graph-based Approach**: Treats road networks as graphs with intersections as nodes
- **Deep Learning**: GraphSAGE neural network with 3 layers and 208K+ parameters
- **Risk Assessment**: Identifies high-risk locations and times

## Results

| Metric | Performance |
|--------|-------------|
| **F1 Score** | 0.113 |
| **AUC-ROC** | 0.764 |
| **Recall** | 0.751 |
| **Precision** | 0.061 |
| **Accuracy** | 67.52% |

The model successfully identifies **75% of actual crashes** (high recall) but has challenges with precision due to extreme class imbalance (only 2.8% of samples are positive crashes).

## Architecture

### Data Pipeline

1. **Data Collection**: Downloads collision data from NYC Open Data API
2. **Road Network**: Extracts road topology using OSMnx (OpenStreetMap)
3. **Spatial Join**: Maps crashes to nearest road network nodes
4. **Negative Sampling**: Creates balanced temporal windows (20% sampling rate)
5. **Feature Engineering**: Extracts node features and temporal patterns

### Model Architecture

```
Input Features (11) → GraphSAGE Layer 1 (256) → BatchNorm → ReLU → Dropout(0.4)
→ GraphSAGE Layer 2 (256) → BatchNorm → ReLU → Dropout(0.4)
→ GraphSAGE Layer 3 (128) → BatchNorm → ReLU → Dropout(0.4)
→ FC Layer (32) → ReLU → Dropout(0.4)
→ Output (2 classes)
```


**Loss Function**: Weighted Cross-Entropy with 15.6x weight on positive class

## Dataset

### Data Sources

- **NYC Motor Vehicle Collisions**: [NYC Open Data Portal](https://data.cityofnewyork.us/)
- **Road Network**: OpenStreetMap via OSMnx
- **Coverage**: August 2024 - October 2025
- **Sample Size**: 100,000 collision records

### Features

**Node Features (7)**:
- `degree`: Node connectivity
- `num_edges`: Number of connected roads
- `highway_class`: Road classification (0-5)
- `avg_lanes`: Average number of lanes
- `avg_speed`: Average speed limit
- `longitude`, `latitude`: Geographic coordinates

**Temporal Features (4)**:
- `hour`: Hour of day (0-23)
- `day_of_week`: Day of week (0-6)
- `is_weekend`: Weekend indicator
- `month`: Month (1-12)

**Historical Features (2)**:
- `past_30d_crashes`: Crashes in past 30 days
- `days_since_last`: Days since last crash

## Getting Started

### Prerequisites

```
pip install osmnx geopandas networkx torch torch-geometric folium sodapy pandas numpy matplotlib seaborn scikit-learn
```

### Installation

1. Clone the repository:
```
git clone 
```

2. Open the notebook:
```
jupyter notebook nyc_traffic_accident_prediction.ipynb
```


3. Run all cells sequentially

### Usage

The notebook is organized into clear sections:

1. **Setup & Data Collection**: Downloads NYC collision data
2. **Road Network Processing**: Builds graph from OpenStreetMap
3. **Feature Engineering**: Creates node and temporal features
4. **Model Training**: Trains GraphSAGE model
5. **Evaluation**: Tests model performance
6. **Visualization**: Creates risk heatmaps

## 📈 Training Details

- **Train/Val/Test Split**: 60% / 20% / 20% (temporal split)
- **Optimizer**: Adam (lr=0.0005, weight_decay=1e-4)
- **Batch Size**: 50,000 training samples (balanced)
- **Epochs**: 60 (early stopping with patience=20)
- **GPU**: Trained on Google Colab with T4 GPU

### Class Imbalance Handling

- Weighted loss function (95% weight on minority class)
- Balanced mini-batch sampling (10% positive ratio)
- Focal loss experimentation

## Visualizations

The notebook generates interactive maps showing:
- High-risk intersections (top 10% risk scores)
- Temporal risk patterns
- Historical crash hotspots

## Key Insights

1. **Temporal Patterns**: Accident risk varies significantly by hour and day
2. **Network Effects**: Well-connected intersections show different risk profiles
3. **Historical Context**: Past crashes are strong predictors of future events
4. **Extreme Imbalance**: Only 0.041% of node-time pairs result in crashes

## Limitations

- **Class Imbalance**: Extreme imbalance leads to low precision
- **Data Coverage**: Limited to NYC and recent time period
- **Feature Engineering**: Could benefit from weather, traffic volume data
- **Sampling**: 20% time window sampling for memory efficiency

## References

- **GraphSAGE**: Hamilton et al., "Inductive Representation Learning on Large Graphs"
- **NYC Open Data**: [data.cityofnewyork.us](https://data.cityofnewyork.us/)
- **OSMnx**: Boeing, G., "OSMnx: New methods for acquiring, constructing, analyzing, and visualizing complex street networks"

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## Author
Created as a deep learning project for spatiotemporal prediction on graph-structured data.

## Acknowledgments
- NYC Open Data for collision data
- OpenStreetMap contributors
- PyTorch Geometric team

---

**Note**: This is a research/educational project. For production traffic safety systems, additional validation, real-time data, and domain expert consultation would be required.
