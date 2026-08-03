# 🏀 NBA Game Predictor

A machine learning API that computes win probabilities for NBA matchups using historical player and team statistics, with a JavaScript frontend for live matchup display.

## Tech Stack

- **Python** — core ML pipeline and API logic
- **Flask** — REST API and endpoint routing
- **Pandas / NumPy** — data ingestion, cleaning, and feature standardization
- **scikit-learn** — model training and serialization
- **JavaScript / HTML / CSS** — frontend client for live matchups

## Project Structure

basketball-app/
├── src/
│ ├── api_basketball/
│ │ ├── data/
│ │ │ ├── raw/
│ │ │ │ └── games.csv # Raw NBA game records
│ │ │ └── processed/
│ │ │ └── game_features.csv # Cleaned, model-ready data
│ │ ├── models/game_outcome/
│ │ │ ├── model.pkl # Trained model
│ │ │ ├── features.json # Feature definitions
│ │ │ └── metrics.json # Model performance metrics
│ │ ├── api/ # Flask route definitions
│ │ └── main.py # App entry point
│ ├── index.html # Frontend UI
│ ├── script.js # Frontend logic
│ └── styles.css # Styling
├── scripts/ # Utility and preprocessing scripts
├── .vscode/
└── README.md

## Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
git clone https://github.com/kylershih0614/basketball-app.git
cd basketball-app
pip install -r requirements.txt
```

### Running the API

```bash
python src/api_basketball/main.py
```

### Running the Frontend

Open `src/index.html` in your browser or serve it with a local server.

## Data Pipeline

Raw game records in `data/raw/games.csv` are processed through a preprocessing pipeline that cleans, standardizes, and engineers features before outputting to `data/processed/game_features.csv` for model training.

## Model

The trained model is serialized to `models/game_outcome/model.pkl`. Feature definitions and performance metrics are stored alongside it in `features.json` and `metrics.json`.

## Author

**Kyler Shih**
[github.com/kylershih0614](https://github.com/kylershih0614) · [linkedin.com/in/kyler-shih](https://linkedin.com/in/kyler-shih)
