# numerai-sfl-pipeline

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Numerai SFL Pipeline - Systemic Functional Linguistics-based prediction model with GitHub Actions automation. Multi-agent orchestration system for Numerai tournament predictions leveraging linguistic theory for feature analysis and model selection.

## Overview

This project applies Systemic Functional Linguistics (SFL) principles to quantitative finance prediction in the Numerai tournament. By treating financial data as a meaning-making system, the pipeline analyzes market features through linguistic lenses to identify predictive patterns and semantic relationships traditional models might miss.

### Core Innovation

**SFL-Enhanced Feature Analysis**
- Treats financial features as semantic units with ideational (content), interpersonal (stance), and textual (structure) meaning
- Applies transitivity analysis to identify process types in market behavior
- Uses appraisal theory to evaluate feature significance and relationships
- Employs cohesion analysis to detect feature co-occurrence patterns

**Multiagent Architecture**
- Orchestrated agent system for parallel feature processing
- Master agent coordinates specialized analysis agents
- Event-driven communication for real-time model updates
- Fallback mechanisms for robust production deployment

## Features

- 🤖 **Automated Daily Predictions**: GitHub Actions workflow for tournament submissions
- 🧠 **SFL-Based Feature Engineering**: Linguistic analysis of market data semantics
- 📊 **Multi-Model Ensemble**: LightGBM, XGBoost, and Neural Network integration
- 🔄 **Agent Orchestration**: Multiagent system for parallel processing
- 📈 **Performance Tracking**: Historical prediction analysis and model diagnostics
- 🔐 **Secure Credential Management**: GitHub Secrets integration
- 🚀 **Zero-Config Deployment**: Automated setup and execution

## Architecture

```
numerai-sfl-pipeline/
├── .github/
│   └── workflows/
│       └── daily_submission.yml    # Automated prediction workflow
├── sfl_daily.py                     # Main prediction pipeline
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
└── .gitignore                       # Git ignore rules
```

## SFL Theory Application

### Transitivity Analysis
Identifies process types in feature behavior:
- **Material processes**: Direct feature transformations (e.g., price changes)
- **Mental processes**: Market sentiment indicators
- **Relational processes**: Feature correlations and dependencies
- **Verbal processes**: News/communication-based features

### Appraisal Framework
Evaluates feature importance through three dimensions:
- **Attitude**: Feature valence (positive/negative market indicators)
- **Engagement**: Feature consistency (monogloss vs. heterogloss patterns)
- **Graduation**: Feature intensity (force and focus measurements)

### Cohesion Mapping
Detects feature relationships:
- **Reference chains**: Features that co-occur across time
- **Conjunction patterns**: Logical relationships between indicators
- **Lexical cohesion**: Semantic similarity clustering

## Installation

### Prerequisites

- Python 3.8+
- Numerai API credentials
- GitHub account (for automated deployment)

### Local Setup

```bash
# Clone repository
git clone https://github.com/simon-drury/numerai-sfl-pipeline.git
cd numerai-sfl-pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure credentials (see Configuration section)
```

## Configuration

### GitHub Secrets (for automated workflow)

Set the following secrets in your repository settings:

```
NUMERAI_PUBLIC_KEY    # Your Numerai API public key
NUMERAI_SECRET_KEY    # Your Numerai API secret key  
MODEL_ID              # Your Numerai model ID (optional)
```

### Local Environment Variables

Create a `.env` file (not tracked in git):

```bash
NUMERAI_PUBLIC_KEY=your_public_key_here
NUMERAI_SECRET_KEY=your_secret_key_here
MODEL_ID=your_model_id_here
```

## Usage

### Automated Daily Predictions

The GitHub Actions workflow runs automatically every day at 16:00 UTC during tournament rounds. No manual intervention required.

### Manual Execution

```bash
# Run prediction pipeline
python sfl_daily.py

# Test without submission
python sfl_daily.py --dry-run

# Verbose logging
python sfl_daily.py --verbose
```

### Python API

```python
from sfl_daily import NumeraiSFLPipeline

# Initialize pipeline
pipeline = NumeraiSFLPipeline(
    public_key="your_public_key",
    secret_key="your_secret_key",
    model_id="your_model_id"
)

# Run SFL feature analysis
sfl_features = pipeline.analyze_features_sfl(data)

# Generate predictions
predictions = pipeline.predict()

# Submit to tournament
pipeline.submit(predictions)
```

## SFL Feature Engineering Pipeline

```python
def analyze_features_sfl(data):
    """
    Apply SFL analysis to financial features
    
    1. Transitivity Mapping:
       - Classify features by process type
       - Extract participant roles (actors, goals, etc.)
       - Identify circumstantial information
    
    2. Appraisal Analysis:
       - Measure attitude (positive/negative valence)
       - Assess engagement (certainty/hedging)
       - Quantify graduation (intensity)
    
    3. Cohesion Detection:
       - Build feature co-occurrence networks
       - Identify reference chains
       - Map lexical relationships
    
    Returns:
        Enhanced feature matrix with SFL-derived signals
    """
    pass
```

## Model Architecture

### Ensemble Components

1. **LightGBM** (Primary)
   - Gradient boosting with SFL-enhanced features
   - Fast training, handles missing values
   - Feature importance for SFL validation

2. **XGBoost** (Secondary)
   - Alternative gradient boosting
   - Cross-validation for robustness
   - Feature interaction capture

3. **Neural Network** (Experimental)
   - Deep learning for non-linear SFL patterns
   - Embedding layers for categorical features
   - Attention mechanisms for feature relationships

### Ensemble Strategy

```python
predictions = (
    0.6 * lightgbm_pred +
    0.3 * xgboost_pred +
    0.1 * neural_net_pred
)
```

## Performance Metrics

The pipeline tracks:
- **Correlation**: Primary Numerai metric
- **MMC (Meta Model Contribution)**: Orthogonality to meta-model
- **FNC (Feature Neutral Correlation)**: Feature exposure risk
- **Sharpe Ratio**: Risk-adjusted returns
- **Max Drawdown**: Worst-case performance

## Multiagent Orchestration

### Agent Types

- **Master Agent**: Coordinates workflow, manages state
- **Data Agent**: Fetches and preprocesses tournament data
- **SFL Analysis Agent**: Performs linguistic feature engineering
- **Model Agent**: Trains and generates predictions
- **Submission Agent**: Uploads predictions to Numerai
- **Monitor Agent**: Tracks performance and alerts

### Communication Protocol

```python
{
    "event_type": "feature_analysis_complete",
    "agent_id": "sfl_analyzer_01",
    "payload": {
        "sfl_features": [...],
        "transitivity_map": {...},
        "appraisal_scores": {...}
    },
    "timestamp": "2026-01-22T21:30:00Z"
}
```

## GitHub Actions Workflow

```yaml
name: Daily Numerai Prediction
on:
  schedule:
    - cron: '0 16 * * *'  # 16:00 UTC daily
  workflow_dispatch:       # Manual trigger

jobs:
  predict:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
      - run: pip install -r requirements.txt
      - run: python sfl_daily.py
        env:
          NUMERAI_PUBLIC_KEY: ${{ secrets.NUMERAI_PUBLIC_KEY }}
          NUMERAI_SECRET_KEY: ${{ secrets.NUMERAI_SECRET_KEY }}
```

## Development

### Running Tests

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Coverage report
pytest --cov=sfl_daily tests/
```

### Code Style

```bash
# Format code
black sfl_daily.py

# Lint
flake8 sfl_daily.py
pylint sfl_daily.py

# Type checking
mypy sfl_daily.py
```

## Roadmap

- [ ] Expand SFL analysis to news/sentiment data
- [ ] Implement adaptive agent routing based on market conditions
- [ ] Add explainability dashboard for SFL feature contributions
- [ ] Multi-model stacking with SFL meta-features
- [ ] Real-time prediction updates via webhook agents
- [ ] Comparative analysis across multiple Numerai models

## CV/Portfolio Context

This repository demonstrates:

- **Applied Linguistics**: Practical implementation of SFL theory in quantitative finance
- **Data Science**: Feature engineering, ensemble modeling, performance optimization
- **MLOps**: Automated pipelines, CI/CD, reproducible workflows
- **Multiagent Systems**: Orchestrated architecture for complex task decomposition
- **Software Engineering**: Clean code, testing, documentation, version control

Ideal for roles requiring:
- Computational linguistics with real-world applications
- Quantitative research and model development
- Production ML system design and deployment
- Interdisciplinary problem-solving (linguistics + finance + ML)

## License

MIT License - see [LICENSE](LICENSE) file for details

## Citation

If using this system in research or derivative works:

```bibtex
@software{numerai_sfl_pipeline,
  author = {Drury, Simon James},
  title = {Numerai SFL Pipeline: Linguistic Theory for Financial Prediction},
  year = {2026},
  url = {https://github.com/simon-drury/numerai-sfl-pipeline}
}
```

## Contact

**Simon James Drury**
- Email: simondrury2010@gmail.com
- LinkedIn: [simon-drury-60b881293](https://www.linkedin.com/in/simon-drury-60b881293)
- GitHub: [@simon-drury](https://github.com/simon-drury)

## Acknowledgments

- Numerai tournament for providing the data platform
- SFL research community for theoretical foundations
- Open-source ML libraries (scikit-learn, LightGBM, XGBoost, etc.)

## Related Projects

- [sfl-translation-agent](https://github.com/simon-drury/sfl-translation-agent) - SFL-based translation system
- [appraisal-whisperer](https://github.com/simon-drury/appraisal-whisperer) - Performance review analysis
- [agentic-orchestration-sys](https://github.com/simon-drury/agentic-orchestration-sys.-genAI-community_works) - GenAI agent infrastructure
