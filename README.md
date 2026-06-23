# Twitter Bot Detector

> Detects automated bot accounts on Twitter using behavioral and profile features.

## Features Used
- Follower/Following ratio
- Tweet frequency
- Account age
- Profile completeness
- Engagement patterns

## Dataset
[Twitter Bot Accounts — Kaggle](https://www.kaggle.com/datasets/davidmartngutierrez/twitter-bots-accounts)

## Tech Stack
- Python, Scikit-learn, Pandas, Matplotlib

## Quick Start
```bash
git clone git@github.com:Mohdasham/twitter-bot-detector.git
cd twitter-bot-detector
pip install -r requirements.txt
python bot_detector.py
```

## Results
| Metric | Score |
|--------|-------|
| Accuracy | 93.5% |
| ROC-AUC | 0.96 |

