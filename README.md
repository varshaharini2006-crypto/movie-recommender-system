# 🎬 Movie Recommender System

A movie recommendation engine built on the MovieLens dataset, comparing three approaches: popularity-based baseline, item-based collaborative filtering, and SVD matrix factorization. Deployed as a live, interactive Streamlit app.

**🔗 Live App:** [movie-recommender-system-asmunk9u6zwttm8xlxqxcv.streamlit.app](https://movie-recommender-system-asmunk9u6zwttm8xlxqxcv.streamlit.app)

## Overview

Given a user's rating history, the app recommends movies they're likely to enjoy but haven't seen yet. Built and evaluated three models to compare trade-offs between prediction accuracy and recommendation relevance.

## Dataset

MovieLens `ml-latest-small`: 100,836 ratings from 610 users across 9,742 movies (98.3% sparse).

## Models & Results

| Model | RMSE ↓ | Coverage | Precision@10 |
|---|---|---|---|
| Popularity Baseline | 0.9675 | 96.03% | **0.0464** |
| **Item-Based CF** | **0.8253** | 78.95% | 0.0157 |
| SVD (tuned) | 0.8630 | ~100% | — |

**Key finding:** Item-based CF achieved the best rating-prediction accuracy (15% lower RMSE than the baseline), even outperforming a hyperparameter-tuned SVD model — likely due to the dataset's relatively small size, where SVD's advantages are less pronounced. However, the popularity baseline achieved ~3x higher precision@10, highlighting a classic accuracy-vs-discoverability trade-off: personalized recommendations can be more accurate on average while still missing more "sure bets" than popular, broadly-liked titles.

## Approach

1. **EDA** — analyzed rating distribution, sparsity, and long-tail popularity patterns
2. **Baseline model** — popularity ranking (avg rating, filtered for a minimum rating count)
3. **Item-based collaborative filtering** — cosine similarity on a mean-centered user-item matrix, with a minimum-evidence threshold to avoid overconfident predictions from thin data
4. **SVD matrix factorization** — trained and hyperparameter-tuned via grid search (scikit-surprise)
5. **Evaluation** — RMSE, coverage, and precision@10 on a held-out 80/20 split
6. **Deployment** — wrapped the best model in a Streamlit app; compacted model artifacts (top-50 neighbors per item instead of the full similarity matrix) to cut file size by over 95% for deployment

## Tech Stack

Python, Pandas, NumPy, scikit-learn, scikit-surprise, Streamlit

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Future Improvements

- Hybrid model combining content-based features (genres) with collaborative filtering
- Address cold-start problem for new users/movies
- Scale to the full MovieLens dataset (25M+ ratings)
