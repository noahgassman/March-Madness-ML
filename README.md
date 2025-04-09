# NCAA March Madness Tournament Predictions

## Project Overview

This project builds a predictive model for NCAA Men’s March Madness tournament games leveraging advanced machine learning approaches and statistical analyses. The objective is not just to predict tournament outcomes but to understand why certain teams win — uncovering insights from team-level performance metrics, opponent-adjusted statistics, and pre-game contextual features like seeding and rankings.

The broader purpose of this work is twofold:
- Showcase how machine learning can be applied to real-world decision-making under uncertainty (in this case, tournament brackets).
- Create a well-engineered, transparent modeling pipeline that enables performance evaluation, interpretability, and iteration.

## Problem Area & Motivation

Each March, the NCAA tournament captivates fans with its chaos — underdogs win, brackets bust, and analytics are tested in the real world. The tournament’s format — single elimination, 64 teams, limited inter-conference overlap — creates a unique modeling challenge: a binary classification problem with small sample sizes, high variance, and deeply imbalanced priors.

The scale of the problem is staggering: With 63 games and two possible outcomes per game, there are over 9.2 quintillion potential ways to fill out a complete bracket. This makes the probability of randomly predicting a perfect bracket effectively zero. Even expert intuition, advanced statistics, and historical heuristics struggle to consistently outperform randomness in this domain.

From a data science lens, March Madness represents a rare opportunity to apply machine learning to a real-world, high-stakes problem characterized by:
- Highly imbalanced priors (e.g., 16-seeds rarely beat 1-seeds)
- Nonlinear interactions between play styles, rankings, and metrics
- A uniquely constrained prediction target (binary win/loss in tournament context)

Our goal is to construct a robust feature pipeline and predictive model that quantifies pre-game team strength, adjusts for opponent context, and generates probabilities for each potential matchup — moving beyond anecdotes and toward evidence-based bracket forecasting.

This is not just a modeling exercise — it’s a test of how well statistical learning can capture the subtleties of competition, adapt to unpredictable environments, and bring signal to one of the most beloved spectacles in sports.

## Stakeholders & Users
- Basketball analysts and fans looking to quantify team strengths and evaluate matchups beyond seeding.
- Bracketologists and betting markets interested in improving forecast accuracy.
- Sports data science professionals and organizations leveraging modeling for player/team evaluation or content creation.

## Machine Learning Approach

The project takes a supervised learning approach to binary classification, where each observation is a historical NCAA tournament game and the target variable is whether a given team won. This modeling approach involves:
- Constructing a rich feature set from regular season and tournament data.
- Engineering domain-specific metrics: efficiency stats, win/loss splits, opponent-adjusted differentials, tempo, and more.
- Benchmarking performance using a rule-based baseline model (higher seed wins).
- Tuning and evaluating multiple models, including:
    - Logistic Regression (baseline and regularized)
    - XGBoost (gradient boosting)
    - ***TBD in next steps***

Each model is assessed with classification metrics such as AUC, accuracy, F1, and confusion matrices, with performance compared against the baseline.

## Business & Societal Impact

- Improved tournament modeling contributes to a growing field of sports analytics, where data-driven decisions are transforming how teams are built and evaluated.
- In public and private bracket pools, probabilistic modeling offers a statistically grounded alternative to gut feel or historical bias.
- Educational value: the project demonstrates how to apply advanced ML workflows (feature engineering, cross-validation, model interpretation) to real, messy data in a familiar domain.

## Data Description

The dataset was built from multiple sources:
- Historical NCAA tournament and regular season results (via Kaggle and March Madness datasets).
- Advanced stats from KenPom and Massey Ordinals.
- Custom-engineered features derived from per-game box score data.

We structured the data as matchup-level observations, with team-specific features from regular season performance carried into the tournament. Features include:

**Game Context**
- Season, DayNum, Team IDs and Names, Location, Overtime

**Box Score Stats**
- Scoring, Shooting (FG, 3P, FT), Rebounds, Assists, Turnovers, Steals, Blocks, Fouls
- All stats are available for both team and opponent

**Advanced Metrics**
- Tempo-adjusted Possessions, Offensive/Defensive Efficiency
- Percentage-based stats (FG%, FT%, AST/TO ratio)
- Differential and opponent-adjusted features

**Engineered Aggregates**
- Season-level per-game averages for wins, losses, and total games
- Delta features (team minus opponent)
- Ranking features (KenPom, Massey Ordinals)

**Target Variable**
- Binary win/loss flag for tournament games

## Next Steps
- Potential dimension reduction and rigous feature selection process
- Add additional advance metrics
- Build a bracket simulation framework

---
