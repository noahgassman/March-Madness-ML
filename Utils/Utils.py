import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split,GridSearchCV
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, recall_score, precision_score, f1_score, classification_report, precision_recall_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import statsmodels.api as sm
from sklearn.feature_selection import RFECV

import warnings
warnings.filterwarnings("ignore")
#-----------------------------------------------------------------------------------------------------------------------

def create_percentage_col(df, numerator, denominator, new_col):
    """
    Adds a percentage column to the DataFrame based on two existing columns.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - numerator (str): Column name representing the numerator (e.g., FGM).
    - denominator (str): Column name representing the denominator (e.g., FGA).
    - new_col (str): Name of the new percentage column to be created.

    Returns:
    - pd.DataFrame: DataFrame with the new percentage column added.
    """
    if numerator not in df.columns or denominator not in df.columns:
        raise ValueError(f"One or both columns ({numerator}, {denominator}) not found in DataFrame.")

    # Handle scenarios of division by zero
    df[new_col] = np.where(df[denominator] == 0, 0, df[numerator] / df[denominator])
    
    return df

#-----------------------------------------------------------------------------------------------------------------------

def create_difference_col(df, col1, col2, new_col):
    """
    Creates a new column as difference of two (2) columns.
    """
    if col1 not in df.columns or col2 not in df.columns:
        raise ValueError(f"Columns ({col1}, {col2}) not found in DataFrame.")

    df[new_col] = df[col1] - df[col2]
    
    return df

#-----------------------------------------------------------------------------------------------------------------------

def compute_game_features(df):
    """
    Computes per-game efficiency stats and difference metrics before aggregation.
    """
    
    # Efficiency Stats (FG%, 3P%, FT%, etc.)
    for new_col, (num, den) in stat_dict.items():
        df = create_percentage_col(df, num, den, new_col)  # Team efficiency

    # Opponent Efficiency Stats
    # Note: Doing opponent stats in separate loop for ordering purposes-team stats first, then opponent stats, then differences between team and opponent
    for new_col, (num, den) in stat_dict.items():
            df = create_percentage_col(df, f"Opp{num}", f"Opp{den}", f"Opp{new_col}") # Opponent efficiency stats
    
    # Possessions & Ratings
    df["Poss"] = df["FGA"] - df["OR"] + df["TO"] + (0.475 * df["FTA"])
    df["OppPoss"] = df["OppFGA"] - df["OppOR"] + df["OppTO"] + (0.475 * df["OppFTA"])

    df["OffEff"] = (df["Score"] / df["Poss"])
    df["DefEff"] = (df["OppScore"] / df["OppPoss"])

    # Opponent Efficiency Ratings are inverse of team ratings
    df["OppOffEff"] = df["DefEff"]
    df["OppDefEff"] = df["OffEff"]

    # Difference-Based Features (e.g., Margin of Victory, Rebound Diff)
    for stat in diff_stats:
        df = create_difference_col(df, stat, f"Opp{stat}", f"{stat}_Diff")  # Difference from opponent

    return df
#-----------------------------------------------------------------------------------------------------------------------
numeric_cols = []

def compute_aggregates(df):
    """
    Aggregates stats at the season level.
    - Computes total stats (all games)
    - Computes stats for wins/losses separately
    - Computes opponent aggregate stats
    """
    
    # Total season averages (all games)
    total_stats = df.groupby(["Season", "TeamID"])[numeric_cols].mean().reset_index()
    total_stats = total_stats.rename(columns=lambda x: f"{x}_PerGame" if x not in ["Season", "TeamID"] else x)

    # Winning game averages
    win_stats = df[df["Win"] == 1].groupby(["Season", "TeamID"])[numeric_cols].mean().reset_index()
    win_stats = win_stats.rename(columns=lambda x: f"{x}_PerWin" if x not in ["Season", "TeamID"] else x)

    # Losing game averages
    loss_stats = df[df["Win"] == 0].groupby(["Season", "TeamID"])[numeric_cols].mean().reset_index()
    loss_stats = loss_stats.rename(columns=lambda x: f"{x}_PerLoss" if x not in ["Season", "TeamID"] else x)

    # Total Games Played (count occurrences of TeamID)
    total_games = df.groupby(["Season", "TeamID"]).size().reset_index(name="TotalGames")
    total_wins = df.groupby(["Season", "TeamID"])["Win"].sum().reset_index().rename(columns={"Win": "TotalWins"})
    summary_stats = total_games.merge(total_wins, on=["Season", "TeamID"])
    summary_stats["TotalLosses"] = summary_stats["TotalGames"] - summary_stats["TotalWins"]
    summary_stats["WinningPercentage"] = summary_stats["TotalWins"] / summary_stats["TotalGames"]

    # # Merge all aggregations
    team_season_stats = (
        summary_stats
        .merge(total_stats, on=["Season", "TeamID"], how="left")  # Add win count and winning %
        .merge(win_stats, on=["Season", "TeamID"], how="left")
        .merge(loss_stats, on=["Season", "TeamID"], how="left")

    )

    # Fill missing values with 0 for teams that had 0 wins or 0 losses
    team_season_stats = team_season_stats.fillna(0)

    return team_season_stats

#-----------------------------------------------------------------------------------------------------------------------
def false_positive_rate(y_true, y_pred):
    """
    Calculate the False Positive Rate:
    FPR = FP / (FP + TN)

    Args:
        y_true (np.ndarray or pd.Series): Ground truth labels
        y_pred (np.ndarray or pd.Series): Predicted labels

    Returns:
        float: False Positive Rate, or np.nan if undefined
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    fp = np.sum(np.logical_and(y_true == 0, y_pred == 1))
    tn = np.sum(np.logical_and(y_true == 0, y_pred == 0))

    denom = fp + tn
    return np.round(fp / denom, 3) if denom != 0 else np.nan