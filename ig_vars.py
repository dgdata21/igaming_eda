# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 09:37:07 2026

@author: flotp
"""

from pathlib import Path

PL_TXT = [
    "country",
    "currency",
    "language",
    "acquisition_channel",
    "device_reg",
    "gender",
    "vip_level",
    "status",
]

DATES_DT = ["granted_at", "completed_at", "ts", "created_at", "birth_date"]
N64 = ["player_id"]

DIR = "D:/code/igaming_eda/data/"

FILES = {
    "players": "players.parquet",
    "casino": "casino.parquet",
    "poker": "poker.parquet",
    "bonus": "bonus.parquet",
    "tzx": "tzx.parquet",
    "fx_rates": "fx_rates.parquet",
}
TABLES = {name: Path(DIR) / file for name, file in FILES.items()}

CATS_ARG = {
    "players": {"TEXTS": PL_TXT, "DT": DATES_DT},
    "casino": {"DT": DATES_DT},
    "poker": {"DT": DATES_DT},
    "bonus": {"DT": DATES_DT, "N64": N64},
    "tzx": {"DT": DATES_DT, "N64": N64},
    "fx_rates": {"DT": DATES_DT},
}

MEMORY = "2GB"
THREADS = 2
ROWS = 5_000_000

FTD_GR = ["acquisition_channel", "reg_week"]
FTD_AG = {
    "regs": ("player_id", "nunique"),
    "ftd_players": ("is_ftd_7d", "sum"),
    "ftd_players_all": ("ftd_date", "count"),
    "ftd_amount": ("amount_7d_eur", "sum"),
    "ftd_median_dep": ("amount_7d_eur", "median"),
}

PLAYERS_COLS = [
    "player_id",
    "registration_date",
    "country",
    "currency",
    "acquisition_channel",
    "birth_date",
    "status",
]
CASINO_COLS = ["ts", "session_id", "player_id", "game_id"]
BONUS_COLS = ["bonus_id", "granted_at", "player_id"]

AGE_BINS = [-1, 17, 25, 35, 50, 200]
AGE_LBLS = ["under 18", "18-25", "26-35", "36-50", "51+"]

REP_COLS = [
    "acquisition_channel",
    "reg_week",
    "regs",
    "ftd_players",
    "conv",
    "conv_base",
    "z_binom",
    "z_roll",
    "avg_dep",
    "ftd_median_dep",
]


# players_df["age"] = (
#     (players_df["registration_date"] - players_df["birth_date"]).dt.days
#     // 365.25
# ).astype(int)

# age_bins = [17, 25, 35, 50, 100]
# age_labels = ["18-25", "26-35", "36-50", "51+"]

# players_df["age_group"] = pd.cut(
#     players_df["age"], bins=age_bins, labels=age_labels
# )
