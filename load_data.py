# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 09:15:08 2026
@author: flotp

load_data.py
"""

import pandas as pd
from igaming_eda.ig_core import load, print_opt, check_df


def main(verbose: bool = False) -> tuple[pd.DataFrame, ...]:
    from igaming_eda.ig_vars import PLAYERS_COLS, CASINO_COLS, BONUS_COLS

    print_opt()
    casino_df = load("casino", info=False)
    check_df(casino_df, "CASINO", CASINO_COLS)
    poker_df = load("poker", info=False)
    check_df(poker_df, "POKER", ["ts", "player_id"])
    players_df = load("players", columns=PLAYERS_COLS, info=False)
    check_df(players_df, "PLAYERS", ["player_id"])
    bonus_df = load("bonus", info=False)
    check_df(bonus_df, "BONUS", BONUS_COLS)
    tzx_df = load("tzx", info=False)
    check_df(tzx_df, "TRANSACTIONS", ["tx_id", "player_id"])
    rates_df = load("fx_rates")

    return casino_df, poker_df, players_df, bonus_df, tzx_df, rates_df


if __name__ == "__main__":
    main(verbose=True)
    print()
