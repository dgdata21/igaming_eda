# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 10:14:29 2026

@author: flotp
"""

"""
FTD EDA: недельные когорты регистраций, 
конверсия в FTD за 7 дней, детектор аномалий.
"""

import numpy as np
import pandas as pd

from igaming_eda.ig_vars import FTD_GR, FTD_AG, AGE_BINS, AGE_LBLS, REP_COLS
from igaming_eda.ig_core import section
from igaming_eda.load_data import main as pf

# --------------------------------------------------------------------------
# Параметры
# --------------------------------------------------------------------------
# FTD считается, если депозит в дни 0..7 после регистрации
FTD_WINDOW_DAYS = 7
MATURITY_DAYS = 6 + FTD_WINDOW_DAYS  # последний регистрант недели + окно FTD
ROLL_WINDOW = 12  # недель в базовом окне
ROLL_MIN_PERIODS = 4
Z_THRESHOLD = 3.0  # 3 вместо 2: много проверок -> меньше ложных срабатываний
# если курса на дату нет, берём последний известный, но не старше N дней
FX_MAX_STALE_DAYS = 7


def _to_ns(s: pd.Series) -> pd.Series:
    """Единый dtype дат: merge_asof падает,
    если у ключей разное разрешение (s/us/ns)."""
    return pd.to_datetime(s).astype("datetime64[ns]")


# --------------------------------------------------------------------------
# 1. Курсы валют
# --------------------------------------------------------------------------
def prepare_fx(fx_df: pd.DataFrame) -> pd.DataFrame:
    """Чистая таблица курсов:
    currency, date, rate_to_eur (amount * rate = EUR)."""
    fx = fx_df[["currency", "date", "rate_to_eur"]].copy()
    fx["date"] = _to_ns(fx["date"])
    fx = fx.dropna(subset=["rate_to_eur"])
    fx = fx[fx["rate_to_eur"] > 0]
    # дубли (currency, date): оставляем последнюю запись
    fx = fx.drop_duplicates(subset=["currency", "date"], keep="last")
    return fx.sort_values("date").reset_index(drop=True)


def attach_fx(ftd: pd.DataFrame, fx: pd.DataFrame) -> pd.DataFrame:
    """Курс на дату FTD в валюте депозита;
    при отсутствии - последний известный."""
    left = ftd.loc[ftd["ftd_date"].notna(), ["ftd_date", "ftd_curr"]]
    left = left.assign(ftd_date=_to_ns(left["ftd_date"])).sort_values(
        "ftd_date"
    )

    merged = pd.merge_asof(
        left,
        fx,
        left_on="ftd_date",
        right_on="date",
        left_by="ftd_curr",
        right_by="currency",
        direction="backward",
        tolerance=pd.Timedelta(days=FX_MAX_STALE_DAYS),
    )
    # merge_asof сбрасывает индекс; порядок строк сохранён, возвращаем метки
    merged.index = left.index

    # присваивание по индексу: у игроков без FTD получится NaN
    ftd["fx_rate"] = merged["rate_to_eur"]
    ftd["fx_gap_days"] = (merged["ftd_date"] - merged["date"]).dt.days
    return ftd


# --------------------------------------------------------------------------
# 2. Таблица игроков с первым депозитом
# --------------------------------------------------------------------------
def build_ftd(
    players_df: pd.DataFrame, tzx_df: pd.DataFrame, fx_df: pd.DataFrame
) -> pd.DataFrame:
    deposits = tzx_df[
        (tzx_df["type"] == "deposit") & (tzx_df["status"] == "success")
    ]

    # tx_id вторым ключом: created_at без времени, внутри дня порядок иначе произвольный
    first_dep = (
        deposits.sort_values(["created_at", "tx_id"])
        .drop_duplicates(subset="player_id", keep="first")[
            ["player_id", "tx_id", "created_at", "currency", "amount"]
        ]
        .rename(
            columns={
                "tx_id": "ftd_tx_id",
                "created_at": "ftd_date",
                "currency": "ftd_curr",
            }
        )
    )

    ftd = players_df.merge(first_dep, how="left", on="player_id").rename(
        columns={"registration_date": "reg_date"}
    )
    for col in ("reg_date", "birth_date", "ftd_date"):
        ftd[col] = pd.to_datetime(ftd[col])

    # возраст на момент регистрации; NaN-безопасно (без astype(int))
    ftd["age"] = np.floor(
        (ftd["reg_date"] - ftd["birth_date"]).dt.days / 365.25
    )
    ftd["age_group"] = pd.cut(ftd["age"], bins=AGE_BINS, labels=AGE_LBLS)

    ftd["reg_week"] = ftd["reg_date"].dt.to_period("W").dt.start_time

    # фиксированное окно FTD: когорты становятся сравнимыми
    ftd["days_to_ftd"] = (ftd["ftd_date"] - ftd["reg_date"]).dt.days
    ftd["is_ftd_7d"] = ftd["days_to_ftd"].between(
        0, FTD_WINDOW_DAYS
    )  # NaT -> False

    # сумма в EUR по курсу на дату депозита, только внутри окна
    ftd = attach_fx(ftd, prepare_fx(fx_df))
    ftd["amount_7d_eur"] = (ftd["amount"] * ftd["fx_rate"]).where(
        ftd["is_ftd_7d"]
    )

    return ftd


# --------------------------------------------------------------------------
# 3. Проверки качества данных
# --------------------------------------------------------------------------
def check_fx(fx_df: pd.DataFrame) -> None:
    print("=" * 20, "FX CHECKS", "=" * 20)
    dates = pd.to_datetime(fx_df["date"])
    print(
        f"Строк: {len(fx_df)}, валют: {fx_df['currency'].nunique()}, "
        f"период: {dates.min().date()} - {dates.max().date()}"
    )
    print(
        f"Пропусков курса: {fx_df['rate_to_eur'].isna().sum()}, "
        f"курсов <= 0: {(fx_df['rate_to_eur'] <= 0).sum()}, "
        f"дублей (currency, date):"
        f"{fx_df.duplicated(['currency', 'date']).sum()}"
    )
    med = (
        fx_df.groupby("currency", observed=True)["rate_to_eur"]
        .median()
        .round(4)
    )
    print("Медианный курс к EUR (sanity: 1 единица валюты = N EUR):")
    print(med.to_string())
    print()


def check_ftd(ftd: pd.DataFrame) -> None:
    section("FTD CHECKS")
    #    print("=" * 20, "FTD CHECKS", "=" * 20)
    print(
        f"Игроков: {len(ftd)}, с FTD (всего):"
        f"{ftd['ftd_date'].notna().sum()}, "
        f"с FTD за {FTD_WINDOW_DAYS} дн.: {ftd['is_ftd_7d'].sum()}"
    )

    n_bad = (ftd["days_to_ftd"] < 0).sum()
    print(f"FTD раньше регистрации (ошибка данных): {n_bad}")

    has_ftd = ftd["ftd_date"].notna()
    no_fx = ftd[has_ftd & ftd["fx_rate"].isna()]
    print(
        f"FTD без курса: {len(no_fx)}"
        + (
            f" (валюты: {list(no_fx['ftd_curr'].unique())})"
            if len(no_fx)
            else ""
        )
    )
    n_stale = int((ftd["fx_gap_days"] > 0).sum())
    print(
        f"FTD с курсом не на дату депозита"
        f"(взят последний известный): {n_stale}"
    )

    n_age_nan = ftd["age_group"].isna().sum()
    n_minor = (ftd["age_group"] == "under 18").sum()
    print(f"Возраст не определён/вне диапазона: {n_age_nan}")
    print(f"Игроков under 18 (проверить на комплаенс): {n_minor}")
    print()


# --------------------------------------------------------------------------
# 4. Недельная агрегация
# --------------------------------------------------------------------------
def build_weekly(
    ftd: pd.DataFrame, max_date: pd.Timestamp
) -> tuple[pd.DataFrame, int]:
    weekly = ftd.groupby(FTD_GR, observed=True).agg(**FTD_AG).reset_index()

    # только зрелые когорты, фильтр ДО rolling
    mature = weekly["reg_week"] + pd.Timedelta(days=MATURITY_DAYS) <= max_date
    n_dropped = int((~mature).sum())
    weekly = weekly[mature]

    weekly = weekly.sort_values(
        ["acquisition_channel", "reg_week"]
    ).reset_index(drop=True)

    weekly["conv"] = weekly["ftd_players"] / weekly["regs"] * 100.0
    weekly["conv_all"] = (
        weekly["ftd_players_all"] / weekly["regs"] * 100.0
    )  # для сравнения
    weekly["avg_dep"] = weekly["ftd_amount"] / weekly["ftd_players"].replace(
        0, np.nan
    )
    return weekly, n_dropped


# --------------------------------------------------------------------------
# 5. Детектор аномалий
# --------------------------------------------------------------------------
def add_anomaly_flags(weekly: pd.DataFrame) -> pd.DataFrame:
    g = weekly.groupby("acquisition_channel", observed=True)

    def past_roll(col: str, func: str) -> pd.Series:
        """Скользящая статистика только по прошлым неделям (shift(1))."""
        return g[col].transform(
            lambda s: s.shift(1)
            .rolling(ROLL_WINDOW, min_periods=ROLL_MIN_PERIODS)
            .agg(func)
        )

    # эмпирическая база (для справки)
    weekly["conv_rolling_avg"] = past_roll("conv", "mean")
    weekly["conv_rolling_std"] = past_roll("conv", "std")
    weekly["z_roll"] = (weekly["conv"] - weekly["conv_rolling_avg"]) / weekly[
        "conv_rolling_std"
    ].replace(0, np.nan)

    # биномиальная база:
    # pooled-доля за прошлые недели, SE зависит от размера когорты
    base_ftd = past_roll("ftd_players", "sum")
    base_regs = past_roll("regs", "sum")
    p = base_ftd / base_regs
    se_pct = np.sqrt(p * (1 - p) / weekly["regs"]) * 100.0
    weekly["conv_base"] = p * 100.0
    weekly["z_binom"] = (
        weekly["conv"] - weekly["conv_base"]
    ) / se_pct.replace(0, np.nan)

    weekly["is_anomaly"] = weekly["z_binom"].abs() > Z_THRESHOLD
    return weekly


# --------------------------------------------------------------------------
# 6. Отчёт
# --------------------------------------------------------------------------
def report(
    ftd_weekly: pd.DataFrame, anomalies: pd.DataFrame, n_dropped: int
) -> None:
    print(
        f"Незрелых когорт отброшено: "
        f"{n_dropped} из {len(ftd_weekly) + n_dropped}"
    )

    n_eval = int(ftd_weekly["z_binom"].notna().sum())
    share = len(anomalies) / max(n_eval, 1)
    print(
        f"Оценено когорт: {n_eval}, аномалий (|z|>{Z_THRESHOLD}): "
        f"{len(anomalies)} ({share:.1%})"
    )

    if anomalies.empty:
        return

    out = anomalies[REP_COLS].copy()
    out["reg_week"] = out["reg_week"].dt.date  # дата вместо datetime
    # float_format вместо .round(): не трогает нечисловые колонки
    print()
    print(out.to_string(index=False, float_format="{:.2f}".format))


# --------------------------------------------------------------------------
# 7. Точка входа
# --------------------------------------------------------------------------
def main(verbose: bool = True):
    """Весь пайплайн. Единственное место, где что-то печатается.
    Данные грузятся тихо (pf(verbose=False)): head и проверки пропусков/дублей
    печатает load_data, здесь они не нужны.
    verbose=False: те же расчёты без вывода (для вызова из других модулей).
    Возвращает (ftd, ftd_weekly, anomalies).
    """
    _, _, players_df, _, tzx_df, rates_df = pf(verbose=False)

    ftd = build_ftd(players_df, tzx_df, rates_df)

    max_date = pd.to_datetime(tzx_df["created_at"]).max()
    ftd_weekly, n_dropped = build_weekly(ftd, max_date)
    ftd_weekly = add_anomaly_flags(ftd_weekly)

    anomalies = ftd_weekly[ftd_weekly["is_anomaly"]].sort_values(
        "z_binom", key=lambda s: s.abs(), ascending=False
    )

    if verbose:
        check_fx(rates_df)
        check_ftd(ftd)
        report(ftd_weekly, anomalies, n_dropped)

    return ftd, ftd_weekly, anomalies


if __name__ == "__main__":
    main()
print()

"""
import pandas as pd
from igaming_eda.ig_vars import FTD_GR, FTD_AG
from igaming_eda.load_data import main as pf

_, _, players_df, _, tzx_df = pf()

ftd = (
    tzx_df[(tzx_df["type"] == "deposit") & (tzx_df["status"] == "success")]
    .sort_values(by=["created_at"])
    .drop_duplicates(subset=["player_id"], keep="first")
    .reset_index(drop=True)
)
ftd = ftd.rename(
    columns={
        "created_at": "ftd_date",
        "status": "ftd_status",
        "currency": "ftd_curr",
    }
)
ftd = pd.merge(players_df, ftd, how="left", on="player_id")
ftd = ftd.rename(columns={"registration_date": "reg_date"})
ftd["age"] = ((ftd["reg_date"] - ftd["birth_date"]).dt.days // 365.25).astype(
    int
)

age_bins = [15, 17, 25, 35, 50, 100]
age_labels = ["under 18", "18-25", "26-35", "36-50", "51+"]

ftd["age_group"] = pd.cut(ftd["age"], bins=age_bins, labels=age_labels)
ftd["reg_week"] = (
    pd.to_datetime(ftd["reg_date"]).dt.to_period("W").dt.start_time
)

ftd_weekly = ftd.groupby(FTD_GR, observed=True).agg(**FTD_AG).reset_index()
ftd_weekly["conv"] = ftd_weekly["ftd_players"] / ftd_weekly["regs"] * 100.0
ftd_weekly["avg_dep"] = ftd_weekly["ftd_amount"] / ftd_weekly["ftd_players"]


ftd_weekly = ftd_weekly.sort_values(
    ["acquisition_channel", "reg_week"]
).reset_index(drop=True)

ftd_weekly["conv_rolling_avg"] = (
    ftd_weekly.groupby("acquisition_channel")["conv"]
    .rolling(window=12, min_periods=4)
    .mean()
    .values
)

ftd_weekly["conv_rolling_std"] = (
    ftd_weekly.groupby("acquisition_channel")["conv"]
    .rolling(window=12, min_periods=4)
    .std()
    .values
)

ftd_weekly["conv_upper"] = ftd_weekly["conv_rolling_avg"] + (
    2 * ftd_weekly["conv_rolling_std"]
)
ftd_weekly["conv_lower"] = ftd_weekly["conv_rolling_avg"] - (
    2 * ftd_weekly["conv_rolling_std"]
)

anomalies = ftd_weekly[
    (ftd_weekly["conv"] > ftd_weekly["conv_upper"])
    | (ftd_weekly["conv"] < ftd_weekly["conv_lower"])
]

print(anomalies)
"""
# ftd["ftd_conv"] = (
#     ftd["ftd_date"].notna().groupby(by=ftd["reg_date"]).transform("mean")
#     * 100.0
# )
# ftd['cohort'] =
