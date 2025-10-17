#!/usr/bin/env python3
"""Pipeline module coordinating data collection, feature engineering and modeling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from data_collector import DataCollector
from feature_engineer import FeatureEngineer
from betting_model import BettingModel
from backtester import Backtester


def _pick_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    """Return the first matching column name from *candidates* that exists in *df*."""
    if df is None or df.empty:
        return None
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _safe_get(row: pd.Series, candidates: Sequence[str]) -> Optional[str]:
    """Safely pull the first non-null value from *row* using the provided column names."""
    for col in candidates:
        if col in row and pd.notna(row[col]):
            value = row[col]
            if isinstance(value, str):
                return value.strip()
            return value
    return None


def _df_to_map(df: pd.DataFrame,
               key_candidates: Sequence[str],
               val_candidates: Sequence[str],
               default_val: float = 0.0) -> Dict[str, float]:
    """Convert a DataFrame to a lookup dictionary using the best available columns."""
    mapping: Dict[str, float] = {}
    if df is None or df.empty:
        return mapping

    key_col = _pick_col(df, key_candidates)
    val_col = _pick_col(df, val_candidates)
    if key_col is None or val_col is None:
        return mapping

    for _, row in df[[key_col, val_col]].dropna().iterrows():
        key = str(row[key_col]).strip()
        try:
            value = float(row[val_col])
        except (TypeError, ValueError):
            value = default_val
        if key:
            mapping[key] = value
    return mapping


def _coerce_numeric(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Cast the requested columns to numeric types in-place."""

    if df is None or df.empty:
        return df

    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _normalize_game_id(value: Optional[object]) -> Optional[str]:
    """Convert a game identifier to a consistent string key."""

    if value is None:
        return None

    if isinstance(value, (int, np.integer)):
        return str(int(value))

    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return None
        return str(int(value))

    text = str(value).strip()
    if not text:
        return None

    try:
        return str(int(float(text)))
    except (TypeError, ValueError):
        return text


def _extract_line_values(values: Iterable) -> Tuple[List[float], List[float]]:
    """Helper to pull spread and total numbers from an iterable of line objects/dicts."""
    spreads: List[float] = []
    totals: List[float] = []

    for item in values:
        if item is None:
            continue
        if isinstance(item, dict):
            spread = item.get("spread")
            total = (item.get("over_under") or item.get("overUnder") or
                     item.get("total"))
        else:
            spread = getattr(item, "spread", None)
            total = (getattr(item, "over_under", None) or
                     getattr(item, "overUnder", None) or
                     getattr(item, "total", None))

        if spread is not None and not pd.isna(spread):
            try:
                spreads.append(float(spread))
            except (TypeError, ValueError):
                pass
        if total is not None and not pd.isna(total):
            try:
                totals.append(float(total))
            except (TypeError, ValueError):
                pass

    return spreads, totals


def _build_lines_dict(lines_df: pd.DataFrame) -> Dict[str, Dict[str, Optional[float]]]:
    """Create a mapping of game id -> median spread/total from betting lines."""
    result: Dict[str, Dict[str, Optional[float]]] = {}
    if lines_df is None or lines_df.empty:
        return result

    id_col = _pick_col(lines_df, ["id", "game_id", "gameId"])
    spread_col = _pick_col(lines_df, ["spread", "spread_open", "spread_close"])
    total_col = _pick_col(lines_df, ["over_under", "overUnder", "total", "total_open", "total_close"])

    for _, row in lines_df.iterrows():
        game_id = _normalize_game_id(_safe_get(row, [id_col] if id_col else []))
        if not game_id:
            continue

        spreads: List[float] = []
        totals: List[float] = []

        # Lines may already be flattened or nested inside a "lines" column
        if "lines" in row and isinstance(row["lines"], Iterable):
            line_spreads, line_totals = _extract_line_values(row["lines"])
            spreads.extend(line_spreads)
            totals.extend(line_totals)

        if spread_col and pd.notna(row.get(spread_col)):
            try:
                spreads.append(float(row[spread_col]))
            except (TypeError, ValueError):
                pass
        if total_col and pd.notna(row.get(total_col)):
            try:
                totals.append(float(row[total_col]))
            except (TypeError, ValueError):
                pass

        result[game_id] = {
            "spread": float(np.nanmedian(spreads)) if spreads else None,
            "total": float(np.nanmedian(totals)) if totals else None,
        }

    return result


@dataclass
class WeeklyPrediction:
    """Container for a single game prediction output."""

    game: str
    home_team: str
    away_team: str
    predicted_spread: Optional[float]
    predicted_spread_raw: Optional[float]
    spread_capped: bool
    betting_spread: Optional[float]
    spread_edge: Optional[float]
    predicted_total: Optional[float]
    predicted_total_raw: Optional[float]
    total_capped: bool
    betting_total: Optional[float]
    total_edge: Optional[float]
    home_win_prob: Optional[float]
    away_win_prob: Optional[float]
    confidence_calibrated: Optional[float]
    confidence_raw: Optional[float]
    home_conf: Optional[str]
    away_conf: Optional[str]
    is_weeknight: bool
    both_g5: int

    def as_dict(self) -> Dict[str, Optional[float]]:
        return {
            "game": self.game,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "predicted_spread": self.predicted_spread,
            "predicted_spread_raw": self.predicted_spread_raw,
            "spread_capped": self.spread_capped,
            "betting_spread": self.betting_spread,
            "spread_edge": self.spread_edge,
            "predicted_total": self.predicted_total,
            "predicted_total_raw": self.predicted_total_raw,
            "total_capped": self.total_capped,
            "betting_total": self.betting_total,
            "total_edge": self.total_edge,
            "home_win_prob": self.home_win_prob,
            "away_win_prob": self.away_win_prob,
            "confidence_calibrated": self.confidence_calibrated,
            "confidence_raw": self.confidence_raw,
            "home_conf": self.home_conf,
            "away_conf": self.away_conf,
            "is_weeknight": self.is_weeknight,
            "both_g5": self.both_g5,
        }


class NCAABettingPipeline:
    """End-to-end pipeline that orchestrates collection, training and predictions."""

    def __init__(self, api_key: Optional[str]):
        self.collector = DataCollector(api_key)
        self.engineer = FeatureEngineer()
        self.model = BettingModel()
        self.backtester = Backtester(self.model)

    # ------------------------------------------------------------------
    # Utility helpers
    def _build_conference_lookup(self, games_df: pd.DataFrame) -> Dict[str, str]:
        """Try to infer conference membership from the game data and API lookup."""
        lookup: Dict[str, str] = {}
        if games_df is not None and not games_df.empty:
            home_conf_col = _pick_col(games_df, ["home_conference", "homeConference"])
            away_conf_col = _pick_col(games_df, ["away_conference", "awayConference"])
            home_team_col = _pick_col(games_df, ["home_team", "homeTeam"])
            away_team_col = _pick_col(games_df, ["away_team", "awayTeam"])

            if home_conf_col and home_team_col:
                for _, row in games_df[[home_team_col, home_conf_col]].dropna().iterrows():
                    lookup[str(row[home_team_col]).strip()] = str(row[home_conf_col]).strip()
            if away_conf_col and away_team_col:
                for _, row in games_df[[away_team_col, away_conf_col]].dropna().iterrows():
                    lookup[str(row[away_team_col]).strip()] = str(row[away_conf_col]).strip()

        if not lookup:
            lookup = self.collector.build_conference_lookup()
        return lookup

    @staticmethod
    def _is_weeknight(date_str: Optional[str]) -> bool:
        """Return True when the game date lands on a Tuesday or Wednesday."""
        if not date_str:
            return False
        try:
            dt = pd.to_datetime(date_str)
            return dt.dayofweek in (1, 2)
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Data collection & preparation
    def collect_data(self, years: List[int]) -> pd.DataFrame:
        """Collect historical data across the requested years and engineer features."""
        conference_lookup = self.collector.build_conference_lookup()
        all_games: List[Dict] = []

        for year in years:
            print(f"\n[Collecting {year} data]")
            games_df = self.collector.get_games(year)
            lines_df = self.collector.get_betting_lines(year)
            sp_df = self.collector.get_sp_ratings(year)
            talent_df = self.collector.get_talent_rankings(year)
            fpi_df = self.collector.get_fpi_ratings(year)

            if games_df.empty:
                print("  ⚠️  No games returned")
                continue

            sp_lookup = _df_to_map(sp_df, ["team", "school"], ["rating", "sp", "overall"], 0.0)
            talent_lookup = _df_to_map(talent_df, ["school", "team"], ["talent", "talent_composite"], 0.0)
            fpi_lookup = _df_to_map(fpi_df, ["team", "school"], ["fpi"], 0.0)
            lines_lookup = _build_lines_dict(lines_df)

            gid_col = _pick_col(games_df, ["id", "game_id"])
            week_col = _pick_col(games_df, ["week"])
            season_col = _pick_col(games_df, ["season", "year"])
            date_col = _pick_col(games_df, ["start_date", "startDate", "date"])
            home_col = _pick_col(games_df, ["home_team", "homeTeam"])
            away_col = _pick_col(games_df, ["away_team", "awayTeam"])
            home_pts_col = _pick_col(games_df, ["home_points", "homeScore"])
            away_pts_col = _pick_col(games_df, ["away_points", "awayScore"])

            normalized_games_df = games_df.copy()
            rename_map = {}
            if season_col and season_col != "season":
                rename_map[season_col] = "season"
            if gid_col and gid_col != "game_id":
                rename_map[gid_col] = "game_id"
            if week_col and week_col != "week":
                rename_map[week_col] = "week"
            if home_col and home_col != "home_team":
                rename_map[home_col] = "home_team"
            if away_col and away_col != "away_team":
                rename_map[away_col] = "away_team"
            if home_pts_col and home_pts_col != "home_points":
                rename_map[home_pts_col] = "home_points"
            if away_pts_col and away_pts_col != "away_points":
                rename_map[away_pts_col] = "away_points"
            if rename_map:
                normalized_games_df = normalized_games_df.rename(columns=rename_map)

            # Keep references aligned with the normalized column names so downstream
            # lookups use the standardized schema regardless of the original casing.
            gid_col = rename_map.get(gid_col, gid_col)
            week_col = rename_map.get(week_col, week_col)
            season_col = rename_map.get(season_col, season_col)
            date_col = rename_map.get(date_col, date_col)
            home_col = rename_map.get(home_col, home_col)
            away_col = rename_map.get(away_col, away_col)
            home_pts_col = rename_map.get(home_pts_col, home_pts_col)
            away_pts_col = rename_map.get(away_pts_col, away_pts_col)

            _coerce_numeric(
                normalized_games_df,
                ["season", "week", "home_points", "away_points"],
            )

            if "season" in normalized_games_df.columns:
                normalized_games_df["season"] = normalized_games_df["season"].fillna(int(year))

            required_cols = [col for col in [home_col, away_col, home_pts_col, away_pts_col, week_col]
                             if col]
            if required_cols:
                filtered_games_df = normalized_games_df.dropna(subset=required_cols).copy()
            else:
                filtered_games_df = normalized_games_df.copy()

            if filtered_games_df.empty:
                print("  ⚠️  Games returned without scores; skipping year")
                continue

            for _, game in filtered_games_df.iterrows():
                home_team = _safe_get(game, [home_col])
                away_team = _safe_get(game, [away_col])
                week_val = _safe_get(game, [week_col])
                season_val = _safe_get(game, [season_col])
                home_pts = _safe_get(game, [home_pts_col])
                away_pts = _safe_get(game, [away_pts_col])

                if home_team is None or away_team is None:
                    continue

                if pd.isna(home_pts) or pd.isna(away_pts):
                    continue

                try:
                    week_int = int(float(week_val))
                except (TypeError, ValueError):
                    continue

                season_int: Optional[int] = None
                try:
                    season_int = int(float(season_val)) if season_val is not None else None
                except (TypeError, ValueError):
                    season_int = None

                try:
                    home_pts_val = float(home_pts)
                    away_pts_val = float(away_pts)
                except (TypeError, ValueError):
                    continue

                home_metrics = self.engineer.calculate_team_metrics(home_team, filtered_games_df, week_int, int(year))
                away_metrics = self.engineer.calculate_team_metrics(away_team, filtered_games_df, week_int, int(year))

                home_sp = sp_lookup.get(home_team, 0.0)
                away_sp = sp_lookup.get(away_team, 0.0)
                home_talent = talent_lookup.get(home_team, 0.0)
                away_talent = talent_lookup.get(away_team, 0.0)
                home_fpi = fpi_lookup.get(home_team, 0.0)
                away_fpi = fpi_lookup.get(away_team, 0.0)

                home_conf = conference_lookup.get(home_team, "Independent")
                away_conf = conference_lookup.get(away_team, "Independent")

                features = self.engineer.create_matchup_features(
                    home_metrics,
                    away_metrics,
                    home_sp,
                    away_sp,
                    home_talent,
                    away_talent,
                    home_fpi,
                    away_fpi,
                    home_conf,
                    away_conf,
                )

                gid = _normalize_game_id(_safe_get(game, [gid_col]) if gid_col else None)
                date_val = _safe_get(game, [date_col]) if date_col else None

                record = {
                    "game_id": gid,
                    "season": season_int if season_int is not None else int(year),
                    "week": week_int,
                    "date": date_val,
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_points": home_pts_val,
                    "away_points": away_pts_val,
                    "actual_spread": home_pts_val - away_pts_val,
                    "total_points": home_pts_val + away_pts_val,
                    "home_win": 1 if home_pts_val > away_pts_val else 0,
                    "betting_spread": None,
                    "betting_total": None,
                }

                if gid and gid in lines_lookup:
                    record["betting_spread"] = lines_lookup[gid]["spread"]
                    record["betting_total"] = lines_lookup[gid]["total"]

                record.update(features)
                all_games.append(record)

        df = pd.DataFrame(all_games)
        print(f"\n  Total games collected: {len(df)}")
        print(f"  API calls used: {self.collector.call_count}")
        return df

    # ------------------------------------------------------------------
    # Weekly predictions
    def predict_week(self,
                     year: int,
                     week: int,
                     historical_data: pd.DataFrame,
                     injury_data: Optional[Dict[str, Dict[str, float]]] = None) -> pd.DataFrame:
        """Generate predictions for a given week using optional injury context."""
        print(f"\n[Predicting Week {week} - {year}]")

        week_games = self.collector.get_games(year, week, season_type="regular")
        lines_df = self.collector.get_betting_lines(year, week, season_type="regular")
        sp_df = self.collector.get_sp_ratings(year)
        talent_df = self.collector.get_talent_rankings(year)
        fpi_df = self.collector.get_fpi_ratings(year)

        if week_games.empty:
            print("  Found 0 games")
            return pd.DataFrame()

        sp_lookup = _df_to_map(sp_df, ["team", "school"], ["rating", "sp", "overall"], 0.0)
        talent_lookup = _df_to_map(talent_df, ["school", "team"], ["talent", "talent_composite"], 0.0)
        fpi_lookup = _df_to_map(fpi_df, ["team", "school"], ["fpi"], 0.0)
        lines_lookup = _build_lines_dict(lines_df)
        conference_lookup = self._build_conference_lookup(week_games)

        gid_col = _pick_col(week_games, ["id", "game_id"])
        week_col = _pick_col(week_games, ["week"])
        date_col = _pick_col(week_games, ["start_date", "startDate", "date", "start_time", "startTime"])
        home_col = _pick_col(week_games, ["home_team", "homeTeam", "home"])
        away_col = _pick_col(week_games, ["away_team", "awayTeam", "away"])

        if not week_games.empty:
            week_games = week_games.copy()
            _coerce_numeric(week_games, [col for col in [gid_col, week_col] if col])

        normalized_hist = pd.DataFrame()

        if historical_data is None or historical_data.empty:
            hist_full = self.collector.get_games(year, season_type="regular")
            hp = _pick_col(hist_full, ["home_points", "homeScore"])
            ap = _pick_col(hist_full, ["away_points", "awayScore"])
            wk_col_hist = _pick_col(hist_full, ["week"])
            if hist_full.empty or hp is None or ap is None or wk_col_hist is None:
                normalized_hist = pd.DataFrame()
            else:
                mask = (
                    hist_full[hp].notna() &
                    hist_full[ap].notna() &
                    hist_full[wk_col_hist].notna() &
                    (hist_full[wk_col_hist] < week)
                )
                normalized_hist = hist_full.loc[mask].copy()
        
        else:
            normalized_hist = historical_data.copy()

        if not normalized_hist.empty:
            rename_map_hist = {}
            hist_week_col = _pick_col(normalized_hist, ["week"])
            hist_season_col = _pick_col(normalized_hist, ["season", "year"])
            hist_home_col = _pick_col(normalized_hist, ["home_team", "homeTeam"])
            hist_away_col = _pick_col(normalized_hist, ["away_team", "awayTeam"])
            hist_home_pts_col = _pick_col(normalized_hist, ["home_points", "homeScore"])
            hist_away_pts_col = _pick_col(normalized_hist, ["away_points", "awayScore"])

            if hist_week_col and hist_week_col != "week":
                rename_map_hist[hist_week_col] = "week"
            if hist_season_col and hist_season_col != "season":
                rename_map_hist[hist_season_col] = "season"
            if hist_home_col and hist_home_col != "home_team":
                rename_map_hist[hist_home_col] = "home_team"
            if hist_away_col and hist_away_col != "away_team":
                rename_map_hist[hist_away_col] = "away_team"
            if hist_home_pts_col and hist_home_pts_col != "home_points":
                rename_map_hist[hist_home_pts_col] = "home_points"
            if hist_away_pts_col and hist_away_pts_col != "away_points":
                rename_map_hist[hist_away_pts_col] = "away_points"
            if rename_map_hist:
                normalized_hist = normalized_hist.rename(columns=rename_map_hist)

            _coerce_numeric(
                normalized_hist,
                ["season", "week", "home_points", "away_points"],
            )

        predictions: List[WeeklyPrediction] = []

        for _, game in week_games.iterrows():
            try:
                home_team = _safe_get(game, [home_col])
                away_team = _safe_get(game, [away_col])
                week_val = _safe_get(game, [week_col])
                gid = _normalize_game_id(_safe_get(game, [gid_col]) if gid_col else None)
                date_val = _safe_get(game, [date_col])

                if not home_team or not away_team:
                    continue

                home_conf = conference_lookup.get(str(home_team), "Independent")
                away_conf = conference_lookup.get(str(away_team), "Independent")

                is_rivalry = home_conf == away_conf and home_conf in self.engineer.POWER5
                is_divisional = home_conf == away_conf

                injury_lookup = injury_data or {}
                home_injury = injury_lookup.get(str(home_team), {})
                away_injury = injury_lookup.get(str(away_team), {})

                try:
                    week_for_metrics = int(float(week_val))
                except (TypeError, ValueError):
                    week_for_metrics = week

                home_metrics = self.engineer.calculate_team_metrics(home_team, normalized_hist, week_for_metrics, year)
                away_metrics = self.engineer.calculate_team_metrics(away_team, normalized_hist, week_for_metrics, year)

                features = self.engineer.create_matchup_features(
                    home_metrics,
                    away_metrics,
                    sp_lookup.get(str(home_team), 0.0),
                    sp_lookup.get(str(away_team), 0.0),
                    talent_lookup.get(str(home_team), 0.0),
                    talent_lookup.get(str(away_team), 0.0),
                    fpi_lookup.get(str(home_team), 0.0),
                    fpi_lookup.get(str(away_team), 0.0),
                    home_conf,
                    away_conf,
                    game_date=date_val,
                    is_rivalry=is_rivalry,
                    is_divisional=is_divisional,
                    travel_distance=500.0,
                    rest_differential=0,
                    home_qb_confidence=home_injury.get("qb_conf", 1.0),
                    away_qb_confidence=away_injury.get("qb_conf", 1.0),
                    home_ol_starters=home_injury.get("ol_starters", 5),
                    away_ol_starters=away_injury.get("ol_starters", 5),
                    home_key_injuries=home_injury.get("key_injuries", 0),
                    away_key_injuries=away_injury.get("key_injuries", 0),
                )

                features_df = pd.DataFrame([features])
                line_info = lines_lookup.get(gid, {}) if gid else {}

                prediction = self.model.predict(
                    features_df,
                    market_spread=line_info.get("spread"),
                    market_total=line_info.get("total"),
                )

                spread_edge = None
                if prediction.get("predicted_spread") is not None and line_info.get("spread") is not None:
                    spread_edge = prediction["predicted_spread"] - line_info.get("spread")
                total_edge = None
                if prediction.get("predicted_total") is not None and line_info.get("total") is not None:
                    total_edge = prediction["predicted_total"] - line_info.get("total")

                predictions.append(
                    WeeklyPrediction(
                        game=f"{away_team} @ {home_team}",
                        home_team=str(home_team),
                        away_team=str(away_team),
                        predicted_spread=prediction.get("predicted_spread"),
                        predicted_spread_raw=prediction.get("predicted_spread_raw"),
                        spread_capped=bool(prediction.get("spread_capped")),
                        betting_spread=line_info.get("spread"),
                        spread_edge=spread_edge,
                        predicted_total=prediction.get("predicted_total"),
                        predicted_total_raw=prediction.get("predicted_total_raw"),
                        total_capped=bool(prediction.get("total_capped")),
                        betting_total=line_info.get("total"),
                        total_edge=total_edge,
                        home_win_prob=prediction.get("home_win_prob"),
                        away_win_prob=prediction.get("away_win_prob"),
                        confidence_calibrated=prediction.get("confidence_calibrated"),
                        confidence_raw=prediction.get("confidence_raw"),
                        home_conf=home_conf,
                        away_conf=away_conf,
                        is_weeknight=self._is_weeknight(date_val),
                        both_g5=1 if (home_conf in self.engineer.GROUP5 and away_conf in self.engineer.GROUP5) else 0,
                    )
                )
            except Exception as exc:  # pragma: no cover - defensive logging
                print(f"      Error processing game: {exc}")

        return pd.DataFrame([pred.as_dict() for pred in predictions])

    # ------------------------------------------------------------------
    # Reporting helpers
    def print_predictions(self, predictions_df: pd.DataFrame, min_edge: float = 2.5) -> None:
        """Pretty-print predictions with betting recommendations."""
        if predictions_df is None or predictions_df.empty:
            print("\nNo predictions available to display.")
            return

        print("\n" + "=" * 100)
        print("WEEKLY PREDICTIONS")
        print("=" * 100)

        spread_edges = pd.to_numeric(predictions_df["spread_edge"], errors="coerce")
        total_edges = pd.to_numeric(predictions_df["total_edge"], errors="coerce")
        ml_conf = pd.to_numeric(predictions_df["confidence_calibrated"], errors="coerce")

        spread_bets = predictions_df[spread_edges.abs() >= min_edge].copy()
        total_bets = predictions_df[total_edges.abs() >= min_edge].copy()
        ml_bets = predictions_df[ml_conf > 0.72].copy()

        def _format_line(value: Optional[float], fmt: str) -> str:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                numeric = float("nan")
            if np.isnan(numeric):
                numeric = 0.0
            return format(numeric, fmt)

        def _print_bets(title: str, bets: pd.DataFrame, formatter) -> None:
            print(f"\n🎯 {title}")
            print("-" * 100)
            if bets.empty:
                print("  No bets with sufficient edge")
                return
            for _, game in bets.iterrows():
                formatter(game)

        _print_bets(
            f"SPREAD BETS (Edge >= {min_edge} pts)",
            spread_bets,
            lambda game: print(
                f"\n{game['game']}\n"
                f"  Model: {game['predicted_spread']:+.1f} | "
                f"Line: {_format_line(game['betting_spread'], '+.1f')}\n"
                f"  ⭐ BET: {'HOME' if game['spread_edge'] > 0 else 'AWAY'} | Edge: {abs(game['spread_edge']):.1f} pts"
            ),
        )

        _print_bets(
            f"TOTAL BETS (Edge >= {min_edge} pts)",
            total_bets,
            lambda game: print(
                f"\n{game['game']}\n"
                f"  Model: {game['predicted_total']:.1f} | "
                f"Line: {_format_line(game['betting_total'], '.1f')}\n"
                f"  ⭐ BET: {'OVER' if game['total_edge'] > 0 else 'UNDER'} | Edge: {abs(game['total_edge']):.1f} pts"
            ),
        )

        _print_bets(
            "MONEYLINE BETS (High Confidence >72%)",
            ml_bets,
            lambda game: print(
                f"\n{game['game']}\n"
                f"  Win Prob: {game['home_team']} {game['home_win_prob']:.1%} | "
                f"{game['away_team']} {game['away_win_prob']:.1%}\n"
                f"  ⭐ BET: {game['home_team'] if game['home_win_prob'] > 0.5 else game['away_team']} ML | "
                f"Calibrated Confidence: {game['confidence_calibrated']:.1%}"
            ),
        )

        print("\n" + "=" * 100)


__all__ = ["NCAABettingPipeline"]
