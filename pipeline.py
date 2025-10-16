    def predict_week(self, year: int, week: int, historical_data: pd.DataFrame,
                    injury_data: Optional[Dict] = None) -> pd.DataFrame:
        """
        Predict games for a specific week with injury/uncertainty data
        
        Args:
            year: Season year
            week: Week number
            historical_data: Historical game data for metrics
            injury_data: Dict mapping team -> {'qb_conf': float, 'ol_starters': int, 'key_injuries': int}
        """
        
        print(f"\n[Predicting Week {week} - {year}]")
        
        week_games = self.collector.get_games(year, week, season_type="regular")
        betting_df = self.collector.get_betting_lines(year, week, season_type="regular")
        sp_df = self.collector.get_sp_ratings(year)
        talent_df = self.collector.get_talent_rankings(year)
        fpi_df = self.collector.get_fpi_ratings(year)
        
        sp_dict = _df_to_map(sp_df, key_candidates=["team", "school"], 
                            val_candidates=["rating", "sp", "overall"], default_val=0.0)
        talent_dict = _df_to_map(talent_df, key_candidates=["school", "team"], 
                                val_candidates=["talent", "talent_composite"], default_val=0.0)
        fpi_dict = _df_to_map(fpi_df, key_candidates=["team", "school"], 
                             val_candidates=["fpi"], default_val=0.0)
        lines_dict = _build_lines_dict(betting_df)
        
        if week_games is None or week_games.empty:
            print("  Found 0 games")
            return pd.DataFrame()
        
        # Build conference lookup
        conference_lookup = self._build_conference_lookup(week_games)
        
        gid_col = _pick_col(week_games, ["id", "game_id"]) or "id"
        week_col = _pick_col(week_games, ["week"]) or "week"
        date_col = _pick_col(week_games, ["start_date", "startTime", "start_time"]) or "start_date"
        home_col = _pick_col(week_games, ["home_team", "homeTeam", "home"]) or "home_team"
        away_col = _pick_col(week_games, ["away_team", "awayTeam", "away"]) or "away_team"
        
        print(f"  Found {len(week_games)} games")
        
        # Build historical if missing
        if historical_data is None or historical_data.empty:
            hist = self.collector.get_games(year, season_type="regular")
            hp = _pick_col(hist, ["home_points", "homePoints", "home_score"]) or "home_points"
            ap = _pick_col(hist, ["away_points", "awayPoints", "away_score"]) or "away_points"
            wk = _pick_col(hist, ["week"]) or "week"
            mask_scored = hist[hp].notna() & hist[ap].notna() & hist[wk].notna() & (hist[wk] < week)
            historical_data = hist.loc[mask_scored].copy()
        
        predictions = []
        
        for _, game in week_games.iterrows():
            try:
                home_team = _safe_get(game, [home_col])
                away_team = _safe_get(game, [away_col])
                week_val = _safe_get(game, [week_col])
                gid = _safe_get(game, [gid_col])
                date_val = _safe_get(game, [date_col])
                
                if not home_team or not away_team:
                    continue
                
                # Get conference info
                home_conf = conference_lookup.get(str(home_team), 'Independent')
                away_conf = conference_lookup.get(str(away_team), 'Independent')
                
                # Rivalry/divisional
                is_rivalry = (home_conf == away_conf and home_conf in self.engineer.POWER5)
                is_divisional = (home_conf == away_conf)
                
                # Get injury data if provided
                home_injury = (injury_data or {}).get(str(home_team), {})
                away_injury = (injury_data or {}).get(str(away_team), {})
                
                home_qb_conf = home_injury.get('qb_conf', 1.0)
                away_qb_conf = away_injury.get('qb_conf', 1.0)
                home_ol = home_injury.get('ol_starters', 5)
                away_ol = away_injury.get('ol_starters', 5)
                home_injuries = home_injury.get('key_injuries', 0)
                away_injuries = away_injury.get('key_injuries', 0)
                
                # Calculate metrics
                home_metrics = self.engineer.calculate_team_metrics(home_team, historical_data, week_val, year)
                away_metrics = self.engineer.calculate_team_metrics(away_team, historical_data, week_val, year)
                
                home_sp = sp_dict.get(str(home_team), 0.0)
                away_sp = sp_dict.get(str(away_team), 0.0)
                home_talent = talent_dict.get(str(home_team), 0.0)
                away_talent = talent_dict.get(str(away_team), 0.0)
                home_fpi = fpi_dict.get(str(home_team), 0.0)
                away_fpi = fpi_dict.get(str(away_team), 0.0)
                
                # Create features with all enhancements
                features = self.engineer.create_matchup_features(
                    home_metrics, away_metrics,
                    home_sp, away_sp,
                    home_talent, away_talent,
                    home_fpi, away_fpi,
                    home_conf, away_conf,
                    game_date=date_val,
                    is_rivalry=is_rivalry,
                    is_divisional=is_divisional,
                    travel_distance=500.0,  # Can enhance
                    rest_differential=0,
                    home_qb_confidence=home_qb_conf,
                    away_qb_confidence=away_qb_conf,
                    home_ol_starters=home_ol,
                    away_ol_starters=away_ol,
                    home_key_injuries=home_injuries,
                    away_key_injuries=away_injuries
                )
                
                features_df = pd.DataFrame([features])
                
                # Get market lines
                line_pack = lines_dict.get(gid, {})
                spread_line = line_pack.get("spread")
                total_line = line_pack.get("total")
                
                # Make prediction with market guardrails
                try:
                    pred = self.model.predict(features_df, 
                                             market_spread=spread_line,
                                             market_total=total_line)
                except Exception as e:
                    print(f"      Prediction error for {home_team} vs {away_team}: {e}")
                    pred = {
                        "predicted_spread": np.nan,
                        "predicted_total": np.nan,
                        "home_win_prob": np.nan,
                        "away_win_prob": np.nan,
                        "confidence_calibrated": np.nan
                    }
                
                prediction = {
                    "game": f"{away_team} @ {home_team}",
                    "home_team": home_team,
                    "away_team": away_team,
                    "predicted_spread": pred.get("predicted_spread"),
                    "predicted_spread_raw": pred.get("predicted_spread_raw"),
                    "spread_capped": pred.get("spread_capped", False),
                    "betting_spread": spread_line,
                    "spread_edge": (pred.get("predicted_spread") - spread_line) if spread_line is not None and pd.notna(pred.get("predicted_spread")) else np.nan,
                    "predicted_total": pred.get("predicted_total"),
                    "predicted_total_raw": pred.get("predicted_total_raw"),
                    "total_capped": pred.get("total_capped", False),
                    "betting_total": total_line,
                    "total_edge": (pred.get("predicted_total") - total_line) if total_line is not None and pd.notna(pred.get("predicted_total")) else np.nan,
                    "home_win_prob": pred.get("home_win_prob"),
                    "away_win_prob": pred.get("away_win_prob"),
                    "confidence_calibrated": pred.get("confidence_calibrated"),
                    "confidence_raw": pred.get("confidence_raw"),
                    # Add metadata
                    "home_conf": home_conf,
                    "away_conf": away_conf,
                    "is_weeknight": self._is_weeknight(date_val),
                    "both_g5": 1 if (home_conf in self.engineer.GROUP5 and away_conf in self.engineer.GROUP5) else 0,
                }
                predictions.append(prediction)
                
            except Exception as e:
                print(f"      Error processing game: {e}")
                continue
        
        return pd.DataFrame(predictions)
    
    def _is_weeknight(self, date_str: Optional[str]) -> bool:
        """Check if game is on Tuesday or Wednesday"""
        if not date_str:
            return False
        try:
            dt = pd.to_datetime(date_str)
            return dt.dayofweek in [1, 2]  # Tuesday=1, Wednesday=2
        except:
            return False"""
pipeline.py
Main pipeline that orchestrates data collection, training, and prediction
"""

import numpy as np
import pandas as pd
from typing import List
from data_collector import CFBDDataCollector
from feature_engineer import FeatureEngineer
from betting_model import BettingModel
from backtester import Backtester


class NCAABettingPipeline:
    """Complete pipeline"""
    
    def __init__(self, api_key: str):
        self.collector = CFBDDataCollector(api_key)
        self.engineer = FeatureEngineer()
        self.model = BettingModel()
        self.backtester = Backtester(self.model)
    
    def collect_data(self, years: List[int]) -> pd.DataFrame:
        """Collect all data for specified years"""
        
        # Get conference lookup (only once)
        print("\n  Fetching team/conference data...")
        conference_lookup = self.collector.build_conference_lookup()
        
        all_games_data = []
        
        for year in years:
            print(f"\n[Collecting {year} data]")
            
            # Fetch all data
            print("  Fetching games...")
            games = self.collector.get_games(year)
            
            print("  Fetching betting lines...")
            betting_games = self.collector.get_betting_lines(year)
            
            print("  Fetching SP+ ratings...")
            sp_ratings = self.collector.get_sp_ratings(year)
            
            print("  Fetching talent rankings...")
            talent = self.collector.get_talent_rankings(year)
            
            print("  Fetching FPI ratings...")
            fpi_ratings = self.collector.get_fpi_ratings(year)
            
            # Create lookups
            sp_dict = {r.team: r.rating if r.rating else 0 for r in sp_ratings}
            talent_dict = {t.school: t.talent if t.talent else 0 for t in talent}
            fpi_dict = {f.team: f.fpi if f.fpi else 0 for f in fpi_ratings}
            
            # Create betting lines lookup
            lines_dict = {}
            for bg in betting_games:
                if bg.lines and len(bg.lines) > 0:
                    spreads = [line.spread for line in bg.lines if line.spread]
                    totals = [line.over_under for line in bg.lines if line.over_under]
                    
                    lines_dict[bg.id] = {
                        'spread': float(np.median(spreads)) if spreads else None,
                        'total': float(np.median(totals)) if totals else None
                    }
            
            # Convert games to DataFrame
            games_list = []
            for g in games:
                games_list.append({
                    'game_id': g.id,
                    'season': g.season,
                    'week': g.week,
                    'date': g.start_date,
                    'home_team': g.home_team,
                    'away_team': g.away_team,
                    'home_points': g.home_points,
                    'away_points': g.away_points
                })
            
            games_df = pd.DataFrame(games_list)
            
            print(f"  Processing {len(games_df)} games...")
            
            # Process each game
            for _, game in games_df.iterrows():
                if pd.isna(game['home_points']) or pd.isna(game['away_points']):
                    continue
                
                home_team = game['home_team']
                away_team = game['away_team']
                week = game['week']
                
                # Calculate metrics
                home_metrics = self.engineer.calculate_team_metrics(home_team, games_df, week, year)
                away_metrics = self.engineer.calculate_team_metrics(away_team, games_df, week, year)
                
                # Get ratings
                home_sp = sp_dict.get(home_team, 0)
                away_sp = sp_dict.get(away_team, 0)
                home_talent = talent_dict.get(home_team, 0)
                away_talent = talent_dict.get(away_team, 0)
                home_fpi = fpi_dict.get(home_team, 0)
                away_fpi = fpi_dict.get(away_team, 0)
                
                # Get conference data
                home_conf = conference_lookup.get(home_team, 'Independent')
                away_conf = conference_lookup.get(away_team, 'Independent')
                home_conf_strength = self.engineer.get_conference_strength(home_team, conference_lookup)
                away_conf_strength = self.engineer.get_conference_strength(away_team, conference_lookup)
                home_is_p5 = self.engineer.is_power5(home_conf)
                away_is_p5 = self.engineer.is_power5(away_conf)
                
                # Create features
                features = self.engineer.create_matchup_features(
                    home_metrics, away_metrics,
                    home_sp, away_sp,
                    home_talent, away_talent,
                    home_fpi, away_fpi,
                    home_conf_strength, away_conf_strength,
                    home_is_p5, away_is_p5
                )
                
                # Combine
                game_data = {
                    'game_id': game['game_id'],
                    'season': year,
                    'week': week,
                    'date': game['date'],
                    'home_team': home_team,
                    'away_team': away_team,
                    'home_points': game['home_points'],
                    'away_points': game['away_points'],
                    'actual_spread': game['home_points'] - game['away_points'],
                    'total_points': game['home_points'] + game['away_points'],
                    'home_win': 1 if game['home_points'] > game['away_points'] else 0
                }
                
                # Add betting lines
                if game['game_id'] in lines_dict:
                    game_data['betting_spread'] = lines_dict[game['game_id']]['spread']
                    game_data['betting_total'] = lines_dict[game['game_id']]['total']
                else:
                    game_data['betting_spread'] = None
                    game_data['betting_total'] = None
                
                game_data.update(features)
                all_games_data.append(game_data)
        
        df = pd.DataFrame(all_games_data)
        print(f"\n  Total games: {len(df)}")
        print(f"  API calls used: {self.collector.call_count}")
        
        return dfaway_points': g.away_points
                })
            
            games_df = pd.DataFrame(games_list)
            
            print(f"  Processing {len(games_df)} games...")
            
            # Process each game
            for _, game in games_df.iterrows():
                if pd.isna(game['home_points']) or pd.isna(game['away_points']):
                    continue
                
                home_team = game['home_team']
                away_team = game['away_team']
                week = game['week']
                
                # Calculate metrics
                home_metrics = self.engineer.calculate_team_metrics(home_team, games_df, week, year)
                away_metrics = self.engineer.calculate_team_metrics(away_team, games_df, week, year)
                
                # Get ratings
                home_sp = sp_dict.get(home_team, 0)
                away_sp = sp_dict.get(away_team, 0)
                home_talent = talent_dict.get(home_team, 0)
                away_talent = talent_dict.get(away_team, 0)
                home_fpi = fpi_dict.get(home_team, 0)
                away_fpi = fpi_dict.get(away_team, 0)
                
                # Create features
                features = self.engineer.create_matchup_features(
                    home_metrics, away_metrics,
                    home_sp, away_sp,
                    home_talent, away_talent,
                    home_fpi, away_fpi
                )
                
                # Combine
                game_data = {
                    'game_id': game['game_id'],
                    'season': year,
                    'week': week,
                    'date': game['date'],
                    'home_team': home_team,
                    'away_team': away_team,
                    'home_points': game['home_points'],
                    'away_points': game['away_points'],
                    'actual_spread': game['home_points'] - game['away_points'],
                    'total_points': game['home_points'] + game['away_points'],
                    'home_win': 1 if game['home_points'] > game['away_points'] else 0
                }
                
                # Add betting lines
                if game['game_id'] in lines_dict:
                    game_data['betting_spread'] = lines_dict[game['game_id']]['spread']
                    game_data['betting_total'] = lines_dict[game['game_id']]['total']
                else:
                    game_data['betting_spread'] = None
                    game_data['betting_total'] = None
                
                game_data.update(features)
                all_games_data.append(game_data)
        
        df = pd.DataFrame(all_games_data)
        print(f"\n  Total games: {len(df)}")
        print(f"  API calls used: {self.collector.call_count}")
        
        return df
    
    def predict_week(self, year: int, week: int, historical_data: pd.DataFrame) -> pd.DataFrame:
        """Predict games for a specific week"""
        
        print(f"\n[Predicting Week {week} - {year}]")
        
        # Fetch week's games
        week_games = self.collector.get_games(year, week)
        betting_games = self.collector.get_betting_lines(year, week)
        sp_ratings = self.collector.get_sp_ratings(year)
        talent = self.collector.get_talent_rankings(year)
        fpi_ratings = self.collector.get_fpi_ratings(year)
        
        # Create lookups
        sp_dict = {r.team: r.rating if r.rating else 0 for r in sp_ratings}
        talent_dict = {t.school: t.talent if t.talent else 0 for t in talent}
        fpi_dict = {f.team: f.fpi if f.fpi else 0 for f in fpi_ratings}
        
        lines_dict = {}
        for bg in betting_games:
            if bg.lines and len(bg.lines) > 0:
                spreads = [line.spread for line in bg.lines if line.spread]
                totals = [line.over_under for line in bg.lines if line.over_under]
                
                lines_dict[bg.id] = {
                    'spread': float(np.median(spreads)) if spreads else None,
                    'total': float(np.median(totals)) if totals else None
                }
        
        print(f"  Found {len(week_games)} games")
        
        predictions = []
        
        for game in week_games:
            home_team = game.home_team
            away_team = game.away_team
            
            # Calculate metrics
            home_metrics = self.engineer.calculate_team_metrics(home_team, historical_data, week, year)
            away_metrics = self.engineer.calculate_team_metrics(away_team, historical_data, week, year)
            
            home_sp = sp_dict.get(home_team, 0)
            away_sp = sp_dict.get(away_team, 0)
            home_talent = talent_dict.get(home_team, 0)
            away_talent = talent_dict.get(away_team, 0)
            home_fpi = fpi_dict.get(home_team, 0)
            away_fpi = fpi_dict.get(away_team, 0)
            
            features = self.engineer.create_matchup_features(
                home_metrics, away_metrics,
                home_sp, away_sp,
                home_talent, away_talent,
                home_fpi, away_fpi
            )
            
            features_df = pd.DataFrame([features])
            pred = self.model.predict(features_df)
            
            betting_line = lines_dict.get(game.id, {})
            
            prediction = {
                'game': f"{away_team} @ {home_team}",
                'home_team': home_team,
                'away_team': away_team,
                'predicted_spread': pred['predicted_spread'],
                'betting_spread': betting_line.get('spread'),
                'spread_edge': pred['predicted_spread'] - betting_line.get('spread') if betting_line.get('spread') else None,
                'predicted_total': pred['predicted_total'],
                'betting_total': betting_line.get('total'),
                'total_edge': pred['predicted_total'] - betting_line.get('total') if betting_line.get('total') else None,
                'home_win_prob': pred['home_win_prob'],
                'away_win_prob': pred['away_win_prob']
            }
            
            predictions.append(prediction)
        
        return pd.DataFrame(predictions)
    
    def print_predictions(self, predictions_df: pd.DataFrame, min_edge: float = 2.5):
        """Print predictions with recommendations"""
        
        print("\n" + "="*100)
        print(f"WEEK 8 PREDICTIONS - 2025")
        print("="*100)
        
        spread_bets = predictions_df[predictions_df['spread_edge'].abs() >= min_edge]
        total_bets = predictions_df[predictions_df['total_edge'].abs() >= min_edge]
        # Use calibrated confidence for ML bets
        ml_bets = predictions_df[predictions_df['confidence_calibrated'] > 0.72]
        
        print(f"\n🎯 SPREAD BETS (Edge >= {min_edge} pts)")
        print("-"*100)
        if len(spread_bets) > 0:
            for _, game in spread_bets.iterrows():
                edge = game['spread_edge']
                bet_side = game['home_team'] if edge > 0 else game['away_team']
                print(f"\n{game['game']}")
                print(f"  Model: {game['predicted_spread']:+.1f} | Line: {game['betting_spread']:+.1f}")
                print(f"  ⭐ BET: {bet_side} | Edge: {abs(edge):.1f} pts")
        else:
            print("  No bets with sufficient edge")
        
        print(f"\n🎯 TOTAL BETS (Edge >= {min_edge} pts)")
        print("-"*100)
        if len(total_bets) > 0:
            for _, game in total_bets.iterrows():
                edge = game['total_edge']
                bet_side = "OVER" if edge > 0 else "UNDER"
                print(f"\n{game['game']}")
                print(f"  Model: {game['predicted_total']:.1f} | Line: {game['betting_total']:.1f}")
                print(f"  ⭐ BET: {bet_side} | Edge: {abs(edge):.1f} pts")
        else:
            print("  No bets with sufficient edge")
        
        print(f"\n🎯 MONEYLINE BETS (High Confidence >72%)")
        print("-"*100)
        if len(ml_bets) > 0:
            for _, game in ml_bets.iterrows():
                winner = game['home_team'] if game['home_win_prob'] > 0.5 else game['away_team']
                conf = game['confidence_calibrated']
                print(f"\n{game['game']}")
                print(f"  Win Prob: {game['home_team']} {game['home_win_prob']:.1%} | {game['away_team']} {game['away_win_prob']:.1%}")
                print(f"  ⭐ BET: {winner} ML | Calibrated Confidence: {conf:.1%}")
        else:
            print("  No high confidence bets")
        
        print("\n" + "="*100)):.1f} pts")
        else:
            print("  No bets with sufficient edge")
        
        print(f"\n🎯 TOTAL BETS (Edge >= {min_edge} pts)")
        print("-"*100)
        if len(total_bets) > 0:
            for _, game in total_bets.iterrows():
                edge = game['total_edge']
                bet_side = "OVER" if edge > 0 else "UNDER"
                print(f"\n{game['game']}")
                print(f"  Model: {game['predicted_total']:.1f} | Line: {game['betting_total']:.1f}")
                print(f"  ⭐ BET: {bet_side} | Edge: {abs(edge):.1f} pts")
        else:
            print("  No bets with sufficient edge")
        
        print(f"\n🎯 MONEYLINE BETS (High Confidence)")
        print("-"*100)
        if len(ml_bets) > 0:
            for _, game in ml_bets.iterrows():
                winner = game['home_team'] if game['home_win_prob'] > 0.5 else game['away_team']
                prob = max(game['home_win_prob'], game['away_win_prob'])
                print(f"\n{game['game']}")
                print(f"  Win Prob: {game['home_team']} {game['home_win_prob']:.1%} | {game['away_team']} {game['away_win_prob']:.1%}")
                print(f"  ⭐ BET: {winner} ML | Confidence: {prob:.1%}")
        else:
            print("  No high confidence bets")
        
        print("\n" + "="*100)