"""
feature_engineer.py
Enhanced with weekday, conference, dynamic HFA, and uncertainty adjustments
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from datetime import datetime
from scipy.stats import hmean


class FeatureEngineer:
    """Creates features from game data with advanced adjustments"""
    
    # Conference tiers
    POWER5 = {'SEC', 'Big Ten', 'Big 12', 'ACC', 'Pac-12'}
    GROUP5 = {'American Athletic', 'Mountain West', 'Sun Belt', 'Conference USA', 'MAC'}
    
    CONFERENCE_STRENGTH = {
        'SEC': 1.0, 'Big Ten': 0.95, 'Big 12': 0.90, 'ACC': 0.85, 'Pac-12': 0.80,
        'American Athletic': 0.60, 'Mountain West': 0.55, 'Sun Belt': 0.50,
        'Conference USA': 0.45, 'MAC': 0.40, 'FBS Independents': 0.70,
        'Independent': 0.70
    }
    
    # League median tempo (possessions per game)
    LEAGUE_MEDIAN_PACE = 70.0
    
    def __init__(self):
        self.game_metadata = {}  # Store game day, injuries, etc.
    
    def calculate_team_metrics(self, team: str, games_df: pd.DataFrame, 
                               current_week: int, year: int) -> Dict:
        """Calculate team performance metrics from historical games"""
        
        team_games = games_df[
            ((games_df.get('home_team', pd.Series()) == team) | 
             (games_df.get('away_team', pd.Series()) == team)) &
            (games_df.get('week', pd.Series(-1)) < current_week) &
            (games_df.get('season', pd.Series(-1)) == year)
        ].sort_values('week')
        
        if len(team_games) == 0:
            return self._get_default_metrics()
        
        metrics = {}
        points_scored = []
        points_allowed = []
        possessions = []
        wins = 0
        home_games = 0
        away_games = 0
        conf_games = 0
        
        for _, game in team_games.iterrows():
            is_home = game['home_team'] == team
            
            if is_home:
                points_scored.append(game['home_points'])
                points_allowed.append(game['away_points'])
                home_games += 1
                if game['home_points'] > game['away_points']:
                    wins += 1
            else:
                points_scored.append(game['away_points'])
                points_allowed.append(game['home_points'])
                away_games += 1
                if game['away_points'] > game['home_points']:
                    wins += 1
            
            # Estimate possessions (more accurate formula)
            total_pts = game['home_points'] + game['away_points']
            estimated_poss = (total_pts / 1.4) + 8  # Better baseline estimate
            possessions.append(estimated_poss)
        
        total_games = len(team_games)
        
        # Core metrics with regression
        metrics['ppg'] = np.mean(points_scored)
        metrics['papg'] = np.mean(points_allowed)
        metrics['margin'] = metrics['ppg'] - metrics['papg']
        metrics['win_pct'] = wins / total_games
        
        # PACE METRICS (using harmonic mean for tempo blending)
        avg_pace = np.mean(possessions) if possessions else self.LEAGUE_MEDIAN_PACE
        metrics['pace'] = avg_pace
        metrics['pace_harmonic'] = hmean(possessions) if len(possessions) > 1 else avg_pace
        metrics['points_per_possession'] = metrics['ppg'] / avg_pace if avg_pace > 0 else 0.34
        metrics['points_allowed_per_possession'] = metrics['papg'] / avg_pace if avg_pace > 0 else 0.34
        
        # Tempo classification (regressed toward league median)
        raw_tempo_factor = avg_pace / self.LEAGUE_MEDIAN_PACE
        metrics['tempo_factor'] = 0.7 * raw_tempo_factor + 0.3 * 1.0  # 30% regression
        
        # Efficiency (adjusted)
        metrics['off_efficiency'] = metrics['points_per_possession'] / 0.40
        metrics['def_efficiency'] = 1 - (metrics['points_allowed_per_possession'] / 0.40)
        
        # Home/away splits for dynamic HFA
        metrics['home_game_pct'] = home_games / total_games if total_games > 0 else 0.5
        
        # Recent form (last 3-4 games)
        if len(points_scored) >= 3:
            metrics['recent_ppg'] = np.mean(points_scored[-3:])
            metrics['recent_papg'] = np.mean(points_allowed[-3:])
            metrics['recent_margin'] = metrics['recent_ppg'] - metrics['recent_papg']
            metrics['momentum'] = (metrics['recent_ppg'] - metrics['ppg']) * 0.5  # Dampened
        else:
            metrics['recent_ppg'] = metrics['ppg']
            metrics['recent_papg'] = metrics['papg']
            metrics['recent_margin'] = metrics['margin']
            metrics['momentum'] = 0
        
        # Consistency & volatility
        metrics['scoring_variance'] = np.std(points_scored) if len(points_scored) > 1 else 10
        metrics['defense_variance'] = np.std(points_allowed) if len(points_allowed) > 1 else 10
        metrics['pace_variance'] = np.std(possessions) if len(possessions) > 1 else 5
        
        # Sample size penalty
        metrics['games_played'] = total_games
        metrics['sample_confidence'] = min(total_games / 8.0, 1.0)  # Full confidence at 8+ games
        
        return metrics
    
    @staticmethod
    def _get_default_metrics() -> Dict:
        """Default metrics with neutral priors"""
        return {
            'ppg': 24.0, 'papg': 24.0, 'margin': 0.0, 'win_pct': 0.5,
            'pace': 70.0, 'pace_harmonic': 70.0, 'points_per_possession': 0.34,
            'points_allowed_per_possession': 0.34, 'tempo_factor': 1.0,
            'off_efficiency': 0.5, 'def_efficiency': 0.5,
            'home_game_pct': 0.5, 'recent_ppg': 24.0, 'recent_papg': 24.0,
            'recent_margin': 0.0, 'momentum': 0.0,
            'scoring_variance': 10.0, 'defense_variance': 10.0, 'pace_variance': 5.0,
            'games_played': 0, 'sample_confidence': 0.0
        }
    
    def calculate_dynamic_hfa(self, home_team: str, away_team: str,
                             home_conf: str, away_conf: str,
                             is_rivalry: bool = False,
                             is_divisional: bool = False,
                             travel_distance: float = 500.0,
                             rest_differential: int = 0) -> float:
        """
        Calculate dynamic home field advantage based on multiple factors
        Base HFA = 1.5 pts, adjusted by context
        """
        
        base_hfa = 1.5
        
        # Conference familiarity adjustment
        if home_conf == away_conf:
            # Conference game - reduce HFA
            if is_divisional or is_rivalry:
                base_hfa *= 0.7  # Rivals/division = minimal HFA
            else:
                base_hfa *= 0.85  # Conference game
        
        # G5 vs G5 - further reduction (especially short travel)
        if home_conf in self.GROUP5 and away_conf in self.GROUP5:
            if travel_distance < 300:  # Regional matchup
                base_hfa *= 0.75
        
        # Travel distance adjustment (longer = more HFA)
        if travel_distance > 1500:  # Cross-country
            base_hfa *= 1.15
        elif travel_distance < 200:  # Very short
            base_hfa *= 0.85
        
        # Rest differential (negative = away team more rested)
        if rest_differential < 0:  # Away team extra rest
            base_hfa *= 0.9
        elif rest_differential > 0:  # Home team extra rest
            base_hfa *= 1.1
        
        # Cap HFA between 0.5 and 3.0
        return max(0.5, min(3.0, base_hfa))
    
    def get_weekday_adjustment(self, game_date: Optional[str], 
                               home_conf: str, away_conf: str,
                               combined_pace: float) -> Dict[str, float]:
        """
        Calculate weekday penalties for Tuesday/Wednesday G5 games
        Returns adjustments for total and spread
        """
        
        adjustments = {'total_adj': 0.0, 'spread_adj': 0.0, 'pace_adj': 0.0}
        
        if not game_date:
            return adjustments
        
        try:
            # Parse date and get day of week
            dt = pd.to_datetime(game_date)
            weekday = dt.dayofweek  # Monday=0, Sunday=6
            
            # Tuesday (1) or Wednesday (2) games
            if weekday in [1, 2]:
                # Base weeknight penalty
                is_g5_game = (home_conf in self.GROUP5 and away_conf in self.GROUP5)
                
                if is_g5_game:
                    # Stronger penalty for G5 midweek
                    adjustments['total_adj'] = -3.5  # Less scoring
                    adjustments['spread_adj'] = -1.0  # Favorite underperforms
                    adjustments['pace_adj'] = -3.0   # Slower game
                else:
                    # Moderate penalty for P5 midweek
                    adjustments['total_adj'] = -2.0
                    adjustments['spread_adj'] = -0.5
                    adjustments['pace_adj'] = -1.5
                
                # Extra penalty for fast-paced teams on short week
                if combined_pace > 75:
                    adjustments['total_adj'] -= 1.5  # Fast teams struggle more
                    adjustments['pace_adj'] -= 2.0
        
        except Exception:
            pass
        
        return adjustments
    
    def get_uncertainty_penalty(self, qb_confidence: float = 1.0,
                               ol_starters: int = 5,
                               key_injuries: int = 0) -> Dict[str, float]:
        """
        Calculate penalties for uncertainty (injuries, lineup questions)
        
        Args:
            qb_confidence: 0.0-1.0, how certain we are about starting QB
            ol_starters: number of starting OL available (out of 5)
            key_injuries: number of other key injuries
        
        Returns:
            Dict with total and win prob adjustments
        """
        
        penalty = {'total_adj': 0.0, 'win_prob_regression': 0.0}
        
        # QB uncertainty (major impact)
        if qb_confidence < 0.95:
            qb_penalty_factor = (0.95 - qb_confidence) * 6  # Up to -6 pts
            penalty['total_adj'] -= qb_penalty_factor
            penalty['win_prob_regression'] += 0.1 * (1 - qb_confidence)
        
        # OL injuries (significant impact on scoring)
        if ol_starters < 5:
            ol_missing = 5 - ol_starters
            penalty['total_adj'] -= ol_missing * 1.5  # -1.5 per missing starter
            penalty['win_prob_regression'] += 0.05 * ol_missing
        
        # Other key injuries
        if key_injuries > 0:
            penalty['total_adj'] -= key_injuries * 0.8
            penalty['win_prob_regression'] += 0.03 * key_injuries
        
        # Cap adjustments
        penalty['total_adj'] = max(-8.0, penalty['total_adj'])
        penalty['win_prob_regression'] = min(0.25, penalty['win_prob_regression'])
        
        return penalty
    
    def create_matchup_features(
        self,
        home_metrics: Dict,
        away_metrics: Dict,
        home_sp: float,
        away_sp: float,
        home_talent: float,
        away_talent: float,
        home_fpi: float,
        away_fpi: float,
        home_conf: str = 'Independent',
        away_conf: str = 'Independent',
        game_date: Optional[str] = None,
        is_rivalry: bool = False,
        is_divisional: bool = False,
        travel_distance: float = 500.0,
        rest_differential: int = 0,
        home_qb_confidence: float = 1.0,
        away_qb_confidence: float = 1.0,
        home_ol_starters: int = 5,
        away_ol_starters: int = 5,
        home_key_injuries: int = 0,
        away_key_injuries: int = 0
    ) -> Dict:
        """Create comprehensive matchup features with all adjustments"""
        
        # Calculate conference strengths
        home_conf_strength = self.CONFERENCE_STRENGTH.get(home_conf, 0.5)
        away_conf_strength = self.CONFERENCE_STRENGTH.get(away_conf, 0.5)
        home_is_p5 = 1 if home_conf in self.POWER5 else 0
        away_is_p5 = 1 if away_conf in self.POWER5 else 0
        
        # DYNAMIC HOME FIELD ADVANTAGE
        dynamic_hfa = self.calculate_dynamic_hfa(
            '', '',  # team names not needed for base calculation
            home_conf, away_conf,
            is_rivalry, is_divisional,
            travel_distance, rest_differential
        )
        
        # PACE FEATURES (using harmonic mean for blending)
        pace_harmonic = hmean([home_metrics['pace_harmonic'], away_metrics['pace_harmonic']])
        combined_pace = (home_metrics['pace'] + away_metrics['pace']) / 2
        
        # Regress extreme paces toward league median (especially for weeknights)
        pace_regressed = 0.75 * combined_pace + 0.25 * self.LEAGUE_MEDIAN_PACE
        
        # WEEKDAY ADJUSTMENTS
        weekday_adj = self.get_weekday_adjustment(game_date, home_conf, away_conf, combined_pace)
        
        # UNCERTAINTY PENALTIES
        home_uncertainty = self.get_uncertainty_penalty(
            home_qb_confidence, home_ol_starters, home_key_injuries
        )
        away_uncertainty = self.get_uncertainty_penalty(
            away_qb_confidence, away_ol_starters, away_key_injuries
        )
        
        # Combined uncertainty impact
        total_uncertainty_adj = (home_uncertainty['total_adj'] + away_uncertainty['total_adj']) / 2
        win_prob_uncertainty = abs(home_uncertainty['win_prob_regression'] - 
                                   away_uncertainty['win_prob_regression'])
        
        features = {
            # Dynamic HFA
            'home_field': 1,
            'dynamic_hfa': dynamic_hfa,
            'is_rivalry': 1 if is_rivalry else 0,
            'is_divisional': 1 if is_divisional else 0,
            'travel_distance': travel_distance / 1000.0,  # Normalize to thousands of miles
            'rest_differential': rest_differential,
            
            # Core differentials (heavily regressed)
            'ppg_diff': (home_metrics['ppg'] - away_metrics['ppg']) * 0.7,
            'papg_diff': (home_metrics['papg'] - away_metrics['papg']) * 0.7,
            'margin_diff': (home_metrics['margin'] - away_metrics['margin']) * 0.6,
            'win_pct_diff': (home_metrics['win_pct'] - away_metrics['win_pct']) * 0.6,
            
            # PACE FEATURES (with harmonic mean and regression)
            'pace_harmonic': pace_harmonic,
            'pace_combined': combined_pace,
            'pace_regressed': pace_regressed,
            'pace_diff': home_metrics['pace'] - away_metrics['pace'],
            'tempo_factor_home': home_metrics['tempo_factor'],
            'tempo_factor_away': away_metrics['tempo_factor'],
            'tempo_mismatch': abs(home_metrics['tempo_factor'] - away_metrics['tempo_factor']),
            
            # Weekday adjustments
            'weekday_total_adj': weekday_adj['total_adj'],
            'weekday_spread_adj': weekday_adj['spread_adj'],
            'weekday_pace_adj': weekday_adj['pace_adj'],
            
            # Conference context
            'conf_strength_diff': home_conf_strength - away_conf_strength,
            'both_g5': 1 if (home_conf in self.GROUP5 and away_conf in self.GROUP5) else 0,
            'both_p5': home_is_p5 * away_is_p5,
            'p5_vs_g5': abs(home_is_p5 - away_is_p5),
            
            # Matchup efficiency (dampen)
            'home_off_vs_away_def': (home_metrics['off_efficiency'] - away_metrics['def_efficiency']) * 0.75,
            'away_off_vs_home_def': (away_metrics['off_efficiency'] - home_metrics['def_efficiency']) * 0.75,
            
            # External ratings (primary predictors)
            'sp_rating_diff': home_sp - away_sp,
            'talent_diff': (home_talent - away_talent) * 0.4,  # Reduced weight
            'fpi_diff': home_fpi - away_fpi,
            
            # Sample size confidence
            'home_sample_confidence': home_metrics['sample_confidence'],
            'away_sample_confidence': away_metrics['sample_confidence'],
            'combined_confidence': (home_metrics['sample_confidence'] + away_metrics['sample_confidence']) / 2,
            
            # Uncertainty factors
            'total_uncertainty_adj': total_uncertainty_adj,
            'win_prob_uncertainty': win_prob_uncertainty,
            'home_qb_confidence': home_qb_confidence,
            'away_qb_confidence': away_qb_confidence,
            'home_ol_health': home_ol_starters / 5.0,
            'away_ol_health': away_ol_starters / 5.0,
            
            # Expected points (conservative, with all adjustments)
            'expected_total_base': pace_regressed * 0.64,  # Conservative PPP
            'expected_total_adj': pace_regressed * 0.64 + weekday_adj['total_adj'] + total_uncertainty_adj,
            
            # Spread expectation (with dynamic HFA and weekday adj)
            'expected_spread_base': (home_metrics['margin'] - away_metrics['margin']) * 0.5 + dynamic_hfa,
            'expected_spread_adj': ((home_metrics['margin'] - away_metrics['margin']) * 0.5 + 
                                   dynamic_hfa + weekday_adj['spread_adj']),
            
            # Volatility indicators
            'combined_variance': (home_metrics['scoring_variance'] + away_metrics['scoring_variance']) / 2,
            'pace_volatility': (home_metrics['pace_variance'] + away_metrics['pace_variance']) / 2,
            
            # Regression signals (prevent overconfidence)
            'extreme_favorite': 1 if abs(home_sp - away_sp) > 24 else 0,
            'large_talent_gap': 1 if abs(home_talent - away_talent) > 100 else 0,
        }
        
        return features