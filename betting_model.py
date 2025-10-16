"""
betting_model.py
ML models with probability calibration and market sanity guardrails
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Optional
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve


class BettingModel:
    """ML models with advanced calibration and guardrails"""
    
    def __init__(self):
        self.spread_model = GradientBoostingRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            min_samples_split=20, subsample=0.8, random_state=42
        )
        self.total_model = GradientBoostingRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            min_samples_split=20, subsample=0.8, random_state=42
        )
        self.moneyline_model = GradientBoostingClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            min_samples_split=20, subsample=0.8, random_state=42
        )
        
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.is_trained = False
        
        # Probability calibration
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        self.is_calibrated = False
        
        # Market sanity bounds
        self.MAX_SPREAD_EDGE = 6.0  # Don't deviate more than 6 pts from market spread
        self.MAX_TOTAL_EDGE = 7.0   # Don't deviate more than 7 pts from market total
        
    def train(self, training_df: pd.DataFrame):
        """Train all models with calibration"""
        
        print("\n  Training models...")
        
        # Define features
        exclude_cols = [
            'game_id', 'season', 'week', 'date', 'home_team', 'away_team',
            'home_points', 'away_points', 'actual_spread', 'total_points',
            'home_win', 'betting_spread', 'betting_total'
        ]
        
        self.feature_columns = [col for col in training_df.columns if col not in exclude_cols]
        print(f"    Features: {len(self.feature_columns)}")
        
        # Train spread model
        if 'actual_spread' in training_df.columns:
            spread_data = training_df[training_df['actual_spread'].notna()].copy()
            if len(spread_data) > 100:
                X = spread_data[self.feature_columns].fillna(0)
                y = spread_data['actual_spread']
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                X_train_scaled = self.scaler.fit_transform(X_train)
                X_test_scaled = self.scaler.transform(X_test)
                
                self.spread_model.fit(X_train_scaled, y_train)
                preds = self.spread_model.predict(X_test_scaled)
                mae = mean_absolute_error(y_test, preds)
                print(f"    Spread MAE: {mae:.2f} pts")
            else:
                print("    Not enough spread data (need > 100 rows)")
        else:
            print("    'actual_spread' column not found")
        
        # Train total model
        if 'total_points' in training_df.columns:
            total_data = training_df[training_df['total_points'].notna()].copy()
            if len(total_data) > 100:
                X = total_data[self.feature_columns].fillna(0)
                y = total_data['total_points']
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                X_train_scaled = self.scaler.transform(X_train)
                X_test_scaled = self.scaler.transform(X_test)
                
                self.total_model.fit(X_train_scaled, y_train)
                preds = self.total_model.predict(X_test_scaled)
                mae = mean_absolute_error(y_test, preds)
                print(f"    Total MAE: {mae:.2f} pts")
            else:
                print("    Not enough total data (need > 100 rows)")
        else:
            print("    'total_points' column not found")
        
        # Train moneyline model AND calibrate probabilities
        if 'home_win' in training_df.columns:
            ml_data = training_df[training_df['home_win'].notna()].copy()
            if len(ml_data) > 100:
                X = ml_data[self.feature_columns].fillna(0)
                y = ml_data['home_win']
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                X_train_scaled = self.scaler.transform(X_train)
                X_test_scaled = self.scaler.transform(X_test)
                
                self.moneyline_model.fit(X_train_scaled, y_train)
                
                # Get raw probabilities on test set
                raw_probs = self.moneyline_model.predict_proba(X_test_scaled)[:, 1]
                
                # Train isotonic calibrator
                self.calibrator.fit(raw_probs, y_test)
                self.is_calibrated = True
                
                # Evaluate calibration
                calibrated_probs = self.calibrator.predict(raw_probs)
                acc_raw = accuracy_score(y_test, (raw_probs > 0.5).astype(int))
                acc_cal = accuracy_score(y_test, (calibrated_probs > 0.5).astype(int))
                
                print(f"    Moneyline Accuracy (raw): {acc_raw:.1%}")
                print(f"    Moneyline Accuracy (calibrated): {acc_cal:.1%}")
                
                # Print calibration curve buckets
                self._print_calibration_stats(raw_probs, calibrated_probs, y_test)
            else:
                print("    Not enough moneyline data (need > 100 rows)")
        else:
            print("    'home_win' column not found")
        
        self.is_trained = True
    
    def _print_calibration_stats(self, raw_probs, calibrated_probs, y_true):
        """Print calibration statistics by probability bucket"""
        print("\n    Calibration Analysis:")
        
        buckets = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
        
        for low, high in buckets:
            # Raw probabilities
            mask_raw = (raw_probs >= low) & (raw_probs < high)
            if mask_raw.sum() > 0:
                actual_rate_raw = y_true[mask_raw].mean()
                avg_pred_raw = raw_probs[mask_raw].mean()
                count = mask_raw.sum()
                print(f"      {low:.0%}-{high:.0%} (raw): {count} games, "
                      f"predicted {avg_pred_raw:.1%}, actual {actual_rate_raw:.1%}")
            
            # Calibrated probabilities
            mask_cal = (calibrated_probs >= low) & (calibrated_probs < high)
            if mask_cal.sum() > 0:
                actual_rate_cal = y_true[mask_cal].mean()
                avg_pred_cal = calibrated_probs[mask_cal].mean()
                count_cal = mask_cal.sum()
                print(f"      {low:.0%}-{high:.0%} (cal): {count_cal} games, "
                      f"predicted {avg_pred_cal:.1%}, actual {actual_rate_cal:.1%}")
    
    def predict(self, features_df: pd.DataFrame, 
                market_spread: Optional[float] = None,
                market_total: Optional[float] = None) -> Dict:
        """
        Make prediction with calibration and market sanity guardrails
        
        Args:
            features_df: Game features
            market_spread: Market spread (for guardrails)
            market_total: Market total (for guardrails)
        """
        
        X = features_df[self.feature_columns].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        # Raw predictions
        spread_raw = self.spread_model.predict(X_scaled)[0]
        total_raw = self.total_model.predict(X_scaled)[0]
        raw_probs = self.moneyline_model.predict_proba(X_scaled)[0]
        
        # APPLY MARKET SANITY GUARDRAILS
        spread_pred = spread_raw
        if market_spread is not None:
            spread_edge = spread_raw - market_spread
            if abs(spread_edge) > self.MAX_SPREAD_EDGE:
                # Cap edge at max
                spread_pred = market_spread + np.sign(spread_edge) * self.MAX_SPREAD_EDGE
                if abs(spread_edge) > 10:  # Log extreme cases
                    print(f"      [Spread guardrail applied: raw={spread_raw:+.1f}, "
                          f"market={market_spread:+.1f}, capped={spread_pred:+.1f}]")
        
        total_pred = total_raw
        if market_total is not None:
            total_edge = total_raw - market_total
            if abs(total_edge) > self.MAX_TOTAL_EDGE:
                # Cap edge at max
                total_pred = market_total + np.sign(total_edge) * self.MAX_TOTAL_EDGE
                if abs(total_edge) > 10:
                    print(f"      [Total guardrail applied: raw={total_raw:.1f}, "
                          f"market={market_total:.1f}, capped={total_pred:.1f}]")
        
        # CALIBRATE PROBABILITIES
        home_prob_raw = raw_probs[1]
        away_prob_raw = raw_probs[0]
        
        if self.is_calibrated:
            # Use isotonic calibration
            home_prob_calibrated = self.calibrator.predict([home_prob_raw])[0]
            away_prob_calibrated = 1 - home_prob_calibrated
        else:
            # Fallback: move 30% toward 0.5 (reduce overconfidence)
            home_prob_calibrated = 0.5 + (home_prob_raw - 0.5) * 0.70
            away_prob_calibrated = 1 - home_prob_calibrated
        
        # Additional uncertainty adjustment from features
        if 'win_prob_uncertainty' in features_df.columns:
            uncertainty = float(features_df['win_prob_uncertainty'].iloc[0])
            # Push probabilities toward 0.5 based on uncertainty
            home_prob_calibrated = 0.5 + (home_prob_calibrated - 0.5) * (1 - uncertainty)
            away_prob_calibrated = 1 - home_prob_calibrated
        
        # Close game check: if spread is tight, cap max probability
        if abs(spread_pred) < 3:
            max_prob = 0.62  # Can't be more confident than 62% in close game
            if home_prob_calibrated > max_prob:
                home_prob_calibrated = max_prob
                away_prob_calibrated = 1 - max_prob
            elif away_prob_calibrated > max_prob:
                away_prob_calibrated = max_prob
                home_prob_calibrated = 1 - max_prob
        
        return {
            'predicted_spread': spread_pred,
            'predicted_spread_raw': spread_raw,
            'spread_capped': abs(spread_raw - spread_pred) > 0.1,
            'predicted_total': total_pred,
            'predicted_total_raw': total_raw,
            'total_capped': abs(total_raw - total_pred) > 0.1,
            'home_win_prob': home_prob_calibrated,
            'away_win_prob': away_prob_calibrated,
            'home_win_prob_raw': home_prob_raw,
            'confidence_calibrated': max(home_prob_calibrated, away_prob_calibrated),
            'confidence_raw': max(home_prob_raw, away_prob_raw)
        }
    
    def save(self, directory: str = "models"):
        """Save models and calibrator"""
        os.makedirs(directory, exist_ok=True)
        
        with open(f"{directory}/spread_model.pkl", "wb") as f:
            pickle.dump(self.spread_model, f)
        with open(f"{directory}/total_model.pkl", "wb") as f:
            pickle.dump(self.total_model, f)
        with open(f"{directory}/moneyline_model.pkl", "wb") as f:
            pickle.dump(self.moneyline_model, f)
        with open(f"{directory}/scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)
        with open(f"{directory}/feature_columns.pkl", "wb") as f:
            pickle.dump(self.feature_columns, f)
        
        if self.is_calibrated:
            with open(f"{directory}/calibrator.pkl", "wb") as f:
                pickle.dump(self.calibrator, f)
        
        print(f"    Models saved to {directory}/")
    
    def load(self, directory: str = "models"):
        """Load models and calibrator"""
        with open(f"{directory}/spread_model.pkl", "rb") as f:
            self.spread_model = pickle.load(f)
        with open(f"{directory}/total_model.pkl", "rb") as f:
            self.total_model = pickle.load(f)
        with open(f"{directory}/moneyline_model.pkl", "rb") as f:
            self.moneyline_model = pickle.load(f)
        with open(f"{directory}/scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)
        with open(f"{directory}/feature_columns.pkl", "rb") as f:
            self.feature_columns = pickle.load(f)
        
        # Try to load calibrator
        calibrator_path = f"{directory}/calibrator.pkl"
        if os.path.exists(calibrator_path):
            with open(calibrator_path, "rb") as f:
                self.calibrator = pickle.load(f)
            self.is_calibrated = True
        else:
            self.is_calibrated = False
        
        self.is_trained = True
        print(f"    Models loaded from {directory}/ (calibrated: {self.is_calibrated})")