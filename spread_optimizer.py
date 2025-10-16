#!/usr/bin/env python3
"""
spread_optimizer.py
Advanced training optimization specifically for spread predictions
Includes hyperparameter tuning, feature selection, and ensemble methods
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression, RFE, RFECV
import pickle
import warnings
warnings.filterwarnings('ignore')


class SpreadModelOptimizer:
    """
    Optimizes the spread prediction model through:
    1. Hyperparameter tuning
    2. Feature selection
    3. Ensemble methods
    4. Time-series cross-validation
    """
    
    def __init__(self):
        self.best_model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.feature_importance = None
        self.optimization_results = {}
        
    def prepare_data(self, training_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for spread model training"""
        
        # Filter to games with actual spreads
        spread_data = training_df[training_df['actual_spread'].notna()].copy()
        
        if len(spread_data) < 100:
            raise ValueError(f"Insufficient data: {len(spread_data)} games (need >100)")
        
        # Define features to exclude
        exclude_cols = [
            'game_id', 'season', 'week', 'date', 'home_team', 'away_team',
            'home_points', 'away_points', 'actual_spread', 'total_points',
            'home_win', 'betting_spread', 'betting_total'
        ]
        
        self.feature_columns = [col for col in spread_data.columns if col not in exclude_cols]
        
        X = spread_data[self.feature_columns].fillna(0)
        y = spread_data['actual_spread']
        
        print(f"📊 Training data: {len(X)} games, {len(self.feature_columns)} features")
        
        return X, y
    
    def time_series_split(self, X: pd.DataFrame, y: pd.Series, n_splits: int = 5):
        """
        Time-series aware cross-validation split
        Respects temporal ordering (don't train on future, test on past)
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        return tscv.split(X)
    
    def evaluate_model(self, model, X_train, X_test, y_train, y_test, model_name: str):
        """Comprehensive model evaluation"""
        
        # Scale data
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train and predict
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Betting accuracy (within 3 points)
        within_3 = np.abs(y_test - y_pred) <= 3
        accuracy_3pt = within_3.mean()
        
        # ATS simulation (against market if available)
        results = {
            'model': model_name,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'accuracy_3pt': accuracy_3pt,
            'n_test': len(y_test)
        }
        
        return results, model
    
    def grid_search_gbm(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Grid search for Gradient Boosting hyperparameters
        Optimized for spread prediction
        """
        
        print("\n🔍 Running Grid Search for Gradient Boosting...")
        
        param_grid = {
            'n_estimators': [200, 300, 400, 500],
            'max_depth': [3, 4, 5, 6],
            'learning_rate': [0.01, 0.05, 0.1],
            'min_samples_split': [10, 20, 30, 50],
            'min_samples_leaf': [5, 10, 15],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'max_features': ['sqrt', 0.5, 0.7, 1.0]
        }
        
        base_model = GradientBoostingRegressor(random_state=42, loss='huber')
        
        # Use TimeSeriesSplit for CV
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Randomized search (faster than grid search)
        random_search = RandomizedSearchCV(
            base_model,
            param_distributions=param_grid,
            n_iter=50,  # Try 50 combinations
            cv=tscv,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            verbose=1,
            random_state=42
        )
        
        # Scale data
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit
        random_search.fit(X_scaled, y)
        
        print(f"\n✅ Best MAE: {-random_search.best_score_:.2f}")
        print(f"✅ Best params: {random_search.best_params_}")
        
        return {
            'best_model': random_search.best_estimator_,
            'best_params': random_search.best_params_,
            'best_score': -random_search.best_score_,
            'cv_results': random_search.cv_results_
        }
    
    def feature_selection(self, X: pd.DataFrame, y: pd.Series, method: str = 'rfe') -> List[str]:
        """
        Select most important features for spread prediction
        
        Methods:
        - 'rfe': Recursive Feature Elimination
        - 'kbest': SelectKBest with F-statistic
        - 'importance': Model-based feature importance
        """
        
        print(f"\n🎯 Feature Selection using {method}...")
        
        X_scaled = self.scaler.fit_transform(X)
        
        if method == 'rfe':
            # Recursive Feature Elimination with CV
            estimator = GradientBoostingRegressor(n_estimators=100, random_state=42)
            selector = RFECV(
                estimator, 
                step=5, 
                cv=TimeSeriesSplit(n_splits=3),
                scoring='neg_mean_absolute_error',
                n_jobs=-1
            )
            selector.fit(X_scaled, y)
            selected_features = [f for f, selected in zip(self.feature_columns, selector.support_) if selected]
            
            print(f"  Optimal features: {len(selected_features)} / {len(self.feature_columns)}")
            
        elif method == 'kbest':
            # Select K best features
            k = min(40, len(self.feature_columns))  # Top 40 or all if less
            selector = SelectKBest(f_regression, k=k)
            selector.fit(X_scaled, y)
            selected_features = [f for f, selected in zip(self.feature_columns, selector.get_support()) if selected]
            
            print(f"  Selected top {k} features")
            
        elif method == 'importance':
            # Model-based importance
            model = GradientBoostingRegressor(n_estimators=200, random_state=42)
            model.fit(X_scaled, y)
            importances = model.feature_importances_
            
            # Select features with importance > threshold
            threshold = np.percentile(importances, 25)  # Top 75%
            selected_features = [f for f, imp in zip(self.feature_columns, importances) if imp > threshold]
            
            # Store importance for later
            self.feature_importance = dict(zip(self.feature_columns, importances))
            
            print(f"  Selected {len(selected_features)} features above importance threshold")
        
        else:
            selected_features = self.feature_columns
        
        return selected_features
    
    def create_ensemble(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Create ensemble of multiple models for spread prediction
        Combines GBM, Random Forest, and Ridge Regression
        """
        
        print("\n🎭 Creating Ensemble Model...")
        
        X_scaled = self.scaler.fit_transform(X)
        
        # Split for ensemble training
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Train multiple models
        models = {
            'gbm': GradientBoostingRegressor(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                min_samples_split=20, subsample=0.8, random_state=42
            ),
            'rf': RandomForestRegressor(
                n_estimators=200, max_depth=10, min_samples_split=20,
                random_state=42, n_jobs=-1
            ),
            'extra_trees': ExtraTreesRegressor(
                n_estimators=200, max_depth=10, min_samples_split=20,
                random_state=42, n_jobs=-1
            ),
            'ridge': Ridge(alpha=10.0)
        }
        
        predictions = {}
        weights = {}
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            mae = mean_absolute_error(y_test, preds)
            
            predictions[name] = preds
            # Weight inversely proportional to MAE
            weights[name] = 1 / mae
            
            print(f"  {name:15s} MAE: {mae:.2f}")
        
        # Normalize weights
        total_weight = sum(weights.values())
        weights = {k: v/total_weight for k, v in weights.items()}
        
        # Weighted ensemble prediction
        ensemble_pred = sum(predictions[name] * weights[name] for name in models.keys())
        ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
        
        print(f"\n  📊 Ensemble MAE: {ensemble_mae:.2f}")
        print(f"  Weights: {', '.join([f'{k}: {v:.2%}' for k, v in weights.items()])}")
        
        return {
            'models': models,
            'weights': weights,
            'ensemble_mae': ensemble_mae,
            'test_mae_individual': {k: mean_absolute_error(y_test, v) for k, v in predictions.items()}
        }
    
    def optimize_for_ats(self, X: pd.DataFrame, y: pd.Series, betting_spreads: pd.Series) -> Dict:
        """
        Optimize model specifically for Against The Spread (ATS) performance
        Custom loss function that penalizes wrong side of spread more heavily
        """
        
        print("\n💰 Optimizing for ATS Performance...")
        
        X_scaled = self.scaler.fit_transform(X)
        
        # Custom scorer: ATS win rate
        def ats_scorer(y_true, y_pred, spreads):
            """
            Calculate ATS win rate
            Bet on favorite if our pred is MORE positive than spread
            Bet on underdog if our pred is LESS positive than spread
            """
            correct = 0
            total = 0
            
            for true, pred, spread in zip(y_true, y_pred, spreads):
                if pd.notna(spread):
                    # Did we beat the spread?
                    bet_favorite = pred > spread
                    actual_covered = true > spread
                    
                    if bet_favorite == actual_covered:
                        correct += 1
                    total += 1
            
            return correct / total if total > 0 else 0.5
        
        # Train model and evaluate ATS
        model = GradientBoostingRegressor(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.03,
            min_samples_split=30,
            subsample=0.85,
            loss='huber',  # Robust to outliers
            alpha=0.9,  # Huber parameter
            random_state=42
        )
        
        # Time-series split
        tscv = TimeSeriesSplit(n_splits=5)
        ats_scores = []
        
        for train_idx, test_idx in tscv.split(X_scaled):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            spreads_test = betting_spreads.iloc[test_idx]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            ats_score = ats_scorer(y_test, y_pred, spreads_test)
            ats_scores.append(ats_score)
        
        avg_ats = np.mean(ats_scores)
        print(f"  Average ATS Win Rate: {avg_ats:.1%}")
        
        # Train final model on all data
        model.fit(X_scaled, y)
        
        return {
            'model': model,
            'ats_cv_scores': ats_scores,
            'avg_ats_score': avg_ats
        }
    
    def run_full_optimization(self, training_df: pd.DataFrame, 
                             optimize_method: str = 'ensemble') -> Dict:
        """
        Run complete optimization pipeline
        
        optimize_method:
        - 'grid_search': Hyperparameter tuning
        - 'feature_selection': Feature selection + training
        - 'ensemble': Ensemble of multiple models
        - 'ats': Optimize for against-the-spread performance
        - 'all': Run all methods and compare
        """
        
        print("\n" + "="*80)
        print("SPREAD MODEL OPTIMIZATION")
        print("="*80)
        
        # Prepare data
        X, y = self.prepare_data(training_df)
        
        # Get betting spreads if available
        betting_spreads = training_df.loc[X.index, 'betting_spread'] if 'betting_spread' in training_df.columns else pd.Series([None]*len(X))
        
        results = {}
        
        if optimize_method in ['grid_search', 'all']:
            results['grid_search'] = self.grid_search_gbm(X, y)
            self.best_model = results['grid_search']['best_model']
        
        if optimize_method in ['feature_selection', 'all']:
            selected_features = self.feature_selection(X, y, method='importance')
            X_selected = X[selected_features]
            
            # Train on selected features
            model = GradientBoostingRegressor(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                min_samples_split=20, subsample=0.8, random_state=42
            )
            X_scaled = self.scaler.fit_transform(X_selected)
            model.fit(X_scaled, y)
            
            results['feature_selection'] = {
                'selected_features': selected_features,
                'model': model,
                'n_features': len(selected_features)
            }
        
        if optimize_method in ['ensemble', 'all']:
            results['ensemble'] = self.create_ensemble(X, y)
        
        if optimize_method in ['ats', 'all']:
            results['ats'] = self.optimize_for_ats(X, y, betting_spreads)
            if optimize_method == 'ats':
                self.best_model = results['ats']['model']
        
        # Compare all methods if 'all'
        if optimize_method == 'all':
            self._compare_methods(X, y, results)
        
        self.optimization_results = results
        
        print("\n" + "="*80)
        print("OPTIMIZATION COMPLETE")
        print("="*80)
        
        return results
    
    def _compare_methods(self, X: pd.DataFrame, y: pd.Series, results: Dict):
        """Compare all optimization methods"""
        
        print("\n📊 COMPARISON OF METHODS")
        print("-"*80)
        
        X_scaled = self.scaler.fit_transform(X)
        split_idx = int(len(X) * 0.8)
        X_test, y_test = X_scaled[split_idx:], y.iloc[split_idx:]
        
        comparison = []
        
        # Grid search
        if 'grid_search' in results:
            pred = results['grid_search']['best_model'].predict(X_test)
            mae = mean_absolute_error(y_test, pred)
            comparison.append(('Grid Search GBM', mae))
        
        # Feature selection
        if 'feature_selection' in results:
            X_test_selected = self.scaler.transform(
                X.iloc[split_idx:][results['feature_selection']['selected_features']]
            )
            pred = results['feature_selection']['model'].predict(X_test_selected)
            mae = mean_absolute_error(y_test, pred)
            comparison.append((f"Feature Selection ({results['feature_selection']['n_features']} features)", mae))
        
        # Ensemble
        if 'ensemble' in results:
            comparison.append(('Ensemble', results['ensemble']['ensemble_mae']))
        
        # ATS optimized
        if 'ats' in results:
            pred = results['ats']['model'].predict(X_test)
            mae = mean_absolute_error(y_test, pred)
            comparison.append((f"ATS Optimized (ATS: {results['ats']['avg_ats_score']:.1%})", mae))
        
        # Sort by MAE
        comparison.sort(key=lambda x: x[1])
        
        print(f"\n{'Method':<50} {'Test MAE':<10}")
        print("-"*80)
        for method, mae in comparison:
            print(f"{method:<50} {mae:>8.2f} pts")
        
        # Select best model
        best_method = comparison[0][0]
        print(f"\n🏆 Best Method: {best_method}")
        
        if 'Grid Search' in best_method:
            self.best_model = results['grid_search']['best_model']
        elif 'ATS' in best_method:
            self.best_model = results['ats']['model']
        elif 'Feature Selection' in best_method:
            self.best_model = results['feature_selection']['model']
            self.feature_columns = results['feature_selection']['selected_features']
    
    def save_optimized_model(self, filepath: str = "models/spread_model_optimized.pkl"):
        """Save the optimized model"""
        
        if self.best_model is None:
            print("⚠️  No model to save. Run optimization first.")
            return
        
        save_dict = {
            'model': self.best_model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'feature_importance': self.feature_importance,
            'optimization_results': self.optimization_results
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_dict, f)
        
        print(f"✅ Optimized model saved to {filepath}")
    
    def load_optimized_model(self, filepath: str = "models/spread_model_optimized.pkl"):
        """Load the optimized model"""
        
        with open(filepath, 'rb') as f:
            save_dict = pickle.load(f)
        
        self.best_model = save_dict['model']
        self.scaler = save_dict['scaler']
        self.feature_columns = save_dict['feature_columns']
        self.feature_importance = save_dict.get('feature_importance')
        self.optimization_results = save_dict.get('optimization_results', {})
        
        print(f"✅ Optimized model loaded from {filepath}")


# Example usage
if __name__ == "__main__":
    
    # Load your training data
    import pandas as pd
    training_data = pd.read_csv("training_data.csv")
    
    # Initialize optimizer
    optimizer = SpreadModelOptimizer()
    
    # Run full optimization (this will take 10-30 minutes)
    results = optimizer.run_full_optimization(
        training_data, 
        optimize_method='all'  # Try all methods
    )
    
    # Save the best model
    optimizer.save_optimized_model("models/spread_model_optimized.pkl")
    
    print("\n✅ Optimization complete! Best model saved.")
