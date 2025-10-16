"""
backtester.py
Backtesting functionality for betting strategy
"""

import pandas as pd
from typing import Dict


class Backtester:
    """Backtest betting strategy"""
    
    def __init__(self, model):
        self.model = model
    
    def run(self, test_df: pd.DataFrame, edge_threshold: float = 2.5) -> Dict:
        """Backtest with edge threshold - using conservative thresholds"""
        
        print("\n  Running backtest...")
        
        results = {
            'spread': {'bets': 0, 'wins': 0, 'units': 0},
            'total': {'bets': 0, 'wins': 0, 'units': 0},
            'moneyline': {'bets': 0, 'wins': 0, 'units': 0}
        }
        
        for idx, row in test_df.iterrows():
            if pd.isna(row['betting_spread']) or pd.isna(row['betting_total']):
                continue
            
            pred = self.model.predict(row.to_frame().T)
            
            # Spread betting - more conservative edge required
            spread_edge = abs(pred['predicted_spread'] - row['betting_spread'])
            if spread_edge >= edge_threshold:
                results['spread']['bets'] += 1
                bet_home = pred['predicted_spread'] > row['betting_spread']
                won = (bet_home and row['actual_spread'] > row['betting_spread']) or \
                      (not bet_home and row['actual_spread'] < row['betting_spread'])
                
                if won:
                    results['spread']['wins'] += 1
                    results['spread']['units'] += 1
                else:
                    results['spread']['units'] -= 1.1
            
            # Total betting - more conservative
            total_edge = abs(pred['predicted_total'] - row['betting_total'])
            if total_edge >= edge_threshold:
                results['total']['bets'] += 1
                bet_over = pred['predicted_total'] > row['betting_total']
                won = (bet_over and row['total_points'] > row['betting_total']) or \
                      (not bet_over and row['total_points'] < row['betting_total'])
                
                if won:
                    results['total']['wins'] += 1
                    results['total']['units'] += 1
                else:
                    results['total']['units'] -= 1.1
            
            # Moneyline (MUCH higher confidence required to reduce overconfidence)
            # Only bet if calibrated confidence is very high
            if pred['confidence_calibrated'] > 0.72:  # Raised from 0.65
                results['moneyline']['bets'] += 1
                bet_home = pred['home_win_prob'] > 0.5
                won = (row['home_win'] == 1 and bet_home) or (row['home_win'] == 0 and not bet_home)
                
                if won:
                    results['moneyline']['wins'] += 1
                    results['moneyline']['units'] += 1
                else:
                    results['moneyline']['units'] -= 1
        
        # Calculate stats
        for bet_type in results:
            if results[bet_type]['bets'] > 0:
                results[bet_type]['win_pct'] = results[bet_type]['wins'] / results[bet_type]['bets']
                results[bet_type]['roi'] = (results[bet_type]['units'] / results[bet_type]['bets']) * 100
            else:
                results[bet_type]['win_pct'] = 0
                results[bet_type]['roi'] = 0
        
        return results
    
    def print_results(self, results: Dict):
        """Print backtest results"""
        print("\n" + "="*70)
        print("BACKTEST RESULTS")
        print("="*70)
        
        for bet_type in ['spread', 'total', 'moneyline']:
            r = results[bet_type]
            print(f"\n{bet_type.upper()}:")
            print(f"  Bets: {r['bets']} | Wins: {r['wins']} | Win Rate: {r['win_pct']:.1%}")
            print(f"  Units: {r['units']:+.1f} | ROI: {r['roi']:+.1f}%")
        
        total_units = sum(r['units'] for r in [results['spread'], results['total'], results['moneyline']])
        print(f"\n{'='*70}")
        print(f"TOTAL UNITS: {total_units:+.1f}")
        print("="*70)