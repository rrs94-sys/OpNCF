#!/usr/bin/env python3
"""
main.py - FINAL VERSION
Complete NCAA Football Betting Model Pipeline
Optimized for maximum accuracy with all enhancements
"""

import os
import sys
import pandas as pd
from pathlib import Path

from data_collector import DataCollector
from feature_engineer import FeatureEngineer
from betting_model import BettingModel
from backtester import Backtester
from pipeline import NCAABettingPipeline


def main():
    print("\n" + "="*100)
    print("NCAA FOOTBALL BETTING MODEL - V2.0 (ENHANCED)")
    print("="*100)
    print("\nEnhancements:")
    print("  ✓ Dynamic HFA (context-dependent)")
    print("  ✓ Weekday adjustments (Tue/Wed penalties)")
    print("  ✓ Pace tempo blending (harmonic mean)")
    print("  ✓ Probability calibration (isotonic regression)")
    print("  ✓ Market sanity guardrails (±6 spread, ±7 total)")
    print("  ✓ Uncertainty penalties (injuries/lineup)")
    print("="*100)
    
    # ========== CONFIGURATION ==========
    API_KEY = os.getenv('CFBD_API_KEY')
    
    if not API_KEY:
        print("\n❌ ERROR: CFBD_API_KEY not set!")
        print("\nSet your API key:")
        print("  export CFBD_API_KEY='your_key_here'")
        print("\nOr pass it in code:")
        print("  API_KEY = 'your_key_here'")
        return 1
    
    # Years to train on (2-3 years recommended)
    TRAIN_YEARS = [2023, 2024]
    
    # Current season and week
    CURRENT_YEAR = 2025
    CURRENT_WEEK = 8
    
    # Edge thresholds (higher = more selective)
    MIN_SPREAD_EDGE = 3.0
    MIN_TOTAL_EDGE = 3.0
    
    # Backtest parameters
    BACKTEST_WEEKS = 4  # Test on last 4 weeks
    BACKTEST_EDGE = 2.5
    
    # Output directories
    OUTPUT_DIR = Path("output")
    MODELS_DIR = Path("models")
    OUTPUT_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    
    # ========== OPTIONAL: INJURY DATA ==========
    # Define injury/uncertainty data for teams this week
    # Format: {'Team': {'qb_conf': 0.0-1.0, 'ol_starters': 0-5, 'key_injuries': int}}
    
    injury_data = {
        # Example - replace with actual data:
        # 'Alabama': {
        #     'qb_conf': 0.85,      # 85% certain about starting QB
        #     'ol_starters': 4,     # 4 of 5 OL starters healthy
        #     'key_injuries': 1     # 1 other key injury (WR1, etc)
        # },
        # 'LSU': {
        #     'qb_conf': 1.0,       # Fully certain
        #     'ol_starters': 5,     # Full healthy OL
        #     'key_injuries': 0     # No injuries
        # },
    }
    
    # ========== INITIALIZE PIPELINE ==========
    print("\n[Step 1/6] Initializing pipeline...")
    try:
        pipeline = NCAABettingPipeline(API_KEY)
        print("  ✓ Pipeline initialized")
        print(f"  ✓ API calls used so far: {pipeline.collector.call_count}")
    except Exception as e:
        print(f"  ❌ Error initializing: {e}")
        return 1
    
    # ========== COLLECT HISTORICAL DATA ==========
    print(f"\n[Step 2/6] Collecting historical data ({TRAIN_YEARS})...")
    try:
        historical_data = pipeline.collect_data(TRAIN_YEARS)
        
        if historical_data.empty:
            print("  ❌ No historical data collected!")
            return 1
        
        print(f"  ✓ Collected {len(historical_data)} games")
        print(f"  ✓ API calls used: {pipeline.collector.call_count}")
        
        # Save raw data
        historical_data.to_csv(OUTPUT_DIR / "historical_data.csv", index=False)
        print(f"  ✓ Saved: {OUTPUT_DIR}/historical_data.csv")
        
    except Exception as e:
        print(f"  ❌ Error collecting data: {e}")
        return 1
    
    # ========== COLLECT CURRENT SEASON DATA ==========
    print(f"\n[Step 3/6] Collecting {CURRENT_YEAR} season data (through Week {CURRENT_WEEK-1})...")
    try:
        current_season_data = pipeline.collect_data([CURRENT_YEAR])
        
        if not current_season_data.empty:
            # Filter to games before current week
            current_season_data = current_season_data[
                current_season_data['week'] < CURRENT_WEEK
            ]
            print(f"  ✓ Collected {len(current_season_data)} games from {CURRENT_YEAR}")
        else:
            print(f"  ⚠️  No {CURRENT_YEAR} data yet")
            current_season_data = pd.DataFrame()
        
        # Combine all data
        all_data = pd.concat([historical_data, current_season_data], ignore_index=True)
        print(f"  ✓ Total training data: {len(all_data)} games")
        
    except Exception as e:
        print(f"  ❌ Error collecting current season: {e}")
        all_data = historical_data
    
    # ========== TRAIN MODELS ==========
    print("\n[Step 4/6] Training models...")
    try:
        pipeline.model.train(all_data)
        print("  ✓ All models trained successfully")
        
        # Save models
        pipeline.model.save(str(MODELS_DIR))
        print(f"  ✓ Models saved to {MODELS_DIR}/")
        
    except Exception as e:
        print(f"  ❌ Error training models: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # ========== BACKTEST ==========
    print(f"\n[Step 5/6] Running backtest (last {BACKTEST_WEEKS} weeks)...")
    try:
        # Get test data
        if 'week' in all_data.columns and not all_data.empty:
            max_week = all_data['week'].max()
            test_data = all_data[all_data['week'] > max_week - BACKTEST_WEEKS]
            
            if len(test_data) > 20:  # Need decent sample
                backtest_results = pipeline.backtester.run(test_data, edge_threshold=BACKTEST_EDGE)
                pipeline.backtester.print_results(backtest_results)
                
                # Save backtest results
                backtest_summary = {
                    'spread_roi': backtest_results['spread']['roi'],
                    'total_roi': backtest_results['total']['roi'],
                    'ml_roi': backtest_results['moneyline']['roi'],
                    'spread_bets': backtest_results['spread']['bets'],
                    'total_bets': backtest_results['total']['bets'],
                    'ml_bets': backtest_results['moneyline']['bets'],
                }
                pd.DataFrame([backtest_summary]).to_csv(OUTPUT_DIR / "backtest_summary.csv", index=False)
            else:
                print(f"  ⚠️  Not enough test data ({len(test_data)} games)")
                backtest_results = None
        else:
            print("  ⚠️  Cannot run backtest (missing week data)")
            backtest_results = None
            
    except Exception as e:
        print(f"  ⚠️  Backtest error: {e}")
        backtest_results = None
    
    # ========== PREDICT CURRENT WEEK ==========
    print(f"\n[Step 6/6] Generating Week {CURRENT_WEEK} predictions...")
    try:
        week_predictions = pipeline.predict_week(
            CURRENT_YEAR, 
            CURRENT_WEEK, 
            all_data,
            injury_data=injury_data if injury_data else None
        )
        
        if week_predictions is not None and not week_predictions.empty:
            print(f"  ✓ Generated {len(week_predictions)} predictions")
            
            # Save predictions
            week_predictions.to_csv(OUTPUT_DIR / f"week{CURRENT_WEEK}_predictions.csv", index=False)
            print(f"  ✓ Saved: {OUTPUT_DIR}/week{CURRENT_WEEK}_predictions.csv")
            
            # Print formatted predictions
            pipeline.print_predictions(week_predictions, min_edge=MIN_SPREAD_EDGE)
            
        else:
            print("  ⚠️  No predictions generated (no games scheduled?)")
            week_predictions = pd.DataFrame()
            
    except Exception as e:
        print(f"  ❌ Error generating predictions: {e}")
        import traceback
        traceback.print_exc()
        week_predictions = pd.DataFrame()
    
    # ========== SUMMARY ==========
    print("\n" + "="*100)
    print("EXECUTION SUMMARY")
    print("="*100)
    print(f"API Calls Used: {pipeline.collector.call_count} / 60,000")
    print(f"Training Games: {len(all_data)}")
    print(f"Training Years: {TRAIN_YEARS} + {CURRENT_YEAR}")
    
    if backtest_results:
        print(f"\nBacktest Performance:")
        print(f"  Spread: {backtest_results['spread']['bets']} bets, "
              f"{backtest_results['spread']['win_pct']:.1%} win rate, "
              f"{backtest_results['spread']['roi']:+.1f}% ROI")
        print(f"  Total:  {backtest_results['total']['bets']} bets, "
              f"{backtest_results['total']['win_pct']:.1%} win rate, "
              f"{backtest_results['total']['roi']:+.1f}% ROI")
        print(f"  ML:     {backtest_results['moneyline']['bets']} bets, "
              f"{backtest_results['moneyline']['win_pct']:.1%} win rate, "
              f"{backtest_results['moneyline']['roi']:+.1f}% ROI")
    
    print(f"\nWeek {CURRENT_WEEK} Predictions: {len(week_predictions)} games")
    
    if not week_predictions.empty:
        # Count recommendations
        spread_recs = week_predictions[week_predictions['spread_edge'].abs() >= MIN_SPREAD_EDGE]
        total_recs = week_predictions[week_predictions['total_edge'].abs() >= MIN_TOTAL_EDGE]
        ml_recs = week_predictions[week_predictions.get('confidence_calibrated', 0) > 0.72]
        
        print(f"  Spread Recommendations: {len(spread_recs)}")
        print(f"  Total Recommendations: {len(total_recs)}")
        print(f"  Moneyline Recommendations: {len(ml_recs)}")
    
    print("\nFiles Created:")
    print(f"  📊 {OUTPUT_DIR}/historical_data.csv - All training data")
    print(f"  🎯 {OUTPUT_DIR}/week{CURRENT_WEEK}_predictions.csv - Current week predictions")
    if backtest_results:
        print(f"  📈 {OUTPUT_DIR}/backtest_summary.csv - Backtest performance")
    print(f"  🤖 {MODELS_DIR}/ - Trained models (spread, total, moneyline, calibrator)")
    
    print("\n" + "="*100)
    print("✅ Pipeline Complete!")
    print("="*100)
    
    print("\n💡 Tips:")
    print("  - Review predictions in output/week{}_predictions.csv".format(CURRENT_WEEK))
    print("  - Check 'spread_capped' and 'total_capped' columns for guardrail activations")
    print("  - Use 'confidence_calibrated' for moneyline bet sizing")
    print("  - Update injury_data dict in this file for better accuracy")
    print("  - Retrain weekly with updated data")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
