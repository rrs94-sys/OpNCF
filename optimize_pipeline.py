#!/usr/bin/env python3
"""
optimize_pipeline.py
Run this after collecting data to optimize all models for maximum accuracy
"""

import sys
import pandas as pd
from pathlib import Path
from spread_optimizer import SpreadModelOptimizer
from betting_model import BettingModel


def main():
    print("\n" + "="*80)
    print("MODEL OPTIMIZATION PIPELINE")
    print("="*80)
    
    # Check for training data
    data_file = Path("output/historical_data.csv")
    if not data_file.exists():
        print("\n❌ Error: No training data found!")
        print("   Run 'python main.py' first to collect data")
        return 1
    
    # Load data
    print("\n[1/4] Loading training data...")
    training_data = pd.read_csv(data_file)
    print(f"  ✓ Loaded {len(training_data)} games")
    
    # Check data quality
    spread_games = training_data[training_data['actual_spread'].notna()]
    print(f"  ✓ Games with spreads: {len(spread_games)}")
    
    if len(spread_games) < 100:
        print(f"\n⚠️  Warning: Only {len(spread_games)} games with spreads")
        print("   Recommendation: Collect more years of data for better optimization")
        print("   Minimum recommended: 200+ games")
        
        response = input("\n   Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return 0
    
    # Ask user which optimization to run
    print("\n[2/4] Select optimization method:")
    print("  1. Quick (Feature Selection) - 10 min, 0.5-1.5 pts improvement")
    print("  2. Standard (Ensemble) - 20 min, 1-3 pts improvement")
    print("  3. Full (All Methods) - 60 min, 2-4 pts improvement - RECOMMENDED")
    print("  4. ATS-Focused (Against The Spread) - 20 min, +3-5% ATS win rate")
    
    choice = input("\nEnter choice (1-4) [default: 3]: ").strip() or "3"
    
    method_map = {
        '1': 'feature_selection',
        '2': 'ensemble',
        '3': 'all',
        '4': 'ats'
    }
    
    optimize_method = method_map.get(choice, 'all')
    
    # Run optimization
    print(f"\n[3/4] Running {optimize_method} optimization...")
    print("     This may take a while - go grab coffee! ☕")
    
    optimizer = SpreadModelOptimizer()
    
    try:
        results = optimizer.run_full_optimization(
            training_data,
            optimize_method=optimize_method
        )
    except Exception as e:
        print(f"\n❌ Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Save optimized model
    print("\n[4/4] Saving optimized models...")
    
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Save spread model
    optimizer.save_optimized_model("models/spread_model_optimized.pkl")
    
    # Print results summary
    print("\n" + "="*80)
    print("OPTIMIZATION RESULTS")
    print("="*80)
    
    if optimize_method == 'all':
        print("\n📊 Comparison of Methods:")
        print("-"*80)
        
        # Feature selection
        if 'feature_selection' in results:
            fs = results['feature_selection']
            print(f"\nFeature Selection:")
            print(f"  Selected Features: {fs['n_features']}")
            print(f"  Improvement: Removed {len(optimizer.feature_columns) - fs['n_features']} less important features")
        
        # Ensemble
        if 'ensemble' in results:
            ens = results['ensemble']
            print(f"\nEnsemble:")
            print(f"  Test MAE: {ens['ensemble_mae']:.2f} pts")
            print(f"  Model Weights:")
            for model_name, weight in ens['weights'].items():
                print(f"    {model_name:15s} {weight:>6.1%}")
        
        # ATS
        if 'ats' in results:
            ats = results['ats']
            print(f"\nATS Optimization:")
            print(f"  ATS Win Rate: {ats['avg_ats_score']:.1%}")
            print(f"  CV Scores: {', '.join([f'{s:.1%}' for s in ats['ats_cv_scores']])}")
        
        # Grid search
        if 'grid_search' in results:
            gs = results['grid_search']
            print(f"\nGrid Search:")
            print(f"  Best MAE: {gs['best_score']:.2f} pts")
            print(f"  Best Parameters:")
            for param, value in gs['best_params'].items():
                print(f"    {param:20s} {value}")
    
    else:
        # Single method results
        if optimize_method in results:
            result = results[optimize_method]
            
            if optimize_method == 'feature_selection':
                print(f"\n✓ Selected {result['n_features']} most important features")
                print(f"  Removed {len(optimizer.feature_columns) - result['n_features']} features")
            
            elif optimize_method == 'ensemble':
                print(f"\n✓ Ensemble MAE: {result['ensemble_mae']:.2f} pts")
                print(f"\n  Model Weights:")
                for model_name, weight in result['weights'].items():
                    print(f"    {model_name:15s} {weight:>6.1%}")
            
            elif optimize_method == 'ats':
                print(f"\n✓ ATS Win Rate: {result['avg_ats_score']:.1%}")
                print(f"  CV Scores: {', '.join([f'{s:.1%}' for s in result['ats_cv_scores']])}")
            
            elif optimize_method == 'grid_search':
                print(f"\n✓ Best MAE: {result['best_score']:.2f} pts")
                print(f"\n  Best Parameters:")
                for param, value in result['best_params'].items():
                    print(f"    {param:20s} {value}")
    
    # Feature importance
    if optimizer.feature_importance:
        print("\n📊 Top 10 Most Important Features:")
        print("-"*80)
        sorted_features = sorted(
            optimizer.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for i, (feature, importance) in enumerate(sorted_features[:10], 1):
            bar_length = int(importance * 50)
            bar = '█' * bar_length
            print(f"{i:2d}. {feature:35s} {bar} {importance:.4f}")
    
    print("\n" + "="*80)
    print("✅ OPTIMIZATION COMPLETE!")
    print("="*80)
    
    print("\n📁 Files Created:")
    print("  models/spread_model_optimized.pkl - Optimized spread model")
    
    print("\n🚀 Next Steps:")
    print("  1. Update betting_model.py to use optimized model")
    print("  2. Run backtest to verify improvement:")
    print("     python -c \"from betting_model import BettingModel; m = BettingModel(); m.load_optimized()\"")
    print("  3. Generate predictions for current week:")
    print("     python main.py")
    
    print("\n💡 Tips:")
    print("  - Retrain weekly with updated data")
    print("  - Monitor performance vs baseline")
    print("  - Use ensemble method for big games")
    print("  - Track feature importance changes over time")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
