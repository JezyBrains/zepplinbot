#!/usr/bin/env python3
"""
Cryptographic Data Collection Tool for Provably Fair Games

Collects:
- Round numbers
- Crash coefficients
- UUIDs (server seeds)
- SHA-256 hashes
- Timestamps

This data enables cryptanalysis of the randomness source.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
from datetime import datetime
from src.crypto_analyzer import CryptoPatternDetector
import json

def collect_crypto_data():
    """Interactive collection of cryptographic game data."""
    
    print("="*70)
    print("CRYPTOGRAPHIC DATA COLLECTION - Zeppelin Provably Fair System")
    print("="*70)
    print("\nFor each round, enter:")
    print("  1. Round Number (e.g., 7669677)")
    print("  2. Coefficient (e.g., 1.15)")
    print("  3. UUID (e.g., d36f37b5-d6e9-4de2-a6f6-31b53b7c6efd)")
    print("  4. Hash (optional - for verification)")
    print("\nCommands:")
    print("  - Type 'done' when finished")
    print("  - Type 'analyze' to run cryptanalysis on collected data")
    print("  - Type 'simple' to enter only coefficients (like before)")
    print("="*70)
    
    data = []
    
    while True:
        print(f"\n--- Round {len(data) + 1} ---")
        
        # Round number
        round_num = input("Round number (or command): ").strip()
        
        if round_num.lower() == 'done':
            break
        elif round_num.lower() == 'analyze':
            if data:
                analyze_data(data)
            else:
                print("❌ No data to analyze yet!")
            continue
        elif round_num.lower() == 'simple':
            simple_mode()
            return
        
        # Coefficient
        try:
            coeff = input("Coefficient (e.g., 1.15): ").strip()
            coeff = float(coeff.replace('x', ''))
        except ValueError:
            print("❌ Invalid coefficient. Skipping round.")
            continue
        
        # UUID
        uuid = input("UUID (press Enter to skip): ").strip()
        if not uuid:
            uuid = None
        
        # Hash
        hash_val = input("Hash (press Enter to skip): ").strip()
        if not hash_val:
            hash_val = None
        
        # Store data
        entry = {
            'timestamp': datetime.now(),
            'round_number': round_num,
            'coefficient': coeff,
            'uuid': uuid,
            'hash': hash_val
        }
        data.append(entry)
        
        print(f"✅ Added round {round_num}: {coeff}x")
    
    if data:
        save_and_analyze(data)
    else:
        print("\n❌ No data collected.")

def simple_mode():
    """Fallback to simple coefficient-only collection."""
    print("\n" + "="*70)
    print("SIMPLE MODE - Coefficient Only")
    print("="*70)
    print("Enter coefficients one per line. Type 'done' when finished.")
    print("="*70)
    
    data = []
    
    while True:
        coeff_input = input(f"\nCoefficient {len(data) + 1}: ").strip()
        
        if coeff_input.lower() == 'done':
            break
        
        try:
            coeff = float(coeff_input.replace('x', ''))
            data.append({
                'timestamp': datetime.now(),
                'coefficient': coeff
            })
            print(f"✅ Added: {coeff}x")
        except ValueError:
            print("❌ Invalid number. Try again.")
    
    if data:
        df = pd.DataFrame(data)
        df.to_csv('data/zeppelin_data.csv', index=False)
        print(f"\n✅ Saved {len(df)} coefficients to data/zeppelin_data.csv")

def save_and_analyze(data):
    """Save data and run cryptanalysis."""
    df = pd.DataFrame(data)
    
    # Save to CSV
    filename = 'data/zeppelin_crypto_data.csv'
    df.to_csv(filename, index=False)
    
    print(f"\n✅ Saved {len(df)} rounds to {filename}")
    
    # Run analysis
    print("\n" + "="*70)
    print("RUNNING CRYPTANALYSIS...")
    print("="*70)
    
    analyze_data(data)

def analyze_data(data):
    """Run cryptographic analysis on collected data."""
    df = pd.DataFrame(data)
    
    detector = CryptoPatternDetector()
    results = detector.full_analysis(df)
    
    print("\n" + "="*70)
    print("CRYPTOGRAPHIC ANALYSIS REPORT")
    print("="*70)
    
    # Data summary
    summary = results['data_summary']
    print(f"\n📊 Dataset:")
    print(f"   Total rounds: {summary['total_rounds']}")
    print(f"   Coefficient range: {summary['coefficient_range'][0]:.2f}x - {summary['coefficient_range'][1]:.2f}x")
    print(f"   Mean coefficient: {summary['mean_coefficient']:.2f}x")
    
    # Autocorrelation (KEY METRIC)
    if 'autocorrelation' in results:
        autocorr = results['autocorrelation']
        print(f"\n🔍 Sequence Analysis:")
        print(f"   Max autocorrelation: {autocorr['max_autocorr']:.4f}")
        
        if autocorr['pattern_detected']:
            print(f"   ⚠️  PATTERN DETECTED at lags: {autocorr['significant_lags']}")
            print(f"   → A Transformer model may exploit this!")
        else:
            print(f"   ✅ No significant patterns (strong randomness)")
    
    # Temporal bias
    if 'temporal_bias' in results:
        temporal = results['temporal_bias']
        print(f"\n⏰ Temporal Analysis:")
        print(f"   Chi-square: {temporal['chi_square']:.2f}")
        
        if temporal['bias_detected']:
            print(f"   ⚠️  TIME-BASED BIAS DETECTED")
            print(f"   → Consider time-of-day features in model")
        else:
            print(f"   ✅ No temporal bias")
    
    # UUID entropy
    if 'uuid_entropy' in results:
        uuid_ent = results['uuid_entropy']
        print(f"\n🔐 UUID Entropy:")
        print(f"   Shannon entropy: {uuid_ent['entropy']:.4f}")
        print(f"   Bit balance: {uuid_ent['bit_balance']:.4f}")
        
        if not uuid_ent['is_random']:
            print(f"   ⚠️  LOW ENTROPY - Predictable seed generation")
        else:
            print(f"   ✅ High entropy (good randomness)")
    
    # Hash verification
    if 'hash_verification' in results:
        hash_ver = results['hash_verification']
        print(f"\n🔒 Hash Verification:")
        print(f"   Verified: {hash_ver['total_verified']} rounds")
        
        if hash_ver['all_valid']:
            print(f"   ✅ All hashes valid (provably fair)")
        else:
            print(f"   ⚠️  {hash_ver['invalid_count']} invalid hashes!")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    for i, rec in enumerate(results['recommendations'], 1):
        print(f"   {i}. {rec}")
    
    print("\n" + "="*70)
    
    # Save analysis report
    report_file = 'outputs/crypto_analysis_report.json'
    os.makedirs('outputs', exist_ok=True)
    with open(report_file, 'w') as f:
        # Convert non-serializable objects
        serializable_results = {
            k: v for k, v in results.items() 
            if k not in ['autocorrelation']  # Skip numpy arrays
        }
        json.dump(serializable_results, f, indent=2, default=str)
    
    print(f"\n📄 Full report saved to: {report_file}")

if __name__ == "__main__":
    try:
        collect_crypto_data()
    except KeyboardInterrupt:
        print("\n\n❌ Collection cancelled by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
