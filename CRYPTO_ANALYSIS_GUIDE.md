# Cryptographic Analysis Guide for Provably Fair Systems

## Understanding the Challenge

You're dealing with a **SHA-256 based provably fair system**. This guide explains how to properly analyze it.

---

## 1. The System Architecture

### How Zeppelin Generates Results

```
Round Key = "{round_number}_{coefficient}_{uuid}"
                    ↓
              SHA-256 Hash
                    ↓
         Reliability Code (Proof)
```

**Example:**
```
Input:  "7669677_1.15_d36f37b5-d6e9-4de2-a6f6-31b53b7c6efd"
Output: "91d39461d0f2931f8a364fcf60304d2db39ea544978ade3fd33e9bdd4cfb433f"
```

### Why Direct Prediction is Impossible

- **SHA-256 is a one-way function** - Cannot reverse the hash
- **Avalanche effect** - Tiny input change = completely different output
- **Cryptographically secure** - No polynomial-time algorithm can predict next bit with >50% accuracy

---

## 2. What We CAN Analyze

Instead of breaking SHA-256, we look for **implementation flaws**:

### A. Seed Generation Weaknesses

**Target:** The UUID/seed generation process

**Vulnerabilities to check:**
- ✅ **Time-based seeds** - If UUID correlates with server time
- ✅ **Low entropy** - Predictable patterns in UUID generation
- ✅ **Sequential dependencies** - Current result depends on previous ones

### B. Coefficient Distribution Biases

**Target:** The mapping from hash to coefficient

**What to look for:**
- ✅ **Temporal patterns** - Different outcomes at different times
- ✅ **Frequency anomalies** - Certain values appear more often
- ✅ **Autocorrelation** - Sequence shows non-random patterns

---

## 3. Professional Analysis Workflow

### Step 1: Data Collection

Collect the **full cryptographic data**, not just coefficients:

```bash
python3 crypto_collect.py
```

**Required fields:**
- Round number
- Coefficient (crash point)
- UUID (server seed)
- Hash (for verification)
- Timestamp

### Step 2: Verify Provable Fairness

```python
from src.crypto_analyzer import ProvenFairnessVerifier

verifier = ProvenFairnessVerifier()
is_valid, hash = verifier.verify_round(
    round_number="7669677",
    coefficient=1.15,
    uuid="d36f37b5-d6e9-4de2-a6f6-31b53b7c6efd",
    provided_hash="91d39461d0f2931f8a364fcf60304d2db39ea544978ade3fd33e9bdd4cfb433f"
)
```

If `is_valid = True`, the game is **provably fair** (not cheating).

### Step 3: Run Cryptanalysis

```bash
python3 -c "
from src.crypto_analyzer import CryptoPatternDetector
import pandas as pd

df = pd.read_csv('data/zeppelin_crypto_data.csv')
detector = CryptoPatternDetector()
results = detector.full_analysis(df)
"
```

**Key metrics:**
1. **Autocorrelation** - Does current coefficient depend on previous ones?
2. **Temporal bias** - Does time-of-day affect outcomes?
3. **UUID entropy** - Is the seed generation truly random?

---

## 4. Model Architecture for Cryptanalysis

### Current System (Basic LSTM/XGBoost)

❌ **Problem:** Only looks at coefficient values  
❌ **Missing:** Binary representation, seed analysis, temporal features

### Recommended: Transformer with Binary Features

```python
# Feature engineering
features = [
    coefficient_binary,      # 32 bits
    uuid_binary,             # 128 bits
    hour_of_day,             # Temporal
    day_of_week,             # Temporal
    previous_5_coeffs,       # Sequence context
    time_since_last_high     # Pattern detection
]
```

**Why Transformers?**
- **Attention mechanism** - Sees relationships across entire sequence
- **Long-range dependencies** - Can detect patterns 100+ rounds back
- **Binary thinking** - Processes data at bit level (like crypto functions)

---

## 5. The Mathematical Reality

### What Success Looks Like

If you find an **exploitable pattern**, you'll see:

1. **Autocorrelation > 0.1** - Sequence is not random
2. **Chi-square > 23.685** - Temporal bias exists
3. **UUID entropy < 7.5** - Seed generation is weak

### What Failure Looks Like (Strong System)

- **Autocorrelation ≈ 0** - No sequence patterns
- **Uniform temporal distribution** - No time-based bias
- **UUID entropy ≈ 8.0** - Perfect randomness

### The "Next-Bit Test"

In cryptography, a PRNG is secure if:

```
P(predict next bit) ≤ 0.5 + ε
```

Where `ε` (your advantage) is negligibly small.

**Your goal:** Find if `ε > 0` by detecting implementation flaws.

---

## 6. Practical Strategy

### Phase 1: Reconnaissance (Current)

✅ Collect 100+ rounds with full crypto data  
✅ Run cryptanalysis to find vulnerabilities  
✅ Identify which attack vector is most promising

### Phase 2: Exploitation (If Pattern Found)

**If autocorrelation detected:**
→ Train Transformer on sequence patterns

**If temporal bias detected:**
→ Add time-based features to model

**If UUID entropy low:**
→ Reverse-engineer seed generation algorithm

### Phase 3: Validation

- Test predictions on new data
- Calculate actual advantage `ε`
- If `ε > 0.05`, you have an edge

---

## 7. Current Data Analysis

Run analysis on your 111 crash points:

```bash
# Convert existing data to crypto format
python3 -c "
import pandas as pd
df = pd.read_csv('data/zeppelin_data.csv')
print(f'Analyzing {len(df)} rounds...')

from src.crypto_analyzer import CryptoPatternDetector
detector = CryptoPatternDetector()
results = detector.full_analysis(df)

for rec in results['recommendations']:
    print(rec)
"
```

---

## 8. Next Steps

### Immediate Actions:

1. **Scrape full crypto data** from Zeppelin website:
   - Use OBS capture to get round numbers, UUIDs, hashes
   - Or manually collect 20-30 rounds with full data

2. **Run cryptanalysis:**
   ```bash
   python3 crypto_collect.py
   ```

3. **Based on results:**
   - **Pattern found** → Build Transformer model
   - **No pattern** → System is cryptographically secure (can't be predicted)

### Long-term Strategy:

If the system is **truly secure** (SHA-256 + CSPRNG):
- No amount of data will help
- Mathematical advantage `ε ≈ 0`
- Focus on bankroll management, not prediction

If **implementation flaw exists**:
- Transformer model can exploit it
- Advantage `ε > 0` is achievable
- Profitable strategy possible

---

## 9. Tools Available

| Tool | Purpose |
|------|---------|
| `crypto_collect.py` | Collect full cryptographic data |
| `src/crypto_analyzer.py` | Run cryptanalysis |
| `manual_collect.py` | Simple coefficient collection |
| `obs_capture.py` | Automated screen capture |

---

## 10. Disclaimer

This analysis is for **educational purposes** and understanding provably fair systems. 

**Key points:**
- Properly implemented SHA-256 systems are **mathematically secure**
- We're looking for **implementation bugs**, not breaking cryptography
- Even with patterns, gambling carries inherent risk
- Use for research and learning, not guaranteed profit

---

## Summary

**You cannot break SHA-256.**

**You CAN find:**
- Weak seed generation
- Temporal biases
- Sequence patterns
- Implementation flaws

**Next step:** Collect full crypto data and run analysis to see if exploitable patterns exist.
