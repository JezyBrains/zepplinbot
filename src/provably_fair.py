import hashlib
import hmac

class ProvablyFair:
    """
    Utilities for verifying Provably Fair crash games.
    Based on standard implementation used by many crash sites:
    Hash = SHA256(Key)
    Multiplier = calculated from Hash (or Key)
    """

    @staticmethod
    def verify_hash(round_key: str, provided_hash: str) -> bool:
        """
        Verify if the round key matches the provided hash.
        Most sites use SHA256(round_key) -> hash.
        """
        if not round_key or not provided_hash:
            return False
        
        calculated_hash = hashlib.sha256(round_key.encode('utf-8')).hexdigest()
        return calculated_hash == provided_hash

    @staticmethod
    def calculate_multiplier(round_key: str) -> float:
        """
        Calculate the crash multiplier from the round key.
        This is a common algorithm (e.g. used by Busterabit, etc.)
        Note: Exact formula varies by site. This is a generic implementation
        that commonly matches: 
        1. HMAC_SHA256(key, salt)
        2. Convert to float
        3. 1 / (1 - float) -> multiplier
        
        For Zeppelin/Betsolutions, the key format `7675129_12.44_UUID` 
        actually CONTAINS the multiplier in plain text (12.44).
        So for this specific provider, we extract it directly.
        """
        try:
            # Zeppelin format: "RoundNum_Multiplier_UUID"
            # Example: 7675129_12.44_10ee1bc7-c70c-445c-80de-c5826c349713
            parts = round_key.split('_')
            if len(parts) >= 2:
                return float(parts[1])
            return 0.0
        except:
            return 0.0

    @staticmethod
    def extract_serial(round_key: str) -> str:
        """Extract the UUID/Serial part from the round key"""
        try:
            parts = round_key.split('_')
            if len(parts) >= 3:
                return parts[2]
            return ""
        except:
            return ""

    @staticmethod
    def generate_hash(round_key: str) -> str:
        """Generate SHA256 hash from key"""
        return hashlib.sha256(round_key.encode('utf-8')).hexdigest()
