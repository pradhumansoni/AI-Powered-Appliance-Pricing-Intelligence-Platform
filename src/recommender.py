import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Any
import joblib

from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[1]

# Add project root to Python path
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
# Internal imports from your BI layer
from src.bi_layer import build_feature_dataframe, _load_model

# Relative file loading
X_train_path = BASE_DIR / "models" / "dataset" / "X_train.parquet"
y_train_path = BASE_DIR / "models" / "dataset" / "y_train.parquet"

class ApplianceRecommender:
    def __init__(self):
        # 1. Load Model and Data
        self.pipeline, self.preprocessor, self.model, _ = _load_model()
        
        self.X_train = pd.read_parquet(X_train_path)
        self.y_train = pd.read_parquet(y_train_path)
        
        # 2. Pre-calculate the 'Market Pool'
        self.pool = self.X_train.copy()
        self.pool['actual_price'] = np.expm1(self.y_train['log_price'])
        
        # 3. Calculate Fair Prices (NEW: needed for price filtering in recommendations)
        # We predict fair prices for every product in the pool once at startup,
        # then use these to filter recommendations to stay within ±15% of predicted fair price.
        log_preds = self.pipeline.predict(self.X_train)
        self.pool['fair_price'] = np.expm1(log_preds)
        
        # NOTE ON V3 DESIGN:
        # The original V2 used 'deal_score' (fair_price - actual_price) / fair_price
        # to rank recommendations. This had a critical flaw: since the price model
        # has R² = 0.887 (explains 87.7% of variance), rankings were actually
        # rewarding products where the model happened to be WRONG, not products
        # that were objectively better matches for the user.
        #
        # SOLUTION: Use pure content-based similarity (brand, features, star rating)
        # for RANKING. But DO use fair_price to FILTER to a reasonable price band.
        # This ensures recommendations are in the user's expected budget range,
        # while ranking within that band by feature similarity, not by model error.

        # IMPROVEMENT 1: Capacity Tolerance Map
        # Instead of an exact capacity match, allow a realistic +/- range per category.
        # AC is in tons, Refrigerator is in litres, Washing Machine is in kg -- so each needs its own tolerance.
        self.tolerance_map = {
            'AC': 0.2,
            'Refrigerator': 20,
            'Washing Machine': 1
        }

        # IMPROVEMENT 2: Price Range Tolerance for Recommendations (NEW)
        # When recommending, only show products priced within ±15% of fair price.
        # This ensures users get recommendations in their expected budget range,
        # not products that are wildly overpriced or underpriced.
        self.price_tolerance_pct = 0.15  # ±15%

        # IMPROVEMENT 3: Brand Diversity Control
        # Maximum number of products allowed from the same brand in the final recommendation list.
        self.max_per_brand = 2

    def get_recommendations(self, user_input: Dict[str, Any], n=10) -> List[Dict]:
        """
        Recommend appliances based on CONTENT SIMILARITY ONLY.
        
        The recommendation is purely content-based (brand, features, star rating),
        NOT based on price model predictions. This avoids the trap of recommending
        products where the price model happened to be wrong.
        
        Returns top-n products ranked by how closely they match the user's preferences.
        """
        category = user_input.get('category')
        capacity = user_input.get('capacity_value')
        
        # Map category to the specific capacity column in your pool
        category_capacity_map = {
            'AC': 'capacity_ac_tons',
            'Refrigerator': 'capacity_ref_litres',
            'Washing Machine': 'capacity_wm_kg'
        }
        
        # Get the correct column name based on the category
        capacity_col = category_capacity_map.get(category)
        
        if not capacity_col:
            return [] # Invalid category

        # 1. HARD CONSTRAINTS (The "Deal-Breakers")
        # Filter by category AND capacity WITHIN a tolerance range.
        # This ensures we only consider appliances that are actually suitable
        # for the user's stated needs (right category, right size).
        tolerance = self.tolerance_map.get(category, 0)
        capacity_diff = (self.pool[capacity_col] - capacity).abs()

        mask = (self.pool['category'] == category) & (capacity_diff <= tolerance)
        eligible_pool = self.pool[mask].copy()
            
        if eligible_pool.empty:
            return []

        # 2. SOFT CONSTRAINTS (The "Preferences")
        # Rank by cosine similarity on brand, star rating, and features.
        # This tells us: "of the products that fit the user's hard constraints,
        # which ones are most similar to what the user said they want?"
        #
        # We do NOT use price predictions here. The user chose a brand, a star
        # rating, and certain features -- we're measuring how well each product
        # in the eligible pool matches those stated preferences.
        
        user_encoded = self.preprocessor.transform(build_feature_dataframe(user_input))
        # Drop price-related columns before encoding, so we're only measuring
        # feature similarity, not price similarity.
        pool_encoded = self.preprocessor.transform(eligible_pool.drop(columns=['actual_price', 'fair_price']))
        
        similarities = cosine_similarity(user_encoded, pool_encoded).flatten()
        eligible_pool['recommendation_score'] = similarities

        # 3. PRICE FILTER (NEW: ensures recommendations are in expected budget range)
        # Calculate fair price for the user's input, then only keep products
        # priced within ±15% of that fair price. This ensures recommendations
        # are in the user's budget ballpark, not wildly overpriced or underpriced.
        user_fair_price = self.pipeline.predict(user_encoded)[0]
        user_fair_price = np.expm1(user_fair_price)
        
        price_lower = user_fair_price * (1 - self.price_tolerance_pct)
        price_upper = user_fair_price * (1 + self.price_tolerance_pct)
        
        price_mask = (eligible_pool['actual_price'] >= price_lower) & \
                     (eligible_pool['actual_price'] <= price_upper)
        eligible_pool = eligible_pool[price_mask]
        
        if eligible_pool.empty:
            return []

        # 4. RANKING

        # Sort the full eligible pool by score, best first
        ranked_pool = eligible_pool.sort_values('recommendation_score', ascending=False)

        # 5. BRAND DIVERSITY CONTROL
        # Since ranked_pool is already sorted by score (best first), grouping by
        # 'brand_name' and taking head(max_per_brand) keeps only the TOP scoring
        # products for each brand -- group_keys=False stops pandas from adding
        # an extra 'brand_name' index level to the result.
        # NOTE: renamed from 'brand' to 'brand_name' -- app.py confirms the real
        # column name in the dataset is 'brand_name', not 'brand'.
        diverse_pool = ranked_pool.groupby('brand_name', group_keys=False).head(self.max_per_brand)

        # groupby reorders rows by brand, so we re-sort by score before
        # cutting down to the final top-n list.
        final_pool = diverse_pool.sort_values('recommendation_score', ascending=False).head(n)

        return final_pool.to_dict('records')