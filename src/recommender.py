import pandas as pd
import numpy as np
from typing import List, Dict, Any
from pathlib import Path

# Paths
PROCESSED_DATA_PATH = Path(__file__).parent.parent / "data/03.cleaned/multi_appliances_cleaned_engineered.parquet"

class ApplianceRecommender:
    def __init__(self):
        # Load the full cleaned dataset (which includes names and actual prices)
        self.df = pd.read_parquet(PROCESSED_DATA_PATH)
        
    def find_alternatives(self, user_input_features: Dict[str, Any], n_recommendations: int = 3) -> pd.DataFrame:
        """
        Finds the best alternative appliances based on similarity and value.
        """
        category = user_input_features.get('category')
        target_capacity = user_input_features.get('capacity_value')
        
        # 1. Filter by Category
        filtered_df = self.df[self.df['category'].str.lower() == category.lower()].copy()
        
        # 2. Filter by Capacity (±15% range)
        # We need to map capacity_value to the correct column in the dataframe
        cap_col = self._get_capacity_column(category)
        if cap_col and target_capacity:
            lower_bound = target_capacity * 0.85
            upper_bound = target_capacity * 1.15
            filtered_df = filtered_df[
                (filtered_df[cap_col] >= lower_bound) & 
                (filtered_df[cap_col] <= upper_bound)
            ]

        # 3. Calculate "Value Score"
        # Value = (Predicted Price / Actual Price)
        # Higher is better (you get more than what you pay for)
        # Note: We'll need to run predictions on the filtered_df if we haven't stored them
        # For now, let's assume we use 'rating' and 'feature_count' as a proxy for value
        
        # 4. Calculate Similarity
        # Simple Euclidean distance on key features: star_rating, has_inverter, etc.
        # ... similarity logic here ...

        return filtered_df.head(n_recommendations)

    def _get_capacity_column(self, category: str) -> str:
        mapping = {
            'air conditioner': 'capacity_ac_tons',
            'refrigerator': 'capacity_ref_liters',
            'washing machine': 'capacity_wm_kg'
        }
        return mapping.get(category.lower())