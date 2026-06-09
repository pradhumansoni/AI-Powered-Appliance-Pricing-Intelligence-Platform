# instructions_cleaning.md

## Objective

Transform raw Smartprix appliance data into a unified, modeling-ready dataset suitable for:

* Linear Regression
* Ridge Regression
* Lasso Regression
* Decision Trees
* Random Forest
* XGBoost
* SHAP Explainability
* Streamlit Deployment
* Recommendation Engine

The dataset combines three appliance categories:

* Air Conditioners (AC)
* Refrigerators
* Washing Machines

---

# 1. Initial Data Cleaning

## Preserve Original Dataset

Always keep an untouched copy:

```python
df_raw = df.copy()
```

---

## Convert Price to Numeric

Remove currency symbols and commas.

Example:

```
₹42,990 → 42990
```

```python
df['price'] = (
    df['price']
    .str.replace('₹', '', regex=False)
    .str.replace(',', '', regex=False)
    .astype(float)
)
```

---

## Extract User Rating

Raw format:

```
"--rating: 4.3;"
```

Target:

```
4.3
```

```python
df['rating'] = (
    df['rating']
    .str.extract(r'(\d+\.?\d*)')[0]
    .astype(float)
)
```

---

# 2. Extract Features From Product Name

The `name` column contains critical information.

Example:

```
Samsung RT34DG5A4DS8 330 L 3 Star Double Door Refrigerator
```

Extract:

* Capacity
* Star rating
* Appliance subtype
* Model year
* Inverter information

---

## Star Rating

```python
df['star_rating'] = (
    df['name']
    .str.extract(r'(\d)\s*Star')[0]
    .astype(float)
)
```

---

## Model Year

Extract years from 2020–2029.

```python
df['model_year'] = (
    df['name']
    .str.extract(r'(202\d)')[0]
    .astype(float)
)
```

---

## Recent Model Flag

Instead of imputing year:

```python
df['is_recent_model'] = (
    df['model_year'] >= 2024
).astype(int)

df['year_is_imputed'] = (
    df['model_year'].isnull()
).astype(int)
```

---

# 3. Parse Features Column

The features column contains Python-literal lists.

Convert safely:

```python
import ast

df['features'] = df['features'].apply(
    lambda x: ast.literal_eval(x)
)
```

---

# 4. Universal Binary Features

Extract from both name and features.

## Connectivity

```python
has_wifi
has_app_control
has_voice_control
```

---

## Inverter Technology

```python
has_inverter
```

---

## Smart Score

Represents smart-feature tier.

```python
df['smart_score'] = (
    df['has_wifi']
    + df['has_app_control']
    + df['has_voice_control']
)
```

---

# 5. AC-Specific Features

Prefix all AC columns with `ac_`.

Extract:

```text
ac_tonnage
ac_is_split
ac_is_window
ac_is_tower
ac_is_portable
ac_auto_clean
ac_pm25_filter
ac_hepa_filter
ac_4way_swing
ac_hot_cold
ac_copper_condenser
```

Non-AC rows:

```
Fill with 0.
```

---

# 6. Refrigerator Features

Prefix:

```text
ref_
```

Extract:

```text
ref_capacity_liters
ref_is_single_door
ref_is_double_door
ref_is_french_door
ref_is_side_by_side
ref_is_multi_door
ref_frost_free
ref_convertible
ref_water_dispenser
ref_door_alarm
ref_child_lock
```

Non-refrigerator rows:

```
Fill with 0.
```

---

# 7. Washing Machine Features

Prefix:

```text
wm_
```

Extract:

```text
wm_capacity_kg
wm_is_front_load
wm_is_top_load
wm_is_semi_automatic
wm_is_fully_automatic
wm_steam_wash
wm_inbuilt_heater
wm_quick_wash
wm_ss_tub
wm_child_lock
wm_has_dryer
```

Non-WM rows:

```
Fill with 0.
```

---

# 8. Capacity Interaction Features

DO NOT combine capacities into one column.

Incorrect:

```text
capacity
```

Correct:

```python
df['cap_x_tons'] = (
    df['ac_tonnage'].fillna(0)
    * (df['category'] == 'AC').astype(int)
)

df['cap_x_liters'] = (
    df['ref_capacity_liters'].fillna(0)
    * (df['category'] == 'Refrigerator').astype(int)
)

df['cap_x_kg'] = (
    df['wm_capacity_kg'].fillna(0)
    * (df['category'] == 'WashingMachine').astype(int)
)
```

Reason:

* Prevent incompatible unit mixing.
* Essential for linear models.
* Improves tree efficiency.

---

# 9. Category Encoding

Required for unified modeling.

```python
df = pd.get_dummies(
    df,
    columns=['category'],
    prefix='cat'
)
```

Produces:

```text
cat_AC
cat_Refrigerator
cat_WashingMachine
```

---

# 10. Brand Target Encoding

Use target encoding instead of one-hot encoding.

```python
from category_encoders import TargetEncoder

enc = TargetEncoder(
    cols=['brand'],
    smoothing=10
)

X_train['brand_encoded'] = (
    enc.fit_transform(
        X_train['brand'],
        y_train
    )
)

X_test['brand_encoded'] = (
    enc.transform(X_test['brand'])
)
```

Guidelines:

* Use smoothing=10.
* Perform out-of-fold encoding.
* Avoid target leakage.

---

# 11. Interaction Features

## Star Rating × User Rating

```python
df['star_x_rating'] = (
    df['star_rating'].fillna(0)
    * df['rating']
)
```

Captures:

* Energy efficiency
* Customer satisfaction

simultaneously.

---

# 12. Target Transformation

Train on log-transformed price.

```python
df['log_price'] = np.log1p(
    df['price']
)
```

Prediction inversion:

```python
pred_price = np.expm1(pred)
```

Benefits:

* Reduces skewness.
* Stabilizes variance.
* Improves relative error.

---

# 13. Validation Features (EDA Only)

Never use for training.

AC:

```python
price_per_ton
```

Refrigerator:

```python
price_per_liter
```

Washing Machine:

```python
price_per_kg
```

Purpose:

* Sanity checks.
* Recommendation engine.
* Value analysis.

---

# 14. Drop Columns Before Modeling

Remove after extraction:

```text
name
price_raw
rating_raw
features
model_year
```

For unified models also remove:

```text
ac_tonnage
ref_capacity_liters
wm_capacity_kg
```

Use interaction versions instead.

---

# 15. Final Unified Feature Set

## Universal

```text
cat_AC
cat_Refrigerator
cat_WashingMachine
brand_encoded
rating
star_rating
star_x_rating
smart_score
has_inverter
is_recent_model
year_is_imputed
```

---

## Capacity

```text
cap_x_tons
cap_x_liters
cap_x_kg
```

---

## AC

```text
ac_is_split
ac_is_window
ac_is_tower
ac_4way_swing
ac_auto_clean
ac_pm25_filter
ac_copper_condenser
ac_hot_cold
```

---

## Refrigerator

```text
ref_is_single_door
ref_is_double_door
ref_is_french_door
ref_is_side_by_side
ref_frost_free
ref_convertible
ref_child_lock
```

---

## Washing Machine

```text
wm_is_front_load
wm_is_top_load
wm_is_semi_automatic
wm_steam_wash
wm_inbuilt_heater
wm_quick_wash
wm_ss_tub
wm_has_dryer
```

---

# Target Variable

```text
TARGET = log_price
```

---

# 16. Evaluation Strategy

Do NOT rely solely on blended metrics.

Evaluate separately for:

* AC
* Refrigerators
* Washing Machines

Recommended metrics:

```text
R²
RMSE
MAE
MAPE
```

MAPE is especially important because:

* ₹3,000 error on ₹20,000 product = 15%
* ₹3,000 error on ₹150,000 product = 2%

Blended RMSE hides these differences.

---

# Final Principle

Build one unified, leakage-free feature engineering pipeline that supports:

Linear → Ridge → Lasso → Tree → Random Forest → XGBoost → SHAP → Streamlit Deployment → Recommendation System.
