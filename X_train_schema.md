\# X\_TRAIN\_SCHEMA.md



\## Purpose



This document describes the exact schema of `X\_train.parquet`, the feature matrix used to train the final multi-category appliance pricing model.



It serves as the translation layer between:



\* Streamlit UI

\* Agentic systems (Claude/OpenAI)

\* The preprocessing pipeline

\* The trained model



Agents should always consult this file before generating UI components, validating inputs, creating inference payloads, or explaining predictions.



\---



\# Dataset Overview



\## Shape



\* Rows: \*\*2,296\*\*

\* Columns: \*\*65\*\*



This dataset contains engineered features from multiple appliance categories.



\---



\# Categories



\## category



\### Description



Appliance type.



\### dtype



```python

object

```



\### Allowed Values



```python

AC

Refrigerator

WashingMachine

```



\### UI Mapping



| User Selection  | Pipeline Value |

| --------------- | -------------- |

| Air Conditioner | AC             |

| Refrigerator    | Refrigerator   |

| Washing Machine | WashingMachine |



\---



\# General Features



\## rating



\### Description



Average customer rating extracted from Smartprix.



\### dtype



```python

float64

```



\### Unique Values



Approximately 25 unique values.



\### Examples



```python

4.05

4.50

4.45

4.55

```



\### UI Label



Customer Rating



\---



\## brand\_name



\### Description



Brand of appliance.



\### dtype



```python

object

```



\### Unique Brands



57 brands.



\### Examples



```python

LG

Samsung

Whirlpool

Panasonic

Haier

Godrej

Voltas

Daikin

IFB

Bosch

```



\### UI Label



Brand



\---



\## star\_rating



\### Description



Energy efficiency rating.



\### dtype



```python

float64

```



\### Unique Values



```python

0

2

3

4

5

```



\### Interpretation



0 generally means:



> Rating unavailable or not applicable.



\### UI Label



Energy Rating



\---



\## n\_features



\### Description



Total number of features available in the appliance.



\### dtype



```python

int64

```



\### Unique Values



16 distinct values.



\### Interpretation



Higher values generally indicate more premium products.



\---



\# Smart Feature Indicators



All binary indicators use:



```python

0 = No

1 = Yes

```



\---



\## has\_star\_rating



Whether energy rating exists.



\---



\## has\_inverter



Whether inverter technology exists.



\---



\## has\_wifi



Whether Wi-Fi connectivity exists.



\---



\## has\_voice\_control



Whether voice assistant support exists.



\---



\## has\_app\_control



Whether mobile app control exists.



\---



\# smart\_connectivity\_score



\## Description



Combined smart feature score.



\### Formula



```python

has\_wifi

\+ has\_voice\_control

\+ has\_app\_control

```



\### Values



| Score | Meaning               |

| ----- | --------------------- |

| 0     | No smart features     |

| 1     | One smart feature     |

| 2     | Two smart features    |

| 3     | Fully smart appliance |



\### UI Label



Smart Features Score



\---



\# AC Features



Binary indicators.



```python

0 = Absent

1 = Present

```



\## Columns



```python

ac\_split

ac\_window

ac\_pm25\_filter

ac\_hepa\_filter

ac\_auto\_clean

ac\_hot\_and\_cold

ac\_copper\_condenser

ac\_Dehumidification

ac\_Turbo Mode

ac\_Self Diagnosis

```



\## UI Mapping Examples



| User Input       | Pipeline Feature    |

| ---------------- | ------------------- |

| PM 2.5 Filter    | ac\_pm25\_filter      |

| Turbo Mode       | ac\_Turbo Mode       |

| Self Diagnosis   | ac\_Self Diagnosis   |

| Copper Condenser | ac\_copper\_condenser |

| Auto Clean       | ac\_auto\_clean       |



\---



\# Refrigerator Features



Binary indicators.



```python

0 = No

1 = Yes

```



\## Columns



```python

ref\_single\_door

ref\_multi\_door

ref\_chest\_freezer

ref\_side\_door

ref\_french\_door

ref\_double\_door

ref\_triple\_door

ref\_frost\_free

ref\_convertible

ref\_door\_alarm

ref\_door\_lock

ref\_dispenser

ref\_door\_display

ref\_mini

```



\## UI Mapping Examples



| User Input  | Pipeline Feature |

| ----------- | ---------------- |

| Frost Free  | ref\_frost\_free   |

| Convertible | ref\_convertible  |

| Double Door | ref\_double\_door  |

| French Door | ref\_french\_door  |



\---



\# Washing Machine Features



Binary indicators.



```python

0 = No

1 = Yes

```



\## Columns



```python

wm\_fully\_automatic

wm\_semi\_automatic

wm\_with\_dryer

wm\_washer\_only

wm\_dryer\_only

wm\_front\_load

wm\_top\_load

wm\_inbuilt\_heater

wm\_quick\_wash

wm\_ss\_tub

wm\_child\_lock

wm\_shock\_proof

wm\_display

```



\## UI Mapping Examples



| User Input          | Pipeline Feature |

| ------------------- | ---------------- |

| Front Load          | wm\_front\_load    |

| Top Load            | wm\_top\_load      |

| Child Lock          | wm\_child\_lock    |

| Quick Wash          | wm\_quick\_wash    |

| Stainless Steel Tub | wm\_ss\_tub        |



\---



\# Engineered Features



These features are generated during preprocessing.



\---



\## capacity\_sq



Squared capacity term.



\### Purpose



Captures nonlinear capacity effects.



\---



\## n\_features\_sq



Squared feature count.



\### Purpose



Captures premium saturation effects.



\---



\## rating\_sq



Squared rating.



\### Purpose



Captures nonlinear rating relationships.



\---



\## capacity\_n\_features



\### Formula



```python

capacity × n\_features

```



\### Purpose



Measures interaction between product size and feature richness.



\---



\## capacity\_rating



\### Formula



```python

capacity × rating

```



\### Purpose



Captures joint influence of capacity and customer satisfaction.



\---



\## rating\_n\_features



\### Formula



```python

rating × n\_features

```



\### Purpose



Measures interaction between customer perception and features.



\---



\## smart\_intensity



\### Values



```python

0–3

```



\### Description



Represents strength of smart ecosystem.



\---



\## ac\_premium\_features



\### Description



Aggregate premium AC score.



\### Interpretation



Higher values indicate richer AC functionality.



\---



\## ref\_door\_complexity



\### Description



Refrigerator sophistication score.



\### Interpretation



Distinguishes basic and advanced door configurations.



\---



\## wm\_tech\_level



\### Description



Washing machine technology score.



\### Interpretation



Higher values indicate more advanced washing machines.



\---



\# Statistical Features



\## features\_above\_avg



\### Description



Difference between appliance feature count and category average.



\### dtype



```python

float64

```



\---



\## rating\_above\_avg



\### Description



Difference between product rating and category average.



\### dtype



```python

float64

```



\---



\## capacity\_above\_avg



\### Description



Difference between appliance capacity and category average.



\### dtype



```python

float64

```



\---



\## brand\_frequency



\### Description



Frequency of the brand in training data.



\### dtype



```python

int64

```



\### Interpretation



Acts as a proxy for brand popularity and market presence.



\---



\# Capacity Features



Separate capacity columns prevent unit mismatch across categories.



\---



\## capacity\_ac\_tons



\### Description



Air conditioner capacity.



\### Examples



```python

0.8

1.0

1.2

1.5

2.0

```



\### Non-AC Products



```python

0

```



\---



\## capacity\_ref\_liters



\### Description



Refrigerator capacity.



\### Examples



```python

190

294

340

420

```



\### Non-Refrigerator Products



```python

0

```



\---



\## capacity\_wm\_kg



\### Description



Washing machine capacity.



\### Examples



```python

6.0

6.5

7.0

8.0

10.0

```



\### Non-Washing Machines



```python

0

```



\---



\# Agent Instructions



When interacting with users:



1\. Use user-friendly terminology.

2\. Translate user inputs into exact pipeline feature names.

3\. Validate values against the allowed schema.

4\. Never invent new feature names.

5\. Populate irrelevant category features with their default values.

6\. Ensure generated inference payloads exactly match training schema.



\---



\# Example Translations



\## User



> Does this AC have Turbo Mode?



Pipeline



```python

ac\_Turbo Mode = 1

```



\---



\## User



> Is it Wi-Fi enabled?



Pipeline



```python

has\_wifi = 1

```



\---



\## User



> It is a front-load washing machine.



Pipeline



```python

wm\_front\_load = 1

wm\_top\_load = 0

```



\---



\## User



> It is a double-door refrigerator.



Pipeline



```python

ref\_double\_door = 1

```



\---



\# Importance of this File



This document is the single source of truth for:



\* Streamlit UI generation,

\* Agent reasoning,

\* Input validation,

\* Payload construction,

\* Prediction explanations,

\* Human-to-pipeline terminology mapping.



Any future changes to preprocessing or schema must be reflected in this file immediately to keep the application and agents synchronized.



