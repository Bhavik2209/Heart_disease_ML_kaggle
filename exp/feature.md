| Feature                 | Raw Dtype     | Semantic Type       | Clinical Nature            | Modeling Treatment                 |
| ----------------------- | ------------- | ------------------- | -------------------------- | ---------------------------------- |
| id                      | int64         | Identifier          | No clinical meaning        | Drop                               |
| Age                     | int64         | Continuous          | Risk increases with age    | Numeric (possibly nonlinear)       |
| Sex                     | int64 (0/1)   | Binary Categorical  | Biological risk difference | Binary (no scaling required)       |
| Chest pain type         | int64 (1–4)   | Nominal Categorical | Pain classification        | OneHotEncode                       |
| BP                      | int64         | Continuous          | Hypertension risk          | Numeric (+ possible bins)          |
| Cholesterol             | int64         | Continuous          | Lipid risk                 | Numeric (+ possible log transform) |
| FBS over 120            | int64 (0/1)   | Binary              | Diabetes indicator         | Binary                             |
| EKG results             | int64 (0/1/2) | Nominal Categorical | ECG pattern                | OneHotEncode                       |
| Max HR                  | int64         | Continuous          | Cardiac performance        | Numeric                            |
| Exercise angina         | int64 (0/1)   | Binary              | Exercise-induced ischemia  | Binary                             |
| ST depression           | float64       | Continuous          | Ischemia severity          | Numeric (possibly skewed)          |
| Slope of ST             | int64 (0/1/2) | Nominal Categorical | ST behavior                | OneHotEncode                       |
| Number of vessels fluro | int64 (0–3)   | Ordinal             | Blockage severity          | Keep numeric                       |
| Thallium                | int64 (3/6/7) | Nominal Categorical | Stress test result         | OneHotEncode                       |
| Heart Disease           | object        | Target              | Disease presence           | Map to 0/1                         |
