# Bengaluru UHI Predictor

Interactive Streamlit app for predicting Urban Heat Island (UHI) intensity in Bengaluru using Landsat-derived spectral indices and a pre-trained XGBoost model.

## 🚀 Features

- Choose a known city location or input custom coordinates.
- Fetch real-time Landsat index values from Google Earth Engine (NDVI, NDBI, NDWI, MNDWI, BSI, LST).
- Predict UHI intensity using pre-trained model.
- Display UHI zone badge (Low/Medium/High).
- Built map heatmap layer and sample points with folium and streamlit-folium.
- Explainable model output via SHAP feature importance charts.

## 📦 Files

- `app.py` - main Streamlit app
- `features.json` - model feature order
- `uhi_data.csv` - sample dataset used for metrics/heatmap
- `best_model.pkl` - trained model artifact
- `scaler.pkl` - standard scaler artifact
- `UHI1.ipynb` - exploratory notebook

## 🛠️ Setup

1. Clone repo:

```bash
git clone https://github.com/Rakesh060404/UHI.git
cd UHI
```

2. Create virtual environment and install dependencies:

```bash
python -m venv venv
venv\Scripts\activate    # Windows
# or `source venv/bin/activate` on macOS/Linux
pip install --upgrade pip
pip install -r requirements.txt
# if requirements.txt is missing:
pip install streamlit pandas numpy scikit-learn joblib folium streamlit-folium shap earthengine-api matplotlib
```

3. Authenticate Google Earth Engine (required for live fetch):

```bash
earthengine authenticate
```

4. Run app:

```bash
streamlit run app.py
```

## ℹ️ Notes

- `best_model.pkl` and `scaler.pkl` should be present in repo for prediction.
- If Earth Engine access is unavailable, edit `fetch_real_indices` to use local sample indices.
- `st.divider()` requires Streamlit >=1.18.
- Large artifacts (`.pkl`) may be better stored as releases or via Git LFS.

## 🪪 Usage

- Pick location `Whitefield`, `Hebbal`, etc., or `Custom location`.
- Click `Fetch from GEE & Predict`.
- Review UHI intensity, zone, and map.

## 📌 Author

- Rakesh ([@Rakesh060404](https://github.com/Rakesh060404))

---

## Troubleshooting

- If you see `ModuleNotFoundError: No module named 'ee'`:
  `pip install earthengine-api`
- If you see `Earth Engine` authorization error, rerun `earthengine authenticate`.
- For Windows CRLF warnings, run `git config --global core.autocrlf true`.
