import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import shap
import ee

st.set_page_config(
    page_title="Bengaluru UHI Predictor",
    page_icon="🌡️",
    layout="wide"
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .zone-high   { background:#fff1f0; border:1.5px solid #ff4d4f; border-radius:8px;
                   padding:0.6rem 1.2rem; color:#a8071a; font-weight:600; font-size:1.1rem; display:inline-block; }
    .zone-medium { background:#fffbe6; border:1.5px solid #faad14; border-radius:8px;
                   padding:0.6rem 1.2rem; color:#874d00; font-weight:600; font-size:1.1rem; display:inline-block; }
    .zone-low    { background:#f6ffed; border:1.5px solid #52c41a; border-radius:8px;
                   padding:0.6rem 1.2rem; color:#135200; font-weight:600; font-size:1.1rem; display:inline-block; }
    .idx-card    { background:#f0f2f6; border-radius:8px; padding:0.5rem 0.8rem;
                   text-align:center; margin-bottom:0.3rem; }
    .idx-val     { font-size:1.3rem; font-weight:600; color:#1f1f1f; }
    .idx-lbl     { font-size:0.75rem; color:#666; }
</style>
""", unsafe_allow_html=True)

# ── Load artifacts ────────────────────────────────────────────────────────────

def _normalize_xgb_base_score(model):
    if hasattr(model, "get_params") and hasattr(model, "set_params"):
        base_score = model.get_params().get("base_score")
        if isinstance(base_score, str):
            normalized = base_score.strip().strip("[]")
            try:
                model.set_params(base_score=float(normalized))
            except ValueError:
                pass
    return model

@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load("best_model.pkl")
    except ValueError as exc:
        message = str(exc)
        if "could not convert string to float" in message and "base_score" in message:
            st.error("Model load failed due to xgboost base_score format mismatch (e.g., '[3.512663E-2]').\n"
                     "Please use xgboost==1.7.6 or regenerate model with a numeric base_score.")
            raise
        raise

    model = _normalize_xgb_base_score(model)
    scaler = joblib.load("scaler.pkl")
    df = pd.read_csv("uhi_data.csv")
    with open("features.json") as f:
        features = json.load(f)
    return model, scaler, df, features

@st.cache_resource
def init_gee():
    ee.Initialize(project='vocal-affinity-481906-e8')

model, scaler, df, FEATURES = load_artifacts()
init_gee()

# ── Known Bengaluru locations ─────────────────────────────────────────────────
LOCATIONS = {
    "Whitefield":        (12.97, 77.75),
    "Electronic City":   (12.84, 77.67),
    "Marathahalli":      (12.96, 77.70),
    "Hebbal":            (13.04, 77.59),
    "Cubbon Park":       (12.98, 77.59),
    "Lalbagh":           (12.95, 77.58),
    "Yelahanka":         (13.10, 77.60),
    "Bannerghatta":      (12.86, 77.58),
    "Koramangala":       (12.93, 77.62),
    "Rajajinagar":       (12.99, 77.55),
    "KR Puram":          (13.00, 77.68),
    "Hennur":            (13.06, 77.64),
    "Sarjapur":          (12.86, 77.72),
    "HSR Layout":        (12.91, 77.64),
    "Nagarbhavi":        (12.97, 77.51),
    "Custom location":   (13.00, 77.60),
}

# ── GEE fetch function ────────────────────────────────────────────────────────
def fetch_real_indices(lat, lon):
    point = ee.Geometry.Point([lon, lat])
    dataset = (
        ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
        .filterBounds(point)
        .filterDate('2022-03-01', '2022-05-31')
        .filter(ee.Filter.lt('CLOUD_COVER', 10))
        .median()
    )
    ndvi  = dataset.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')
    ndbi  = dataset.normalizedDifference(['SR_B6', 'SR_B5']).rename('NDBI')
    ndwi  = dataset.normalizedDifference(['SR_B3', 'SR_B5']).rename('NDWI')
    mndwi = dataset.normalizedDifference(['SR_B3', 'SR_B6']).rename('MNDWI')
    bsi   = dataset.expression(
        '((SWIR1+RED)-(NIR+BLUE))/((SWIR1+RED)+(NIR+BLUE))',
        {'SWIR1': dataset.select('SR_B6'), 'RED': dataset.select('SR_B4'),
         'NIR':   dataset.select('SR_B5'), 'BLUE': dataset.select('SR_B2')}
    ).rename('BSI')
    lst_img = dataset.select('ST_B10').rename('LST')

    combined = ndvi.addBands([ndbi, ndwi, mndwi, bsi, lst_img])
    values = combined.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=point.buffer(150),
        scale=30
    ).getInfo()
    return values

def uhi_zone(intensity):
    if intensity >= 2.0:   return "High",   "zone-high",   "🔴"
    elif intensity >= 0.5: return "Medium",  "zone-medium", "🟡"
    else:                  return "Low",     "zone-low",    "🟢"

def predict_from_indices(indices, lat, lon):
    inp = {
        'NDVI':  indices.get('NDVI',  0),
        'NDBI':  indices.get('NDBI',  0),
        'NDWI':  indices.get('NDWI',  0),
        'MNDWI': indices.get('MNDWI', 0),
        'BSI':   indices.get('BSI',   0),
        'lat':   lat,
        'lon':   lon,
    }
    row = pd.DataFrame([inp])[FEATURES]
    row_sc = scaler.transform(row)
    return float(model.predict(row_sc)[0]), inp

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🌡️ Bengaluru Urban Heat Island Predictor")
st.markdown("Real satellite index values fetched from **Google Earth Engine** → XGBoost prediction · R²=0.405")
st.divider()

# ── Top metrics ───────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total training samples", f"{len(df):,}")
c2.metric("Avg LST (dataset)",      f"{df['LST_C'].mean():.1f} °C")
c3.metric("Max UHI intensity",      f"+{df['UHI_intensity'].max():.1f} °C")
c4.metric("Model R²",               "0.405")
st.divider()

# ── Main layout ───────────────────────────────────────────────────────────────
left, right = st.columns([1, 1.6], gap="large")

with left:
    st.subheader("Select a location")
    st.caption("Pick a known area or enter custom coordinates. Real spectral index values are fetched live from GEE.")

    area = st.selectbox("Area", list(LOCATIONS.keys()), index=0)
    default_lat, default_lon = LOCATIONS[area]

    if area == "Custom location":
        col_a, col_b = st.columns(2)
        with col_a:
            lat = st.number_input("Latitude",  min_value=12.82, max_value=13.18,
                                  value=default_lat, step=0.01, format="%.4f")
        with col_b:
            lon = st.number_input("Longitude", min_value=77.45, max_value=77.78,
                                  value=default_lon, step=0.01, format="%.4f")
    else:
        lat, lon = default_lat, default_lon
        st.markdown(f"📍 **{area}** — lat `{lat}`, lon `{lon}`")

    st.markdown("")
    predict_btn = st.button("🛰️ Fetch from GEE & Predict", use_container_width=True, type="primary")

    if predict_btn:
        with st.spinner(f"Fetching real Landsat data for {area}..."):
            try:
                indices = fetch_real_indices(lat, lon)
            except Exception as e:
                st.error(f"GEE fetch failed: {e}")
                st.stop()

        intensity, inp = predict_from_indices(indices, lat, lon)
        zone_name, zone_cls, zone_icon = uhi_zone(intensity)
        lst_raw = indices.get('LST', None)
        lst_c   = (lst_raw * 0.00341802 + 149.0 - 273.15) if lst_raw else df['LST_C'].mean() + intensity

        st.markdown("---")
        st.markdown("**Real index values from GEE (Landsat, Mar–May 2022)**")

        # Show actual fetched index values
        idx_cols = st.columns(5)
        idx_names = ['NDVI','NDBI','NDWI','MNDWI','BSI']
        idx_colors = {
            'NDVI':'#52c41a','NDBI':'#ff4d4f',
            'NDWI':'#1890ff','MNDWI':'#13c2c2','BSI':'#fa8c16'
        }
        for col, name in zip(idx_cols, idx_names):
            val = indices.get(name, 0)
            with col:
                st.markdown(
                    f'<div class="idx-card">'
                    f'<div class="idx-val" style="color:{idx_colors[name]}">{val:.3f}</div>'
                    f'<div class="idx-lbl">{name}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

        st.markdown("")
        r1, r2 = st.columns(2)
        r1.metric("UHI intensity",  f"{intensity:+.2f} °C")
        r2.metric("Surface temp",   f"{lst_c:.1f} °C")

        st.markdown(
            f'<div class="{zone_cls}">{zone_icon} Heat zone: {zone_name}</div>',
            unsafe_allow_html=True
        )
        st.markdown("")

        if zone_name == "High":
            st.warning(f"**{area}** is a high heat zone. Dense built-up surfaces with low vegetation are the primary driver.")
        elif zone_name == "Medium":
            st.info(f"**{area}** has moderate UHI intensity — mixed land use with some green cover.")
        else:
            st.success(f"**{area}** is a cool zone. Strong vegetation or water body presence detected.")

        # What drove this prediction
        st.markdown("**Index interpretation**")
        interp_cols = st.columns(2)
        with interp_cols[0]:
            st.markdown(f"- NDVI `{indices.get('NDVI',0):.3f}` → {'🌿 Good vegetation' if indices.get('NDVI',0)>0.3 else '🏙️ Low vegetation'}")
            st.markdown(f"- NDBI `{indices.get('NDBI',0):.3f}` → {'🏢 High built-up' if indices.get('NDBI',0)>0.1 else '🌱 Low built-up'}")
            st.markdown(f"- BSI  `{indices.get('BSI',0):.3f}`  → {'🏜️ Bare/impervious' if indices.get('BSI',0)>0.1 else '🌾 Soil covered'}")
        with interp_cols[1]:
            st.markdown(f"- NDWI  `{indices.get('NDWI',0):.3f}`  → {'💧 Water present' if indices.get('NDWI',0)>0 else '🔆 Dry surface'}")
            st.markdown(f"- MNDWI `{indices.get('MNDWI',0):.3f}` → {'💧 High moisture' if indices.get('MNDWI',0)>0 else '🔆 Low moisture'}")

with right:
    tab1, tab2, tab3 = st.tabs(["🗺️ UHI Grid Map", "📊 Feature Importance", "📈 Data Explorer"])

    with tab1:
        from streamlit_folium import st_folium
        import folium
        from folium.plugins import HeatMap

        st.caption("Real Bengaluru map with UHI intensity heatmap overlay")

    col_map1, col_map2 = st.columns([2, 1])
    with col_map2:
        map_style = st.selectbox("Map style", [
            "OpenStreetMap", "CartoDB Positron", "CartoDB DarkMatter"
        ])
        show_heatmap  = st.checkbox("UHI heatmap layer", value=True)
        show_markers  = st.checkbox("Location markers",  value=True)
        show_grid     = st.checkbox("Sample points",     value=False)
        heat_radius   = st.slider("Heatmap radius", 10, 40, 20)
        heat_blur     = st.slider("Heatmap blur",   10, 30, 15)

    tiles_map = {
        "OpenStreetMap":    "OpenStreetMap",
        "CartoDB Positron": "CartoDB positron",
        "CartoDB DarkMatter": "CartoDB dark_matter"
    }

    with col_map1:
        m = folium.Map(
            location=[13.00, 77.60],
            zoom_start=11,
            tiles=tiles_map[map_style],
            width="100%"
        )

        # UHI heatmap layer from real sampled data
        if show_heatmap:
            heat_data = []
            for _, row in df.iterrows():
                # Normalize intensity to 0-1 for folium heatmap weight
                weight = (row['UHI_intensity'] - df['UHI_intensity'].min()) / \
                         (df['UHI_intensity'].max() - df['UHI_intensity'].min())
                heat_data.append([row['lat'], row['lon'], float(weight)])

            HeatMap(
                heat_data,
                radius=heat_radius,
                blur=heat_blur,
                gradient={
                    '0.0': '#4575b4',
                    '0.4': '#ffffbf',
                    '0.7': '#fc8d59',
                    '1.0': '#d73027'
                },
                min_opacity=0.4,
                name="UHI Heatmap"
            ).add_to(m)

        # Sample point dots
        if show_grid:
            sample_df = df.sample(min(300, len(df)), random_state=42)
            for _, row in sample_df.iterrows():
                intensity = row['UHI_intensity']
                color = '#d73027' if intensity >= 2 else ('#faad14' if intensity >= 0.5 else '#52c41a')
                folium.CircleMarker(
                    location=[row['lat'], row['lon']],
                    radius=3,
                    color=color,
                    fill=True,
                    fill_opacity=0.7,
                    popup=f"UHI: {intensity:+.2f}°C | LST: {row['LST_C']:.1f}°C",
                    tooltip=f"{intensity:+.2f}°C"
                ).add_to(m)

        # Known location markers
        if show_markers:
            zone_colors = {}
            for name, (ly, lx) in LOCATIONS.items():
                if name == "Custom location":
                    continue
                # Predict UHI for this landmark using nearest training data
                near = df.copy()
                near['dist'] = ((near['lat'] - ly)**2 + (near['lon'] - lx)**2)**0.5
                nearest = near.nsmallest(5, 'dist')
                avg_intensity = nearest['UHI_intensity'].mean()
                zone_name, _, zone_icon = uhi_zone(avg_intensity)

                color = '#d73027' if zone_name == 'High' else \
                        '#faad14' if zone_name == 'Medium' else '#52c41a'

                folium.Marker(
                    location=[ly, lx],
                    popup=folium.Popup(
                        f"<b>{name}</b><br>"
                        f"UHI intensity: {avg_intensity:+.2f}°C<br>"
                        f"Zone: {zone_icon} {zone_name}",
                        max_width=200
                    ),
                    tooltip=f"{name} — {avg_intensity:+.2f}°C",
                    icon=folium.Icon(color='red' if zone_name=='High' else
                                         'orange' if zone_name=='Medium' else 'green',
                                    icon='thermometer', prefix='fa')
                ).add_to(m)

        # Legend HTML overlay
        legend_html = """
        <div style="position:fixed; bottom:30px; left:30px; z-index:1000;
                    background:white; padding:12px 16px; border-radius:8px;
                    border:1px solid #ddd; font-size:12px; font-family:sans-serif;">
            <b>UHI Intensity</b><br>
            <span style="color:#d73027">&#9632;</span> High  (&gt;+2°C)<br>
            <span style="color:#fc8d59">&#9632;</span> Medium-high<br>
            <span style="color:#ffffbf;-webkit-text-stroke:0.5px #aaa">&#9632;</span> Near mean<br>
            <span style="color:#91bfdb">&#9632;</span> Medium-low<br>
            <span style="color:#4575b4">&#9632;</span> Low (cool zone)
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))

        st_folium(m, width=700, height=520)

    with tab2:
        st.caption("SHAP values — contribution of each real feature to UHI prediction")
        with st.spinner("Computing SHAP..."):
            X_sample    = df[FEATURES].sample(min(500, len(df)), random_state=42)
            X_sample_sc = scaler.transform(X_sample)
            explainer   = shap.TreeExplainer(model)
            shap_vals   = explainer.shap_values(X_sample_sc)

        chart_type = st.radio("Chart type", ["Bar (mean |SHAP|)", "Beeswarm"], horizontal=True)
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        if chart_type == "Bar (mean |SHAP|)":
            mean_shap  = np.abs(shap_vals).mean(axis=0)
            sorted_idx = np.argsort(mean_shap)
            bar_colors = ['#d73027' if shap_vals.mean(axis=0)[i] > 0 else '#4575b4'
                          for i in sorted_idx]
            ax2.barh([FEATURES[i] for i in sorted_idx], mean_shap[sorted_idx],
                     color=bar_colors, edgecolor='none')
            ax2.set_xlabel("Mean |SHAP value|", fontsize=9)
            ax2.set_title("Feature importance (SHAP)", fontsize=11)
            ax2.tick_params(labelsize=9)
            plt.tight_layout()
            st.pyplot(fig2)
        else:
            plt.close()
            shap.summary_plot(shap_vals, X_sample, feature_names=FEATURES, show=False)
            st.pyplot(plt.gcf())
        plt.close()

        st.markdown("""
        **Reading the chart:**
        - 🔴 Red = feature increases UHI (more heat)
        - 🔵 Blue = feature decreases UHI (cooling)
        - Longer bar = stronger influence
        """)

    with tab3:
        st.caption("Distribution of UHI intensity and spatial spread across sampled pixels")
        col_a, col_b = st.columns(2)

        with col_a:
            fig3, ax3 = plt.subplots(figsize=(5, 3.5))
            ax3.hist(df['UHI_intensity'], bins=40, color='#fc8d59', edgecolor='white', lw=0.4)
            ax3.axvline(0, color='#d73027', lw=1.5, linestyle='--', label='City mean')
            ax3.axvline(2, color='#faad14', lw=1, linestyle=':', label='High threshold')
            ax3.set_xlabel("UHI intensity (°C)", fontsize=9)
            ax3.set_ylabel("Count", fontsize=9)
            ax3.set_title("UHI intensity distribution", fontsize=10)
            ax3.legend(fontsize=8)
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close()

        with col_b:
            fig4, ax4 = plt.subplots(figsize=(5, 3.5))
            colors_zone = df['UHI_intensity'].apply(
                lambda x: '#d73027' if x >= 2 else ('#faad14' if x >= 0.5 else '#52c41a')
            )
            ax4.scatter(df['lon'], df['lat'], c=colors_zone, s=3, alpha=0.5)
            ax4.set_xlabel("Longitude", fontsize=9)
            ax4.set_ylabel("Latitude",  fontsize=9)
            ax4.set_title("Sample pixels by UHI zone", fontsize=10)
            from matplotlib.lines import Line2D
            ax4.legend(handles=[
                Line2D([0],[0], marker='o', color='w', markerfacecolor='#d73027', ms=7, label='High'),
                Line2D([0],[0], marker='o', color='w', markerfacecolor='#faad14', ms=7, label='Medium'),
                Line2D([0],[0], marker='o', color='w', markerfacecolor='#52c41a', ms=7, label='Low'),
            ], fontsize=8)
            plt.tight_layout()
            st.pyplot(fig4)
            plt.close()

        show_cols = ['lat','lon','NDVI','NDBI','NDWI','MNDWI','BSI','LST_C','UHI_intensity']
        st.dataframe(df[show_cols].sample(min(200, len(df))).round(4), height=260)

st.divider()
st.caption("Bengaluru UHI · Landsat LC08 C02 T1 L2 · Mar–May 2022 · XGBoost R²=0.405 · Built with Streamlit + GEE")