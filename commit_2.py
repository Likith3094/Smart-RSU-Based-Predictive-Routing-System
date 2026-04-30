import streamlit as st
import pandas as pd
import numpy as np
import requests
import pickle
import folium
from streamlit_folium import st_folium
from datetime import datetime, timedelta

# 1. INITIAL CONFIGURATION
st.set_page_config(page_title="EV Smart RSU Reroute", layout="wide")

# --- CONFIGURATION ---
TOMTOM_KEY = "akPHWg9CrZg6SDG085lPj5Dlb0vfrUNA" # <--- PASTE YOUR KEY HERE
OSRM_URL = "http://localhost:5001/route/v1/driving/"

# --- DATA LOADING ---
@st.cache_resource
def load_assets():
    try:
        model = pickle.load(open('ev_xgboost_model.pkl', 'rb'))
        stations = pd.read_csv('stations_data.csv')
        live_feed = pd.read_csv('rsu_live_feed.csv')
        live_feed['timestamp'] = pd.to_datetime(live_feed['timestamp'])
        return model, stations, live_feed
    except Exception as e:
        st.error(f"Error loading files: {e}")
        st.stop()

model, df_stations, df_live = load_assets()

# --- 2. HELPER FUNCTIONS ---
def get_real_travel_stats(u_lat, u_lon, s_lat, s_lon, tomtom_key):
    try:
        osrm_query = f"{OSRM_URL}{u_lon},{u_lat};{s_lon},{s_lat}?overview=full&geometries=geojson"
        osrm_res = requests.get(osrm_query, timeout=5).json()
        route_data = osrm_res['routes'][0]
        base_time = route_data['duration'] / 60
        dist = route_data['distance'] / 1000
        path_geometry = route_data['geometry']['coordinates']

        tt_url = f"https://api.tomtom.com/routing/1/calculateRoute/{u_lat},{u_lon}:{s_lat},{s_lon}/json?key={TOMTOM_KEY}&traffic=true"
        tt_res = requests.get(tt_url, timeout=10).json()
        delay = tt_res['routes'][0]['summary']['trafficDelayInSeconds'] / 60
        return base_time + delay, dist, path_geometry
    except Exception as e:
        return 999, 0, None

def get_rsu_features(station_id):
    station_data = df_live[df_live['station_id'] == station_id].tail(3)
    if station_data.empty:
        return {'prev_occupied_1': 0, 'occ_change': 0, 'rolling_avg_3': 0}
    curr_occ = station_data['occupied_now'].iloc[-1]
    prev_occ = station_data['occupied_now'].iloc[-2] if len(station_data) > 1 else curr_occ
    return {'prev_occupied_1': curr_occ, 'occ_change': curr_occ - prev_occ, 'rolling_avg_3': station_data['occupied_now'].mean()}

# --- 3. UI LAYOUT ---
st.title("⚡ RSU-Enabled EV Smart Navigation")

# Sidebar for inputs
with st.sidebar:
    st.header("📍 User Parameters")
    u_lat = st.number_input("Latitude", value=28.6139, format="%.6f")
    u_lon = st.number_input("Longitude", value=77.2090, format="%.6f")
    u_soc = st.slider("Battery SoC (%)", 5, 100, 15)
    search_btn = st.button("Generate Smart Route", use_container_width=True)

# THE FIX: Persistent Storage
if 'results' not in st.session_state:
    st.session_state.results = None
if 'geometries' not in st.session_state:
    st.session_state.geometries = None

if search_btn:
    with st.spinner("Analyzing routes..."):
        df_stations['air_dist'] = np.sqrt((df_stations['latitude']-u_lat)**2 + (df_stations['longitude']-u_lon)**2)
        candidates = df_stations.nsmallest(10, 'air_dist')
        
        final_list = []
        geoms = {}

        for _, stn in candidates.iterrows():
            total_tt, road_dist, path = get_real_travel_stats(u_lat, u_lon, stn['latitude'], stn['longitude'], TOMTOM_KEY)
            rsu = get_rsu_features(stn['station_id'])
            arrival_time = datetime.now() + timedelta(minutes=total_tt)
            
            feat_row = pd.DataFrame([{
                'hour': arrival_time.hour, 'day_of_week': arrival_time.weekday(),
                'is_weekend': 1 if arrival_time.weekday() >= 5 else 0,
                'total_slots': stn['charging_points'], 'prev_occupied_1': rsu['prev_occupied_1'],
                'occ_change': rsu['occ_change'], 'rolling_avg_3': rsu['rolling_avg_3']
            }])
            
            pred_occ = model.predict(feat_row)[0]
            pred_free = max(0, stn['charging_points'] - pred_occ)
            t_wait = (30 / stn['charging_points']) if pred_free < 1 else 0
            
            score = (total_tt * 0.95) + (t_wait * 0.05) if u_soc <= 15 else total_tt + t_wait
            
            final_list.append({
                "Station ID": stn['station_id'], "Distance (km)": round(road_dist, 2),
                "Travel Time (min)": round(total_tt, 2), "Wait Time (min)": round(t_wait, 1),
                "Free Slots": round(pred_free, 1), "Score": score, "lat": stn['latitude'], "lon": stn['longitude']
            })
            geoms[stn['station_id']] = path

        st.session_state.results = pd.DataFrame(final_list).sort_values("Score").head(5)
        st.session_state.geometries = geoms

# Display Results if they exist in memory
if st.session_state.results is not None:
    rec_df = st.session_state.results
    best_choice = rec_df.iloc[0]
    
    col_table, col_map = st.columns([1, 1])
    
    with col_table:
        st.subheader("🎯 Recommended Stations")
        st.dataframe(rec_df.drop(columns=['Score', 'lat', 'lon']), use_container_width=True)
    
    with col_map:
        st.subheader(f"🗺️ Path to {best_choice['Station ID']}")
        m = folium.Map(location=[u_lat, u_lon], zoom_start=13, tiles="CartoDB positron")
        
        path_coords = st.session_state.geometries.get(best_choice['Station ID'])
        if path_coords:
            folium_path = [[p[1], p[0]] for p in path_coords] # Fixed index order [Lat, Lon]
            folium.PolyLine(folium_path, color="red", weight=5).add_to(m)

        folium.Marker([u_lat, u_lon], popup="You", icon=folium.Icon(color='blue')).add_to(m)
        for _, r in rec_df.iterrows():
            folium.Marker([r['lat'], r['lon']], popup=r['Station ID'], icon=folium.Icon(color='green', icon='bolt', prefix='fa')).add_to(m)
        
        st_folium(m, width=700, height=500, key="main_map")
