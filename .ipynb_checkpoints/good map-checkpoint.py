import streamlit as st
import pandas as pd
import numpy as np
import requests
import pickle
import folium
from streamlit_folium import st_folium
from datetime import datetime, timedelta

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="EV Smart Reroute", layout="wide")
TOMTOM_KEY = "akPHWg9CrZg6SDG085lPj5Dlb0vfrUNA" 
OSRM_URL = "http://localhost:5001/route/v1/driving/"

@st.cache_resource
def load_assets():
    model = pickle.load(open('ev_xgboost_model.pkl', 'rb'))
    stations = pd.read_csv('processed_stations.csv')
    live_feed = pd.read_csv('rsu_live_feed.csv')
    live_feed['timestamp'] = pd.to_datetime(live_feed['timestamp'])
    return model, stations, live_feed

model, df_stations, df_live = load_assets()

# --- 2. HELPER FUNCTIONS ---
def get_coords_from_address(address, tomtom_key):
    url = f"https://tomtom.com{address}.json?key={tomtom_key}&lat=28.6139&lon=77.2090&radius=50000&limit=1"
    try:
        res = requests.get(url).json()
        if 'results' in res and len(res['results']) > 0:
            pos = res['results'][0]['position']
            return pos['lat'], pos['lon']
    except: return None, None

def get_real_travel_stats(u_lat, u_lon, s_lat, s_lon, tomtom_key):
    try:
        osrm_query = f"{OSRM_URL}{u_lon},{u_lat};{s_lon},{s_lat}?overview=full&geometries=geojson&radiuses=1000;1000"
        osrm_res = requests.get(osrm_query, timeout=5).json()
        route = osrm_res['routes'][0]
        
        tt_url = f"https://tomtom.com{u_lat},{u_lon}:{s_lat},{s_lon}/json?key={tomtom_key}&traffic=true"
        tt_res = requests.get(tt_url, timeout=10).json()
        delay = tt_res['routes'][0]['summary']['trafficDelayInSeconds'] / 60
        return (route['duration']/60) + delay, route['distance']/1000, route['geometry']['coordinates']
    except: return 999, 0, None

# --- 3. SESSION STATE ---
if 'u_lat' not in st.session_state: st.session_state.u_lat = 28.6129
if 'u_lon' not in st.session_state: st.session_state.u_lon = 77.2295
if 'results' not in st.session_state: st.session_state.results = None
if 'geometries' not in st.session_state: st.session_state.geometries = None
if 'selected_station' not in st.session_state: st.session_state.selected_station = None

# --- 4. SIDEBAR ---
with st.sidebar:
    st.header("📍 Location")
    address_query = st.text_input("Enter Address (e.g. Champa Gali)", "")
    if st.button("Search & Update"):
        lat, lon = get_coords_from_address(address_query, TOMTOM_KEY)
        if lat:
            st.session_state.u_lat, st.session_state.u_lon = lat, lon
            st.success("Location Found!")
        else: st.error("Place not found in Delhi.")

    u_lat = st.number_input("Lat", value=st.session_state.u_lat, format="%.6f")
    u_lon = st.number_input("Lon", value=st.session_state.u_lon, format="%.6f")
    u_soc = st.slider("Battery SoC (%)", 5, 100, 20)
    
    if st.button("Generate Smart Route", use_container_width=True):
        with st.spinner("Analyzing..."):
            df_stations['air_dist'] = np.sqrt((df_stations['latitude']-u_lat)**2 + (df_stations['longitude']-u_lon)**2)
            candidates = df_stations.nsmallest(10, 'air_dist')
            final_list = []; geoms = {}
            for _, stn in candidates.iterrows():
                tt, dist, path = get_real_travel_stats(u_lat, u_lon, stn['latitude'], stn['longitude'], TOMTOM_KEY)
                
                # --- RSU DATA & PREDICTION ---
                s_data = df_live[df_live['station_id'] == stn['station_id']].tail(3)
                curr_occ = s_data['occupied_now'].iloc[-1] if not s_data.empty else 0
                prev_occ = s_data['occupied_now'].iloc[-2] if len(s_data) > 1 else curr_occ
                
                feat = pd.DataFrame([{'hour': datetime.now().hour, 'day_of_week': datetime.now().weekday(), 'is_weekend': 0, 
                                      'total_slots': stn['charging_points'], 'prev_occupied_1': curr_occ, 
                                      'occ_change': curr_occ - prev_occ, 'rolling_avg_3': s_data['occupied_now'].mean() if not s_data.empty else 0}])
                
                # FIX: Added [0] here
                pred_occ = model.predict(feat)[0]
                pred_free = max(0, stn['charging_points'] - pred_occ)
                t_wait = (30 / stn['charging_points']) if pred_free < 1 else 0
                score = (tt * 0.95) + (t_wait * 0.05) if u_soc <= 15 else tt + t_wait
                
                final_list.append({"Station ID": stn['station_id'], "Distance (km)": round(dist, 2), 
                                   "Travel Time (min)": round(tt, 2), "Wait Time (min)": round(t_wait, 1), 
                                   "Free Slots": round(pred_free, 1), "Score": score, 
                                   "lat": stn['latitude'], "lon": stn['longitude']})
                geoms[stn['station_id']] = path
            
            st.session_state.results = pd.DataFrame(final_list).sort_values("Score").head(5)
            st.session_state.geometries = geoms
            st.session_state.selected_station = st.session_state.results.iloc[0]['Station ID']

# --- 5. MAIN DISPLAY ---
if st.session_state.results is not None:
    res_df = st.session_state.results
    col1, col2 = st.columns([1, 1])
    with col1:
        st.dataframe(res_df.drop(columns=['Score', 'lat', 'lon']), hide_index=True)
    with col2:
        m = folium.Map(location=[u_lat, u_lon], zoom_start=14)
        active_id = st.session_state.selected_station
        path_coords = st.session_state.geometries.get(active_id)
        if path_coords:
            folium.PolyLine([[p[1], p[0]] for p in path_coords], color="red", weight=6).add_to(m)
        st_folium(m, width=600, height=400, key="nav_map")
