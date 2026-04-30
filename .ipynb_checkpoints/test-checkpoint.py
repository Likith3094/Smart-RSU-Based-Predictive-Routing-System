import streamlit as st
import pandas as pd
import numpy as np
import requests
import pickle
import folium
import urllib.parse
from streamlit_folium import st_folium
from streamlit_geolocation import streamlit_geolocation
from datetime import datetime, timedelta

# --- 1. INITIAL CONFIGURATION ---
st.set_page_config(page_title="EV Smart Navigation", layout="wide")

TOMTOM_KEY = "akPHWg9CrZg6SDG085lPj5Dlb0vfrUNA" 
OSRM_URL = "http://localhost:5001/route/v1/driving/"

# --- 2. ASSET LOADING ---
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

# --- 3. HELPER FUNCTIONS ---
def get_coords_from_address(address, tomtom_key):
    """Robust Search with Fuzzy Logic and Global Fallback"""
    try:
        query = urllib.parse.quote(address)
        # Try Delhi Bias first
        url = f"https://tomtom.com{query}.json?key={tomtom_key}&lat=28.6139&lon=77.2090&radius=50000&limit=1"
        res = requests.get(url, timeout=5).json()
        
        if res.get('results'):
            pos = res['results'][0]['position']
            return pos['lat'], pos['lon']
        
        # Fallback to Global if Delhi bias fails (for places like Greater Kailash)
        fallback_url = f"https://tomtom.com{query}.json?key={tomtom_key}&limit=1"
        res_fb = requests.get(fallback_url, timeout=5).json()
        if res_fb.get('results'):
            pos = res_fb['results'][0]['position']
            return pos['lat'], pos['lon']
    except: return None, None
    return None, None

def get_real_travel_stats(u_lat, u_lon, s_lat, s_lon, tomtom_key):
    try:
        # OSRM Call with snapping
        osrm_query = f"{OSRM_URL}{u_lon},{u_lat};{s_lon},{s_lat}?overview=full&geometries=geojson&radiuses=1000;1000"
        osrm_res = requests.get(osrm_query, timeout=5).json()
        if osrm_res.get('code') != 'Ok': return 999, 0, None
        
        route = osrm_res['routes'][0]
        # Geometry: Correctly swap [Lon, Lat] to [Lat, Lon] for Folium
        path_coords = [[p[1], p[0]] for p in route['geometry']['coordinates']]

        # TomTom Traffic API
        tt_url = f"https://tomtom.com{u_lat},{u_lon}:{s_lat},{s_lon}/json?key={tomtom_key}&traffic=true"
        tt_res = requests.get(tt_url, timeout=10).json()
        delay = tt_res['routes'][0]['summary']['trafficDelayInSeconds'] / 60
        
        return (route['duration']/60) + delay, route['distance']/1000, path_coords
    except: return 999, 0, None

def get_rsu_features(station_id):
    stn_data = df_live[df_live['station_id'] == station_id].tail(3)
    if stn_data.empty: return {'prev_occupied_1': 0, 'occ_change': 0, 'rolling_avg_3': 0}
    curr = stn_data['occupied_now'].iloc[-1]
    prev = stn_data['occupied_now'].iloc[-2] if len(stn_data) > 1 else curr
    return {'prev_occupied_1': curr, 'occ_change': curr - prev, 'rolling_avg_3': stn_data['occupied_now'].mean()}

# --- 4. SESSION STATE MANAGEMENT ---
if 'u_lat' not in st.session_state: st.session_state.u_lat = 28.6139
if 'u_lon' not in st.session_state: st.session_state.u_lon = 77.2090
if 'results' not in st.session_state: st.session_state.results = None
if 'geometries' not in st.session_state: st.session_state.geometries = None
if 'selected_station' not in st.session_state: st.session_state.selected_station = None

# --- 5. SIDEBAR ---
with st.sidebar:
    st.header("📍 Set Your Location")
    
    # 5a. Address Search
    search_input = st.text_input("Enter Address (e.g., Greater Kailash)", "")
    if st.button("🔍 Search & Update"):
        if search_input:
            lat, lon = get_coords_from_address(search_input, TOMTOM_KEY)
            if lat:
                st.session_state.u_lat, st.session_state.u_lon = lat, lon
                st.success("Location Updated!")
                st.rerun()
            else: st.error("Place not found. Try adding 'Delhi'.")

    # 5b. Live GPS
    st.write("---")
    st.write("Use Current GPS:")
    location = streamlit_geolocation()
    if location.get('latitude'):
        st.session_state.u_lat = location['latitude']
        st.session_state.u_lon = location['longitude']
        st.rerun()
    
    # 5c. Manual Sync
    st.write("---")
    u_lat = st.number_input("Lat", value=st.session_state.u_lat, format="%.6f")
    u_lon = st.number_input("Lon", value=st.session_state.u_lon, format="%.6f")
    st.session_state.u_lat, st.session_state.u_lon = u_lat, u_lon

    u_soc = st.slider("Battery SoC (%)", 5, 100, 20)
    
    if st.button("🚀 Find Optimal Station", use_container_width=True):
        with st.spinner("Analyzing RSU patterns..."):
            df_stations['air_dist'] = np.sqrt((df_stations['latitude']-u_lat)**2 + (df_stations['longitude']-u_lon)**2)
            candidates = df_stations.nsmallest(10, 'air_dist')
            final_list, geoms = [], {}
            for _, stn in candidates.iterrows():
                tt, dist, path = get_real_travel_stats(u_lat, u_lon, stn['latitude'], stn['longitude'], TOMTOM_KEY)
                rsu = get_rsu_features(stn['station_id'])
                feat = pd.DataFrame([{'hour': datetime.now().hour, 'day_of_week': datetime.now().weekday(), 'is_weekend': 0, 'total_slots': stn['charging_points'], 'prev_occupied_1': rsu['prev_occupied_1'], 'occ_change': rsu['occ_change'], 'rolling_avg_3': rsu['rolling_avg_3']}])
                
                pred_occ = model.predict(feat)[0]
                pred_free = max(0, stn['charging_points'] - pred_occ)
                t_wait = (30 / stn['charging_points']) if pred_free < 1 else 0
                score = (tt * 0.95) + (t_wait * 0.05) if u_soc <= 15 else tt + t_wait
                
                final_list.append({
                    "Station ID": stn['station_id'], "Distance (km)": round(dist, 2), 
                    "Travel Time (min)": round(tt, 2), "Wait Time (min)": round(t_wait, 1), 
                    "Total (min)": round(tt+t_wait, 2), "Free Slots": round(pred_free, 1), 
                    "Score": score, "lat": stn['latitude'], "lon": stn['longitude']
                })
                geoms[stn['station_id']] = path
            
            st.session_state.results = pd.DataFrame(final_list).sort_values("Score").head(5)
            st.session_state.geometries = geoms
            st.session_state.selected_station = st.session_state.results.iloc[0]['Station ID']

# --- 6. MAIN DISPLAY ---
st.title("⚡ RSU-Enabled EV Smart Navigation")

if st.session_state.results is not None:
    res_df = st.session_state.results
    
    # sugerencia Table (TOP)
    st.subheader("🎯 Best Recommendations")
    st.dataframe(res_df.drop(columns=['Score', 'lat', 'lon']), hide_index=True, use_container_width=True)
    st.info("💡 Tip: Click any station marker on the map below to switch the red navigation path.")

    st.divider()

    # Map Section (BOTTOM)
    active_id = st.session_state.selected_station
    st.subheader(f"🗺️ Route to {active_id}")
    
    m = folium.Map(location=[st.session_state.u_lat, st.session_state.u_lon], zoom_start=14)
    folium.Marker([st.session_state.u_lat, st.session_state.u_lon], tooltip="You", icon=folium.Icon(color='blue', icon='car', prefix='fa')).add_to(m)

    for _, r in res_df.iterrows():
        folium.Marker(
            [r['lat'], r['lon']], 
            tooltip=f"{r['Station ID']}",
            icon=folium.Icon(color='green', icon='bolt', prefix='fa')
        ).add_to(m)

    # Drawing red path for the selected station
    path_coords = st.session_state.geometries.get(active_id)
    if path_coords:
        folium.PolyLine(path_coords, color="red", weight=6).add_to(m)

    map_output = st_folium(m, height=500, use_container_width=True, key="main_nav_map")

    # Handle marker clicks to change path
    if map_output['last_object_clicked_tooltip']:
        clicked_id = map_output['last_object_clicked_tooltip']
        if clicked_id in st.session_state.geometries and clicked_id != st.session_state.selected_station:
            st.session_state.selected_station = clicked_id
            st.rerun()
