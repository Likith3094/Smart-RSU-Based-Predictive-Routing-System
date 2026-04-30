import streamlit as st
import pandas as pd
import numpy as np
import requests
import pickle
import folium
from streamlit_folium import st_folium
from datetime import datetime, timedelta

# --- 1. INITIAL CONFIGURATION ---
st.set_page_config(page_title="EV Smart RSU Reroute", layout="wide")

TOMTOM_KEY = "akPHWg9CrZg6SDG085lPj5Dlb0vfrUNA" 
OSRM_URL = "http://localhost:5001/route/v1/driving/"

# --- 2. ASSET LOADING ---
@st.cache_resource
def load_assets():
    try:
        model = pickle.load(open('ev_xgboost_model.pkl', 'rb'))
        # Ensure your CSV name matches your local file (e.g., processed_stations.csv)
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
    """Fuzzy search to find locations like 'Champa Gali'"""
    url = f"https://tomtom.com{address}.json?key={tomtom_key}&limit=1"
    try:
        res = requests.get(url).json()
        if res['results']:
            pos = res['results']['position']
            return pos['lat'], pos['lon']
    except: return None, None

def get_real_travel_stats(u_lat, u_lon, s_lat, s_lon, tomtom_key):
    """Calculates Road Time, Distance, and Geometry Path"""
    try:
        # Added radiuses=1000 to help snap 'off-road' points to nearest road
        osrm_query = f"{OSRM_URL}{u_lon},{u_lat};{s_lon},{s_lat}?overview=full&geometries=geojson&radiuses=1000;1000"
        osrm_res = requests.get(osrm_query, timeout=5).json()
        
        # Access the first route index [0]
        route_data = osrm_res['routes'][0]
        base_time = route_data['duration'] / 60
        dist = route_data['distance'] / 1000
        path_geometry = route_data['geometry']['coordinates'] # List of [lon, lat]

        tt_url = f"https://api.tomtom.com/routing/1/calculateRoute/{u_lat},{u_lon}:{s_lat},{s_lon}/json?key={tomtom_key}&traffic=true"
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

# --- 4. SESSION STATE ---
if 'results' not in st.session_state: st.session_state.results = None
if 'geometries' not in st.session_state: st.session_state.geometries = None
if 'selected_station' not in st.session_state: st.session_state.selected_station = None
if 'u_lat' not in st.session_state: st.session_state.u_lat = 28.6139
if 'u_lon' not in st.session_state: st.session_state.u_lon = 77.2090

# --- 5. SIDEBAR ---
with st.sidebar:
    st.header("📍 Set Your Location")
    address_query = st.text_input("Search (e.g. Champa Gali, Saket)", "Connaught Place, Delhi")
    if st.button("Search & Update"):
        lat, lon = get_coords_from_address(address_query, TOMTOM_KEY)
        if lat:
            st.session_state.u_lat, st.session_state.u_lon = lat, lon
            st.success("Location Found!")
        else: st.error("Place not found.")

    u_lat = st.number_input("Latitude", value=st.session_state.u_lat, format="%.6f")
    u_lon = st.number_input("Longitude", value=st.session_state.u_lon, format="%.6f")
    u_soc = st.slider("Battery SoC (%)", 5, 100, 20)
    
    if st.button("Generate Smart Route", use_container_width=True):
        with st.spinner("Analyzing routes..."):
            df_stations['air_dist'] = np.sqrt((df_stations['latitude']-u_lat)**2 + (df_stations['longitude']-u_lon)**2)
            candidates = df_stations.nsmallest(10, 'air_dist')
            
            final_list = []; geoms = {}

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
                
                # --- COST FUNCTION (Optimal vs Safety) ---
                score = (total_tt * 0.95) + (t_wait * 0.05) if u_soc <= 15 else total_tt + t_wait
                
                final_list.append({
                    "Station ID": stn['station_id'], "Distance (km)": round(road_dist, 2),
                    "Travel Time (min)": round(total_tt, 2), "Wait Time (min)": round(t_wait, 1),
                    "Free Slots": round(pred_free, 1), "Score": score, "lat": stn['latitude'], "lon": stn['longitude']
                })
                geoms[stn['station_id']] = path

            # Sort by SCORE so the best station is at index 0
            sorted_res = pd.DataFrame(final_list).sort_values("Score").head(5)
            st.session_state.results = sorted_res
            st.session_state.geometries = geoms
            st.session_state.selected_station = sorted_res.iloc[0]['Station ID']

# --- 6. MAIN INTERFACE ---
st.title("⚡ RSU-Enabled EV Smart Navigation")

if st.session_state.results is not None:
    res_df = st.session_state.results
    col_table, col_map = st.columns([1, 1])
    
    with col_table:
        st.subheader("🎯 Optimal Choices")
        st.dataframe(res_df.drop(columns=['Score', 'lat', 'lon']), hide_index=True)
        st.info("💡 **Interactive:** Click any station marker on the map to switch navigation paths.")
    
    with col_map:
        # Default to the best choice or whatever was clicked last
        active_id = st.session_state.selected_station
        st.subheader(f"🗺️ Navigation: {active_id}")
        
        m = folium.Map(location=[u_lat, u_lon], zoom_start=13, tiles="CartoDB positron")
        
        # Draw path for the SELECTED station
        path_coords = st.session_state.geometries.get(active_id)
        if path_coords:
            folium_path = [[p[1], p[0]] for p in path_coords] # Swap Lon/Lat to Lat/Lon
            folium.PolyLine(folium_path, color="red", weight=6, opacity=0.8).add_to(m)

        # Markers
        folium.Marker([u_lat, u_lon], icon=folium.Icon(color='blue', icon='user', prefix='fa')).add_to(m)
        for _, r in res_df.iterrows():
            folium.Marker(
                [r['lat'], r['lon']], 
                tooltip=r['Station ID'], # Tooltip used for click detection
                icon=folium.Icon(color='green', icon='bolt', prefix='fa')
            ).add_to(m)
        
        # Render map and detect clicks
        map_output = st_folium(m, width=700, height=500, key="main_map")

        # Update path on marker click
        if map_output['last_object_clicked_tooltip']:
            clicked_id = map_output['last_object_clicked_tooltip']
            if clicked_id in st.session_state.geometries and clicked_id != st.session_state.selected_station:
                st.session_state.selected_station = clicked_id
                st.rerun()
