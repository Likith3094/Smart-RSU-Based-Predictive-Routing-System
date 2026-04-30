import streamlit as st
import pandas as pd
import numpy as np
import requests
import pickle
import folium
import os
import urllib.parse
from streamlit_folium import st_folium
from streamlit_geolocation import streamlit_geolocation
from datetime import datetime, timedelta
from dotenv import load_dotenv
# --- 1. INITIAL CONFIGURATION ---
st.set_page_config(page_title="EV Smart RSU Reroute", layout="wide")

load_dotenv()

TOMTOM_KEY = os.getenv("TOMTOM_API_KEY")
OSRM_URL = os.getenv("OSRM_URL")
if not TOMTOM_KEY:
    st.error("TOMTOM API KEY not found. Check your .env file")
    st.stop()

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
    try:
        query = urllib.parse.quote(address)
        url = f"https://api.tomtom.com/search/2/geocode/{query}.json"

        params = {
            "key": tomtom_key,
            "limit": 1
        }

        res = requests.get(url, params=params, timeout=5)
        data = res.json()

        if data.get('results'):
            pos = data['results'][0]['position']
            return pos['lat'], pos['lon']

    except Exception as e:
        st.error(f"Search error: {e}")

    return None, None


def get_real_travel_stats(u_lat, u_lon, s_lat, s_lon, tomtom_key):
    try:
        osrm_query = f"{OSRM_URL}{u_lon},{u_lat};{s_lon},{s_lat}?overview=full&geometries=geojson"
        osrm_res = requests.get(osrm_query, timeout=5).json()
        route_data = osrm_res['routes'][0]
        base_time = route_data['duration'] / 60
        dist = route_data['distance'] / 1000
        path_geometry = route_data['geometry']['coordinates']

        tt_url = f"https://api.tomtom.com/routing/1/calculateRoute/{u_lat},{u_lon}:{s_lat},{s_lon}/json?key={tomtom_key}&traffic=true"
        tt_res = requests.get(tt_url, timeout=10).json()
        delay = tt_res['routes'][0]['summary']['trafficDelayInSeconds'] / 60

        return base_time + delay, dist, path_geometry
    except:
        return 999, 0, None


def get_rsu_features(station_id):
    station_data = df_live[df_live['station_id'] == station_id].tail(3)
    if station_data.empty:
        return {'prev_occupied_1': 0, 'occ_change': 0, 'rolling_avg_3': 0}

    curr_occ = station_data['occupied_now'].iloc[-1]
    prev_occ = station_data['occupied_now'].iloc[-2] if len(station_data) > 1 else curr_occ

    return {
        'prev_occupied_1': curr_occ,
        'occ_change': curr_occ - prev_occ,
        'rolling_avg_3': station_data['occupied_now'].mean()
    }

# --- 4. SESSION STATE ---
if 'results' not in st.session_state: st.session_state.results = None
if 'geometries' not in st.session_state: st.session_state.geometries = None
if 'selected_station' not in st.session_state: st.session_state.selected_station = None

#  Start with NO location
if 'u_lat' not in st.session_state: st.session_state.u_lat = None
if 'u_lon' not in st.session_state: st.session_state.u_lon = None

# --- 5. SIDEBAR ---
with st.sidebar:
    st.header(" User Location")

    # --- Address Search ---
    search_input = st.text_input("Enter Address")

    if st.button(" Set Location"):
        if search_input:
            lat, lon = get_coords_from_address(search_input, TOMTOM_KEY)

            if lat is not None:
                st.session_state.u_lat = lat
                st.session_state.u_lon = lon
                st.success("Location Updated!")
                st.rerun()
            else:
                st.error("Address not found")

    st.write("---")

    # --- GPS ---
    st.write("Or capture your current position:")
    location = streamlit_geolocation()

    if location.get('latitude'):
        st.session_state.u_lat = location['latitude']
        st.session_state.u_lon = location['longitude']
        st.success("GPS Location Captured!")

    st.write("---")

    u_lat = st.session_state.u_lat
    u_lon = st.session_state.u_lon

    # --- Disable logic ---
    generate_disabled = (u_lat is None or u_lon is None)

    if generate_disabled:
        st.caption(" Please set your location using address or GPS")

    u_soc = st.slider("Battery SoC (%)", 5, 100, 20)

    if st.button(" Find Best Charging Station", use_container_width=True, disabled=generate_disabled):
        with st.spinner("Analyzing routes..."):

            df_stations['air_dist'] = np.sqrt(
                (df_stations['latitude'] - u_lat) ** 2 +
                (df_stations['longitude'] - u_lon) ** 2
            )

            candidates = df_stations.nsmallest(10, 'air_dist')

            final_list = []
            geoms = {}

            for _, stn in candidates.iterrows():
                total_tt, road_dist, path = get_real_travel_stats(
                    u_lat, u_lon,
                    stn['latitude'], stn['longitude'],
                    TOMTOM_KEY
                )

                rsu = get_rsu_features(stn['station_id'])
                arrival_time = datetime.now() + timedelta(minutes=total_tt)

                feat_row = pd.DataFrame([{
                    'hour': arrival_time.hour,
                    'day_of_week': arrival_time.weekday(),
                    'is_weekend': 0,
                    'total_slots': stn['charging_points'],
                    'prev_occupied_1': rsu['prev_occupied_1'],
                    'occ_change': rsu['occ_change'],
                    'rolling_avg_3': rsu['rolling_avg_3']
                }])

                pred_occ = model.predict(feat_row)[0]
                pred_free = max(0, stn['charging_points'] - pred_occ)

                t_wait = (30 / stn['charging_points']) if pred_free < 1 else 0

                score = (total_tt * 0.95 + t_wait * 0.05) if u_soc <= 15 else total_tt + t_wait

                final_list.append({
                    "Station ID": stn['station_id'],
                    "Distance (km)": round(road_dist, 2),
                    "Travel Time (min)": round(total_tt, 2),
                    "Wait Time (min)": round(t_wait, 1),
                    "Free Slots": round(pred_free, 1),
                    "Score": score,
                    "lat": stn['latitude'],
                    "lon": stn['longitude']
                })

                geoms[stn['station_id']] = path

            st.session_state.results = pd.DataFrame(final_list).sort_values("Score").head(5)
            st.session_state.geometries = geoms
            st.session_state.selected_station = st.session_state.results.iloc[0]['Station ID']

# --- 6. MAIN INTERFACE ---
st.title("⚡ RSU-Enabled EV Smart Navigation")

# --- Initial Guidance ---
if st.session_state.results is None:
    st.warning(" Set your location and click 'Find Best Charging Station' to begin.")

if st.session_state.results is not None:
    res_df = st.session_state.results

    st.subheader(" Optimal Choices")
    st.dataframe(res_df.drop(columns=['Score', 'lat', 'lon']), hide_index=True, use_container_width=True)

    st.divider()

    active_id = st.session_state.selected_station
    st.subheader(f" Navigation: {active_id}")

    m = folium.Map(location=[u_lat, u_lon], zoom_start=13)

    path_coords = st.session_state.geometries.get(active_id)
    if path_coords:
        folium_path = [[p[1], p[0]] for p in path_coords]
        folium.PolyLine(folium_path, color="red", weight=5).add_to(m)

    folium.Marker([u_lat, u_lon], icon=folium.Icon(color='blue')).add_to(m)

    for _, r in res_df.iterrows():
        folium.Marker(
            [r['lat'], r['lon']],
            tooltip=r['Station ID'],
            icon=folium.Icon(color='green')
        ).add_to(m)

    map_output = st_folium(m, height=500, use_container_width=True)

    if map_output and map_output.get('last_object_clicked_tooltip'):
        clicked_id = map_output['last_object_clicked_tooltip']
        if clicked_id in st.session_state.geometries and clicked_id != st.session_state.selected_station:
            st.session_state.selected_station = clicked_id
            st.rerun()