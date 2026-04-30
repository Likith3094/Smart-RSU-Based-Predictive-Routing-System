import streamlit as st
import pandas as pd
import numpy as np
import requests
import pickle
from datetime import datetime, timedelta

# 1. MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(page_title="EV Smart Reroute", layout="wide")

# --- CONFIGURATION ---
# Replace with your actual TomTom key
TOMTOM_KEY = "akPHWg9CrZg6SDG085lPj5Dlb0vfrUNA" 
OSRM_URL = "http://localhost:5001/route/v1/driving/"

# --- DATA LOADING ---
@st.cache_resource
def load_assets():
    try:
        # Load the trained XGBoost model
        model = pickle.load(open('ev_xgboost_model.pkl', 'rb'))
        # Load the unique station hubs (1961 stations)
        stations = pd.read_csv('stations_data.csv')
        # Load the 24hr Live Simulation feed
        live_feed = pd.read_csv('rsu_live_feed.csv')
        live_feed['timestamp'] = pd.to_datetime(live_feed['timestamp'])
        return model, stations, live_feed
    except Exception as e:
        st.error(f"Error loading files: {e}. Check if CSVs and .pkl are in the folder.")
        st.stop()

model, df_stations, df_live = load_assets()

# --- HELPER FUNCTIONS ---
def get_real_travel_stats(u_lat, u_lon, s_lat, s_lon, tomtom_key):
    try:
        # 1. OSRM CALL (Local Docker)
        # OSRM expects: {longitude},{latitude}
        osrm_url = f"http://localhost:5001/route/v1/driving/{u_lon},{u_lat};{s_lon},{s_lat}?overview=false"
        osrm_res = requests.get(osrm_url, timeout=5).json()
        
        # Accessing the first route [0]
        base_time_mins = osrm_res['routes'][0]['duration'] / 60
        road_dist_km = osrm_res['routes'][0]['distance'] / 1000

        # 2. TOMTOM CALL (Cloud) - FIXED URL STRUCTURE
        # We must include 'api.' and the full path '/routing/1/calculateRoute/'
        tt_url = f"https://api.tomtom.com/routing/1/calculateRoute/{u_lat},{u_lon}:{s_lat},{s_lon}/json?key={TOMTOM_KEY}&traffic=true"
        
        tt_res = requests.get(tt_url, timeout=10).json()
        
        # Accessing the first route [0] and its summary
        traffic_delay_mins = tt_res['routes'][0]['summary']['trafficDelayInSeconds'] / 60
        
        return (base_time_mins + traffic_delay_mins), road_dist_km

    except Exception as e:
        # This will show you exactly what's wrong in your Anaconda terminal
        print(f"DEBUG: Error at station {s_lat}: {e}")
        return 999, 0






def get_rsu_features(station_id):
    """Fetches real-time trend data from the RSU Live Feed for XGBoost"""
    # Find the last recorded logs for this station to determine trends
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

# --- USER INTERFACE ---
st.title("⚡ RSU-Enabled EV Recommendation System")
st.markdown("Predictive Rerouting using **XGBoost**, **Local OSRM**, and **TomTom Traffic**.")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("📍 User Parameters")
    u_lat = st.number_input("Current Latitude (Delhi)", value=28.6139, format="%.6f")
    u_lon = st.number_input("Current Longitude (Delhi)", value=77.2090, format="%.6f")
    u_soc = st.slider("Battery SoC (%)", 5, 100, 20)
    
    if u_soc <= 15:
        st.warning("⚠️ **Safety Mode**: Prioritizing nearest chargers.")
    else:
        st.info("✅ **Optimal Mode**: Balancing traffic and station load.")
    
    search_btn = st.button("Generate Smart Recommendations", use_container_width=True)

if search_btn:
    with st.spinner("Analyzing routes and predicting hub occupancy..."):
        # Step 1: Broad filter (Straight line distance) to save API calls
        df_stations['air_dist'] = np.sqrt((df_stations['latitude']-u_lat)**2 + (df_stations['longitude']-u_lon)**2)
        candidates = df_stations.nsmallest(10, 'air_dist')
        
        final_results = []
        
        for _, stn in candidates.iterrows():
            # Step 2: Call APIs (Unpack EXACTLY 2 values)
            total_tt, road_dist = get_real_travel_stats(u_lat, u_lon, stn['latitude'], stn['longitude'], TOMTOM_KEY)
            
            # Step 3: Prepare RSU features
            rsu = get_rsu_features(stn['station_id'])
            arrival_time = datetime.now() + timedelta(minutes=total_tt)
            
            feat_row = pd.DataFrame([{
                'hour': arrival_time.hour,
                'day_of_week': arrival_time.weekday(),
                'is_weekend': 1 if arrival_time.weekday() >= 5 else 0,
                'total_slots': stn['charging_points'],
                'prev_occupied_1': rsu['prev_occupied_1'],
                'occ_change': rsu['occ_change'],
                'rolling_avg_3': rsu['rolling_avg_3']
            }])
            
            # Step 4: XGBoost Prediction
            pred_occ = model.predict(feat_row)[0]
            pred_free = max(0, stn['charging_points'] - pred_occ)
            
            # Step 5: Wait Time Penalty
            t_wait = (30 / stn['charging_points']) if pred_free < 1 else 0
            
            # Step 6: Cost Function based on SoC
            if u_soc <= 15:
                score = (total_tt * 0.95) + (t_wait * 0.05)
            else:
                score = total_tt + t_wait
            
            final_results.append({
                "Station ID": stn['station_id'],
                "Distance (km)": round(road_dist, 2),
                "Travel Time (min)": round(total_tt, 2),
                "Predicted Wait (min)": round(t_wait, 1),
                "Free Slots @ Arrival": round(pred_free, 1),
                "lat": stn['latitude'],
                "lon": stn['longitude'],
                "Final Score": score
            })
            
        # Select Top 5
        rec_df = pd.DataFrame(final_results).sort_values("Final Score").head(5)
        
        with col2:
            st.header("🎯 Recommended Stations")
            display_df = rec_df[['Station ID', 'Distance (km)', 'Travel Time (min)', 'Predicted Wait (min)', 'Free Slots @ Arrival']]
            st.dataframe(display_df, use_container_width=True)
            
            # Simple Point Map
            st.map(rec_df[['lat', 'lon']].rename(columns={'lat':'latitude', 'lon':'longitude'}))
