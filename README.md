# ⚡ RSU-Enabled Predictive EV Navigation Framework

[![Python](https://shields.io)](https://python.org)
[![XGBoost](https://shields.io)](https://readthedocs.io)
[![Docker](https://shields.io)](https://docker.com)
[![Streamlit](https://shields.io)](https://streamlit.io)

An intelligent, predictive routing system for Electric Vehicles (EVs) that minimizes waiting times by forecasting station occupancy using Road-Side Unit (RSU) data and machine learning.


## Project Overview

Most EV navigation systems are **reactive**, showing you what is free *now*. This project addresses the **Predictive Occupancy Gap** in high-density urban environments (modeled for Delhi-NCR). 

By the time a driver reaches a station, it might already be full. Our system uses an **XGBoost Regressor** to predict station availability specifically for the driver's **Estimated Time of Arrival (ETA)**, effectively load-balancing the charging grid.

### Key Features
*   **Predictive Intelligence:** Forecasts station saturation at the exact moment of arrival.
*   **Context-Aware Routing:** Automatically switches to "Safety Mode" if Battery SoC is <15% to prevent road breakdowns.
*   **Hybrid Engine:** Uses local **Dockerized OSRM** for fast routing and **TomTom API** for real-time traffic deltas.
*   **Live Dashboard:** Interactive Streamlit interface with real-world map visuals and GPS integration.


## System Architecture

The framework operates on a hybrid-cloud architecture to ensure scalability and cost-efficiency:
1.  **RSU Layer (Data):** Collects live occupancy momentum and rolling averages.
2.  **ML Engine (Brain):** An XGBoost model trained on temporal lag features (Shifted T-1, T-2 data) to ensure zero data leakage.
3.  **Routing Engine:** 
    *   **OSRM (Local):** Calculates precise road geometry and base durations.
    *   **TomTom (Cloud):** Refines ETAs based on live traffic congestion.


## Experimental Results

We conducted a Monte Carlo simulation involving **1,000 unique users** to compare our system against the industry-standard "Naive Nearest" strategy.


| Metric | Nearest Station (Naive) | Smart RSU Model | Improvement |
| :--- | :--- | :--- | :--- |
| **Avg. Wait Time** | 15.10 mins | 11.35 mins | **24.9% Reduction** |
| **Total Time to Charge** | 21.40 mins | 18.30 mins | **14.5% Efficiency Gain** |


## Tech Stack

*   **Language:** Python 3.9+
*   **Machine Learning:** XGBoost, Scikit-learn, Pandas, NumPy
*   **Infrastructure:** Docker (OSRM Backend)
*   **APIs:** TomTom Routing & Search APIs
*   **Visualization:** Folium, Streamlit-Folium
*   **Deployment:** Streamlit


## Getting Started

### 1. Prerequisites
*   Install [Docker Desktop](https://docker.comproducts/docker-desktop/)
*   Obtain a free API Key from the [TomTom Developer Portal](https://tomtom.com)

### 2. Setup OSRM (Docker)
Download the map data and run the OSRM container:
```powershell
docker run -d -p 5001:5000 --name osrm -v "C:/your_data_path:/data" osrm/osrm-backend osrm-routed --algorithm mld /data/your_map_file.osrm
```

### 3. Install Dependencies
```bash
pip install streamlit pandas numpy xgboost requests folium streamlit-folium streamlit-geolocation
```

### 4. Run the App
```bash
streamlit run app.py
```

---

## Future Scope
*   **V2G Integration:** Incorporating Vehicle-to-Grid pricing signals.
*   **Energy Sourcing:** Prioritizing stations powered by renewable energy sources.
*   **Multi-Vehicle Coordination:** Communicating between vehicles to prevent swarm-convergence on the same empty station.

---

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
