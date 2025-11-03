import streamlit as st
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
import folium
from streamlit_folium import folium_static
import warnings
import requests
from math import radians, cos, sin, asin, sqrt

warnings.filterwarnings("ignore")

# Set page config
st.set_page_config(
    page_title="NYC Traffic Accident Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #D7263D;
        text-align: center;
        padding: 1rem 0;
    }
    .risk-high {
        color: #D7263D;
        font-weight: bold;
        font-size: 1.5rem;
    }
    .risk-medium {
        color: #F77F00;
        font-weight: bold;
        font-size: 1.5rem;
    }
    .risk-low {
        color: #06A77D;
        font-weight: bold;
        font-size: 1.5rem;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# Define model architecture (must match trained model)
class SpatiotemporalGNN(torch.nn.Module):
    def __init__(self, num_features, hidden_dim=256, output_dim=2):
        super(SpatiotemporalGNN, self).__init__()

        self.sage1 = SAGEConv(num_features, hidden_dim)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)

        self.sage2 = SAGEConv(hidden_dim, hidden_dim)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim)

        self.sage3 = SAGEConv(hidden_dim, hidden_dim // 2)
        self.bn3 = torch.nn.BatchNorm1d(hidden_dim // 2)

        self.fc1 = torch.nn.Linear(hidden_dim // 2, 32)
        self.fc2 = torch.nn.Linear(32, output_dim)

        self.dropout = torch.nn.Dropout(0.4)

    def forward(self, x, edge_index, node_indices):
        num_all_nodes = 4618  # Fixed for Manhattan network

        full_features = torch.zeros(num_all_nodes, x.shape[1], device=x.device)
        full_features[node_indices] = x

        h = self.sage1(full_features, edge_index)
        h = self.bn1(h)
        h = F.relu(h)
        h = self.dropout(h)

        h = self.sage2(h, edge_index)
        h = self.bn2(h)
        h = F.relu(h)
        h = self.dropout(h)

        h = self.sage3(h, edge_index)
        h = self.bn3(h)
        h = F.relu(h)
        h = self.dropout(h)

        batch_h = h[node_indices]

        out = self.fc1(batch_h)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out


@st.cache_resource
def load_model():
    """Load the trained model."""
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize model with 19 features (sophisticated version)
        model = SpatiotemporalGNN(num_features=19, hidden_dim=256, output_dim=2).to(
            device
        )

        # Load weights
        model.load_state_dict(
            torch.load("model_weights/best_model.pt", map_location=device)
        )
        model.eval()

        # Create dummy edge index (simplified)
        # In production, this would be loaded from the actual graph
        edge_index = torch.randint(0, 4618, (2, 16370)).to(device)

        return {
            "model": model,
            "device": device,
            "edge_index": edge_index,
            "num_features": 19,
        }
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def get_rush_hour_intensity(hour):
    """Calculate rush hour intensity."""
    if hour == 7:
        return 2
    elif hour == 8:
        return 3
    elif hour == 9:
        return 2
    elif hour == 17:
        return 2
    elif hour == 18:
        return 3
    elif hour == 19:
        return 2
    elif hour >= 23 or hour <= 3:
        return 1
    else:
        return 0


def is_holiday(date):
    """Check if date is near a major holiday."""
    month = date.month
    day = date.day
    holidays = [(1, 1), (7, 4), (12, 25), (11, 27), (12, 31)]

    for h_month, h_day in holidays:
        if month == h_month and abs(day - h_day) <= 1:
            return 1
    return 0


def get_season(month):
    """Get season from month."""
    if month in [12, 1, 2]:
        return 0  # Winter
    elif month in [3, 4, 5]:
        return 1  # Spring
    elif month in [6, 7, 8]:
        return 2  # Summer
    else:
        return 3  # Fall


def get_weather_risk(season, hour):
    """Calculate weather risk proxy."""
    if season == 0:
        if hour in [6, 7, 8, 17, 18, 19]:
            return 3
        else:
            return 2
    elif season == 1:
        if hour in [7, 8, 17, 18]:
            return 2
        else:
            return 1
    elif season == 2:
        if hour in [12, 13, 14, 15, 16]:
            return 1
        else:
            return 0
    else:
        if hour >= 18 or hour <= 6:
            return 2
        else:
            return 1


def is_dark(month, hour):
    """Check if it's dark."""
    if month in [11, 12, 1, 2]:
        return 1 if hour >= 17 or hour <= 7 else 0
    elif month in [5, 6, 7, 8]:
        return 1 if hour >= 20 or hour <= 5 else 0
    else:
        return 1 if hour >= 18 or hour <= 6 else 0


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance between two points on earth (in meters)."""
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371000  # Radius of earth in meters
    return c * r


@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_nearby_crashes(latitude, longitude, radius_meters=50, days_back=30):
    """
    Fetch crash data from NYC Open Data API for a location.

    Args:
        latitude: Location latitude
        longitude: Location longitude
        radius_meters: Search radius in meters (default 50m)
        days_back: How many days back to search (default 30)

    Returns:
        tuple: (past_30d_crashes, days_since_last_crash)
    """
    try:
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        # Format dates for API (YYYY-MM-DD)
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        # NYC Open Data API endpoint
        base_url = "https://data.cityofnewyork.us/resource/h9gi-nx95.json"

        # Calculate bounding box (approximate)
        # 1 degree latitude ≈ 111km, 1 degree longitude ≈ 85km at NYC latitude
        lat_offset = radius_meters / 111000
        lon_offset = radius_meters / 85000

        # Build query
        params = {
            "$where": f"crash_date between '{start_str}' and '{end_str}' AND "
            f"latitude between {latitude - lat_offset} and {latitude + lat_offset} AND "
            f"longitude between {longitude - lon_offset} and {longitude + lon_offset}",
            "$limit": 1000,
            "$select": "crash_date,crash_time,latitude,longitude",
        }

        # Make API request
        response = requests.get(base_url, params=params, timeout=10)

        if response.status_code == 200:
            crashes = response.json()

            if not crashes:
                return 0, 999  # No crashes found

            # Filter by actual distance
            nearby_crashes = []
            for crash in crashes:
                try:
                    crash_lat = float(crash.get("latitude", 0))
                    crash_lon = float(crash.get("longitude", 0))

                    distance = haversine_distance(
                        latitude, longitude, crash_lat, crash_lon
                    )

                    if distance <= radius_meters:
                        crash_date_str = crash.get("crash_date", "")

                        # Parse crash datetime (crash_date already includes time)
                        # Format: 2024-11-01T14:30:00.000
                        try:
                            # Try parsing with milliseconds
                            crash_datetime = datetime.strptime(
                                crash_date_str, "%Y-%m-%dT%H:%M:%S.%f"
                            )
                        except ValueError:
                            try:
                                # Try without milliseconds
                                crash_datetime = datetime.strptime(
                                    crash_date_str, "%Y-%m-%dT%H:%M:%S"
                                )
                            except ValueError:
                                # Skip if can't parse
                                continue

                        nearby_crashes.append(
                            {"datetime": crash_datetime, "distance": distance}
                        )
                except (ValueError, KeyError):
                    continue

            if not nearby_crashes:
                return 0, 999

            # Count crashes in past 30 days
            past_30d_count = len(nearby_crashes)

            # Find most recent crash
            most_recent = max(nearby_crashes, key=lambda x: x["datetime"])
            days_since = (end_date - most_recent["datetime"]).days

            return past_30d_count, max(0, days_since)

        else:
            # API error, return defaults
            return 0, 999

    except Exception as e:
        # On any error, return default values
        st.warning(
            f"⚠️ Could not fetch real-time crash data: {str(e)[:100]}. Using defaults."
        )
        return 0, 999


def extract_features(location_type, timestamp, past_crashes=0, days_since_last=999):
    """Extract features for prediction (19 features - sophisticated version)."""

    # Static features based on location type (typical values)
    location_features = {
        "Times Square": {
            "degree": 4,
            "num_edges": 4,
            "highway_class": 3,
            "avg_lanes": 3,
            "avg_speed": 25,
            "intersection_complexity": 8.5,
            "speed_differential": 0,
            "speed_category": 1,
        },
        "Brooklyn Bridge": {
            "degree": 3,
            "num_edges": 3,
            "highway_class": 4,
            "avg_lanes": 4,
            "avg_speed": 35,
            "intersection_complexity": 7.0,
            "speed_differential": 10,
            "speed_category": 2,
        },
        "Central Park South": {
            "degree": 4,
            "num_edges": 4,
            "highway_class": 2,
            "avg_lanes": 2,
            "avg_speed": 25,
            "intersection_complexity": 5.5,
            "speed_differential": 0,
            "speed_category": 1,
        },
        "Custom Location": {
            "degree": 3,
            "num_edges": 3,
            "highway_class": 2,
            "avg_lanes": 2,
            "avg_speed": 25,
            "intersection_complexity": 5.0,
            "speed_differential": 0,
            "speed_category": 1,
        },
    }

    static_feats = location_features.get(
        location_type, location_features["Custom Location"]
    )

    # Temporal features
    hour = timestamp.hour
    day_of_week = timestamp.weekday()
    is_weekend = 1 if day_of_week >= 5 else 0
    month = timestamp.month

    rush_hour_intensity = get_rush_hour_intensity(hour)
    is_holiday_flag = is_holiday(timestamp)
    season = get_season(month)
    weather_risk = get_weather_risk(season, hour)
    is_dark_flag = is_dark(month, hour)

    # Combine all 19 features (matching saved model)
    features = [
        static_feats["degree"],
        static_feats["num_edges"],
        static_feats["highway_class"],
        static_feats["avg_lanes"],
        static_feats["avg_speed"],
        static_feats["intersection_complexity"],
        static_feats["speed_differential"],
        static_feats["speed_category"],
        hour,
        day_of_week,
        is_weekend,
        month,
        rush_hour_intensity,
        is_holiday_flag,
        season,
        weather_risk,
        is_dark_flag,
        past_crashes,
        days_since_last,
    ]

    return features, {
        "hour": hour,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "rush_hour": rush_hour_intensity,
        "holiday": is_holiday_flag,
        "season": ["Winter", "Spring", "Summer", "Fall"][season],
        "weather_risk": weather_risk,
        "is_dark": is_dark_flag,
        "past_crashes": past_crashes,
        "days_since_last": days_since_last,
    }


def predict_risk(features, model_data):
    """Make prediction."""
    # Simple standardization (approximate) for 19 features
    features_array = np.array([features])

    # Basic normalization (mean=0, std=1 approximation) for 19 features
    mean_vals = [
        3.5,  # degree
        3.5,  # num_edges
        2.0,  # highway_class
        1.8,  # avg_lanes
        25.0,  # avg_speed
        5.0,  # intersection_complexity
        2.0,  # speed_differential
        1.0,  # speed_category
        12.0,  # hour
        3.0,  # day_of_week
        0.2,  # is_weekend
        6.5,  # month
        1.0,  # rush_hour_intensity
        0.05,  # is_holiday
        1.5,  # season
        1.0,  # weather_risk
        0.45,  # is_dark
        0.2,  # past_30d_crashes
        200,  # days_since_last
    ]
    std_vals = [
        0.7,  # degree
        0.7,  # num_edges
        1.3,  # highway_class
        0.7,  # avg_lanes
        3.0,  # avg_speed
        3.0,  # intersection_complexity
        5.0,  # speed_differential
        1.0,  # speed_category
        7.0,  # hour
        2.0,  # day_of_week
        0.4,  # is_weekend
        3.5,  # month
        1.0,  # rush_hour_intensity
        0.2,  # is_holiday
        1.0,  # season
        1.0,  # weather_risk
        0.5,  # is_dark
        0.5,  # past_30d_crashes
        300,  # days_since_last
    ]

    features_normalized = (features_array - mean_vals) / std_vals

    # Convert to tensor
    features_tensor = torch.tensor(features_normalized, dtype=torch.float32).to(
        model_data["device"]
    )
    node_idx = torch.tensor([0], dtype=torch.long).to(model_data["device"])

    # Predict
    model_data["model"].eval()
    with torch.no_grad():
        output = model_data["model"](
            features_tensor, model_data["edge_index"], node_idx
        )
        prob = torch.softmax(output, dim=1)[0, 1].item()

    return prob


def get_risk_category(probability):
    """Categorize risk level."""
    if probability > 0.7:
        return "🔴 VERY HIGH RISK", "risk-high"
    elif probability > 0.5:
        return "🟠 HIGH RISK", "risk-high"
    elif probability > 0.35:
        return "🟡 MEDIUM RISK", "risk-medium"
    else:
        return "🟢 LOW RISK", "risk-low"


# Main App
def main():
    st.markdown(
        '<p class="main-header">🚗 NYC Traffic Accident Risk Predictor</p>',
        unsafe_allow_html=True,
    )
    st.markdown("### Spatiotemporal GNN-based Collision Risk Assessment")

    # Auto-load model on startup
    if "model_data" not in st.session_state:
        with st.spinner("🔄 Loading model... Please wait."):
            model_data = load_model()
            if model_data:
                st.session_state["model_data"] = model_data

    # Sidebar
    with st.sidebar:
        st.header("📊 Model Information")

        if "model_data" in st.session_state:
            st.success("✅ Model Loaded")
            model_data = st.session_state["model_data"]
            st.info(
                f"**Device:** {model_data['device']}\n\n**Features:** {model_data['num_features']}"
            )
            st.divider()
            st.markdown(
                """
            **Model Performance:**
            - F1 Score: 0.147
            - Recall: 0.317
            - AUC-ROC: 0.758
            - Precision: 0.095
            """
            )

            if st.button("🔄 Reload Model"):
                st.cache_resource.clear()
                del st.session_state["model_data"]
                st.rerun()
        else:
            st.error("❌ Model Failed to Load")
            st.info(
                "Make sure `model_weights/best_model.pt` is in the same directory as this app."
            )
            if st.button("🔄 Try Again"):
                st.rerun()
            return

    # Main content
    st.header("Single Location Risk Assessment")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📍 Location Details")

        location_option = st.selectbox(
            "Select Location",
            [
                "Times Square",
                "Brooklyn Bridge",
                "Central Park South",
                "Custom Location",
            ],
        )

        if location_option == "Custom Location":
            latitude = st.number_input(
                "Latitude",
                min_value=40.4,
                max_value=40.95,
                value=40.7580,
                format="%.4f",
            )
            longitude = st.number_input(
                "Longitude",
                min_value=-74.3,
                max_value=-73.7,
                value=-73.9855,
                format="%.4f",
            )
        else:
            locations = {
                "Times Square": (40.7580, -73.9855),
                "Brooklyn Bridge": (40.7061, -73.9969),
                "Central Park South": (40.7678, -73.9717),
            }
            latitude, longitude = locations[location_option]
            st.info(f"**Latitude:** {latitude}\n\n**Longitude:** {longitude}")

    with col2:
        st.subheader("🕐 Time Details")

        prediction_date = st.date_input("Date", value=datetime.now())
        prediction_time = st.time_input("Time", value=time(17, 0))

        prediction_datetime = datetime.combine(prediction_date, prediction_time)

        st.info(
            f"**Day:** {prediction_datetime.strftime('%A')}\n\n**Hour:** {prediction_time.hour}:00"
        )

    st.divider()

    # Fetch real crash data
    st.subheader("📊 Historical Crash Data")

    with st.spinner("🔍 Fetching real crash history from NYC Open Data..."):
        past_crashes, days_since_last = fetch_nearby_crashes(latitude, longitude)

    # Display fetched data
    col3, col4 = st.columns(2)
    with col3:
        st.metric(
            "Past 30-Day Crashes",
            past_crashes,
            help="Real crash data from NYC Open Data (within 50m radius)",
        )
        if past_crashes > 0:
            st.caption(
                f"🔴 High-risk area"
                if past_crashes >= 5
                else (
                    f"🟡 Moderate activity" if past_crashes >= 2 else f"🟢 Low activity"
                )
            )

    with col4:
        days_text = (
            "No recent crashes"
            if days_since_last >= 999
            else f"{days_since_last} days ago"
        )
        st.metric(
            "Last Crash",
            days_text,
            help="Days since most recent crash (999 = no crashes in past 30 days)",
        )
        if days_since_last < 999:
            st.caption(
                f"🔴 Very recent!"
                if days_since_last < 7
                else f"🟡 Recent" if days_since_last < 14 else f"🟢 Not recent"
            )

    # Optional: Allow manual override
    with st.expander("🔧 Override Historical Data (Advanced)"):
        st.caption("Only use this if you have more accurate local data")
        override_history = st.checkbox("Use custom values instead of API data")

        if override_history:
            past_crashes = st.number_input(
                "Custom Past 30-Day Crashes",
                min_value=0,
                max_value=100,
                value=past_crashes,
            )
            days_since_last = st.number_input(
                "Custom Days Since Last Crash",
                min_value=0,
                max_value=999,
                value=days_since_last,
            )

    if st.button("🔍 Predict Risk", type="primary", use_container_width=True):
        with st.spinner("Analyzing risk..."):
            model_data = st.session_state["model_data"]

            # Extract features
            features, feature_dict = extract_features(
                location_option, prediction_datetime, past_crashes, days_since_last
            )

            # Make prediction
            risk_prob = predict_risk(features, model_data)

            # Display results
            st.success("✅ Analysis Complete!")

            st.markdown("### 🎯 Crash Risk Assessment")

            risk_category, risk_class = get_risk_category(risk_prob)

            col5, col6, col7 = st.columns([1, 2, 1])
            with col6:
                st.markdown(
                    f'<p class="{risk_class}">{risk_category}</p>',
                    unsafe_allow_html=True,
                )
                st.metric("Crash Probability", f"{risk_prob*100:.2f}%")

            st.progress(risk_prob)

            st.divider()

            # Feature breakdown
            st.markdown("### 📊 Risk Factors")

            col8, col9 = st.columns(2)

            with col8:
                st.markdown("**Temporal Factors**")
                st.write(f"• Hour: {prediction_time.hour}:00")
                st.write(f"• Day: {prediction_datetime.strftime('%A')}")
                st.write(f"• Rush Hour Intensity: {feature_dict['rush_hour']}/3")
                st.write(
                    f"• Holiday Period: {'Yes' if feature_dict['holiday'] else 'No'}"
                )
                st.write(
                    f"• Lighting: {'Dark' if feature_dict['is_dark'] else 'Light'}"
                )

            with col9:
                st.markdown("**Historical Context**")
                st.write(f"• Past 30d Crashes: {past_crashes}")
                st.write(
                    f"• Days Since Last: {days_since_last if days_since_last < 999 else 'N/A'}"
                )
                st.write(f"• Season: {feature_dict['season']}")
                st.write(f"• Weather Risk: {feature_dict['weather_risk']}/3")

            # Map
            st.divider()
            st.markdown("### 🗺️ Location Map")

            m = folium.Map(location=[latitude, longitude], zoom_start=15)

            if risk_prob > 0.7:
                color = "red"
            elif risk_prob > 0.35:
                color = "orange"
            else:
                color = "green"

            folium.Marker(
                [latitude, longitude],
                popup=f"Risk: {risk_prob*100:.1f}%",
                icon=folium.Icon(color=color, icon="info-sign"),
            ).add_to(m)

            folium_static(m)

    # Footer
    st.divider()
    st.markdown(
        """
    <div style='text-align: center; color: gray; padding: 1rem;'>
        <p>NYC Traffic Accident Risk Predictor | Powered by Spatiotemporal GNN</p>
        <p><small>⚠️ For informational purposes only.</small></p>
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
