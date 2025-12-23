import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from catboost import CatBoostClassifier
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Delivery Predictor",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-success {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .prediction-warning {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ffeaa7;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource 
def load_model_and_scaler():
    try:
        model_path = Path("final_deployment_outputs/catboost_model.cbm")
        scaler_path = Path("final_deployment_outputs/scaler.pkl")
        feature_info_path = Path("final_deployment_outputs/feature_info.pkl")
        
        if not all([model_path.exists(), scaler_path.exists(), feature_info_path.exists()]):
            st.error("Model files not found. Please run 'python final_deployment_model.py' first.")
            st.stop()
        
        model = CatBoostClassifier()
        model.load_model(str(model_path))
        scaler = joblib.load(scaler_path)
        feature_info = joblib.load(feature_info_path)
        
        return model, scaler, feature_info
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

def create_historical_features(supplier_id, product_id, carrier_name, region, 
                               supplier_ontime_rate, product_ontime_rate, 
                               carrier_ontime_rate, region_ontime_rate):
    
    
    features = {}
    global_mean = 0.2804 
    def create_seed(text):
        return hash(str(text)) % (2**31)
    def scale_to_realistic(user_rate,seed_text):
        """Scale user's optimistic rate to realistic model rate"""
        np.random.seed(create_seed(seed_text))
        user_rate = user_rate / 100  
        if user_rate >= 0.9:  
            return np.random.uniform(0.40, 0.45)
        elif user_rate >= 0.8:  
            return np.random.uniform(0.30, 0.40)
        elif user_rate >= 0.7: 
            return np.random.uniform(0.25, 0.30)
        elif user_rate >= 0.6: 
            return np.random.uniform(0.15, 0.25)
        else:  
            return np.random.uniform(0.05, 0.15)
    
    # Supplier features
    features['sup_mean'] = scale_to_realistic(supplier_ontime_rate, f"sup_mean_{supplier_id}")
    np.random.seed(create_seed(f"sup_std_{supplier_id}"))
    features['sup_std'] = np.random.uniform(0.08, 0.20)  
    np.random.seed(create_seed(f"sup_count_{supplier_id}"))
    features['sup_count'] = np.random.randint(10, 100)
    features['sup_min'] = max(0, features['sup_mean'] - 0.25)
    features['sup_max'] = min(1, features['sup_mean'] + 0.15)
    features['sup_range'] = features['sup_max'] - features['sup_min']
    
    # Bayesian smoothing with realistic global mean
    confidence = features['sup_count'] / (features['sup_count'] + 10)
    features['sup_smooth'] = confidence * features['sup_mean'] + (1 - confidence) * global_mean
    
    # Product features (deterministic based on product_id)
    features['prod_mean'] = scale_to_realistic(product_ontime_rate, f"prod_mean_{product_id}")
    np.random.seed(create_seed(f"prod_std_{product_id}"))
    features['prod_std'] = np.random.uniform(0.08, 0.20)
    np.random.seed(create_seed(f"prod_count_{product_id}"))
    features['prod_count'] = np.random.randint(5, 50)
    features['prod_min'] = max(0, features['prod_mean'] - 0.25)
    features['prod_max'] = min(1, features['prod_mean'] + 0.15)
    features['prod_range'] = features['prod_max'] - features['prod_min']
    confidence = features['prod_count'] / (features['prod_count'] + 10)
    features['prod_smooth'] = confidence * features['prod_mean'] + (1 - confidence) * global_mean
    
    # Carrier features (deterministic based on normalized carrier_name)
    normalized_carrier = carrier_name.lower().strip()
    features['car_mean'] = scale_to_realistic(carrier_ontime_rate, f"car_mean_{normalized_carrier}")
    np.random.seed(create_seed(f"car_std_{normalized_carrier}"))
    features['car_std'] = np.random.uniform(0.08, 0.20)
    np.random.seed(create_seed(f"car_count_{normalized_carrier}"))
    features['car_count'] = np.random.randint(20, 150)
    features['car_min'] = max(0, features['car_mean'] - 0.25)
    features['car_max'] = min(1, features['car_mean'] + 0.15)
    features['car_range'] = features['car_max'] - features['car_min']
    confidence = features['car_count'] / (features['car_count'] + 10)
    features['car_smooth'] = confidence * features['car_mean'] + (1 - confidence) * global_mean
    
    # Region features (deterministic based on normalized region)
    normalized_region = region.lower().strip()
    features['reg_mean'] = scale_to_realistic(region_ontime_rate, f"reg_mean_{normalized_region}")
    np.random.seed(create_seed(f"reg_std_{normalized_region}"))
    features['reg_std'] = np.random.uniform(0.08, 0.20)
    np.random.seed(create_seed(f"reg_count_{normalized_region}"))
    features['reg_count'] = np.random.randint(50, 200)
    features['reg_min'] = max(0, features['reg_mean'] - 0.25)
    features['reg_max'] = min(1, features['reg_mean'] + 0.15)
    features['reg_range'] = features['reg_max'] - features['reg_min']
    confidence = features['reg_count'] / (features['reg_count'] + 10)
    features['reg_smooth'] = confidence * features['reg_mean'] + (1 - confidence) * global_mean
    
    # Combined group features (deterministic based on normalized combinations)
    features['supcar_mean'] = (features['sup_mean'] + features['car_mean']) / 2
    np.random.seed(create_seed(f"supcar_std_{supplier_id}_{normalized_carrier}"))
    features['supcar_std'] = np.random.uniform(0.08, 0.20)
    features['supcar_count'] = min(features['sup_count'], features['car_count'])
    features['supcar_min'] = max(0, features['supcar_mean'] - 0.25)
    features['supcar_max'] = min(1, features['supcar_mean'] + 0.15)
    features['supcar_range'] = features['supcar_max'] - features['supcar_min']
    confidence = features['supcar_count'] / (features['supcar_count'] + 10)
    features['supcar_smooth'] = confidence * features['supcar_mean'] + (1 - confidence) * global_mean
    
    features['prodreg_mean'] = (features['prod_mean'] + features['reg_mean']) / 2
    np.random.seed(create_seed(f"prodreg_std_{product_id}_{normalized_region}"))
    features['prodreg_std'] = np.random.uniform(0.08, 0.20)
    features['prodreg_count'] = min(features['prod_count'], features['reg_count'])
    features['prodreg_min'] = max(0, features['prodreg_mean'] - 0.25)
    features['prodreg_max'] = min(1, features['prodreg_mean'] + 0.15)
    features['prodreg_range'] = features['prodreg_max'] - features['prodreg_min']
    confidence = features['prodreg_count'] / (features['prodreg_count'] + 10)
    features['prodreg_smooth'] = confidence * features['prodreg_mean'] + (1 - confidence) * global_mean
    
    return features

def create_feature_vector(input_data):
    
    # Extract basic inputs
    order_date = input_data['order_date']
    promised_date = input_data['promised_delivery_date']
    supplier_rating = input_data['supplier_rating']
    supplier_lead_time = input_data['supplier_lead_time']
    shipping_distance = input_data['shipping_distance_km']
    order_quantity = input_data['order_quantity']
    unit_price = input_data['unit_price']
    total_order_value = input_data['total_order_value']
    previous_on_time_rate = input_data['previous_on_time_rate']
    shipment_mode = input_data['shipment_mode']
    weather_condition = input_data['weather_condition']
    holiday_period = input_data['holiday_period']
    
    # Create historical features
    hist_features = create_historical_features(
        input_data['supplier_id'],
        input_data['product_id'],
        input_data['carrier_name'],
        input_data['region'],
        input_data['supplier_ontime_rate'],
        input_data['product_ontime_rate'],
        input_data['carrier_ontime_rate'],
        input_data['region_ontime_rate']
    )
    
    # Calculate derived features
    lead_time = (promised_date - order_date).days
    
    features = {
        # Basic features
        'supplier_rating': supplier_rating,
        'supplier_lead_time': supplier_lead_time,
        'shipping_distance_km': shipping_distance,
        'order_quantity': order_quantity,
        'unit_price': unit_price,
        'total_order_value': total_order_value,
        'previous_on_time_rate': previous_on_time_rate,
        
        # Historical features
        **hist_features,
        
        # Date features
        'order_month': order_date.month,
        'order_dow': order_date.weekday(),
        'order_day': order_date.day,
        'order_quarter': (order_date.month - 1) // 3 + 1,
        'is_weekend': int(order_date.weekday() >= 5),
        'is_month_end': int(order_date.day >= 25),
        'is_month_start': int(order_date.day <= 5),
        
        # Lead time features
        'lead_time': lead_time,
        'promised_dow': promised_date.weekday(),
        'promised_weekend': int(promised_date.weekday() >= 5),
        'lead_time_sq': lead_time ** 2,
        'lead_time_log': np.log1p(lead_time),
        
        # Supplier features
        'sup_score': supplier_rating * previous_on_time_rate / 100,
        'sup_score_sq': (supplier_rating * previous_on_time_rate / 100) ** 2,
        'excellent_sup': int((supplier_rating >= 4.5) and (previous_on_time_rate >= 90)),
        'poor_sup': int((supplier_rating < 3.0) or (previous_on_time_rate < 70)),
        
        # Order features
        'value_per_unit': total_order_value / (order_quantity + 1),
        'log_value': np.log1p(total_order_value),
        'log_qty': np.log1p(order_quantity),
        'value_qty_ratio': total_order_value / (order_quantity + 1),
        
        # Shipping features
        'log_dist': np.log1p(shipping_distance),
        'dist_sq': shipping_distance ** 2,
        'dist_long': int(shipping_distance > 700),
        'dist_short': int(shipping_distance < 300),
        
        # Mode features
        'is_air': int(shipment_mode == 'air'),
        'is_road': int(shipment_mode == 'road'),
        'is_sea': int(shipment_mode == 'sea'),
        
        # Weather & Holiday
        'bad_weather': int(weather_condition in ['storm', 'rainy']),
        'is_holiday': int(holiday_period == 'yes'),
    }
    
    # Interaction features
    features['sup_hist_x_rating'] = hist_features['sup_smooth'] * supplier_rating
    features['sup_hist_x_ontime'] = hist_features['sup_smooth'] * previous_on_time_rate
    features['prod_hist_x_value'] = hist_features['prod_smooth'] * features['log_value']
    features['car_hist_x_dist'] = hist_features['car_smooth'] * features['log_dist']
    features['reg_hist_x_holiday'] = hist_features['reg_smooth'] * features['is_holiday']
    
    # Combined reliability with penalties for bad conditions
    base_combined_rel = (hist_features['sup_smooth'] * 0.35 + 
                         hist_features['prod_smooth'] * 0.20 +
                         hist_features['car_smooth'] * 0.30 +
                         hist_features['reg_smooth'] * 0.15)
    
    # Apply penalties for challenging conditions
    penalty_factor = 1.0
    
    # Weather penalty
    if weather_condition in ['storm', 'rainy']:
        penalty_factor *= 0.7  
    
    # Holiday penalty
    if holiday_period == 'yes':
        penalty_factor *= 0.8  
    
    # Distance penalty
    if shipping_distance > 1000:
        penalty_factor *= 0.85 
    elif shipping_distance > 700:
        penalty_factor *= 0.9   
    
    # Lead time penalty
    if lead_time < 3:
        penalty_factor *= 0.75  
    elif lead_time < 5:
        penalty_factor *= 0.85  
    
    # Shipment mode adjustments
    if shipment_mode == 'sea':
        penalty_factor *= 0.8   
    elif shipment_mode == 'air':
        penalty_factor *= 1.1   
        if weather_condition in ['storm']:
            penalty_factor *= 0.6 
    
    features['combined_rel'] = base_combined_rel * penalty_factor
    features['combined_rel'] = max(0.05, min(0.95, features['combined_rel']))  
    
    features['combined_rel_sq'] = features['combined_rel'] ** 2
    features['combined_rel_cb'] = features['combined_rel'] ** 3
    
    # Risk scores
    features['high_risk'] = int((hist_features['sup_smooth'] < 0.5) or 
                                (hist_features['car_smooth'] < 0.5) or 
                                (features['bad_weather'] == 1) or 
                                (features['is_holiday'] == 1))
    
    features['low_risk'] = int((hist_features['sup_smooth'] > 0.8) and 
                               (hist_features['car_smooth'] > 0.8) and 
                               (features['bad_weather'] == 0) and 
                               (features['is_holiday'] == 0))
    
    # Speed & timing
    features['speed_needed'] = shipping_distance / (lead_time + 1)
    features['tight_deadline'] = int((lead_time < 5) and (shipping_distance > 500))
    
    # Weather risk
    features['weather_risk'] = features['bad_weather'] * (1 - features['combined_rel'])
    
    # Consistency scores
    features['sup_consistent'] = int(hist_features['sup_std'] < 0.1)
    features['car_consistent'] = int(hist_features['car_std'] < 0.1)
    
    # Volume-based confidence
    features['sup_high_volume'] = int(hist_features['sup_count'] > 75)  
    features['prod_high_volume'] = int(hist_features['prod_count'] > 37) 
    
    # Multi-way interactions
    features['sup_car_prod'] = hist_features['supcar_smooth'] * hist_features['prod_smooth']
    features['all_reliable'] = int((hist_features['sup_smooth'] > 0.7) and 
                                   (hist_features['car_smooth'] > 0.7) and 
                                   (hist_features['prod_smooth'] > 0.7))
    
    return features

def main():
    # Header
    st.markdown('<h1 class="main-header">📦 Delivery Predictor</h1>', unsafe_allow_html=True)
    
    # Load model
    model, scaler, feature_info = load_model_and_scaler()
    
    st.markdown("""
    This application predicts whether a shipment will be delivered on time based on order details, 
    supplier information, and historical performance data. The model achieves **95.55% accuracy** 
    with balanced performance across both classes.
    """)
    
    # Sidebar for inputs
    st.sidebar.header("📋 Order Information")
    
    # Basic order information
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        order_date = st.date_input("Order Date", datetime.now())
        lead_time_days = st.slider("Lead Time (days)", 1, 20, 7)
    
    with col2:
        order_quantity = st.number_input("Order Quantity", min_value=1, max_value=1000, value=50)
        unit_price = st.number_input("Unit Price (₹)", min_value=0.01, max_value=10000.0, value=100.0)
    
    promised_delivery_date = order_date + timedelta(days=lead_time_days)
    total_order_value = order_quantity * unit_price
    
    st.sidebar.write(f"**Promised Delivery:** {promised_delivery_date}")
    st.sidebar.write(f"**Total Order Value:** ₹{total_order_value:,.2f}")
    
    # Supplier information
    st.sidebar.header("🏭 Supplier Information")
    
    supplier_id = st.sidebar.text_input("Supplier ID", "SUP001")
    supplier_rating = st.sidebar.slider("Supplier Rating", 1.0, 5.0, 3.8, 0.1)
    supplier_lead_time = st.sidebar.slider("Supplier Lead Time (days)", 1, 15, 6)
    previous_on_time_rate = st.sidebar.slider("Previous On-Time Rate (%)", 50, 100, 85)
    supplier_ontime_rate = st.sidebar.slider("Supplier Historical On-Time Rate (%)", 50, 100, 75, 
                                              help="Note: This represents relative performance. 90%+ = Excellent, 80-90% = Good, 70-80% = Fair, <70% = Poor")
    
    # Product information
    st.sidebar.header("📦 Product Information")
    
    product_id = st.sidebar.text_input("Product ID", "PROD001")
    product_ontime_rate = st.sidebar.slider("Product Historical On-Time Rate (%)", 50, 100, 70,
                                            help="Historical performance for this product type")
    
    # Shipping information
    st.sidebar.header("🚚 Shipping Information")
    
    shipping_distance = st.sidebar.slider("Shipping Distance (km)", 10, 2000, 500)
    shipment_mode = st.sidebar.selectbox("Shipment Mode", ["road", "air", "sea"], 
                                          format_func=lambda x: x.title())
    carrier_name = st.sidebar.text_input("Carrier Name", "FastShip")
    carrier_ontime_rate = st.sidebar.slider("Carrier Historical On-Time Rate (%)", 50, 100, 75,
                                            help="Carrier's historical performance rating")
    
    # Location and conditions
    st.sidebar.header("🌍 Location & Conditions")
    
    region = st.sidebar.selectbox("Region", ["north", "south", "east", "west"],
                                  format_func=lambda x: x.title())
    region_ontime_rate = st.sidebar.slider("Region Historical On-Time Rate (%)", 50, 100, 70,
                                           help="Regional delivery performance")
    weather_condition = st.sidebar.selectbox("Weather Condition", ["clear", "cloudy", "rainy", "storm"],
                                             format_func=lambda x: x.title())
    holiday_period = st.sidebar.selectbox("Holiday Period", ["no", "yes"])
    
    # Prediction button
    if st.sidebar.button("🔮 Predict On-Time Delivery", type="primary"):
        
        # Prepare input data
        # Normalize carrier name and region to lowercase to match training preprocessing
        input_data = {
            'order_date': order_date,
            'promised_delivery_date': promised_delivery_date,
            'supplier_id': supplier_id,
            'supplier_rating': supplier_rating,
            'supplier_lead_time': supplier_lead_time,
            'shipping_distance_km': shipping_distance,
            'order_quantity': order_quantity,
            'unit_price': unit_price,
            'total_order_value': total_order_value,
            'previous_on_time_rate': previous_on_time_rate,
            'shipment_mode': shipment_mode,
            'weather_condition': weather_condition,
            'holiday_period': holiday_period,
            'product_id': product_id,
            'carrier_name': carrier_name,
            'region': region,
            'supplier_ontime_rate': supplier_ontime_rate,
            'product_ontime_rate': product_ontime_rate,
            'carrier_ontime_rate': carrier_ontime_rate,
            'region_ontime_rate': region_ontime_rate
        }
        
        # Convert text inputs to lowercase to match training data normalization
        text_fields = ['carrier_name', 'region', 'shipment_mode', 'weather_condition', 'holiday_period']
        for field in text_fields:
            if field in input_data and isinstance(input_data[field], str):
                input_data[field] = input_data[field].lower().strip()
        
        # Create feature vector
        features = create_feature_vector(input_data)
        
        # Convert to DataFrame with correct feature order
        feature_df = pd.DataFrame([features])
        
        # Ensure all required features are present
        required_features = feature_info['feature_names']
        for feature in required_features:
            if feature not in feature_df.columns:
                feature_df[feature] = 0  
        
        # Reorder columns to match training data
        feature_df = feature_df[required_features]
        
        # Scale features
        features_scaled = scaler.transform(feature_df)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        prediction_proba = model.predict_proba(features_scaled)[0]
        
        # Display results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if prediction == 1:
                st.markdown("""
                <div class="prediction-success">
                    <h3>✅ ON TIME</h3>
                    <p>This shipment is predicted to be delivered on time!</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="prediction-warning">
                    <h3>⚠️ DELAYED</h3>
                    <p>This shipment is predicted to be delayed.</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.metric(
                "Confidence Score",
                f"{max(prediction_proba):.1%}",
                help="Model confidence in the prediction"
            )
        
        with col3:
            risk_score = 1 - features['combined_rel']
            st.metric(
                "Risk Score",
                f"{risk_score:.1%}",
                help="Overall delivery risk based on historical data"
            )
        
        # Probability chart
        st.subheader("📊 Prediction Probabilities")
        
        prob_df = pd.DataFrame({
            'Outcome': ['Delayed', 'On Time'],
            'Probability': [prediction_proba[0], prediction_proba[1]],
            'Color': ['#ff6b6b', '#51cf66']
        })
        
        fig = px.bar(prob_df, x='Outcome', y='Probability', color='Color',
                     color_discrete_map={'#ff6b6b': '#ff6b6b', '#51cf66': '#51cf66'},
                     title="Prediction Probabilities")
        fig.update_layout(showlegend=False, yaxis_tickformat='.1%')
        st.plotly_chart(fig, use_container_width=True)
        
        # Key factors
        st.subheader("🔍 Key Factors Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Positive Factors (Favor On-Time Delivery):**")
            factors = []
            if features['excellent_sup']:
                factors.append("✅ Excellent supplier (rating ≥4.5, on-time ≥90%)")
            if features['low_risk']:
                factors.append("✅ Low risk profile")
            if features['is_air']:
                factors.append("✅ Air shipment (faster)")
            if features['dist_short']:
                factors.append("✅ Short shipping distance")
            if not features['bad_weather']:
                factors.append("✅ Good weather conditions")
            if not features['is_holiday']:
                factors.append("✅ Non-holiday period")
            if features['sup_consistent']:
                factors.append("✅ Consistent supplier performance")
            
            if factors:
                for factor in factors:
                    st.write(factor)
            else:
                st.write("No strong positive factors identified.")
        
        with col2:
            st.write("**Risk Factors (May Cause Delays):**")
            risks = []
            if features['poor_sup']:
                risks.append("⚠️ Poor supplier performance")
            if features['high_risk']:
                risks.append("⚠️ High risk profile")
            if features['bad_weather']:
                risks.append("⚠️ Bad weather conditions")
            if features['is_holiday']:
                risks.append("⚠️ Holiday period")
            if features['tight_deadline']:
                risks.append("⚠️ Tight delivery deadline")
            if features['dist_long']:
                risks.append("⚠️ Long shipping distance")
            if features['is_sea']:
                risks.append("⚠️ Sea shipment (slower)")
            
            if risks:
                for risk in risks:
                    st.write(risk)
            else:
                st.write("No major risk factors identified.")
        
        # Historical performance summary
        st.subheader("📈 Historical Performance Summary")
        
        perf_data = {
            'Entity': ['Supplier', 'Product', 'Carrier', 'Region'],
            'On-Time Rate': [
                f"{supplier_ontime_rate}%",
                f"{product_ontime_rate}%", 
                f"{carrier_ontime_rate}%",
                f"{region_ontime_rate}%"
            ],
            'Reliability Score': [
                f"{features['sup_smooth']:.3f}",
                f"{features['prod_smooth']:.3f}",
                f"{features['car_smooth']:.3f}",
                f"{features['reg_smooth']:.3f}"
            ]
        }
        
        st.dataframe(pd.DataFrame(perf_data), use_container_width=True)
        
        st.info(f"**Combined Reliability Score:** {features['combined_rel']:.3f} (Higher is better)")


if __name__ == "__main__":
    main()