"""
Generate sample supply chain data for the On-Time Delivery Prediction System.
This creates realistic sample data to demonstrate the application.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def generate_sample_data(n_samples=1000, random_state=42):
    """Generate sample supply chain data."""
    np.random.seed(random_state)
    
    # Define categorical options
    shipment_modes = ['Air', 'Road', 'Sea']
    weather_conditions = ['Clear', 'Rainy', 'Storm']
    regions = ['North', 'South', 'East', 'West']
    carriers = ['BlueDart', 'DHL', 'FedEx', 'LocalTruckers', 'UPS']
    holiday_periods = ['Yes', 'No']
    delayed_reason_codes = ['None', 'Weather', 'Traffic', 'Supplier', 'Customs']
    
    # Generate features
    data = {
        'order_id': range(1, n_samples + 1),
        'supplier_id': np.random.randint(1, 51, n_samples),
        'supplier_rating': np.round(np.random.uniform(1, 5, n_samples), 1),
        'supplier_lead_time': np.random.randint(1, 30, n_samples),
        'shipping_distance_km': np.random.randint(50, 5000, n_samples),
        'order_quantity': np.random.randint(1, 500, n_samples),
        'unit_price': np.round(np.random.uniform(10, 500, n_samples), 2),
        'total_order_value': np.round(np.random.uniform(100, 50000, n_samples), 2),
        'previous_on_time_rate': np.round(np.random.uniform(0.5, 1.0, n_samples), 2),
        'shipment_mode': np.random.choice(shipment_modes, n_samples),
        'weather_condition': np.random.choice(weather_conditions, n_samples),
        'region': np.random.choice(regions, n_samples),
        'carrier_name': np.random.choice(carriers, n_samples),
        'holiday_period': np.random.choice(holiday_periods, n_samples),
        'delayed_reason_code': np.random.choice(delayed_reason_codes, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Create target variable based on features (simulating realistic patterns)
    # Higher supplier rating, higher on-time rate, shorter distance = more likely on-time
    on_time_prob = (
        0.3 +
        0.15 * (df['supplier_rating'] / 5) +
        0.20 * df['previous_on_time_rate'] +
        0.10 * (1 - df['shipping_distance_km'] / 5000) +
        0.10 * (1 - df['supplier_lead_time'] / 30) +
        0.05 * (df['shipment_mode'] == 'Air').astype(int) +
        0.05 * (df['weather_condition'] == 'Clear').astype(int) +
        0.05 * (df['holiday_period'] == 'No').astype(int)
    )
    
    # Add some randomness and clamp to [0.2, 0.95]
    on_time_prob = np.clip(on_time_prob + np.random.normal(0, 0.1, n_samples), 0.2, 0.95)
    
    # Generate binary target
    df['on_time_delivery'] = (np.random.random(n_samples) < on_time_prob).astype(int)
    
    return df


def preprocess_and_save(df, output_dir):
    """Preprocess data and save all required files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    label_encoders = {}
    
    # Encode categorical variables
    categorical_cols = ['shipment_mode', 'weather_condition', 'region', 'holiday_period', 'carrier_name', 'delayed_reason_code']
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
            print(f"  Encoded {col}: {dict(zip(le.classes_, range(len(le.classes_))))}")
    
    # Feature Engineering
    if 'supplier_rating' in df.columns and 'previous_on_time_rate' in df.columns:
        df['supplier_reliability_score'] = df['supplier_rating'] * df['previous_on_time_rate']
    
    # Prepare features for training
    feature_cols = [
        'supplier_rating', 'supplier_lead_time', 'shipping_distance_km',
        'order_quantity', 'unit_price', 'total_order_value', 'previous_on_time_rate',
        'shipment_mode_encoded', 'weather_condition_encoded', 'region_encoded',
        'holiday_period_encoded', 'carrier_name_encoded'
    ]
    
    X = df[feature_cols]
    y = df['on_time_delivery']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Initialize and fit scaler (but we'll use unscaled features for models)
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    # Save all files
    df.to_csv(output_dir / 'processed_data.csv', index=False)
    X_train.to_csv(output_dir / 'X_train.csv', index=False)
    X_test.to_csv(output_dir / 'X_test.csv', index=False)
    y_train.to_csv(output_dir / 'y_train.csv', index=False)
    y_test.to_csv(output_dir / 'y_test.csv', index=False)
    
    # Save correlation matrix
    corr_matrix = X.corr()
    corr_matrix.to_csv(output_dir / 'correlation_matrix.csv')
    
    with open(output_dir / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    with open(output_dir / 'label_encoders.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)
    
    print(f"\n✓ Saved files to {output_dir}")
    print(f"  - Training samples: {len(X_train)}")
    print(f"  - Test samples: {len(X_test)}")
    print(f"  - Features: {len(feature_cols)}")
    
    return X_train, X_test, y_train, y_test, label_encoders, scaler


if __name__ == "__main__":
    print("=" * 60)
    print("  GENERATING SAMPLE SUPPLY CHAIN DATA")
    print("=" * 60)
    
    # Generate sample data
    print("\n1. Generating sample data...")
    df = generate_sample_data(n_samples=1000)
    print(f"   Generated {len(df)} samples")
    
    # Preprocess and save
    print("\n2. Preprocessing and saving files...")
    base_dir = Path(__file__).parent
    output_dir = base_dir / 'outputs'
    
    X_train, X_test, y_train, y_test, label_encoders, scaler = preprocess_and_save(df, output_dir)
    
    print("\n" + "=" * 60)
    print("  SAMPLE DATA GENERATION COMPLETE")
    print("=" * 60)
    print("\nNow you can run the Streamlit app!")
