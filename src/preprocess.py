import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def split_data(df):
    """
    Splits data into 80% Train, 10% Validation, and 10% Test.
    ~ Train(39116), Val(4889), Test(4890)
    """
    # Split 80% Train and 20% Remainder
    train_df, remainder_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Split the 20% Remainder into 10% Val and 10% Test
    val_df, test_df = train_test_split(remainder_df, test_size=0.5, random_state=42)
    
    print(f"Split complete: Train({len(train_df)}), Val({len(val_df)}), Test({len(test_df)})")
    return train_df, val_df, test_df

def preprocess_pipeline(train_df, val_df, test_df):
    """
    Stateful preprocessing: Fits only on training data to prevent leakage.
    """
    
    # --------- 1. STATELESS TRANSFORMATIONS
    def basic_clean(data):
        temp = data.copy()

        # Dropping zero values 
        temp = temp[temp['price'] > 0] 
        # stabilize variance and handle skewness
        temp['price_log'] = np.log1p(temp['price'])

        # Impute missing values 
        temp['reviews_per_month'] = temp['reviews_per_month'].fillna(0)

        # Cap outliers
        temp['minimum_nights'] = temp['minimum_nights'].clip(upper=30)

        # Normalize availability to a ratio
        temp['availability_ratio'] = temp['availability_365'] / 365
        
        # Manual Mappings for unique categories to avoid one-hot encoding
        neigh_map = {'Manhattan': 5, 'Brooklyn': 4, 'Queens': 3, 'Staten Island': 2, 'Bronx': 1}
        room_map = {'Entire home/apt': 3, 'Private room': 2, 'Shared room': 1}
        temp['neighbourhood_group'] = temp['neighbourhood_group'].map(neigh_map)
        temp['room_type'] = temp['room_type'].map(room_map)

        # Drop columns that are not useful for modeling
        cols_to_drop = [
            # non-informative
            'id', 'host_id', 'name', 'host_name', 'last_review', 'latitude', 'longitude',
            # since we keep 'price_log' and 'availability_ratio', we drop the original columns 
                         'price', 'availability_365'] 
        
        return temp.drop(columns=cols_to_drop)

    train_base = basic_clean(train_df)
    val_base = basic_clean(val_df)
    test_base = basic_clean(test_df)


    # --------- 2. STATEFUL: TARGET ENCODING for "neighbourhood" (221 unique values)
    # Calculate means from TRAIN data only, then apply to all three
    target_means = train_base.groupby('neighbourhood')['price_log'].mean()


    # we might encounter a "Neighbourhood" in the Test dataset that never appeared in Training dataset.
    # Since target_means won't have a value for it, we use the global_mean as a safe fallback.
    global_mean = train_base['price_log'].mean()

    for data in [train_base, val_base, test_base]:
        data['neighbourhood_enc'] = data['neighbourhood'].map(target_means).fillna(global_mean)
        data.drop(columns=['neighbourhood'], inplace=True)


    # --------- 3. STATEFUL: SCALING (Fit on Train data only)
    scaler = StandardScaler()
    num_cols = ['minimum_nights', 'number_of_reviews', 'reviews_per_month', 
                'calculated_host_listings_count', 'availability_ratio', 'neighbourhood_enc']
    
    # FIT on train, then TRANSFORM all three
    train_base[num_cols] = scaler.fit_transform(train_base[num_cols])
    val_base[num_cols] = scaler.transform(val_base[num_cols])
    test_base[num_cols] = scaler.transform(test_base[num_cols])

    return train_base, val_base, test_base