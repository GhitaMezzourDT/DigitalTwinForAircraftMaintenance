import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import weibull_min

def extract_data_stream_sequence(asset_data, window_size):
   
    asset_data = asset_data.sort_index()

    sequences = []
    for i in range(len(asset_data) - window_size + 1):
        # Extract window_size data points for each condition variable
        window_sequence = asset_data.iloc[i:i + window_size][['pump_flow', 'pressure', 'head']]
        sequences.append(window_sequence.values.flatten())  
        
    return sequences

def construct_time_series_features(asset_data, classifier):

    features = asset_data[['pump_flow', 'pressure', 'head']].values
    
    # Classifictaion based on a RandomForestClassifier
    behavior_class = classifier.predict(features.reshape(1, -1))[0]

    #Failure probability calculation based on the Weibull distribution for each condition variable
    failure_probabilities = []
    for i in range(features.shape[1]):
        beta_1, eta_1, beta_2, eta_2 = get_weibull_parameters(features[:, i])
        lambda_i_d = calculate_failure_probability(features[:, i], beta_1, eta_1, beta_2, eta_2)
        failure_probabilities.append(lambda_i_d)

    return behavior_class, failure_probabilities

def get_weibull_parameters(data):
    # Function to calculate Weibull distribution parameters
    # beta and eta are estimated for each condition variable and fault type 
    beta_1, eta_1, beta_2, eta_2 = 563, 0.88, 1260, 1.09 # Example of Impeller Breakdown 
    return beta_1, eta_1, beta_2, eta_2

def calculate_failure_probability(data, beta_1, eta_1, beta_2, eta_2):
    # Function to calculate failure probability based on Weibull distribution
    t = np.arange(1, len(data) + 1)
    lambda_i_d = (beta_1 / eta_1**(beta_1)) * t**(beta_1 - 1) + (beta_2 / eta_2**(beta_2)) * t**(beta_2 - 1)
    return lambda_i_d

n = 3  # Number of assets
T = 100  # Number of time steps
window_size = 10 

# Initialize RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)


date_rng = pd.date_range(start='2022-01-01', end='2022-04-10', freq='D')
asset_data = pd.DataFrame(index=date_rng)
asset_data['pump_flow'] = np.random.rand(len(date_rng))
asset_data['pressure'] = np.random.rand(len(date_rng))
asset_data['head'] = np.random.rand(len(date_rng))

# Main loop
for k in range(T):
    for i in range(1, n+1):
        # Extract data stream sequence for condition variables
        data_sequence = extract_data_stream_sequence(asset_data, window_size)

        # Construct time series features matrix and classify behavior
        behavior_class, failure_probabilities = construct_time_series_features(pd.DataFrame(data_sequence), classifier)

        if behavior_class == "anomaly":
            lambda_i = np.sum(failure_probabilities)  # Update lambda_i for asset i
            y_out = f"C_NORMAL^(m={i})"
            S_HCA = "Anomaly Detected"
        else:
            lambda_i = 0
            S_HCA = "Normal Behavior"
            y_out = "C_NORMAL^(m=0)"

# Compute failure probabilities
F_H = np.prod(np.array([f_1_H for _ in range(H)]))  # Replace with your actual value for f_1_H
F_R = np.prod(np.array([f_1_R for _ in range(R)]))  # Replace with your actual value for f_1_R
F_L = np.prod(np.array([f_1_L for _ in range(L)]))  # Replace with your actual value for f_1_L

behavior_rules.append({
    "F_H": F_H,
    "F_R": F_R,
    "F_L": F_L,
    "S_HCA": S_HCA,
    "lambda_i": lambda_i,
    "y_out": y_out
})

for rule in behavior_rules:
    print("Rule:")
    print(f"F_H: {rule['F_H']}, F_R: {rule['F_R']}, F_L: {rule['F_L']}")
    print(f"S_HCA: {rule['S_HCA']}, lambda_i: {rule['lambda_i']}, y_out: {rule['y_out']}")
    print("--------------")



