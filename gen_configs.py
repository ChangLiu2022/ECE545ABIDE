import pickle

# Example data: list of dictionaries
configs = [
    {"identifier":"A", "seq": "128-182", "self": False, "dropout":False, "prefix": "combined_v2_X2_"}
]

# --- Write to file using pickle ---
with open('configs.pkl', 'wb') as f:
    pickle.dump(configs, f)



