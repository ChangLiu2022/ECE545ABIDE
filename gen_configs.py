import pickle

# Example data: list of dictionaries
configs = [
     #{"identifier":"A", "seq": "128-182", "self": False, "dropout":False, "prefix": "combined_v2_X2_"},
     #{"identifier":"A", "seq": "128-182", "self": True, "dropout":False, "prefix": "combined_v2_X2_"},
     #{"identifier":"A", "seq": "128-182", "self": False, "dropout":True, "prefix": "combined_v2_X2_"},
     #{"identifier":"A", "seq": "128-182", "self": True, "dropout":True, "prefix": "combined_v2_X2_"},
     #{"identifier":"B", "seq": "128-182", "self": False, "dropout":False, "prefix": "combined_v3_X3_X_"},
     #{"identifier":"B", "seq": "128-182", "self": True, "dropout":False, "prefix": "combined_v3_X3_X_"},
     #{"identifier":"B", "seq": "128-182", "self": False, "dropout":True, "prefix": "combined_v3_X3_X_"},
     #{"identifier":"B", "seq": "128-182", "self": True, "dropout":True, "prefix": "combined_v3_X3_X_"},
     {"identifier":"A", "seq": "128-512", "self": False, "dropout":False, "prefix": "cleaned_combined_v4_X3_X_"},
     {"identifier":"A", "seq": "128-512", "self": True, "dropout":False, "prefix": "cleaned_combined_v4_X3_X_"},
      {"identifier":"A", "seq": "128-512", "self": False, "dropout":True, "prefix": "cleaned_combined_v4_X3_X_"},
      {"identifier":"A", "seq": "128-512", "self": True, "dropout":True, "prefix": "cleaned_combined_v4_X3_X_"},
     {"identifier":"N", "seq": "128-182-512", "self": False, "dropout":False, "prefix": "combined_v5_X3_X_"},
     {"identifier":"N", "seq": "128-182-512", "self": True, "dropout":False, "prefix": "combined_v5_X3_X_"},
     {"identifier":"N", "seq": "128-182-512", "self": False, "dropout":True, "prefix": "combined_v5_X3_X_"},
     {"identifier":"N", "seq": "128-182-512", "self": True, "dropout":True, "prefix": "combined_v5_X3_X_"}
]

# --- Write to file using pickle ---
with open('configs.pkl', 'wb') as f:
    pickle.dump(configs, f)



