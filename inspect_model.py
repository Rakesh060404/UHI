import joblib, pickle, os, xgboost
path = 'best_model.pkl'
print('exists', os.path.exists(path), os.path.getsize(path))
try:
    m = joblib.load(path)
    print('loaded', type(m))
    print(m)
except Exception as e:
    import traceback; traceback.print_exc()
    try:
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        print('pickle load type', type(obj))
    except Exception as e2:
        print('pickle failed', e2)
