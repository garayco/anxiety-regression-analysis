import pickle

with open("modelo_ansiedad.pkl", "rb") as f:
    model = pickle.load(f)

print(model)