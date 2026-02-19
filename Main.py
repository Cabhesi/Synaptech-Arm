import pandas as pd
import matplotlib.pyplot as mpl
import sklearn as s0
from sklearn.model_selection import RandomizedSearchCV
from sklearn.datasets import load_iris
from scipy.stats import  loguniform, uniform, randint
from sklearn.linear_model import LogisticRegression

def makeDataset(filePath):
    df = pd.read_csv(filePath, names=['Unsorted'])
    return df

myData = makeDataset("Evo_Initial_BCI_Data/2026-27-01_Evo_Run04_FiveSets_Gain12.csv")
pd.set_option('display.max_columns', None)
myNewData = myData['Unsorted'].str.split('\t', expand=True)
myNewData = myNewData.astype(float)
myNewData = myNewData.rename(columns = {
    0: "Sample Index",
    1: "EXG Channel 0",
    2: "EXG Channel 1",
    3: "EXG Channel 2",
    4: "EXG Channel 3",
    5: "EXG Channel 4",
    6: "EXG Channel 5",
    7: "EXG Channel 6",
    8: "EXG Channel 7",
    9: "Accel Channel 0",
    10: "Accel Channel 1",
    11: "Accel Channel 2",
    12: "Not Used",
    13: "Digital Channel 0 (D11)",
    14: "Digital Channel 1 (D12)",
    15: "Digital Channel 2 (D13)",
    16: "Digital Channel 3 (D17)",
    17: "Not Used",
    18: "Digital Channel 4 (D18)",
    19: "Analog Channel 0",
    20: "Analog Channel 1",
    21: "Analog Channel 2",
    22: "Timestamp",
    23: "Marker Channel",
    24: "Timestamp (Formatted)",
})
lowest = myNewData.iloc[0, 22]
myNewData['Timestamp'] = myNewData['Timestamp'] - lowest
myNewData['Label'] = (round(myNewData['Timestamp'], 0) % 10) >= 5
print(myNewData)


iris = load_iris(as_frame=True)
df = iris.frame

param_dist = {
    'C': loguniform(1e-4, 1e2),
    'l1_ratio': uniform(0, 1),
    'max_iter': randint(400, 500)
}

model = LogisticRegression(solver="saga", tol=1e-3)

tuner = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=100,
    cv=5,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

x = df.drop(columns=['target'])
y = df['target']
tuner.fit(x, y)

print(f"Best Accuracy: {tuner.best_score_: .4f}")
print(f"Best Parameters: {tuner.best_params_}")