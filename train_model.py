from sklearn.datasets import load_iris
from sklearn.svm import SVC

from joblib import dump

model = SVC()
X, y = load_iris(return_X_y=True)
model.fit(X, y)

dump(model, 'svc.joblib')
