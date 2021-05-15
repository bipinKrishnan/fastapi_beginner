from joblib import load
from fastapi import FastAPI, Form
from pydantic import BaseModel
import uvicorn

class Iris(BaseModel):
	n_1: float
	n_2: float
	n_3: float
	n_4: float
	
class Test(BaseModel):
	a: int
	b: int

app = FastAPI()

@app.post("/")
def get_root(data:Iris):
	data = data.dict()
	n_1, n_2, n_3, n_4 = data['n_1'], data['n_2'], data['n_3'], data['n_4']
	model = load("svc.joblib")
	pred = model.predict([[n_1, n_2, n_3, n_4]])
	return (f"Predicted label: {pred[0]}")
	
@app.post('/form')
def test_form(user=Form(...)):
	return user+" this is the user"
	
if __name__=='__main__':
	uvicorn.run(app, host='127.0.0.1', port=8000)
