import joblib
from pydantic import BaseModel
from fastapi import FastAPI
import uvicorn

# 1. Load the trained model
model = joblib.load('frauddetection.pkl')

# 2. Define the input data schema using Pydantic BaseModel
class InputData(BaseModel):
    Year:int
    Month:int
    UseChip:int
    Amount:int
    MerchantName:int
    MerchantCity:int
    MerchantState:int
    mcc:int
    # Add the rest of the input features (feature4, feature5, ..., feature12)

# 3. Create a FastAPI app
app = FastAPI()

# 4. Define the prediction route
@app.post('/predict/')
def predict(data: InputData):
    # Convert the input data to a dictionary
    input_data = data.dict()

    # Extract the input features from the dictionary
    feature1 = input_data['Year']
    feature2=input_data['Month']
    feature3=input_data['UseChip']
    feature4=input_data['Amount']
    feature5=input_data['MerchantName']
    feature6=input_data['MerchantCity']
    feature7=input_data['MerchantState']
    feature8=input_data['mcc']
    # Extract the rest of the input features (feature4, feature5, ..., feature12)

    # Perform the prediction using the loaded model
    prediction = model.predict([[feature1, feature2, feature3,feature4,feature5,feature6,feature7,feature8]])  # Replace ... with the rest of the features

    # Convert the prediction to a string (or any other format you prefer)
    result = "Fraud" if prediction[0] == 1 else "Not a Fraud"

    return {"prediction": result}
# 4. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=7000)
