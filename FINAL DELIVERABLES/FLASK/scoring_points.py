import requests
import numpy as np
# NOTE: you must manually set API_KEY below using information retrieved from your IBM Cloud account.
API_KEY = "7FGNzBcGOMBIR-LTevrXYzj50iLFF2lJw2jYIkuus7wn"
token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey":
 API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
mltoken = token_response.json()["access_token"]

header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}

# NOTE: manually define and pass the array(s) of values to be scored in the next line
#payload_scoring = {"input_data": [{"field": [array_of_input_fields], "values": [array_of_values_to_be_scored, another_array_of_values_to_be_scored]}]}
payload_scoring = {"input_data": [{"field": ['vehicleType', 'yearOfRegistration', 'gearbox',
                       'powerPS','model', 'kilometer', 'monthOfRegistration', 'fuelType',
                       'brand','notRepairedDamage' ], 
                                  "values":[[0,2011,0,190,0,150000,5,0,0,0]] }]}
response_scoring = requests.post('https://us-south.ml.cloud.ibm.com/ml/v4/deployments/2e6b4079-fdd3-4b9b-8427-f309d0af6b20/predictions?version=2022-11-16', json=payload_scoring,
headers={'Authorization': 'Bearer ' + mltoken})
print("Scoring response")
predictions=response_scoring.json()
df_ev = np.exp(predictions['predictions'][0]['values'][0][0])
print(df_ev)