
import pandas
from flask import Flask, render_template, request

import pandas as pd
import numpy as np 


from sklearn.preprocessing import LabelEncoder

app =Flask(__name__)
import pickle
filename = 'resale_model.sav'
model_rand= pickle.load(open(filename, 'rb'))
@app.route('/') 
def index():
    return render_template('home.html')

@app.route('/predict') 
def predict():
    return render_template('predict.html')
@app.route('/y_predict', methods= ['POST']) 
def y_predict():
    regyear = int(request.form['Registrationyear'])
    powerps = int(request.form['PowerofcarinPS'])
    kms =int(request.form['KilometersDriven'])
    regmonth =int(request.form.get('Registrationmonth'))
    gearbox =(request.form['Geartype'])
    damage =request.form['cd']
    model = request.form.get('model')
    brand =request.form.get('brand') 
    fuelType =request.form.get('fueltype') 
    vehicletype = request.form.get('vechicletype')
    row = {'vehicleType': vehicletype,'yearOfRegistration':regyear, 
           'gearbox': gearbox,'powerPS': powerps, 'model':model,'kilometer': kms, 
           'monthOfRegistration': regmonth,'fuelType': fuelType,
           'brand': brand,'notRepairedDamage': damage}
    print(row)
    new_row = pd.DataFrame([row])
    new_df = pd.DataFrame(columns = ['vehicleType', 'yearOfRegistration', 'gearbox',
                           'powerPS','model', 'kilometer', 'monthOfRegistration', 'fuelType',
                           'brand','notRepairedDamage' ])
    new_df=pd.concat([new_df,new_row], ignore_index = True)
    new_df['monthOfRegistration']= new_df['monthOfRegistration'].astype(int)
    labels = ['gearbox', 'notRepairedDamage', 'model', 'brand', 'fuelType', 'vehicleType'] 
    mapper = {}
    for i in labels:
        mapper[i] = LabelEncoder()
        mapper[i].fit(new_df[i]) 
        mapper[i].classes_ = np.load(str('classes'+i+'.npy'),allow_pickle=True) 
        tr = mapper[i].fit_transform(new_df[i])
        new_df.loc[:, i + '_labels'] = pd.Series (tr, index=new_df.index)
    labeled = new_df[ ['yearOfRegistration', 'powerPS','kilometer','monthOfRegistration']
        + [x+'_labels' for x in labels]]
    X = labeled.values
    print(X)
    y_prediction = model_rand.predict(X)
    print(y_prediction)
    df_ev = np.exp(y_prediction)
    print("df_ev: {} ".format(df_ev))
    return render_template('predict.html',y = 'The resale value predicted is {:.2f}$'.format(df_ev[0]))
    
if __name__=='__main__':
    app.run(debug= False)
   