from app import app
from flask import request, render_template
import os
import pandas as pd
import numpy as np
import pickle






# Adding path to config
app.config['INITIAL_FILE_UPLOADS']='app/static/uploads'
app.config['EXISTING_FILE']='app/static/original'
app.config['GENERATED_FILE']='app/static/generated'

ALLOWED_EXTENSIONS= set(['csv'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

#Route to home page
@app.route("/",methods=["GET","POST"])
def index():

    # Execute if request is get
    if request.method == "GET":
        return render_template("index.html")

    # Execute if request is post
    if request.method== "POST":
        # Get uploaded image
        file_upload = request.files['file_upload']
        if file_upload and allowed_file(file_upload.filename):
            filename = file_upload.filename

            # save uploaded file

            file_upload.save(os.path.join(app.config['INITIAL_FILE_UPLOADS'],'test.csv'))

            # read saved file

            df= pd.read_csv('app/static/uploads/test.csv')
            customer_data=df["Customer_ID"]
            features=df.drop("Customer_ID", axis=1)
            features.drop("Unnamed: 0", axis=1,inplace=True)

            # load model

            model_file="app/vote_soft_model.sav"
            loaded_model = pickle.load(open(model_file, 'rb')) 

            # extract customer_id , features from csv file
            # load data
            customer_data_array=customer_data.to_numpy()
            test_data_array=features.to_numpy()

            # predict the results

            ynew = loaded_model.predict(test_data_array)

            # create zip file 

            final_data=np.dstack((customer_data_array,ynew))

            # create table 
            stringg='<tr><th>Customer ID</th><th>Default Or Not</th></tr>'
            for i in range(len(customer_data_array)):
                if ynew[i]==1:
                    cust="Default"
                    table_data='<td style="color:red;">'  + cust +'</td>'
                else:
                    cust="Not Default"
                    table_data='<td>'  + cust +'</td>'

                stringg=stringg+ '<tr><td>'  + str(customer_data_array[i]) +'</td>' + table_data +'</tr>'
                
                
    return render_template('index.html',pred=stringg)
        