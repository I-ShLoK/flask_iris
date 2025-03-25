#!/usr/bin/env python
# coding: utf-8

# In[1]:


#app.py

from flask import Flask,render_template,request
import joblib
import numpy as np


# In[3]:


app=Flask(__name__)
model=joblib.load('iris_model.pkl')


# In[7]:


@app.route('/')
def home():
    return render_template('home.html')
@app.route('/predict',methods=['POST'])
def predict():
    try:
        features=[float(request.form[f'feature{i}']) for i in range(1,5)]
    except ValueError: 
        return render_template('result.html',prediciton="Invalid input.Please enter numbers")
    prediction=model.predict([features])[0]

    class_names=['Setosa','Versicolor','Virginica']
    result=class_names[prediction]

    return render_template('result.html',prediction=result)
if __name__=='__main__':
    app.run(debug= True)


# In[ ]:





# In[ ]:




