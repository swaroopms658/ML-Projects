from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd
def DecisionTree(Age,gender,Bmi,BloodPressure,FBS,HbA1c,FamilyHistoryofDiabetes,Smoking,Diet,Exercise):
   
   path="C:\\Users\\Sriram\\Downloads\\41_EasiestDiabetesClassification\\DiabetesClassification.csv"
   data=pd.read_csv(path)
   from sklearn.preprocessing import LabelEncoder
   le=LabelEncoder()
   data['Gender']=le.fit_transform(data[['Gender']])
   data['Blood Pressure']=le.fit_transform(data[['Blood Pressure']])
   data['Smoking']=le.fit_transform(data[['Smoking']])
   data['Diet']=le.fit_transform(data[['Diet']])
   data['Exercise']=le.fit_transform(data[['Exercise']])
   data['Diagnosis']=le.fit_transform(data[['Diagnosis']])
   data['Family History of Diabetes']=le.fit_transform(data[['Family History of Diabetes']])
   data['HbA1c']=data['HbA1c'].astype(int)
   inputs = data.drop(['Diagnosis'], axis=1)
   output = data[['Diagnosis']]

   import sklearn
   from sklearn.model_selection import train_test_split
   x_train,x_test,y_train,y_test=train_test_split(inputs,output,test_size=0.2)
   from sklearn import tree
   model=tree.DecisionTreeClassifier()
   model.fit(x_train,y_train)
   y_pred = model.predict(x_test)
   res=model.predict([[Age,gender,Bmi,BloodPressure,FBS,HbA1c,FamilyHistoryofDiabetes,Smoking,Diet,Exercise]])
   return 'Cant be Diagnoised' if res == 0 else 'Can be Diagnoised' 






   
def Classifier(request):
   if(request.method=="POST"):
      data=request.POST
      Age=data.get('txtage')
      gender=data.get('txtgender')
      Bmi=data.get('txtbmi')
      BloodPressure=data.get('txtbp')
      FBS=data.get('txtfbs')
      HbA1c=data.get('txthb1ac')
      FamilyHistoryofDiabetes=data.get('txtfamilyhistory')
      Smoking=data.get('txtsmoking')
      Diet=data.get('txthdiet')
      Exercise=data.get('txtexercise')
      prediction=DecisionTree(Age,gender,Bmi,BloodPressure,FBS,HbA1c,FamilyHistoryofDiabetes,Smoking,Diet,Exercise)
      
      
      
      return render(request,'Classifier.html',context={'prediction':prediction}) 
   
   return render(request,'Classifier.html')





