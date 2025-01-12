from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd

def DecisionTree(Age, gender, Bmi, BloodPressure, FBS, HbA1c, FamilyHistoryofDiabetes, Smoking, Diet, Exercise):
    # Path to dataset
    path = "C:\\Users\\Sriram\\Downloads\\41_EasiestDiabetesClassification\\DiabetesClassification.csv"
    data = pd.read_csv(path)
    
    # Preprocessing
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    data['Gender'] = le.fit_transform(data[['Gender']])
    data['Blood Pressure'] = le.fit_transform(data[['Blood Pressure']])
    data['Smoking'] = le.fit_transform(data[['Smoking']])
    data['Diet'] = le.fit_transform(data[['Diet']])
    data['Exercise'] = le.fit_transform(data[['Exercise']])
    data['Diagnosis'] = le.fit_transform(data[['Diagnosis']])
    data['Family History of Diabetes'] = le.fit_transform(data[['Family History of Diabetes']])
    data['HbA1c'] = data['HbA1c'].astype(int)
    
    # Splitting inputs and outputs
    inputs = data.drop(['Diagnosis'], axis=1)
    output = data['Diagnosis']
    
    # Train-test split
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(inputs, output, test_size=0.2, random_state=42)
    
    # Model training
    from sklearn import tree
    model = tree.DecisionTreeClassifier()
    model.fit(x_train, y_train)
    
    # Calculate accuracy
    accuracy = model.score(x_test, y_test)  # Accuracy on the test set
    
    # Prediction for input values
    res = model.predict([[Age, gender, Bmi, BloodPressure, FBS, HbA1c, FamilyHistoryofDiabetes, Smoking, Diet, Exercise]])
    
    # Return result and accuracy
    return 'Cannot be Diagnosed' if res[0] == 0 else 'Can be Diagnosed', round(accuracy * 100, 2)

def Classifier(request):
    if request.method == "POST":
        data = request.POST
        Age = int(data.get('txtage'))
        gender = int(data.get('txtgender'))
        Bmi = float(data.get('txtbmi'))
        BloodPressure = int(data.get('txtbp'))
        FBS = float(data.get('txtfbs'))
        HbA1c = int(data.get('txthb1ac'))
        FamilyHistoryofDiabetes = int(data.get('txtfamilyhistory'))
        Smoking = int(data.get('txtsmoking'))
        Diet = int(data.get('txthdiet'))
        Exercise = int(data.get('txtexercise'))
        
        # Call the DecisionTree function
        prediction, accuracy = DecisionTree(Age, gender, Bmi, BloodPressure, FBS, HbA1c, 
                                            FamilyHistoryofDiabetes, Smoking, Diet, Exercise)
        
        # Pass prediction and accuracy to the template
        return render(request, 'Classifier.html', context={'prediction': prediction, 'accuracy': accuracy})
    
    return render(request, 'Classifier.html')
