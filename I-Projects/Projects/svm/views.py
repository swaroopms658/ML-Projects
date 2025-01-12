from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def svm(region, tenure, age, income, marital, address, ed, employ, retire, gender, reside):
    # Load data
    path = "C:\\Users\\Sriram\\Downloads\\40_customerClassification\\Telecust1.csv"
    data = pd.read_csv(path)

    # Label encoding
    le = LabelEncoder()
    data['custcat'] = le.fit_transform(data[['custcat']])
    inputs = data.drop(['custcat'], axis=1)
    output = data['custcat']

    # Feature scaling
    scaler = StandardScaler()
    inputs_scaled = scaler.fit_transform(inputs)

    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(inputs_scaled, output, test_size=0.2, random_state=42)

    # Hyperparameter tuning using GridSearchCV
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto']
    }
    grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy', verbose=1)
    grid_search.fit(x_train, y_train)

    # Best model
    best_model = grid_search.best_estimator_

    # Train the model
    best_model.fit(x_train, y_train)

    # Test accuracy
    y_pred = best_model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Prediction for user input
    user_input = scaler.transform([[region, tenure, age, income, marital, address, ed, employ, retire, gender, reside]])
    res = best_model.predict(user_input)

    category = ['Customer is of category A', 'Customer is of category B',
                'Customer is of category C', 'Customer is of category D']
    
    return category[res[0]], accuracy  # Return both prediction and accuracy


def classification(request):
    if request.method == 'POST':
        data = request.POST
        # Collect input data from the form
        region = int(data.get('txtregion'))
        tenure = int(data.get('txttenure'))
        age = int(data.get('txtage'))
        income = float(data.get('txtincome'))
        marital = int(data.get('txtmar'))
        address = int(data.get('txtadd'))
        ed = int(data.get('txted'))
        employ = int(data.get('txtemploy'))
        retire = int(data.get('txtretire'))
        gender = int(data.get('txtgender'))
        reside = int(data.get('txtreside'))

        # Call the SVM function to get prediction and accuracy
        prediction, accuracy = svm(region, tenure, age, income, marital, address, ed, employ, retire, gender, reside)
        
        # Render template with prediction and accuracy
        return render(request, 'Classification.html', context={'prediction': prediction, 'accuracy': round(accuracy * 100, 2)})

    return render(request, 'Classification.html')
