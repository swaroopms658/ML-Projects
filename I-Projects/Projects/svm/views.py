from django.shortcuts import render
from django.http import HttpResponse

def svm(region, tenure, age, income, marital, address, ed, employ, retire, gender, reside):
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC

    # Load data
    path = "C:\\Users\\Sriram\\Downloads\\40_customerClassification\\Telecust1.csv"
    data = pd.read_csv(path)

    # Label encoding
    le = LabelEncoder()
    data['custcat'] = le.fit_transform(data[['custcat']])
    inputs = data.drop(['custcat'], axis=1)
    output = data['custcat']

    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(inputs, output, test_size=0.2, random_state=42)

    # Train the model
    model = SVC()
    model.fit(x_train, y_train)

    # Calculate model accuracy
    accuracy = model.score(x_test, y_test)  # Accuracy on test data

    # Prediction
    res = model.predict([[region, tenure, age, income, marital, address, ed, employ, retire, gender, reside]])
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
