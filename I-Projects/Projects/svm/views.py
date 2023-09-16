from django.shortcuts import render
from django.http import HttpResponse

def svm(region,tenure,age,income,marital,address,ed,employ,retire,gender,reside):
    import pandas as pd
    path="C:\\Users\\Sriram\\Downloads\\40_customerClassification\\Telecust1.csv"
    data=pd.read_csv(path)
    import sklearn
    from sklearn.preprocessing import LabelEncoder
    le=LabelEncoder()
    data['custcat']=le.fit_transform(data[['custcat']])
    inputs=data.drop(['custcat'],'columns')
    output=data.drop(['region','tenure','age','income','marital','address','ed','employ','retire','gender','reside'],'columns')
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(inputs,output,test_size=0.2)
    from sklearn.svm import SVC
    model=SVC()
    model.fit(x_train,y_train)
    res=model.predict([[region,tenure,age,income,marital,address,ed,employ,retire,gender,reside]])
    if res==0:
        return 'Customer is of category A' 
    elif res==1:
        return 'Customer is of category B'
    elif res==2:
        return 'Customer is of category C'
    else:
        return 'Customer is of category D'
    
def classification(request):
    if(request.method=='POST'):
        data=request.POST
        region=data.get('txtregion')
        tenure=data.get('txttenure')
        age=data.get('txtage')
        income=data.get('txtincome')
        marital=data.get('txtmar')
        address=data.get('txtadd')
        ed=data.get('txted')
        employ=data.get('txtemploy')
        retire=data.get('txtretire')
        gender=data.get('txtgender')
        reside=data.get('txtreside')
        prediction=svm(region,tenure,age,income,marital,address,ed,employ,retire,gender,reside)
        return render(request,'Classification.html',context={'prediction':prediction}) 
   
    return render(request,'Classification.html')



