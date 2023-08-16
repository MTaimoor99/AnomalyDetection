from django.shortcuts import render
from django.http import HttpResponse
from anomaly_detection.utils import apply_normality_test

#Import necessary libraries to process uploaded file
import pandas as pd
import matplotlib.pyplot as plt,mpld3
from matplotlib.figure import Figure
import scipy.stats as stats
import numpy as np
from sklearn.model_selection import train_test_split

# Create your views here.
def histogram_plot(request):
    context={}
    if request.method=="POST":
        test_file = request.FILES.get(u'csv_file') #request.FILES has the file contents to be read. the get method gets the contents.
        if test_file: #Check if file is valid.
           df = pd.read_csv(test_file) #Read using pandas
           df=df.iloc[:,:-1]
           hist_list=[]
           for column in df.columns:
                fig=plt.figure()
                plt.hist(df[column])
                plt.title(f'{column} Histogram')
                histogram=mpld3.fig_to_html(fig)
                hist_list.append([histogram])
        context["hist_list"] = hist_list
    
    return render(request,"Histogram.html",context)
    

def sqrt_transformation(request):
    context={}
    if request.method=="POST":
        test_file = request.FILES.get(u'csv_file') #request.FILES has the file contents to be read. the get method gets the contents.
        if test_file: #Check if file is valid.
           df = pd.read_csv(test_file) #Read using pandas
           df=df.iloc[:,:-1]
           hist_list=[]
           for column in df.columns:
                df[column]=df[column]**(1/2)
                fig=plt.figure()
                plt.hist(df[column])
                plt.title(f'Transformed {column} Histogram')
                histogram=mpld3.fig_to_html(fig)
                hist_list.append([histogram])
        context["hist_list"] = hist_list

    return render(request,"SqrtTransformation.html",context)

def cube_root_transformation(request):
    context={}
    if request.method=="POST":
        test_file = request.FILES.get(u'csv_file') #request.FILES has the file contents to be read. the get method gets the contents.
        if test_file: #Check if file is valid.
           df = pd.read_csv(test_file) #Read using pandas
           df=df.iloc[:,:-1]
           hist_list=[]
           for column in df.columns:
               df[column]=df[column]**(1/3)
               fig=plt.figure()
               plt.hist(df[column])
               plt.title(f'Transformed {column} Histogram')
               histogram=mpld3.fig_to_html(fig)
               hist_list.append([histogram])
        context["hist_list"] = hist_list
        
    return render(request,"CubeRootTransformation.html",context)

def log_transformation(request):
    context={}
    if request.method=="POST":
        test_file = request.FILES.get(u'csv_file') #request.FILES has the file contents to be read. the get method gets the contents.
        if test_file: #Check if file is valid.
           df = pd.read_csv(test_file) #Read using pandas
           df=df.iloc[:,:-1]
           hist_list=[]
           for column in df.columns:
               df[column]=np.log(df[column]+0.2) #Adding 0.01 to avoid getting an invalid value upon logarithmic transformation
               fig=plt.figure()
               plt.hist(df[column].dropna().values)
               plt.title(f'Transformed {column} Histogram')
               histogram=mpld3.fig_to_html(fig)
               hist_list.append([histogram])
        context["hist_list"] = hist_list
        
    return render(request,"LogTransformation.html",context)

    
def algorithm_implementation(request):
    context={}
    epsilon=-0.2
    test_file = request.FILES.get(u'csv_file') #request.FILES has the file contents to be read. the get method gets the contents.
    if test_file: #Check if file is valid.
           df = pd.read_csv(test_file) #Read using pandas
           df=df.iloc[:,:-1]
           train_df,test_df=train_test_split(df,test_size=0.2,random_state=42,shuffle=True)
           train_features_mean=train_df.mean().to_numpy()
           train_features_variance=train_df.var().to_numpy()
           train_features_std=train_df.std().to_numpy()
           intermediate_computation=1/(np.sqrt(2*np.pi)*train_features_std)

           train_df_matrix=train_df.to_numpy()
           rows,columns=train_df_matrix.shape
           anomaly_list=[]
           normal_list=[]

           for i in range(rows):
               curr_example=train_df_matrix[i]
               intermediate_computation2=np.exp((curr_example-train_features_mean)**2/(2*train_features_variance))
               result=intermediate_computation*intermediate_computation2
               result_number=np.prod(result)
               if result_number > epsilon:
                   normal_list.append(result_number)
                   context['normal_stats']=len(normal_list)+0
    
               else:
                   anomaly_list.append(result_number)
                   context['anamoly_stats']=len(anomaly_list)+0
         
    return render(request,'Algorithm.html',context)    