#Helper functions for my views.py funcitons

#Libraries to help with my computations.
import scipy.stats as stats
import numpy as np

def pdf_product(input_matrix,example_mean,example_std):
    epsilon=-0.00001
    rows,columns=input_matrix.shape
    anomaly_list=[]
    normal_list=[]
    for i in range(rows):
      curr_example=input_matrix[i]
      curr_example_pdf=stats.norm.pdf(curr_example,example_mean,example_std)
      anomaly_detection_result=np.prod(curr_example_pdf)
      if anomaly_detection_result>epsilon:
          normal_list.append(anomaly_detection_result)
      else:
          anomaly_list.append(anomaly_detection_result)

    return len(normal_list),len(anomaly_list)
    