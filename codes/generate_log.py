import math
import numpy as np

# Function for calculating the laplacian of the gaussian at a given point and with a given variance
def log(x, y, sigma):

    numerator = ( (y**2)+(x**2)-2*(sigma**2) )
    denominator = ( (2*math.pi*(sigma**6) ))
    exponential = math.exp(-((x**2)+(y**2))/(2*(sigma**2)))

    return numerator*exponential/denominator

def generate_log(sigma):

    size = max(1,2*round(sigma*3)+1)
    w = math.ceil(float(size)*float(sigma))
    # If the dimension is an even number, make it odd
    if(w%2 == 0):
        w = w + 1

    log_kernel = []

    w_range = int(math.floor(w/2))
 
    for i in range(-w_range, w_range):

        for j in range(-w_range, w_range):

            log_kernel.append(log(i,j,sigma))

    log_kernel = np.array(log_kernel)
    log_kernel = np.reshape(log_kernel, (w-1,w-1))
    
    return log_kernel