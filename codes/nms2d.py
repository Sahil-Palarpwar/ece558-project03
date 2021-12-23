import numpy as np

def nms2d(slice, threshold):

    copy_slice = np.zeros_like(slice)


    rows = slice.shape[0]
    cols = slice.shape[1]
    #print(slice.shape)
    max_value = 0

    for row in range(1,rows-1):

        for col in range(1,cols-1):
            
            # print("current row is:", row)
            # print("current column is:", col)
            max_value = np.max(slice[row-1:row+2,col-1:col+2])
            # print(max_value, slice[row, col])

            if max_value == slice[row,col] and max_value > threshold:
                # print("Comdiyion")
                copy_slice[row,col] = 1

    return copy_slice


