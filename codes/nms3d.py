import numpy as np

def nms3d(binary_laplacian_scale_space, laplacian_scale_space):

    blss = np.copy(binary_laplacian_scale_space)

    lss = np.copy(laplacian_scale_space)

    for slice in range(len(blss)):
        # if 1st slice, then only compare with slice below

        if slice == 0:
            
            x = np.nonzero(blss[slice])

            for i in range(x[0].shape[0]):

                row_location = x[0][i]
                col_location = x[1][i]
                max_below = np.max(lss[slice+1, row_location-1:row_location+2, col_location-1:col_location+2])

                
                if (lss[slice, row_location,col_location] < max_below):

                    blss[slice, row_location, col_location] = 0

        # if last slice, then only compare with slice above

        if slice == len(blss)-1:

            x = np.nonzero(blss[slice])

            if len(x) == 1:

                return blss
            for i in range(x[0].shape[0]):

                row_location = x[0][i]
                col_location = x[1][i]

                max_above = np.max(lss[slice-1, row_location-1:row_location+2, col_location-1:col_location+2])
            
                if (lss[slice, row_location,col_location] < max_above) :

                    blss[slice, row_location, col_location] = 0

        # if slice in between two extremes(not included), compare with slices above and below

        else:

            x = np.nonzero(blss[slice])

            for i in range(x[0].shape[0]):

                row_location = x[0][i]
                col_location = x[1][i]

                #max value in above slice
                max_above = np.max(lss[slice-1, row_location-1:row_location+2, col_location-1:col_location+2])
                #max value in below slice
                max_below = np.max(lss[slice+1, row_location-1:row_location+2, col_location-1:col_location+2])

                if ((lss[slice, row_location,col_location] < max_below) or (lss[slice, row_location,col_location] < max_above)):

                    blss[slice, row_location, col_location] = 0

    return blss
