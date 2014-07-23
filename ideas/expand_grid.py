# 
# code from 
# http://stackoverflow.com/questions/12130883/r-expand-grid-function-in-python 
#
def expand_grid(x, y):
    xG, yG = np.meshgrid(x, y) # create the actual grid
    xG = xG.flatten() # make the grid 1d
    yG = yG.flatten() # same
    return pd.DataFrame({'x':xG, 'y':yG}) # return a dataframe
