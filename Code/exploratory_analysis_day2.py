#%%
# libraries that we will be using. 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np # we will be using numpy (as np) for numerical operations and to define our exponential growth function (applying it to our large data set)
import scipy # Scipy is another advance math library that will be useful for curve fitting. 
from scipy.optimize import curve_fit # this is the function we will be using to fit our exponential growth curve to the data.

#%%
# Loading the data of the mystery virus outbreak. 
# Note: Kidus -- Using the same data loading code as in day 1, but I will be changing the file path to match where the file is located on my computer, and this is because of the issue I mentioned eariler.
# In the future, I will fix my error with storage saving and change the file path to match both mine and Makayla's file paths.
    # this is the path to the file on my computer: 'C:/Users/kidus/OneDrive/Desktop/Computational BME/Module 02/Module-2-Epidemics-SIR-Modeling/Data/mystery_virus_daily_active_counts_RELEASE#1.csv'
    # This is how the path is saved within Makayla's original start of the code: 'Module-2-Epidemics-SIR-Modeling/Data/mystery_virus_daily_active_counts_RELEASE#1.csv'

data = pd.read_csv('C:/Users/kidus/OneDrive/Desktop/Computational BME/Module 02/Module-2-Epidemics-SIR-Modeling/Data/mystery_virus_daily_active_counts_RELEASE#1.csv', parse_dates=['date'], header=0, index_col=None)

#%%

# We have day number, date, and active cases. We can use the day number and active cases to fit an exponential growth curve to estimate R0.
# Let's define the exponential growth function
def exponential_growth(t, r): # Within our function's parameters, t is the time (day number) and r is the growth rate, which is used to later estimate RO. 
    return np.exp(r * t) 

    # function returns the algebrically equivalent form of the exponential growth model: e^(rt), where e is the base of the natural logarithm, r is the growth rate, and t is time.
    # exp is a function from the numpy library that calculates the exponential of the input value. In this case, it calculates e raised to the power of (r * t).
    

# Copying over the plot from day 1
####Labeling 
plt.xlabel('Day')
plt.ylabel('Active Cases')
plt.title('Active Cases of Mystery Virus Over Time')

####Loading the data
plt.plot(data['day'], data['active reported daily cases'], marker='o', linestyle='-')

plt.show()

# Day 2 of the project: Fitting an exponential growth curve to the data to estimate R0.

# Fit the exponential growth model to the data.
p0 = [0.1]  # Initial guess for the growth rate (r) -- guessed 0.1 as the starting point for the growth rate because the virus seems to be growing at a high rate, but we are not sure at what magnitude this expansion occurs.

params, cv = scipy.optimize.curve_fit(exponential_growth, data['day'], data['active reported daily cases'], p0=p0)
    # the parameters of the scipy.optimize.curve_fit of the function are as follows: 
        # the funcion we want to fit (exponential_growth)
        # the independent variable (X) data (data['day'])
        # the dependent variable (Y) data (data['active reported daily cases'])
        # p0 is the initial guess for the parameters of the model (in this case, the growth rate r).
            # The function returns the optimal parameters (params) and the covariance of the parameters (cv).
            # The optimal parameter (params) are the values of r that best fit the exponential growth model to the data (it will be constantly changing to reflect exponential growth best). The covariance provides an estimate of the uncertainty of the parameter estimates. 

    # scipy.optimize.curve_fit does the heavy lifting of adjusting optimization us to see how an exponential growth model could fit within our graph; this is why we don't have to do newton's methods or method of steepest ascent to find the best fit for our curve.
    # curve_fit is a function that uses non-linear least squares to fit a function (in this case, our exponential growth function) to the data. 

# Hint: Look up the documentation for curve_fit to see how to use it. You will need to provide the day numbers and active cases as inputs, and the exponential_growth function as the model to fit. 
# We'll use a handy function from scipy called CURVE_FIT that allows us to fit any given function to our data. 
# We will fit the exponential growth function to the active cases data. HINT: Look up the documentation for curve_fit to see how to use it.

# Approximate R0 using this fit -- updating our intial estimate of R0 = 0.1 to our new estimate based on the better fitting of the exponential growth curve to the data.
R0 = params[0]
print(f"Estimated R0: {R0}")

# Add the fit as a line on top of your scatterplot -- plotting the original data points and the fitted exponential growth curve on the same graph to visually compare how well the model fits the data.
plt.plot(data['day'], data['active reported daily cases'], marker='o', linestyle='-', label='Data') # this is day 1 of the project, where we are plotting the original data points. We are also adding a label to this plot for the legend.
plt.plot(data['day'], exponential_growth(data['day'], params[0]), label='Exponential Fit') # this is day 2 of the project, where we are plotting the fitted exponential growth curve using the parameters obtained from curve_fit. We are also adding a label to this plot for the legend.

# same x and y labelsas before, but now we are adding a legend to differentiate between the original data points and the fitted curve.
plt.title("Active Cases of Mystery Virus Over Time ") 
plt.xlabel("Day")
plt.ylabel("Active Cases")
plt.legend() # Helps with differentiation between the data points graphed (day 1) and the exponential growth model fitted to the data (day 2). legend() function goes though the plotted elements and creates a legend (Map key) based on the labels we assigned to each plot. In this case, it will create a legend that shows which line corresponds to the original data points and which line corresponds to the exponential fit.
plt.show() 

# R0 value is the growth rate of the exponential curve, which we can use to estimate how quickly the virus is spreading. A higher R0 indicates a faster spread, while a lower R0 indicates a slower spread. By fitting the exponential growth curve to the data, we can get a more accurate estimate of R0, which can help us understand the dynamics of the outbreak and inform public health interventions.
    #  What we got (02/27/2026) -- R0: 0.12144675532571207

