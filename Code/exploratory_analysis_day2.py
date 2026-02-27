#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.optimize import curve_fit
#%%
# Load the data
data = pd.read_csv('Module-2-Epidemics-SIR-Modeling/Data/mystery_virus_daily_active_counts_RELEASE#1.csv', parse_dates=['date'], header=0, index_col=None)
#%%
# We have day number, date, and active cases. We can use the day number and active cases to fit an exponential growth curve to estimate R0.
# Let's define the exponential growth function
def exponential_growth(t, r):
    return np.exp(r * t)

# Copying over the plot from day 1
####Labeling 
plt.xlabel('Day')
plt.ylabel('Active Cases')
plt.title('Active Cases of Mystery Virus Over Time')

####Loading the data
plt.plot(data['day'], data['active reported daily cases'], marker='o', linestyle='-')

plt.show()
# Fit the exponential growth model to the data.
p0 = [0.1]  # Initial guess for the growth rate
params, cv = scipy.optimize.curve_fit(exponential_growth, data['day'], data['active reported daily cases'], p0=p0)

# Hint: Look up the documentation for curve_fit to see how to use it. You will need to provide the day numbers and active cases as inputs, and the exponential_growth function as the model to fit. 
# We'll use a handy function from scipy called CURVE_FIT that allows us to fit any given function to our data. 
# We will fit the exponential growth function to the active cases data. HINT: Look up the documentation for curve_fit to see how to use it.

# Approximate R0 using this fit
R0 = params[0]
print(f"Estimated R0: {R0}")

# Add the fit as a line on top of your scatterplot.
plt.plot(data['day'], data['active reported daily cases'], marker='o', linestyle='-', label='Data')
plt.plot(data['day'], exponential_growth(data['day'], params[0]), label='Exponential Fit')
plt.title("Fitted Exponential Growth Curve")
