#%%
# libraries that we will be using. 
import pandas as pd
import matplotlib.pyplot as plt # another math library
import numpy as np # we will be using numpy (as np) for numerical operations and applying it to our large data set for analysis.
import scipy # Scipy is another advance math library that will be useful for curve fitting.
from scipy.optimize import curve_fit # this is the function we will be using to fit our exponential growth curve to the data.

# Loading the data from Release #2 -- loading the data into a pandas dataframe.
data = pd.read_csv('C:\\Users\\kidus\\OneDrive\\Desktop\\Computational BME\\Module 02\\Module-2-Epidemics-SIR-Modeling\\Data\\mystery_virus_daily_active_counts_RELEASE#2.csv')

# Why initialize 'days' and 'cases' as separate variables?
# 1. Accessibility: It is faster to type 'days' than 'data["day"]' repeatedly.
# 2. Compatibility: Many math functions (like our Euler loop) work best with 
#    simple NumPy arrays (.values) rather than complex Pandas Series objects.
days = data['day'].values
cases = data['active reported daily cases'].values


#%%     
# Objectives for the mini assigment: 
# 2c. Use Euler's method to solve the SEIR model. This section should come from your python code after Data Release #2. 
# This function follows the logic: "Next State = Current State + (Change * Step)" -- Euler's method way of thinking and approximating the solution. 

# As directed by the class notes, we will be using the SEIR model, which has 4 groups: Susceptible (S), Exposed (E), Infectious (I), and Recovered (R).
# To intialize and approximate a SEIR model, need to define the parameters: beta (infection rate), sigma (rate of moving from exposed to infectious), gamma (recovery rate), and the initial population counts for each group (S0, E0, I0, R0).
    # Furthermore, empty arrays are greated to hold the results for S, E, I, and R over days. The first index of each array is set to the initial population counts. 
    # Then, a loop iterates through each day, calculating the change in each group (dS, dE, dI, dR) based on the SEIR equations and updates the counts for the next day using Euler's method. 
    # Finally, the function returns the full history of S, E, I, and R over time.

def run_SEIR_euler(beta, sigma, gamma, S0, E0, I0, R0, timepoints, N):
    # We create empty arrays (filled with zeros) to hold our results for each day.
    # This pre-allocates memory and makes the code run much faster.
    # Author note: Generative AI (Gemini) helped me write parts of this functions, like intalizing the arrays to the length and understanding why (to fit the entire arrays of possible timepoints); how exactly to implement the counter for the loop and update the change values (dS, dE, dI, dR) and how to update the next day values using Euler's method.
    # I still had a hand at writing the code and understanding the logic, but had trouble with some syntax and implimentation of the code. 
    # Citation: https://gemini.google.com/app 

    S = np.zeros(len(timepoints))
    E = np.zeros(len(timepoints))
    I = np.zeros(len(timepoints))
    R = np.zeros(len(timepoints))
    
    # We set the first index [0] to our starting population counts.
    S[0] = S0
    E[0] = E0
    I[0] = I0
    R[0] = R0
    
    # 'h' is our step size. Since our data is recorded daily, h = 1 day.
    h = 1 
    
    # We loop through time. We stop at 'len - 1' because the last day is calculated by the day before it.
    # All equations present from class notes. 
    for t in range(len(timepoints) - 1):
        
        # dS: Change in Susceptible. 
        dS = -beta * S[t] * I[t] / N
        
        # dE: Change in Exposed. 
        dE = (beta * S[t] * I[t] / N) - (sigma * E[t])
        
        # dI: Change in Infectious. ]
        dI = (sigma * E[t]) - (gamma * I[t])
        
        # dR: Change in Recovered. This is simply the number of people who got better.
        dR = (gamma * I[t])
        
        # EULER UPDATE: We take the current value and add the (rate of change * time step).
        # This is essentially drawing a tiny straight line to the next data point.
        S[t+1] = S[t] + h * dS
        E[t+1] = E[t] + h * dE
        I[t+1] = I[t] + h * dI
        R[t+1] = R[t] + h * dR
        
    return S, E, I, R # We return the full history of the outbreak for all 4 groups.


#%%
# 2d. Fit the SEIR model to the data by changing beta, gamma, and sigma. This section should come from your python code after Data Release #2. 
# This is the optimization section. We will be trying to find the best parameters (beta, sigma, gamma) that make our SEIR model's predicted infections (I) match the real data (cases) as closely as possible.
# We don't know the real beta/sigma/gamma, so we try a bunch to see what fits best.

N = 17000 # Total population (constant given from slides -- this is becasue the disease started at UVA within their 17,000 person population)) 
    #-- Start with 5 infected people (I0), 0 exposed (E0), 0 recovered (R0), and suseptible population is N-5 (S0).
S0 = N - 5 
E0 = 0 
I0 = 5 
R0 = 0 

# 'best_sse' starts at infinity. Our goal is to find a beta/sigma/gamma combo that brings this number as close to zero as possible.
# Author note: Had trouble on what to set the initial value of best_sse to, but after looking at the code and understanding that we are trying to minimize the error (SSE), it made sense to set it to infinity so that any calculated SSE would be smaller than this initial value.
# Utilized Generative AI (Gemini) to understand the logic of setting the initial value of best_sse to infinity and how to implement the grid search for finding the best parameters.
# Citation: https://gemini.google.com/app

best_sse = float('inf')  # possible error source??
best_params = (0, 0, 0) # This will hold our winning settings.
sse_history = []        # As per pseudocode, we track every error calculation.

# DEFINE THE SEARCH RANGES (Using your R0 = 0.12 as a guide -- tip from class and R0 value of 0.12 was what we estimated from previous assigment)
# Logic: Since R0 = beta / gamma, and your R0 is roughly 0.12, the true transmission rate (beta) should be about 12% of the recovery rate (gamma).
# We use this "Compass" to set ranges that are realistic, rather than searching through random, massive numbers.

# Generative AI (Gemini) helped me understand how to set the search ranges for beta, sigma, and gamma based on our estimated R0 value and the relationships between these parameters.
# Needed to also understand what the .linspace function does and how to use it to create a range of values for each parameter. 
    # .linspace creates an array of evenly spaced values between a specified start and end point. In this case, we are creating 10 values for each parameter (beta, sigma, gamma) within the defined ranges that are guided by our estimated R0 value and the biological plausibility of the parameters.
    # its parameters are as follows: the start value, the end value, and the number of values to generate. For example, np.linspace(0.01, 0.2, 10) generates 10 values between 0.01 and 0.2 for beta.
# Citation: https://gemini.google.com/app

beta_range = np.linspace(0.05, 0.15, 50)  # Guided by R0: looking for small beta. R0 = beta / gamma, so if R0 is 0.12 and we want to find a beta that is about 12% of gamma, we can set up a range for beta that is small, such as from 0.05 to 0.15, which allows us to explore values around our estimated R0 while keeping the search focused on biologically plausible parameters. We are also generating 50 values within this range to have a good resolution for our grid search.
sigma_range = np.linspace(0.1, 0.3, 50)  # Searching for the incubation speed. This range is chosen based on typical values for the incubation period of infectious diseases, which can vary but often falls within a certain range. By setting sigma to vary between 0.1 and 0.3, we are exploring a range of incubation speeds that could be realistic for the mystery virus, allowing us to find the best fit for our model based on the observed data. We are also generating 50 values within this range to have a good resolution for our grid search.
gamma_range = np.linspace(0.05, 0.5, 50) # Searching for the recovery speed.

# Grid search: We will test every combination of beta, sigma, and gamma within our defined ranges to see which one fits the data best.
# We test 1,000 different combinations of biological parameters.
for b in beta_range:
    for s in sigma_range:
        for g in gamma_range:
            
            # Use the Euler function from 2c to draw a "Predicted" line.
            # We use 'days' (from our CSV) as our x-axis.
            S_m, E_m, I_m, R_m = run_SEIR_euler(b, s, g, S0, E0, I0, R0, days, N)
            
            # 4. Calculate the error (SSE)-- We want to see how far off our model's predicted infections (I_m) are from the real data (cases).
            # We subtract our predicted infections (I_m) from real cases (cases).
            # We square it to make every error positive and penalize big misses.
            current_sse = np.sum((cases - I_m)**2)
                # Doccumentation: ** = power operator in python. 
                # citation: https://softwareengineering.stackexchange.com/questions/131403/what-is-the-name-of-in-python

            
            # Save error to the history list for tracking.
            sse_history.append(current_sse)
            
            # 5. Deterimine the best fit (winner) -- If this is the best fit we've seen, we save the parameters and the error.
            # If the current_sse is the smallest we've seen, it's our new "Best Fit."
            # This combination of b, s, and g is the most biologically accurate.
            if current_sse < best_sse:
                best_sse = current_sse
                best_params = (b, s, g) # storing the best parameters (beta, sigma, gamma) in a tuple for easy access later on.

# Returning the best paramters (beta, sigma, gamma) with the lowest SSE error. 
# These three numbers are the "Biological Profile" of the mystery virus.
best_beta, best_sigma, best_gamma = best_params

print(f"Grid Search Complete. Lowest SSE: {best_sse}")
print(f"The best fit matches your R0 guidance with Beta={best_beta} and Gamma={best_gamma}")

#%%
# 2e. Plot the model-predicted infections over time compared to the data. This section should come from your python code after Data Release #2.

# Choose a timeframe that captures the full "hill" of the infection
# You can change this number (e.g., 100, 200, 300) until the plot looks right!
total_prediction_days = 150 
prediction_time = np.arange(0, total_prediction_days, 1) 
    # Doccumentation purposes: 
        # np.arange creates an array of values starting from the first parameter (0) up to but not including the second parameter (total_prediction_days), with a step size defined by the third parameter (1). In this case, it creates an array of integers from 0 to 149, which represents the days for our prediction timeline.
        # This is important becasue we want to predict the number of active cases over a specific time frame that captures the full progression of the epidemic, including the rise and fall of active cases. 
        # By adjusting total_prediction_days, we can ensure that our plot includes the entire "hill" of the infection curve, allowing us to visualize how well our model fits the observed data over time.
        # Citation: https://gemini.google.com/app

# Run the model with our 'best' settings from 2d
S_p, E_p, I_p, R_p = run_SEIR_euler(best_beta, best_sigma, best_gamma, S0, E0, I0, R0, prediction_time, N)

# --- 2e: Plotting ---
plt.figure(figsize=(10, 6))
plt.scatter(days, cases, color='black', label='Observed Data', zorder=5) # Real dots
plt.plot(prediction_time, I_p, color='red', label='Model Prediction', linewidth=2) # Our line
plt.title(f"Outbreak Projection Over {total_prediction_days} Days")
plt.xlabel("Days")
plt.ylabel("Active Cases")
plt.legend()
plt.show()



#%%
# 2f. Predict the day and amount of active cases at the peak of the epidemic spread. This section should come from your python code after Data Release #2.
# To find the peak of the epidemic, we look for the maximum number of active cases (I_p) predicted by our model and the corresponding day.
# I needed help from generative AI to spot the specific syntax for finding the maximum value and its corresponding index in the array of predicted active cases (I_p).
# I used the np.max function to find the maximum value of active cases, and the np.argmax function to find the index of that maximum value, which corresponds to the day of the peak.
# Citation: https://gemini.google.com/app

# I_p is the array of predicted active cases over time from our SEIR model. We want to find the maximum value in this array, which represents the peak number of active cases during the epidemic spread.
# This will help us understand the approximate severity of outbreak at its worst point, a value derived from here will be utilized to calculate relative error. 

peak_val = np.max(I_p) # corresponds to the maximum number of active cases predicted by our model, which represents the peak of the epidemic spread. This value indicates how many active cases we can expect at the height of the outbreak according to our SEIR model with the best-fitting parameters.
    # peak_val is also the appromimate number of max active cases we can expect at the height of the outbreak according to our SEIR model with the best-fitting parameters.
    # will use this approximate value to compare to the real data and calculate the relative error in the next step (in class activity 03/10/2026).
peak_day = np.argmax(I_p) # corresponed with the index of the maximum value in I_p, which gives us the day number of the peak. 

print(f"The epidemic is predicted to peak on Day {peak_day} with {int(peak_val)} active cases.")

#%% 

# in class activity: 03/10/2026 -- Find the true % relative error between the data and your model prediction at the peak day (compare approximate data value peak_val to the real/true data value on the peak day from the csv file mystery_virus_daily_active_counts_RELEASE#3.csv).
# Load the true data from the CSV file

# Load the True Data -- Ensure the column names match CSV (e.g., 'active reported daily cases' and 'day')
true_data = pd.read_csv("C:\\Users\\kidus\\OneDrive\\Desktop\\Computational BME\\Module 02\\Module-2-Epidemics-SIR-Modeling\\Data\\mystery_virus_daily_active_counts_RELEASE#3.csv")

# 3. Find the Actual Peak in the Real Data
# We look for the maximum value in the actual data column
actual_peak_val = true_data['active reported daily cases'].max() # gives the true maximum number of active cases reported in the real data from data release #3.
actual_peak_day_index = true_data['active reported daily cases'].idxmax() # gives the index of the maximum value in the 'active reported daily cases' column, which corresponds to the day of the peak in the real data.
actual_peak_day = true_data.iloc[actual_peak_day_index]['day']

# 4. Calculate Relative Errors
# Relative Error for Peak Magnitude (Value)
val_relative_error = abs((actual_peak_val - peak_val) / actual_peak_val) * 100

# Relative Error for Peak Timing (Day)
# Using the same formula: (True - Approx) / True
day_relative_error = abs((actual_peak_day - peak_day) / actual_peak_day) * 100

# 5. Output Results
print(f"--- Model Predictions ---")
print(f"Predicted Peak: Day {peak_day} with {int(peak_val)} cases.")

print(f"\n--- True Data Observations ---")
print(f"Actual Peak: Day {actual_peak_day} with {actual_peak_val} cases.")

print(f"\n--- Error Metrics ---")
print(f"Relative Error (Cases): {val_relative_error:.2f}%") # relative error in y axis (number of cases)
print(f"Relative Error (Timing): {day_relative_error:.2f}%") # relative error in x axis (day of the peak)
