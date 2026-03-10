#%%
# libraries that we will be using. 
import pandas as pd
import matplotlib.pyplot as plt # another math library
import numpy as np # we will be using numpy (as np) for numerical operations and applying it to our large data set for analysis.
import scipy # Scipy is another advance math library that will be useful for curve fitting.
from scipy.optimize import curve_fit # this is the function we will be using to fit our exponential growth curve to the data.

# Loading the data from Release #2 -- loading the data into a pandas dataframe.
data = pd.read_csv('C:/Users/kidus/OneDrive/Desktop/Computational BME/Module 02/Module-2-Epidemics-SIR-Modeling/Data/mystery_virus_daily_active_counts_RELEASE#1.csv')

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

N = 100000 # Total population (constant)
S0, E0, I0, R0 = N-5, 0, 5, 0 # Start with 5 infected people, 0 exposed, 0 recovered.
best_sse = float('inf') # We start with "infinite" error so the first attempt is always better.
best_params = (0, 0, 0) # Placeholder for our winners.

# We define ranges based on our earlier R0 calculation (around 0.12).
beta_opts = np.linspace(0.1, 1.0, 20)  # Try 20 values between 0.1 and 1.0
sigma_opts = np.linspace(0.1, 0.5, 10) # Try 10 values for incubation
gamma_opts = np.linspace(0.05, 0.2, 10) # Try 10 values for recovery

# The "Triple Loop": This tests every single combination of the parameters above.
for b in beta_opts:
    for s in sigma_opts:
        for g in gamma_opts:
            # Run the Euler engine using the current "guess" parameters.
            S, E, I, R = run_SEIR_euler(b, s, g, S0, E0, I0, R0, days, N)
            
            # SSE (Sum of Squared Errors): We subtract our model's 'I' from 
            # the real 'cases' data, square it (to remove negatives), and sum it up.
            sse = np.sum((I - cases)**2)
            
            # If this guess has a lower error than our previous best, save it!
            if sse < best_sse:
                best_sse = sse
                best_params = (b, s, g)

# Extract the winning parameters.
best_beta, best_sigma, best_gamma = best_params
print(f"Optimal parameters found: Beta={best_beta}, Sigma={best_sigma}, Gamma={best_gamma}")



#%%
# 2e. Plot the model-predicted infections over time compared to the data. This section should come from your python code after Data Release #2.




#%%
# 2f. Predict the day and amount of active cases at the peak of the epidemic spread. This section should come from your python code after Data Release #2.



