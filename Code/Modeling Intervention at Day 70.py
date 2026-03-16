#%%
#NOTE: majority of the code below is the same as from the "Euler_Optimization_SEIR_model", so this is not as heavily commented, as it would be repetitive 
# LIBRARIES
import os  
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np 
import scipy 
from scipy.optimize import curve_fit 

# Loading the data from Release #2
base_dir = os.path.dirname(__file__) 
path_to_data = os.path.join(base_dir,'..','Data','mystery_virus_daily_active_counts_RELEASE#2.csv') 
data = pd.read_csv(path_to_data)  

# SETTING UP THE SEIR MODEL
days = data['day'].values
cases = data['active reported daily cases'].values

def run_SEIR_euler(beta, sigma, gamma, S0, E0, I0, R0, timepoints, N): 

    S = np.zeros(len(timepoints))
    E = np.zeros(len(timepoints))
    I = np.zeros(len(timepoints))
    R = np.zeros(len(timepoints))
# Emtpy array help with memory allocation, leave at 0
    S[0] = S0
    E[0] = E0
    I[0] = I0
    R[0] = R0
    
    h = 1 # step size = one day 
    
    for t in range(len(timepoints) - 1):
        
        # dS: Change in Susceptible. 
        dS = -beta * S[t] * I[t] / N
        
        # dE: Change in Exposed. 
        dE = (beta * S[t] * I[t] / N) - (sigma * E[t])
        
        # dI: Change in Infectious. ]
        dI = (sigma * E[t]) - (gamma * I[t])
        
        # dR: Change in Recovered. This is simply the number of people who got better.
        dR = (gamma * I[t])
        
        S[t+1] = S[t] + h * dS
        E[t+1] = E[t] + h * dE
        I[t+1] = I[t] + h * dI
        R[t+1] = R[t] + h * dR
        
    return S, E, I, R 

# SEIR POPULATION FOR VTECH
N = 31500 
S0 = N - 5 
E0 = 0 
I0 = 5 
R0 = 0 

best_sse = float('inf')  
best_params = (0, 0, 0) 
sse_history = []       

# Parameter values should stay the same for VTech and UVA data (same disease, same parameters)
beta_range = np.linspace(0.01, 0.5, 20)  
sigma_range = np.linspace(0.01, 0.5, 20)  
gamma_range = np.linspace(0.01, 0.1, 20)

#GRID SEARCH: Going through all combinations of beta, sigma, and gamma within the range that fits data best 
for b in beta_range:
    for s in sigma_range:
        for g in gamma_range:
            
            S_m, E_m, I_m, R_m = run_SEIR_euler(b, s, g, S0, E0, I0, R0, days, N)
            #CALCULATING AND SAVING SSE ERROR
            current_sse = np.sum((cases - I_m)**2)
            
            sse_history.append(current_sse)
            
            if current_sse < best_sse:
                best_sse = current_sse
                best_params = (b, s, g) #BEST PARAMETERS

best_beta, best_sigma, best_gamma = best_params
calculated_R0 = best_beta / best_gamma 
incubation_period = 1 / best_sigma 
infectious_period = 1 / best_gamma 

#PRITNING PARAMETERS
print(f"Grid Search Complete. Lowest SSE: {best_sse}")
print(f'--- Optimized Parameters ---')
print(f'Best Beta: {best_beta:.2f} (Transmission Rate)')
print(f'Best Sigma: {best_sigma:.2f} (Incubation Rate)')
print(f'Best Gamma: {best_gamma:.2f}(Recovery Rate)')
print(f'---Biological Calculations---')
print(f"Calculated R0: {calculated_R0:.2f}")
print(f'Incubation Period: {incubation_period:.1f} days')
print(f'Infectious Period: {infectious_period:.1f} days')

#PLOTTING PREDICTED MODEL OVER THE DATA 
total_prediction_days = 120 #Total prediction is 120 days 
prediction_time = np.arange(0, total_prediction_days, 1) 

S_p, E_p, I_p, R_p = run_SEIR_euler(best_beta, best_sigma, best_gamma, S0, E0, I0, R0, prediction_time, N)

#PLOTTING
plt.figure(figsize=(10, 6))
plt.scatter(days, cases, color='black', label='Observed Data', zorder=5) # Data Points
plt.plot(prediction_time, I_p, color='orange', label='Model Prediction', linewidth=2) # Predicted Graph
plt.title(f"VTech: Outbreak Projection Over {total_prediction_days} Days (NO INTERVENTION)")
plt.xlabel("Days")
plt.ylabel("Active Cases")
plt.legend()
plt.show()

#PREDICTING PEAK DAY AND CASES 
peak_val = np.max(I_p) 
peak_day = np.argmax(I_p)

print(f"The epidemic is predicted to peak on Day {peak_day} with {int(peak_val)} active cases.")

#NOTE: We will not be calculating error because we are using UVA data to predict VTech. Due to the vastly different undergraduate population, an error would be useless to calculate. 
#%% ######################################################################################### 
#MODELING INTERVENTIONS
def run_SEIR_VT_interventions(beta, sigma, gamma, S0, E0, I0, R0, timepoints, N, intervention_type=None):
    S = np.zeros(len(timepoints))
    E = np.zeros(len(timepoints))
    I = np.zeros(len(timepoints))
    R = np.zeros(len(timepoints))

    S[0] = S0
    E[0] = E0
    I[0] = I0
    R[0] = R0

    h = 1 #Step = 1 day 

    for t in range(len(timepoints)-1): 
        #INTERVENTION #1: Immediate Masking Mandate Implemented from Day 70 Onward (reduces transmission by 40%)  
        if intervention_type == "mask" and t>= 70: 
            current_beta = beta * (1- (0.40 * 0.80)) #40% reduction w/ 80% compliance 
        else: 
            current_beta = beta         

        #INTERVENTION #2: Vaccine Rollout (vaccinate 1,000 students on days 70, 80, and 90)
        vaccine_efficacy = 0.80 #Estimated 80% efficacy of vaccine 
        doses_per_day = 1000 
        protection_delay = 10 #10 day delay before vaccine takes effect 
        #Creating the vaccinated bucket 
        if 'vaccine_queue' not in locals(): 
                vaccine_queue = {
                    70 + protection_delay: int(doses_per_day * vaccine_efficacy),
                    80 + protection_delay: int(doses_per_day * vaccine_efficacy),
                    90 + protection_delay: int(doses_per_day * vaccine_efficacy)
                }
        if t in vaccine_queue: 
            num_protected = vaccine_queue[t] #each day checking for number of immune people

            S[t] = max(0,S[t] - num_protected)
            R[t] += num_protected
       
        #INTERVENTION #3: Testing + Quarantine Starting Day 70 (reduced infectious period by 2 days due to delays in testing and low compliance)
        natural_infectious_period = 1/ gamma #regular infection period
        reduction_days = 2 #reduction due to proper testing+quarentine 
        compliance = 0.50 # 50% effective compliance (78% isolate * 64% correctly)

        if intervention_type == 'quarentine' and t >= 70: 
            quarentine_period = natural_infectious_period - reduction_days #period for people who properly isolate 
            average_period = (compliance * quarentine_period) + ((1-compliance) * natural_infectious_period) # (50% * 8 days (50% * 10 days) = 9 day average
            current_gamma = 1 / average_period
        else: 
            current_gamma = gamma 

            # ODE Equations 
        dS = -current_beta * S[t] * I[t] / N 
        dE = (current_beta * S[t] * I[t] / N) - (sigma * E[t])
        dI = (sigma * E[t]) - (current_gamma * I[t])
        dR = (current_gamma * I[t]) 

            # Euler Equations
        S[t+1] = S[t] + dS
        E[t+1] = E[t] + dE
        I[t+1] = I[t] + dI 
        R[t+1] = R[t] + dR 
        
    return S, E, I, R 

#%%
#GENERATE DATA FOR EACH SCENARIO USING BEST PARAMETERS FROM PREVIOUS GRID SEARCH 
S_mask, E_mask, I_mask, R_mask = run_SEIR_VT_interventions(best_beta, best_sigma, best_gamma, S0, E0, I0, R0, prediction_time, N, intervention_type='mask')
S_vaccine, E_vaccine, I_vaccine, R_vaccine = run_SEIR_VT_interventions(best_beta, best_sigma, best_gamma, S0, E0, I0, R0, prediction_time, N, intervention_type='vaccine')
S_quarentine, E_quarentine, I_quarentine, R_quarentine = run_SEIR_VT_interventions(best_beta, best_sigma, best_gamma, S0, E0, I0, R0, prediction_time, N, intervention_type='quarentine')

#PLOTTING COMPARISON
plt.figure(figsize=(10,6))
plt.plot(prediction_time, I_p, label="No Intervention", color='orange', linewidth=2)
plt.plot(prediction_time, I_mask, label="Mask Mandate", linestyle='--')
plt.plot(prediction_time, I_vaccine, label="Vaccine Rollout (Day 70,80,90)", linestyle=':')
plt.plot(prediction_time, I_quarentine, label="Quarantine", linestyle='dotted')

plt.axvline(x=70, color='gray', alpha=0.5, label='Intervention Start') #creates a vertical line at Day 70 
plt.title('VT Outbreak: Comparison of Different Interventions')
plt.xlabel('Days')
plt.ylabel('Active Cases')
plt.legend()
plt.show() 

#FINDING AND PRINTING PEAK VALUES
peak_val_no_intervention = np.max(I_p)
peak_val_mask = np.max(I_mask)
peak_val_vaccine = np.max(I_vaccine)
peak_val_quarentine = np.max(I_quarentine)

peak_day_no_intervention = np.argmax(I_p)
peak_day_mask = np.argmax(I_mask)
peak_day_vaccine = np.argmax(I_vaccine)
peak_day_quarentine = np.argmax(I_quarentine)

print(f"---Values for Peak Day and Cases---")
print(f'No Intervention - Day: {peak_day_no_intervention} Cases: {peak_val_no_intervention:.0f}')
print(f'Mask Mandate - Day: {peak_day_mask} Cases: {peak_val_mask:.0f}')
print(f'Vaccine Rollout - Day: {peak_day_vaccine} Cases: {peak_val_vaccine:.0f}')
print(f'Quarantine and Testing - Day: {peak_day_quarentine} Cases: {peak_val_quarentine:.0f}')