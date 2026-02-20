# drug efficacy optimization example for BME 2315
# made by Lavie, fall 2025

#%% import libraries
import numpy as np
import matplotlib.pyplot as plt


#%% define drug models

# define toxicity levels for each drug (lambda)
metformin_lambda = 0.5

lisinopril_lambda = 0.8

escitalopram_lambda = 0.3

def metformin(x):   # mild toxicity, moderate efficacy
    global metformin_lambda #forces new metformin function to view new lambda when running loop 
    efficacy = 0.8 * np.exp(-0.1*(x-5)**2)
    toxicity = 0.2 * x**2 / 100
    return efficacy - metformin_lambda * toxicity
def lisinopril(x):  # strong efficacy, higher toxicity
    efficacy = np.exp(-0.1*(x-7)**2)
    toxicity = 0.3 * x**2 / 80
    return efficacy - lisinopril_lambda * toxicity
def escitalopram(x):  # weaker efficacy, low toxicity
    efficacy = 0.6 * np.exp(-0.1*(x-4)**2)
    toxicity = 0.1 * x**2 / 120
    return efficacy - escitalopram_lambda * toxicity

def combined(x): 
    return metformin(x) + lisinopril(x) + escitalopram(x)

#%% plot drug efficacies
x = np.linspace(0, 15, 100)
fig, ax = plt.subplots(figsize=(10, 6))
plt.plot(x, metformin(x), label='Metformin', color='blue')
plt.plot(x, lisinopril(x), label='Lisinopril', color='orange')
plt.plot(x, escitalopram(x), label='Escitalopram', color='green')
plt.plot(x, combined(x), label='Combined Effect', color='red')
plt.title('Drug Efficacy vs Dosage')
plt.xlabel('Dosage (mg)')
plt.ylabel('Net Effect')
plt.legend()

# %% Find optimal dosages for each drug

# First method: Steepest Ascent using the update rule

# first, need the first derivative (gradient)
def gradient(f, x, h=1e-4):
    """Central difference approximation for f'(x)."""
    return (f(x + h) - f(x - h)) / (2*h)

def steepest_ascent(f, x0, h_step=0.1, tol=1e-6, max_iter=1000):
    x = x0 # update initial guess
    for i in range(max_iter):
        grad = gradient(f, x)
        x_new = x + h_step * grad     
        
        if abs(x_new - x) < tol:      # convergence condition, when solution is 0
            print(f"Converged in {i+1} iterations.")
            break
            
        x = x_new
    return x, f(x)

# metformin
opt_dose_metformin, opt_effect_metformin = steepest_ascent(metformin, x0=1.0)
print(f"Steepest Ascent Method - Optimal Metformin Dose: {opt_dose_metformin:.2f} mg")
print(f"Steepest Ascent Method - Optimal Metformin Effect: {opt_effect_metformin*100:.2f}%")

# lisinopril
opt_dose_lisinopril, opt_effect_lisinopril = steepest_ascent(lisinopril, x0=1.0)
print(f"Steepest Ascent Method - Optimal Lisinopril Dose: {opt_dose_lisinopril:.2f} mg")
print(f"Steepest Ascent Method - Optimal Lisinopril Effect: {opt_effect_lisinopril*100:.2f}%")

# escitalopram
opt_dose_escitalopram, opt_effect_escitalopram = steepest_ascent(escitalopram, x0=1.0)
print(f"Steepest Ascent Method - Optimal Escitalopram Dose: {opt_dose_escitalopram:.2f} mg")
print(f"Steepest Ascent Method - Optimal Escitalopram Effect: {opt_effect_escitalopram*100:.2f}%")

#Combined Effect 
opt_dose_combined, opt_effect_combined = steepest_ascent(combined, x0=1.0)
print(f"Steepest Ascent Method - Optimal Combined Dose: {opt_dose_combined: .2f} mg")
print(f"Steepest Ascent method - Optimal Combined Effect: {opt_effect_combined*100:.2f}%")

# %% Newton's method

# requires second derivative
def second_derivative(f, x, h=1e-4):
    """Central difference approximation for f''(x)."""
    return (f(x + h) - 2*f(x) + f(x - h)) / (h**2)

def newtons_method(f, x0, tol=1e-6, max_iter=1000):
    x = x0
    for i in range(max_iter):
        grad = gradient(f, x)
        hess = second_derivative(f, x)
        
        if hess == 0:  # avoid division by zero
            print("Zero second derivative. No solution found.")
            return x, f(x)
        
        x_new = x - grad / hess
        
        if abs(x_new - x) < tol:
            print(f"Converged in {i+1} iterations.")
            break
            
        x = x_new
    return x, f(x)

# metformin
opt_dose_metformin_nm, opt_effect_metformin_nm = newtons_method(metformin, x0=1.0)
print(f"Newton's Method - Optimal Metformin Dose: {opt_dose_metformin_nm:.2f} mg")
print(f"Newton's Method - Optimal Metformin Effect: {opt_effect_metformin_nm*100:.2f}%")                

# lisinopril
opt_dose_lisinopril_nm, opt_effect_lisinopril_nm = newtons_method(lisinopril, x0=1.0)
print(f"Newton's Method - Optimal Lisinopril Dose: {opt_dose_lisinopril_nm:.2f} mg")
print(f"Newton's Method - Optimal Lisinopril Effect: {opt_effect_lisinopril_nm*100:.2f}%")

# escitalopram
opt_dose_escitalopram_nm, opt_effect_escitalopram_nm = newtons_method(escitalopram, x0=1.0)
print(f"Newton's Method - Optimal Escitalopram Dose: {opt_dose_escitalopram_nm:.2f} mg")
print(f"Newton's Method - Optimal Escitalopram Effect: {opt_effect_escitalopram_nm*100:.2f}%")

# Combined Effect 
opt_dose_combined_nm, opt_effect_combined_nm = newtons_method(combined, x0=1.0)
print(f"Netwon's Method - Optimal Combined Dose: {opt_dose_combined_nm:.2f} mg")
#Why is this producing 0.00mg?
print(f"Newton's Method - Optimal Combined Effect: {opt_effect_combined_nm*100:.2f}%")


# %% Finding lambda that most closely matches the optimal dose for metformin (4.94mg)

#Test version of metformin, so that lambda can be changed  
def metformin_test(x,test_lmbda):
    efficacy = 0.8 * np.exp(-0.1 * (x - 5)**2)
    toxicity = 0.2 * x**2 / 100
    return efficacy - test_lmbda * toxicity

###Finding Optimal Lambda 
##Test multiple lambdas, choose one that gets optimal dose closest to 4.94mg 

#Range of Lambdas to Test 
lambda_candidates = [0.0, 0.1, 0.5, 0.8, 1.0, 5.0, 10.0]
target_dose = 4.94
best_lambda = lambda_candidates[0]
smallest_error = float('inf')

#Loop through each lambda 
for lmbda in lambda_candidates: 
    #New lambda function that allows for testing specific lambda values 
    temp_function = lambda x: metformin_test(x, test_lmbda=lmbda)
    #run steepest ascent on new function 
    opt_dose_temp, opt_effect_temp = steepest_ascent(temp_function, x0=1.0)
    error = abs(opt_dose_temp - target_dose)

    if error <smallest_error: 
        smallest_error = error 
        best_lambda = lmbda
print(f"Optimal Lambda Found: {best_lambda:.2f}")

#Re-run w/ "winning" lambda 
final_func = lambda x: metformin_test(x, test_lmbda=best_lambda)
opt_dose_metformin, opt_effect_metformin = steepest_ascent(final_func, x0=1.0)

print(f"Steepest Ascent (Optimized Lambda) - Optimal Metformin Dose: {opt_dose_metformin:.2f} mg")
print(f"Steepest Ascent (Optimized Lambda) - Optimal Metformin Effect: {opt_effect_metformin*100:.2f}%")

#Compare to combination drug 
def combined_calibrated(x):
    #use the calibrated optimal lambda for metformin, and original lambdas for other drugs 
    return metformin_test(x,best_lambda) + lisinopril(x) + escitalopram(x)

opt_dose_combined, opt_effect_combined = steepest_ascent(combined_calibrated, x0=1.0)

print(f"Combined Dose Calibration: {opt_dose_combined:.2f} mg")
print(f"Combined Effect Calibrated: {opt_effect_combined*100:.2f}%")