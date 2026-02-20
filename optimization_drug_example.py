# Slide #1 answers: 
## Optimization to find the optimal drug dose: 
    # 1. Increasing the lambda sign represents a higher penalty (negative emphasis) on toxicity. Qualitatively, this causes the Net Effect peak to shift to a lower dosage and results in a lower maximum efficacy; this occurs because patient saftey is prioritized over maximum efficacy, as seen with the lower dosages that acheive lower maximum height values (lower efficacy and success of the drug). 
    # Conversely, decreasing lambda allows for a higher dosage and higher efficacy because the model is more tolerant of toxicity, which can lead to better outcomes for patients who can tolerate higher toxicity levels because the height of the graph is higher and the peak is shifted to the right, indicating a higher optimal dosage.
    
    # 2. Newton's method converges faster than the method of steepest ascent as seen with how both methods respectively converge to the value in 757 iterations and 5 iterations. This is because Newton's method uses second-order information (the curvature of the function) to make more informed updates, while the method of steepest ascent only uses first-order information (the gradient), which can lead to slower convergence, especially near the optimal point where the gradient becomes small.

    # 3. Changing the number of itterations can greatly affect the optimal value that the functions converge to. 
        # For instance, if we changed the number of itterations from 1000 to 100, 
        # the method of steepest ascent would converge to the optimal Escitalopram Dose of 3.95mg with 1000 itterations, but 
        # with only 100 itterations, it would converge to a suboptimal dose of 2.54 mg, which is much lower than the optimal dose previously predicted with the larger itterative limit of 1000. 
            # Percent change wise, it had a 35.7% decrease in the optimal dose, which is a significant change that could have major implications for patient outcomes if this suboptimal dose were used in practice. 

# Slide #2 answers:
##  Continuation of the prevous slide of questions and answers: 
    # 1.  The combined function is plotted within the code 
    # 2.  The combined drugs optimal dosage is 4.58 mg with a net optimal effect of 174.93% with the method of steepest ascent; 
         # Futhermore, with newton's method, the optimal dosage is 5.26 mg with a net 


# drug efficacy optimization example for BME 2315
# made by Lavie, fall 2025

#%% import libraries
import numpy as np
import matplotlib.pyplot as plt


#%% define drug models

# define toxicity levels for each drug (lambda values λ's) -- these are the weights that determine how much we penalize toxicity in our optimization for the diffrent drugs

metformin_lambda = 1.5  # 0.2

lisinopril_lambda = 1.2  # 0.8

escitalopram_lambda = 0.9 # 0.3

# Efficacy (E) = how much the drug helps the patient; 
# Toxicity (T) = how much the drug harms the patient. 
# Lambda (λ) = a weighting factor that determines how much we penalize toxicity in our optimization. 
#   A higher λ means we care more about minimizing toxicity, while a lower λ means we are more tolerant of toxicity in pursuit of efficacy.

def metformin(x):   # mild toxicity, moderate efficacy; where x is the dosage in mg
    efficacy = 0.8 * np.exp(-0.1*(x-5)**2) #
    toxicity = 0.2 * x**2 / 100
    return efficacy - metformin_lambda * toxicity
def lisinopril(x):  # strong efficacy, higher toxicity; where x is the dosage in mg
    efficacy = np.exp(-0.1*(x-7)**2)
    toxicity = 0.3 * x**2 / 80
    return efficacy - lisinopril_lambda * toxicity
def escitalopram(x):  # weaker efficacy, low toxicity; where x is the dosage in mg
    efficacy = 0.6 * np.exp(-0.1*(x-4)**2)
    toxicity = 0.1 * x**2 / 120
    return efficacy - escitalopram_lambda * toxicity

# Define the combined function -- can't just say "combined_drugs = metformin + lisinopril + escitalopram" because these are functions, not constants. We need to define a new function that calls each of the individual drug functions and sums their outputs for a given input x.
def combined_drugs(x):
    return metformin(x) + lisinopril(x) + escitalopram(x)




#%% plot drug efficacies
x = np.linspace(0, 15, 100)
fig, ax = plt.subplots(figsize=(10, 6))
plt.plot(x, metformin(x), label='Metformin', color='blue')
plt.plot(x, lisinopril(x), label='Lisinopril', color='orange')
plt.plot(x, escitalopram(x), label='Escitalopram', color='green')

# Combinted effect of all three drugs: 
# Assuming additive effects, the combined efficacy can be modeled using existing combined_drugs function defined above. 
plt.plot(x, combined_drugs(x), label='Combined Effect', color='red', linestyle='--', linewidth=2)
plt.legend()

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

# combined drugs -- code helped by VScode Copilot
opt_dose_combined, opt_effect_combined = steepest_ascent(combined_drugs, x0=1.0)
print(f"Steepest Ascent Method - Optimal Combined Drug Dose: {opt_dose_combined:.2f} mg")
print(f"Steepest Ascent Method - Optimal Combined Drug Effect: {opt_effect_combined*100:.2f}%")

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

# combined drugs
opt_dose_combined_nm, opt_effect_combined_nm = newtons_method(combined_drugs, x0=1.0)
print(f"Newton's Method - Optimal Combined Drug Dose: {opt_dose_combined_nm:.2f} mg")
print(f"Newton's Method - Optimal Combined Drug Effect: {opt_effect_combined_nm*100:.2f}%")

