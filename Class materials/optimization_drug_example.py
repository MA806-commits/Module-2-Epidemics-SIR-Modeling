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
    # 2.  The combined drugs optimal dosage is 5.26 mg with a net optimal effect of 181.70% with the method of steepest ascent; 
        # Futhermore, with newton's method, the optimal dosage is 5.26 mg with also a net optimal effect of 181.70%.
        # Both methods converge to the same optimal dosage and effect for the combined drugs, which suggests that both optimization techniques are effective in finding the optimal solution for this problem, and that this is the true optimal dosage and effect for the combined drugs given the defined models and parameters.

    # 3. Made the function below to find the lambda value for metformin that would give us an optimal dose of 5.26 mg, which is the optimal dose we found for the combined drugs. By adjusting the lambda value for metformin, we can see how it affects the optimal dose and find the lambda that allows us to achieve a similar optimal dose to that of the combined drugs, which can provide insights into how we might optimize metformin dosing in a clinical setting while considering its toxicity.
        # it is the last part of the code 


# Author: ChatGPT-4 and Google Gemini
# Source: https://chat.openai.com/ and https://gemini.google.com/
    # I want to come forth and acknowlege that I did use Generative AI with some of my understanding for the code and for some of the implenentation of the code. For instance, when working to fix some of the code errors present when I was tyring to generate the combined drug function's plot graphically. 
    # Furthermore, when makign the funtion to optimize the amount of lambda for metformin to acheive the optimal dose of 5.26 mg, I used Google Gemini to help me with the code implementation and understanding of how to approach this problem because I was struggling on how to set up the code to loop thourgh, test, and pick the specific lambda value. 
    # Lastly, I also used Google Gemini to help me understand the Benefit = Effiacy - Lambda * Toxicity equation, how to interpret it with the changes in the graphs, and what it means in a clinical setting in terms of how we might optimize drug dosing while considering toxicity. The writing and code implemmentation is my own, but I needed AI to help me understand the concepts and how to implement them best. 

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



# Question 3 of slide 2: Asked to choose one drug and find its lambda value to acheive the optimal dose value of 5.26 mg seen with the combined drugs, but we are only specifiying it to one drug (we will choose metformin for this example).

# 1. Setting the target to match the combined drugs optimal dosage of 5.26 mg, we will adjust the lambda value for metformin to find the optimal dose that matches this target.
target_dose = 5.26  # This is the 'Combined' peak we want to match

# 2. Creating a list of lambdas to test for the target dosage optimal for chosen drug metaformin within the optimal mix of combined drugs
# We will test 50 different 'penalty' levels from 0.1 to 2.0 -- range set of lambda values to test, from very low penalty (0.1) to high penalty (2.0), with 50 evenly spaced values in between. This allows us to see how different levels of toxicity penalty affect the optimal dosage and find the lambda that best matches our target dose.
    # the range of lambda values chosen are only from 0.1 to 2.0 to ensrure that we don't make the drug too toxic (ex. 100000) or too non-toxic (ex. 0.00001) because these extreme values would not be realistic in a clinical setting and would likely lead to either an unacceptably high toxicity or an unrealistically low toxicity that doesn't reflect real-world drug behavior. By testing a range of lambda values within this more reasonable range, we can find a balance that allows us to achieve the desired optimal dose while still considering patient safety.
    # Furthermore, we picked to go with values that were around 0-15mg of drugs with the range sets going from 0.1 to 2.0 to better represent the scenario of drug dosing seen with the graphs and to accuretly and more prescisely predict where the optimal for the drug (metformin) would be present along with the combined drugs optimal dosage.
    # The choice of 50 values provides a good balance between granularity and computational efficiency, allowing us to find a lambda value that closely matches the target dose without requiring an excessive number of calculations.

test_lambdas = np.linspace(0.1, 2.0, 50)

# Variables to store the best matching lambda and defining the smallest error allowed
best_l = 0 # -- this will store the lambda value that gives us the optimal dose closest to our target dose of 5.26 mg for metformin
smallest_error = 100 # -- this will store the smallest error (the absolute difference between the optimal dose for metformin and the target dose) that we have found so far. We initialize it to a large number (100) to ensure that any actual error we calculate will be smaller than this initial value, allowing us to update it with the first lambda we test and then continue to find smaller errors as we test more lambdas.
    # the smallest error will change and get smaller as we test diffrent lambda values. 

# 3. Looping though each Lambda value and finding the optimal dose for metformin with that specific lambda, then calculating how far away that optimal dose is from our target dose of 5.26 mg, and if it's the closest we've been so far, we save that lambda as our best lambda. This process allows us to find the lambda value that gives us an optimal dose for metformin that is closest to the target dose of 5.26 mg, which is the optimal dose we found for the combined drugs. 
# By doing this, we can see how adjusting the toxicity penalty (lambda) for metformin can help us achieve a similar optimal dose to that of the combined drugs, which can provide insights into how we might optimize metformin dosing in a clinical setting while considering its toxicity.
for l in test_lambdas:
    # A. Define Metformin using the current lambda (l)
    def temp_metformin(x):
        eff = 0.8 * np.exp(-0.1 * (x - 5)**2) # same function as before
        tox = 0.2 * x**2 / 100
        return eff - (l * tox)
    
    # B. Find where the peak is for this specific lambda
    # We use Newton's Method because it's the fastest 'climber'
    current_peak_dose, _ = newtons_method(temp_metformin, x0=5.0) 
        # current_peak_dose will give us the optimal dose for metformin with the current lambda (l) that we are testing. We start our search at x0=5.0 because we know from our previous analysis that the optimal dose for metformin is around 5 mg, so starting our search near this value will help us find the peak more efficiently.
        # _ is used to ignore the second value returned by newtons_method, which is the optimal effect at that dose, because we are only interested in the optimal dose for this part of the analysis.
    
    # C. Calculate how far away we are from the target (5.26)
    error = abs(current_peak_dose - target_dose)

    
    # D. If this is the closest we've been so far, save it!
    if error < smallest_error:
        smallest_error = error # update the smallest error to be the error we just calculated, because it's smaller than the previous smallest error we had, getting us closer to our target dose of combined drugs mix of 5.26 mg, with metformin retaining some of that optimized weight that is more accurate to its porportion to the combined drugs mix weight of 5.26 mg.
        best_l = l

print(f"To match the combined dose of {target_dose}mg...")
print(f"Metformin needs a Lambda of: {best_l:.4f}")