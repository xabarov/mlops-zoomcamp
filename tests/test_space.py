
import math 


# Define the actual ranges for each parameter
ranges = {
    'max_depth': (4, 40),
    'learning_rate': (math.exp(-3), math.exp(-1)), 
    'reg_alpha': (math.exp(-5), math.exp(-2.5)),     
    'reg_lambda':  (math.exp(-6), math.exp(-2.5)),     
    'min_child_weight':  (math.exp(-1), math.exp(4)),     
}

# print min and max values of each parameter
for key in ranges:
    print(f'{key}: {ranges[key][0]} to {ranges[key][1]}')
