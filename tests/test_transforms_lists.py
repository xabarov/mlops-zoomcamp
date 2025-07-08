import yaml

# Load the YAML file
with open('params.yaml', 'r') as file:
    params = yaml.safe_load(file)

# Access the categorical and numerical lists
categorical_features: list[str] = params['transform']['features']['categorical']
numerical_features: list[str] = params['transform']['features']['numerical']

# Print the lists to verify
print(categorical_features)
print(numerical_features)