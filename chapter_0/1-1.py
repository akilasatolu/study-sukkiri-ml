height = input('Enter your height in cm: ')
weight = input('Enter your weight in kg: ')
bmi = int(weight) / (int(height) / 100) ** 2
bmi_as_int = int(bmi)
print(bmi_as_int)