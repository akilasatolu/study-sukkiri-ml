def check_leap_year(year):
    if (year % 4 == 0) and (year % 400 == 0 or year % 100 != 0):
        return True
    else:
        return False
print(check_leap_year(2020))  # True
print(check_leap_year(1900))  # False
print(check_leap_year(2000))  # True
print(check_leap_year(2024))  # True

