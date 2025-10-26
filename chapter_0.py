# 1-1
# height = input('Enter your height in cm: ')
# weight = input('Enter your weight in kg: ')
# bmi = int(weight) / (int(height) / 100) ** 2
# bmi_as_int = int(bmi)
# print(bmi_as_int)


# 2-2
# scoreList = []
# japanese = int(input('Enter your Japanese score: '))
# math = int(input('Enter your Math score: '))
# english = int(input('Enter your English score: '))
# scoreList.append(japanese)
# scoreList.append(math)
# scoreList.append(english)
# totalScore = sum(scoreList)
# print(scoreList)
# print(totalScore)


# 3-1
# number = input('Enter a number: ')
# if int(number) % 2 == 0:
#     print('The number is even.')
# else:
#     print('The number is odd.')

# print('============================')

# string = input('Enter a word: ')
# if string == 'こんにちは':
#     print('ようこそ！')
# elif string == '景気は？':
#     print('ぼちぼちです')
# elif string == 'さようなら':
#     print('お元気で！')
# else:
#     print('どうしました？')


# 4-1
# number = 10
# for i in range(number):
#     print(number - i, end=' ')
# print('ゴー!! シューーーート!!!!')



# 4-2
# data = [71, 67, 73, 61, 79, 59, 83, 87, 72, 79]
# scores = []
# for d in data:
#     scores.append(d)

# final_scores = []
# for s in scores:
#     final_scores.append(0.8 * s + 20)

# avarage_score = sum(final_scores) / len(final_scores)
# print(avarage_score)



# 5-1
def check_leap_year(year):
    if (year % 4 == 0) and (year % 400 == 0 or year % 100 != 0):
        return True
    else:
        return False
print(check_leap_year(2020))  # True
print(check_leap_year(1900))  # False
print(check_leap_year(2000))  # True
print(check_leap_year(2024))  # True