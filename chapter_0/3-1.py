number = input('Enter a number: ')
if int(number) % 2 == 0:
    print('The number is even.')
else:
    print('The number is odd.')

print('============================')

string = input('Enter a word: ')
if string == 'こんにちは':
    print('ようこそ！')
elif string == '景気は？':
    print('ぼちぼちです')
elif string == 'さようなら':
    print('お元気で！')
else:
    print('どうしました？')