import re

text = input('Enter the paragraph: ')
letter = input('Enter starting letter: ')

words = re.findall(r'\b' + re.escape(letter) + r'\w*', text, re.IGNORECASE)

print('Words beginning with the letter', letter, ':')

for word in words:
    print(word)
