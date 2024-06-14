import nltk

ans_path = './ans.txt'
result_path = './result.txt'

ans_file = open(ans_path, 'r')
result_file = open(result_path, 'r')
count = 0
for i in range(1000):
    ans_line = ans_file.readline().split('\t')[1]
    ans_set = set(nltk.word_tokenize(ans_line))
    result_line = result_file.readline().split('\t')[1]
    result_set = set(nltk.word_tokenize(result_line))
    if ans_set == result_set:
        count += 1
print("Accuracy is : %.2f%%" % (count * 1.00 / 10))
