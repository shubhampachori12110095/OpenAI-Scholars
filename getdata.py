import os, io, random, re
from string import punctuation

output = io.open("data-fs2.txt", 'w')
labels = io.open("label-fs2.txt", 'w')

output2 = io.open("data-fs.txt", 'w')
labels2 = io.open("label-fs.txt", 'w')

regex = re.compile('[%s]' % re.escape(punctuation))


def get_data(file_1, file_2):
    with io.open(file_1, 'r', encoding="utf-8") as engFile, io.open(file_2, 'r', encoding="utf-8") as spanFile:
        eng = engFile.readlines()
        span = spanFile.readlines()

        engR = eng[:]
        spanR = span[:]
        random.shuffle(engR), random.shuffle(spanR)
        
        for (el, sl, elR, slR) in zip(eng, span, engR, spanR):
            regex.sub('', el)
            regex.sub('', elR)
            regex.sub('', sl)
            regex.sub('', slR)
            
            output.write("[start] " + el.strip() + " \t " + sl.strip() + " [end]\n")
            labels.write(u"positive\n")
            output.write("[start] " + elR.strip() + " \t " + slR.strip() + " [end]\n")
            labels.write(u"negative\n")
        engFile.close()
        spanFile.close()

get_data("french.txt", "span.txt")

with io.open('data-fs2.txt', 'r', encoding="utf-8") as f:
    sentences = f.readlines()
with io.open('label-fs2.txt', 'r') as f:
    labels = f.readlines()

#shuffle data
temp = list(zip(sentences, labels))
random.shuffle(temp)
sentences, labels = zip(*temp)

for item in sentences:
  output2.write(item)

for item in labels:
  labels2.write(item)

os.remove("data-fs2.txt")
os.remove("label-fs2.txt")