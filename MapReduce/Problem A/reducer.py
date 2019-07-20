#!/usr/bin/env python
import sys
import pdb
 
wordcount = {}
 
for inputt in sys.stdin:
    inputt = inputt.strip()
    word, count = inputt.split(' ', 1)
#    print word
    count = int(count)
    try:
        wordcount[word] = wordcount[word]+count
    except:
        wordcount[word] = count
 
ans = []
for word in wordcount.keys():
    ans.append([(wordcount[word]),word])

ans.sort()
#print(ans)
for i in ans:
    print str(i[0]) + "    " + i[1]


#pdb.set_trace()
#print("delete")


print "Unique words: " + str(len(wordcount.keys()))
sum = 0
for i in wordcount.keys():
    sum += wordcount[i]
print "Total words: " + str(sum)