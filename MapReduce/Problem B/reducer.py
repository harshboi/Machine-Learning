#!/usr/bin/env python
import sys
import pdb

ans = {}

for inputt in sys.stdin:
    inputt = inputt.strip()
    word, inc = inputt.split("   ",1)
#    print word + str(inc)
    inc = int(inc)
#    print(word)

#    print word
    try:
        ans[word] = ans[word] + inc
    except:
        ans[word] = inc
#print(ans.keys())
for i in ans.keys():
    print "(" + i + "): " + str(ans[i])

print "Total Unique Pairs: " + str(len(ans.keys()))