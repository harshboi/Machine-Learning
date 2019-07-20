#!/usr/bin/env python
import sys

# Get input lines from stdin
for inputt in sys.stdin:
	inputt = inputt.strip()
	sentence = inputt.split(",")
	for i in range(len(sentence)):
		for j in range(i,len(sentence)):
			if len(sentence) == 1:
				print sentence[i].lower() + "   " + "1"
			if i == j:
				continue
			if (sentence[i][0] == " "):
				sentence[i] = sentence[i][1:]
				if (len(sentence[i]) > len(sentence[j])):
					temp = sentence[i]
					sentence[i] = sentence[j]
					sentence[j] = temp
				elif (len(sentence[i]) == len(sentence[j])):
					for k in range(len(sentence[i])):
						if (sentence[i][k] > sentence[j][k]):
							temp = sentence[i]
							sentence[i] = sentence[j]
							sentence[j] = temp
				if (sentence[j][0] != " "):
					sentence[j] = " " + sentence[j]
				print sentence[i].lower() + "," + sentence[j].lower() + "   1"
			else:
				if (len(sentence[i]) > len(sentence[j])):
					temp = sentence[i]
					sentence[i] = sentence[j]
					sentence[j] = temp
				elif (len(sentence[i]) == len(sentence[j])):
					for k in range(len(sentence[i])):
						if (sentence[i][k] > sentence[j][k]):
							temp = sentence[i]
							sentence[i] = sentence[j]
							sentence[j] = temp
				if (sentence[j][0] != " "):
					sentence[j] = " " + sentence[j]
				print sentence[i].lower() + "," + sentence[j].lower() + "   " + "1"