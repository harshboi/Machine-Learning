#!/usr/bin/env python
import sys
 
# Get input lines from stdin
for inputt in sys.stdin:
	word = inputt.split()
 
	# Output tuples on stdout
	for each_word in word:
		print each_word.lower() + " 1"