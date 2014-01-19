#!/usr/bin/python

naive_probabilities = map(float, open("probabilities-naive.log").read().split())
optimized_probabilities = map(float, open("probabilities-optimized.log").read().split())
urls = open("web-16384.urls").read().split()

naive_sorted_indices = [pair[0] for pair in sorted(enumerate(naive_probabilities), key=lambda p: -p[1])]
optimized_sorted_indices = [pair[0] for pair in sorted(enumerate(optimized_probabilities), key=lambda p: -p[1])]

print "Top web pages (naive algorithm)"
print "Probability\tURL"
for index in naive_sorted_indices[:10]:
	print "%.5lf%%\t%s" % (100 * naive_probabilities[index], urls[index])

print "Top web pages (optimized algorithm)"
print "Probability\tURL"
for index in optimized_sorted_indices[:10]:
	print "%.5lf%%\t%s" % (100 * optimized_probabilities[index], urls[index])

