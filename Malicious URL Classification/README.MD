The Great Host/Lexical URL Reputation Bake-off ™

Note: Read Writeup for detailed eexplanation on solution

Summary
Modern URL classification systems can classify most URLs without ever needing URL content. This is important because it is far less expensive to classify a URL without having to crawl, download, store, and analyze content. Furthermore, it is often impossible to access content of a URL due to single-shot and auto-cloaking malicious websites.

Typically this classification is done using a combination rule-based and machine-learning techniques. Since we don't have time to delve into machine learning, let's build a rule-based micro-classifier which uses weighted scoring for different URL features. We will run the classification set through our newly built micro URL classifier and see what results.

Instructions
Note that there are two files of URL records, in standard JSON format. One file is the training file, which includes a flag that notes if the URL is pre-known as malicious. The other file is the classification set, which does NOT have a malicious field, which you will be classifying using your micro URL classification system you will build.
Note that there is a utility provided for you, this is a template in pythonPreview the document that you can use . It is a simple parser for the JSON files. You will be extending this script and turning them into your classification system.
Explore the training file and note the features listed for each URL. If you have questions about any of the features please feel free to ask and/or discuss.
Think about what might make a URL more suspicious. Here are some hints to get you thinking:
Young domains are likely MORE malicious than old domains
Domains which don't return IP addresses could be fast-flux domains. These domains are likely to be MORE malicious. For example, how often does a DNS query for google.com fail?
URLs which are listed in the Alexa top 1,000,000 are LIKELY to be LESS malicious than those that are not.
URLs with a very low Alexa rank are likely to be LESS malicious that those with a high Alexa rank. This is known as "URL Prevalence"
Another hint: 50% of the URLs in each file are malicious. Use this to help validate your results.
What about file extension. How often do you *really* download raw .exe file directly from the web, instead of a software package.
What about query string?
How about the number of domain tokens? Path tokens?
What port does the URL use? Do your favorite safe URLs usually use non-standard ports?
What about odd combinations?? If a URL has a keyword in it such as 'paypal', but has a very young domain age and no Alexa rating, is it likely to be malicious? (Think phishing.)
The more features you use, the more accurate your results will likely be.
Think up of additional features, extrapolated from the existing data set, which you can use.
Create a strategy for classifying the raw URL set. Consider using a point system where malicious features add to a score, and safe features detract from a that score. Figure out a threshold to use to decide if a URL has exceeded a score which makes it malicious. Try to have a basic strategy by the end of the lab.
Perform some validation on your strategy by aggregating features of the training set.
These are suggestions. Use your own ideas for classification. Perhaps some statistical analysis on the provided features?

