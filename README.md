# Data-Science-Projects
Repository for all my data science projects - active and completed

# External Links
1. [GloVe Model](http://nlp.stanford.edu/data/glove.6B.zip

# Splitting large file for storing into git
split -b 50m filename.csv
for i in *; do mv "$i" "$i.csv"; done

# Joining the above splitted file
 cat xa{a..f}.csv >filename.csv 
