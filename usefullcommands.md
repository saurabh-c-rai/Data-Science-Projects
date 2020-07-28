# Splitting large file for storing into git
split -b 50m filename.csv
for i in *; do mv "$i" "$i.csv"; done

# Joining the above splitted file
cat xa{a..f}.csv >filename.csv 

# mlflow 
mlflow server --backend-store-uri sqlite:////Users/saurabhrai/Data-Science-Projects/Hackathon/External/Jantahack/PesticideDamagePrediction/databases/mlflow.db --default-artifact-root /Users/saurabhrai/Data-Science-Projects/Hackathon/External/Jantahack/PesticideDamagePrediction/src