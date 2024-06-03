# Group Human intelligence for Computational Optimization
----

# Goals:
The main goal of this project was to be able to perform clustering with Genetic Algorithms. <br>
<br>
We choose to use the [Spotify dataset on Kaggle](https://www.kaggle.com/datasets/vatsalmavani/spotify-dataset) as our data source, as it is cleaned and ready to be used.<br>

----
# Code
* The custom individuals and population can be find [here](genetic/algorithm/mendel.py)
* The selection algorithms used during the project can be find [here](genetic/selection/selection.py)
* The crossover algorithms used during the project can be find [here](genetic/crossover/crossover.py)
* The mutation algorithms used during the project can be find [here](genetic/mutation/mutation.py)
* The logic to run the algorithm (in parallel, by the way) can be find [here](genetic/main.py)
* The notebook with the final analysis and cool graphics can be find [here](notebooks/final_findings.ipynb)
* The complete log of our best results can be find [here](results.txt)
* Finally, the final report can be find [here](about:blank)

---
# Final results
The following image gives a glimpse of our GA clustering results and its comparison to the KMeans algorithm (interestingly enough, the last run found a 3 cluster solution, even though we started with 4): 
![image](https://github.com/vitorsouton/cifo/assets/83122002/5541bf24-2b17-459d-8355-f3515b996da6)

