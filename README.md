## Analysis of Hospital Admissions Data Using k-Means Clustering 

![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white) 
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

### Tejaswini Shekar
________________

### Goal of Analysis

The goal of the analysis is to group patients into meaningful clusters based on their age, daily charge, and additional charges.  
By better understanding patient characteristics and grouping patients in this way, hospital administrators and healthcare workers can prioritise patients and deliver specialised care. 
This will also aid in more intentional and efficient allocation of resources, as well as improved patient outcomes.

### K-Means Clustering Technique
K-means clustering is an unsupervised machine learning algorithm, where "k" represents the number of clusters the data will be classified into. 
The algorithm groups unlabelled data into clusters based on similarity, while trying to minimize the variance within the clusters. 

"K" random points are first initialised, called the means/cluster centroids. Then each data point is classified into a cluster based on which mean it is closest to. 
The coordinates of the mean are updated to be the average of the data points in that cluster. This process is repeated iteratively a specified number of times and until the final clusters are obtained.

The expected outcome of analysis is to meaningfully group the patients into "k" clusters, so that patients with similar characteristics are present in the same cluster. 
This data can be used by hospital stakeholders for categorising patients and tailoring healthcare services accordingly.

### Results and Implications

The k-means model grouped the data set into 4 labelled clusters, with a silhouette score of 0.4019 and an inertia of 8830.9889. The number of patients in each cluster ranged from about 2300-2700. 
The clusters centroids were obtained as an array, using the .cluster_centers method. 
Each list in this array represents the coordinates for a cluster-centroid in a 3-dimensional space (since 3 variables were used in the analysis). 
For example, the first list in the array was [37.42895753,  7372.40876407,  8459.88249949]. 
This is the location of the center of the first cluster, where 37.42895753 is the "age" mean, 7372.40876407 is the "daily_charge" mean and 8459.88249949 is the "addiitonal_charges" mean.

The clusters were then explored to gain insight into their composition.  

- Cluster 1 has middle-aged patients (aged about 37 years), with a high average daily charge (about $7,400) and lower additional charges (about $8,400). 
- Cluster 4 has middle-aged patients (about 37 years) with a comparatively lower daily charge (about $3,200), and similarly low additional charges (about $8,400). 
- Cluster 2 has elderly patients (about 71 years), with a lower average daily charge (about $3,200) and high additional charges (about $18,000).
- Cluster 3 also has elderly patients (about 71 years), with comparatively higher daily charges ($7,400) and high additional charges (about $18,000).

    *The number of patients in each cluster and the exact coordinates of each cluster centroid are given below:*  

    | Cluster    | Number of Patients | Age Centroid | Daily Charge Centroid | Additional Charges Centroid |
    | :-------- | :--------: | :--------: |:--------: |:--------: |
    | **1** | 2588 | 37.42895753 | 7372.40876407 | 8459.882499490 |
    | **2** | 2332 | 71.63866267 | 3283.57973597 | 18092.92597619 |
    | **3** | 2388 | 71.56035205 | 7404.20597668 | 17881.58163656 |
    | **4** | 2692 | 37.27238945 | 3233.05908740 | 8382.735084440 |

The clusters were also visualised, two variables at a time, using scatter plots. The plots show the grouping of patients by color (purple, green, yellow, blue), with the cluster centroids shown as blue diamonds. 

<img width="1582" alt="Screenshot 2025-04-03 at 2 19 16 PM" src="https://github.com/user-attachments/assets/08cb6fcf-3029-4b36-bd6c-eca102bedbc3" />

The scatter plots of the clustered data can also be compared with the distributions of the three individual variables (see section "C3-Data Preparation" for distribution graphs). 
- For example, daily charge has a *bimodal distribution*, with peaks at about $3,200 and $7,400. These peaks coincide with its cluster means, also about $3,200 and $7,400. 
- Similarly, additional charges has a *right-skewed* distribution which can be split into two parts, the left half which peaks at about $9,000 and the right half which has a more flat, trailing shape.  The means for these two halves coincide with the cluster means (about $8,400 and $18,000). 
- The k-means model also appears to have divided the age variable, which has a *uniform distribution*, into 2 halves. The cluster means for age coincide with the midpoints of the two halves (both about 37 years and 71 years).

From exploring the analysis results, we see that the k-means model has grouped the patients into 4 meaningful clusters based on age, daily charge and additional charges. Further, the value of the silhouette score (0.4019) is greater than 0, which indicates that the clusters are distinguishable and contain similar patients.
These clusters can be used by stakeholders to broadly classify patients by age and hospital expenditure, with each cluster requiring different levels of care. Older patients with greater additional charges may need more specialised care during admission and strict follow-up after discharge to prevent readmission. On the other hand, the younger group of patients with lower daily and additional charges may require less intervention and minimal follow-up. 
In this way, the results of the analysis can drive decision-making regarding best practices for healthcare delivery, and efficient follow-up care after discharge. 

### Recommended Course of Action

Based on the analysis results, I recommend a similar k-means clustering analysis on other continuous variables from the data set. 
Expanding the selection of variables used in the k-means clustering can provide additional insight into patient admissions and outcomes. 
A more in-depth analysis can improve healthcare service delivery and optimise resource allocation to ensure the best patient outcomes. 
Also, the silhouette score can be used to determine the optimal number of clusters instead of the elbow plot since it is more objective and may provide more accurate results.
