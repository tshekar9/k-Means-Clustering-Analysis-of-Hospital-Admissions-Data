# k-means clustering analysis



# determine optimal number of clusters
inertia = []
for k in range(1,8):
    kmean_model = KMeans(n_clusters=k, random_state=18, n_init=10)
    kmean_model.fit(scaled_df)
    inertia.append(kmean_model.inertia_)
    
#plot the elbow graph
# optimal number of clusters appars to be 4
plt.plot(range(1,8), inertia, 'o-')
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

# initialise KMeans model, set random state so that same results are obtained each time the model is run
kmeans_model = KMeans(n_clusters=4, random_state=18, n_init=10)

# fit model and get predicted clusters
kmeans_model.fit(scaled_df)
kmeans_model.predict(scaled_df)
labels = kmeans_model.labels_
labels

# check number of patients in each cluster
cluster_counts = pd.Series(kmeans_model.labels_).value_counts()
cluster_counts


# find cluster centroids
centroids = kmeans_model.cluster_centers_     
# unscale data prior to visualising
centroids = scaler.inverse_transform(centroids)
centroids                     


# Create dataframe of centroids
centroids_dict = {
    "age_centroid": [centroids[0][0], centroids[1][0], centroids[2][0], centroids[3][0]], 
    "daily_charge_centroid": [centroids[0][1], centroids[1][1], centroids[2][1], centroids[3][1]], 
    "additional_charges_centroid": [centroids[0][2], centroids[1][2], centroids[2][2], centroids[3][2]]
}
centroids_df = pd.DataFrame(data=centroids_dict, index=["cluster_1", "cluster_2", "cluster_3", "cluster_4"])

# print analysis results
print(centroids_df)


#  Visualise Clustered Data:

age_centroids = (37.42895753, 71.63866267, 71.56035205, 37.27238945)
daily_charge_centroids = (7372.40876407, 3283.57973597, 7404.20597668, 3233.0590874)
additional_charges_centroids = (8459.88249949, 18092.92597619, 17881.58163656, 8382.73508444)

#plot age and daily charge centroids
plt.figure(figsize=[20,5])
plt.subplot(1,3,1)
plt. scatter(df["age"], df["daily_charge"], c=labels, alpha=0.5)
plt.scatter(age_centroids, daily_charge_centroids, marker='D', s=50)
plt.title("Age and Daily Charge")
plt.xlabel("Age")
plt.ylabel("Daily Charge")

#plot daily charge and additional charge centroids
plt.subplot(1,3,2)
plt. scatter(df["daily_charge"], df["additional_charges"], c=labels, alpha=0.5)
plt.scatter(daily_charge_centroids, additional_charges_centroids, marker='D', s=50)
plt.title("Daily Charge and Additional Charges")
plt.xlabel("Daily Charge")
plt.ylabel("Additional Charges")

#plot age and additional charge centroids
plt.subplot(1,3,3)
plt.scatter(df["age"], df["additional_charges"], c=labels, alpha=0.5)
plt.scatter(age_centroids, additional_charges_centroids, marker='D', s=50)
plt.title("Age and Additional Charges")
plt.xlabel("Age")
plt.ylabel("Additional Charges")

# evaluate clusters using silhouette score
silhouette = silhouette_score(scaled_df, labels)
print("Silhouette Score:", silhouette)

# WCSS/inertia of final model
inertia = kmeans_model.inertia_
print("WCSS/Inertia:", inertia)
