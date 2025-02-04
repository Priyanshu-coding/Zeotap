README for eCommerce Transactions Analysis
Overview
This project involves analyzing an eCommerce Transactions dataset consisting of three CSV files: Customers.csv, Products.csv, and Transactions.csv. The goal is to perform exploratory data analysis (EDA), build a lookalike model for customer recommendations, and conduct customer segmentation through clustering.
Dataset Description
The datasets used in this analysis are:
Customers.csv
CustomerID: Unique identifier for each customer.
CustomerName: Name of the customer.
Region: Continent where the customer resides.
SignupDate: Date when the customer signed up.
Products.csv
ProductID: Unique identifier for each product.
ProductName: Name of the product.
Category: Product category.
Price: Product price in USD.
Transactions.csv
TransactionID: Unique identifier for each transaction.
CustomerID: ID of the customer who made the transaction.
ProductID: ID of the product sold.
TransactionDate: Date of the transaction.
Quantity: Quantity of the product purchased.
TotalValue: Total value of the transaction.
Installation
To run this project, ensure you have the following Python packages installed:
pandas
matplotlib
seaborn
scikit-learn
You can install these packages using pip:
bash
pip install pandas matplotlib seaborn scikit-learn
Usage
Mount Google Drive to access your datasets:
python
from google.colab import drive
drive.mount('/content/drive')
Load the datasets:
python
import pandas as pd

customers = pd.read_csv('/content/drive/My Drive/Customers.csv')
products = pd.read_csv('/content/drive/My Drive/Products.csv')
transactions = pd.read_csv('/content/drive/My Drive/Transactions.csv')
Perform exploratory data analysis (EDA):
Check for missing values in each dataset.
Visualize customer distribution by region and product distribution by category.
Analyze monthly sales trends and identify top-selling products and customers.
Build a Lookalike Model:
python
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# Merge datasets and create customer profile
customer_transactions = pd.merge(transactions, customers, on='CustomerID')
customer_profile = customer_transactions.groupby('CustomerID').agg({
    'TotalValue': 'sum',
    'Quantity': 'sum',
    'ProductID': 'nunique'
}).reset_index()

# Normalize features and calculate similarity matrix
scaler = StandardScaler()
customer_profile_scaled = scaler.fit_transform(customer_profile[['TotalValue', 'Quantity', 'ProductID']])
similarity_matrix = cosine_similarity(customer_profile_scaled)

# Function to get top 3 similar customers
def get_top_similar_customers(customer_id, similarity_matrix, top_n=3):
    ...

# Generate lookalike recommendations for first 20 customers
...

# Save recommendations to CSV
lookalike_df.to_csv('/content/drive/My Drive/FirstName_LastName_Lookalike.csv', index=False)
Perform Customer Segmentation using K-Means Clustering:
python
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Perform K-Means clustering
kmeans = KMeans(n_clusters=4, random_state=42)
customer_profile['Cluster'] = kmeans.fit_predict(customer_profile_scaled)

# Calculate Davies-Bouldin Index for cluster evaluation
db_index = davies_bouldin_score(customer_profile_scaled, customer_profile['Cluster'])

# Visualize clusters using PCA
...
Deliverables
Jupyter Notebook/Python scripts containing all code for EDA, Lookalike Model, and Clustering.
CSV file with lookalike recommendations saved in Google Drive.
Results and Insights
The analysis provides insights into:
Customer distribution across regions.
Product popularity by category.
Monthly sales trends over time.
Recommendations for similar customers based on transaction history.
Customer segmentation into distinct clusters based on purchasing behavior.
Conclusion
This project showcases how to leverage data analysis techniques to derive actionable insights from eCommerce transaction data, aiding in marketing strategies and enhancing customer engagement.
Feel free to modify any sections according to your specific needs or findings!
