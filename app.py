from flask import Flask, jsonify, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# Set up Flask application
app = Flask(__name__)

# Load the dataset
data = pd.read_csv('https://raw.githubusercontent.com/sas-7/dest/main/destination.csv')

# Extract relevant features and target variable
X = data[['historical & heritage', 'city', 'pilgrimage', 'hill station', 'beach', 
          'lake & backwater', 'adventure / trekking', 'wildlife', 'waterfall', 
          'nature & scenic', 'price']]
y = data[['city']]  # Target variable

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the KNN model
knn = KNeighborsClassifier(n_neighbors=5)  # Choosing K=5
knn.fit(X_train_scaled, y_train.values.ravel())

# Define function to recommend tourist destinations based on destination type
def recommend_destinations(destination_type):
    # Filter destinations matching the destination type
    filtered_destinations = data[data[destination_type] == 1].head(4)['City']
    if not filtered_destinations.empty:
        return filtered_destinations.tolist()
    else:
        return []

# Define API endpoint to recommend destinations
@app.route('/recommend', methods=['POST'])
def get_recommendations():
    # destination_type = request.args.get('destination_type')
    data = request.get_json()
    destination_type = data['destination_type']
    print(destination_type)
    recommended_destinations = recommend_destinations(destination_type)
    if recommended_destinations:
        return jsonify({'destinations': recommended_destinations})
    else:
        return jsonify({'message': 'No destinations found within the specified criteria'}), 404

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
