from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import psycopg2
import pickle
import numpy
from numpy.linalg import norm
from psycopg2.extensions import register_adapter, AsIs
from sklearn.neighbors import NearestNeighbors


app = Flask(__name__)

# Establish database connection
conn = psycopg2.connect(
    database="user_actions",
    user="rj",
    password="pwd",
    host="localhost",
    port="5432"
)
cursor = conn.cursor()

def addapt_numpy_float64(numpy_float64):
    return AsIs(numpy_float64)

def addapt_numpy_int64(numpy_int64):
    return AsIs(numpy_int64)

register_adapter(numpy.float64, addapt_numpy_float64)
register_adapter(numpy.int64, addapt_numpy_int64)


# Load pre-trained ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False


# Define feature extraction function
def feature_extraction(img, model):
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

# API endpoint to extract features
@app.route('/extract_features', methods=['POST'])
def extract_features():
    try:
        data = request.get_json()
        productId = data['productId']
        imageUrl = data['imageUrl']

        # Open and process the image from the provided URL
        img = Image.open(imageUrl)
        img = img.resize((224, 224))

        # Perform feature extraction
        features = feature_extraction(img, model)

        # Store the features in the product_features table
        cursor.execute("""
            INSERT INTO recommendation_engine.product_features (product_id, feature_vector)
            VALUES (%s, %s)
        """, (productId, pickle.dumps(features)))

        conn.commit()
        return jsonify({"message": "Features extracted and stored successfully!"}), 200
    except Exception as e:
        conn.rollback()
        return jsonify({"error": f"Error: {e}"}), 500


def recommend(productId, feature_list):
    # Fetch features for the given productId from product_features table
    cursor.execute("SELECT feature_vector FROM product_features WHERE product_id=%s", (productId,))
    row = cursor.fetchone()
    if row:
        bytea_data = row[0]
        try:
            # Convert bytea data to bytes
            bytes_data = memoryview(bytea_data).tobytes()

            # Load the bytes using pickle
            query_feature = pickle.loads(bytes_data)

            # Use NearestNeighbors to find similar products
            neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
            neighbors.fit(feature_list)
            distances, indices = neighbors.kneighbors([query_feature])
            return indices.tolist()[0]  # Return recommended product indices
        except Exception as e:
            print(f"Error processing bytea_data: {e}")
    return []  # Return empty list if features for productId not found


# API endpoint to get recommended productIds
@app.route('/get_recommendations', methods=['POST'])
def get_recommendations():
    try:
        data = request.get_json()
        productId = data['productId']

        # Fetching feature vectors for all products
        cursor.execute("SELECT feature_vector FROM product_features")
        rows = cursor.fetchall()

        feature_list = []
        for row in rows:
            bytea_data = row[0]
            try:
                # Convert bytea data to bytes
                bytes_data = memoryview(bytea_data).tobytes()

                # Load the bytes using pickle
                feature_array = pickle.loads(bytes_data)
                feature_list.append(feature_array)
            except Exception as e:
                print(f"Error processing bytea_data: {e}")

        recommended_product_indices = recommend(productId, feature_list)
        return jsonify({"recommendedProductIds": recommended_product_indices}), 200
    except Exception as e:
        return jsonify({"error": f"Error: {e}"}), 500


# Close database connection
@app.teardown_appcontext
def close_connection(exception):
    cursor.close()
    conn.close()


if __name__ == '__main__':
    app.run(debug=True)
