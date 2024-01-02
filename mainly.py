import streamlit
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import psycopg2
import numpy
from psycopg2.extensions import register_adapter, AsIs

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

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

streamlit.title('Fashion Recommender System')

def feature_extraction(img, model):
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

try:
    # Fetch the latest product_id associated with "view_product"
    cursor.execute("SELECT product_id FROM user_action WHERE action_type='view_product' ORDER BY action_datetime DESC LIMIT 1")
    latest_product_id = cursor.fetchone()[0]

    # Fetch the image_path for the latest product_id
    cursor.execute("SELECT image_path FROM product WHERE product_id=%s", (latest_product_id,))
    filename = cursor.fetchone()[0]

    # Open and process the image
    img = Image.open(filename)
    img = img.resize((224, 224))

    # Perform feature extraction
    features = feature_extraction(img, model)
    expected_shape = features.shape

    # Fetching a limited number of feature vectors from the database (10 items)
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

    # Rest of the code remains the same

    indices = recommend(features, feature_list)

    # show
    num_images = 5  # Number of images to display
    if len(indices[0]) >= num_images:
        columns = streamlit.columns(num_images)
        for i in range(num_images):
            cursor.execute("SELECT image_path FROM product WHERE product_id=%s", (indices[0][i] + 1,))
            filename = cursor.fetchone()[0]
            with columns[i]:
                streamlit.image(filename)
    else:
        streamlit.header("Some error occurred in file upload")
except Exception as e:
    streamlit.write(f"Error: {e}")

cursor.close()
conn.close()
