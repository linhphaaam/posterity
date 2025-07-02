import os
import uuid
import numpy as np  # Ensure NumPy is imported
import pandas as pd
from flask import Flask, request, jsonify, render_template, send_file
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
from pipeline import (
    preprocess_metadata,
    prepare_embeddings,
    preprocess_image,
    generate_embedding,
    calculate_similarities,
    # visualize_spiral,
    generate_historical_contexts,
    encode_categories,
    zero_shot_classification,
    summarize_insights,
    summarize_historical_classifications,
    save_and_compress_image,
    spiral_coordinates,
    # preprocess_image_from_base64,
    generate_spiral_data,
    model, preprocess, device
)
from collections import Counter
import logging

logging.basicConfig(level=logging.DEBUG)

###############################
# Global Variables & Setup
###############################

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['STATIC_FOLDER'] = 'static'
app.config['MAX_CONTENT_LENGTH'] = 256 * 1024 * 1024  # 32 MB limit


# Ensure upload and static directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['STATIC_FOLDER'], exist_ok=True)

# Load and preprocess poster dataset
POSTER_DATA_PATH = "static/posters.csv"
POSTER_IMAGE_DIR = "static/poster_images"

poster_data = pd.read_csv(POSTER_DATA_PATH)
poster_data = preprocess_metadata(poster_data)

# Generate embeddings for dataset at startup
dataset_text_embeddings = prepare_embeddings(poster_data, preprocess, model)
poster_image_paths = [os.path.join(POSTER_IMAGE_DIR, img) for img in poster_data['filename']]

# Generate historical context embeddings
historical_contexts = generate_historical_contexts()
category_embeddings = encode_categories(historical_contexts, model, device)

###############################
# Routes
###############################

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process_image():
    file_path = None

    # Check if file is uploaded
    if 'file' in request.files and request.files['file']:
            file = request.files['file']
            logging.debug(f"Uploaded file size: {len(file.read()) / (1024 * 1024):.2f} MB")
            file.seek(0)  # Reset file pointer after size calculation

            # Continue processing file upload
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Compress image
            file_path = save_and_compress_image(file_path)
            input_image_tensor = preprocess_image(file_path)

    elif 'image_data' in request.form and request.form['image_data']:
        try:
            base64_data = request.form['image_data']
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"camera_{uuid.uuid4().hex}.png")
            with open(file_path, "wb") as f:
                f.write(BytesIO(base64.b64decode(base64_data.split(",")[1])).getbuffer())
            logging.debug(f"Saved camera image to: {file_path}")
            if not os.path.exists(file_path):
                raise FileNotFoundError("Camera image not saved correctly.")
            input_image_tensor = preprocess_image(file_path)
        except Exception as e:
            logging.error(f"Error processing camera image: {e}")
            return jsonify({"error": f"Invalid image data: {str(e)}"}), 400
    else:
        return jsonify({"error": "No file uploaded or image captured"}), 400



    # Generate embedding
    input_embedding = generate_embedding(input_image_tensor)

    # Calculate similarities
    similarities = calculate_similarities(input_embedding, dataset_text_embeddings)
    top_n = 25
    top_indices = similarities[0].argsort()[::-1][:top_n]

    # Extract data for top N similar posters
    top_n_images = [poster_image_paths[i] for i in top_indices]
    top_n_similarities = similarities[0][top_indices]
    poster_data_top_n = poster_data.iloc[top_indices].copy()
    poster_data_top_n['similarity'] = top_n_similarities
    top_n_embeddings = dataset_text_embeddings[top_indices]

    # Generate spiral visualization
    # spiral_plot_filename = f"spiral_plot_{uuid.uuid4().hex}.png"
    # spiral_plot_path = os.path.join(app.config['STATIC_FOLDER'], spiral_plot_filename)
    # visualize_spiral(file_path, top_n_images, top_n_similarities, poster_data_top_n, output_path=spiral_plot_path)

    # spiral_info = generate_spiral_data(
    #     file_path, top_n_images, top_n_similarities, poster_data_top_n
    # )

    spiral_info = generate_spiral_data(
        file_path, top_n_images, top_n_similarities, poster_data_top_n
    )

    # Historical classification
    aggregated_embedding = np.mean(top_n_embeddings, axis=0).reshape(1, -1)
    historical_classification_results = zero_shot_classification(
        aggregated_embedding, category_embeddings, historical_contexts
    )

    # Combine metadata and historical insights
    metadata_insights = summarize_insights(poster_data_top_n)
    historical_insights = summarize_historical_classifications(historical_classification_results)
    # combined_insights = f"{metadata_insights}\n\n{historical_insights}"

    combined_insights = {
        "metadata_summary": metadata_insights,
        "historical_summary": historical_insights
    }

    # Calculate decade distribution
    decade_counts = Counter(poster_data_top_n['decade'].dropna())
    decade_data = sorted([(int(decade), int(count)) for decade, count in decade_counts.items()])  # Cast to native int

    # Save input image path
    input_image_url = f"/uploads/{os.path.basename(file_path)}"

    return render_template(
        "results.html",
        # spiral_plot_url=f"/static/{spiral_plot_filename}",
        spiral_data=spiral_info['spiral_data'],
        center_image=spiral_info['center_image'],
        combined_insights=combined_insights,
        decade_data=decade_data,  # Pass decade_data to the template
        input_image_url=input_image_url  # Pass input image URL
    )


###############################
# Static File Handling
###############################

@app.route('/static/<path:filename>')
def serve_static_file(filename):
    return send_file(os.path.join(app.config['STATIC_FOLDER'], filename))

@app.route('/uploads/<path:filename>')
def serve_uploaded_file(filename):
    """Serve files from the uploads folder."""
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))

###############################
# Run the App
###############################

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
