import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import clip
import torch
from PIL import Image
import base64
import logging


# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# ------------------------------
# Step 0: Save and Compress Input Image
# ------------------------------

def save_and_compress_image(file_path):
    """Compress and resize image to reduce payload size."""
    max_size = (1024, 1024)  # Resize to 1024x1024 pixels
    compressed_path = os.path.splitext(file_path)[0] + "_compressed.jpg"

    with Image.open(file_path) as img:
        img = img.convert("RGB")  # Ensure compatibility with JPEG
        img.thumbnail(max_size)  # Resize image
        img.save(compressed_path, "JPEG", quality=75)  # Save as compressed JPEG

    os.remove(file_path)  # Remove the original large file
    return compressed_path


def preprocess_image_from_base64(base64_str):
    """Decode and preprocess base64 image."""
    from io import BytesIO
    image_data = base64.b64decode(base64_str.split(",")[1])  # Strip data URI prefix
    img = Image.open(BytesIO(image_data)).convert("RGB")
    img.thumbnail((1024, 1024))  # Resize directly here
    return preprocess(img).unsqueeze(0).to(device)


# ------------------------------
# Step 1: Data Preprocessing
# ------------------------------

def preprocess_metadata(poster_data):
    """Preprocess metadata fields in the dataset."""
    def preprocess_field(field_string):
        if pd.isna(field_string):
            return []
        return [item.strip().lower() for item in field_string.split(",")]

    def preprocess_single_field(field_string):
        """Process single-value fields like lincoln_theme without splitting."""
        if pd.isna(field_string):
            return []
        return [field_string.strip().lower()]

    def preprocess_subject(subject_string):
        """Extract unique categories from the subject field, removing counts."""
        if pd.isna(subject_string):
            return []
        return [
            item.split()[0].strip().lower()
            for item in subject_string.split(",")
            if item.strip()  # Ensure the item is not empty
        ]

    poster_data['theme_clean'] = poster_data['theme'].apply(preprocess_field)
    poster_data['lincoln_theme_clean'] = poster_data['lincoln_theme'].apply(preprocess_single_field)  # No split
    poster_data['lincoln_theme_2_clean'] = poster_data['lincoln_theme_2'].apply(preprocess_single_field)  # No split
    poster_data['subject_clean'] = poster_data['subject'].apply(preprocess_subject)  # Remove counts

    # Combine all themes into a unified field
    poster_data['all_lincoln_themes'] = poster_data.apply(
        lambda row: row['lincoln_theme_clean'] + row['lincoln_theme_2_clean'], axis=1
    )

    return poster_data


def process_text(row):
    """Combine relevant fields for CLIP embeddings."""
    text_fields = [row['title'], row['theme'], row['description'], row['subject_clean'], row['object'], row['all_lincoln_themes'], row['decade']]
    return ". ".join([
        str(field) for field in text_fields 
        if not isinstance(field, (list, np.ndarray)) and pd.notna(field)
    ])


def prepare_embeddings(poster_data, preprocess_function, model):
    """Generate embeddings for poster text features."""
    poster_data['text_features'] = poster_data.apply(process_text, axis=1)
    texts = poster_data['text_features'].tolist()
    with torch.no_grad():
        text_tokens = clip.tokenize(texts, truncate=True).to(device)
        text_embeddings = model.encode_text(text_tokens).cpu().numpy()
    return text_embeddings

# ------------------------------
# Step 2: Image Similarity
# ------------------------------

def preprocess_image(image_path):
    """Preprocess the user-uploaded image."""
    image = preprocess(Image.open(image_path).convert("RGB"))
    return image.unsqueeze(0).to(device)

def generate_embedding(image_tensor):
    """Generate an embedding for the input image."""
    with torch.no_grad():
        return model.encode_image(image_tensor).cpu().numpy()

def calculate_similarities(input_embedding, dataset_embeddings):
    """Calculate cosine similarities between input and dataset."""
    return cosine_similarity(input_embedding, dataset_embeddings)

# ------------------------------
# Spiral Visualization
# ------------------------------

def spiral_coordinates(num_points, center=(0, 0), spacing=20):
    """Generate coordinates for a two-point spiral layout with constant radial growth."""
    theta = np.linspace(0, 2 * np.pi * num_points / 8, num_points)  # Control angle growth
    radius = np.linspace(20, spacing * num_points / 8, num_points)  # Constant radial spacing
    x = center[0] + radius * np.cos(theta)
    y = center[1] - radius * np.sin(theta)
    return x, y


# def imscatter(x, y, image_paths, ax=None, max_width=60, max_height=60):
#     """Place images at specified coordinates on a plot."""
#     if ax is None:
#         ax = plt.gca()
#     for xx, yy, img_path in zip(x, y, image_paths):
#         if not img_path or not os.path.exists(img_path):
#             logging.warning(f"Invalid or missing image path: {img_path}. Skipping...")
#             continue
#         try:
#             img = Image.open(img_path)
#             img.thumbnail((max_width, max_height), Image.LANCZOS)
#             im = OffsetImage(img, zoom=1)
#             ab = AnnotationBbox(im, (xx, yy), frameon=False)
#             ax.add_artist(ab)
#         except Exception as e:
#             logging.error(f"Failed to process image {img_path}: {e}")


# def visualize_spiral(input_img_path, poster_images, similarities, poster_data_top_n, output_path="static/spiral_plot.png"):
#     if not os.path.exists(input_img_path):
#         raise FileNotFoundError(f"Input image path does not exist: {input_img_path}")
#     valid_poster_images = [img for img in poster_images if os.path.exists(img)]
#     logging.debug(f"Valid poster images: {valid_poster_images}")
    
#     num_points = len(valid_poster_images)
#     spiral_x, spiral_y = spiral_coordinates(num_points, center=(0, 0), spacing=0.5)

#     plt.figure(figsize=(12, 12))
#     imscatter([0], [0], [input_img_path], max_width=60, max_height=60)
#     imscatter(spiral_x, spiral_y, valid_poster_images, max_width=60, max_height=60)

#     plt.plot(spiral_x, spiral_y, 'r-', alpha=0.3)

#     for i in range(num_points):
#         plt.plot([0, spiral_x[i]], [0, spiral_y[i]], 'k--', alpha=0.2)
#         mid_x = (0 + spiral_x[i]) / 2
#         mid_y = (0 + spiral_y[i]) / 2
#         similarity = poster_data_top_n.iloc[i]['similarity']
#         plt.text(mid_x, mid_y, f"{similarity:.3f}", fontsize=8, color='blue', ha='center', va='center')

#     plt.title("Spiral Visualization of Posters with Thumbnails")
#     plt.axis('off')
#     plt.tight_layout()
#     plt.savefig(output_path, dpi=300)
#     plt.close()


# def generate_spiral_data(input_img_path, poster_images, similarities, poster_data_top_n):
#     """Generate spiral layout data for interactive visualization."""
#     num_points = len(poster_images)
#     spiral_x, spiral_y = spiral_coordinates(num_points, center=(0, 0), spacing=0.5)

def generate_spiral_data(input_img_path, poster_images, similarities, poster_data_top_n):
    """Generate spiral layout data for interactive visualization."""
    num_points = len(poster_images)
    spiral_x, spiral_y = spiral_coordinates(num_points, center=(0, 0), spacing=40)  # Use constant spacing

    spiral_data = [
        {
            "x": float(spiral_x[i]),
            "y": float(spiral_y[i]),
            "image_path": poster_images[i],
            "title": poster_data_top_n.iloc[i]['title'],
            "similarity": f"{similarities[i]:.2f}",
            "decade": int(poster_data_top_n.iloc[i]['decade']),
        }
        for i in range(num_points)
    ]
    return {
        "center_image": input_img_path,
        "spiral_data": spiral_data
    }





# ------------------------------
# Historical Context
# ------------------------------

def generate_historical_contexts():
    """Define historical contexts for zero-shot classification."""
    return {
        "labor_movements": [
            "knights of labor", "industrial workers of the world", "great depression",
            "new deal era", "civil rights movement and labor", "farm workers’ movement",
            "post-war union consolidation", "minimum wage campaigns"
        ],
        "historical_events": [
            "may day", "world war ii wartime labor", "haymarket affair",
            "pullman strike", "triangle shirtwaist factory fire",
            "flint sit-down strike", "battle of the overpass", 
            "grape boycott", "patco strike", "occupy wall street"
        ],
        "socio_political_trends": [
            "progressive era reforms", "red scare and labor",
            "feminist labor campaigns", "immigration and labor rights",
            "environmental justice in labor", "rise of worker cooperatives",
            "racial integration in unions"
        ],
        "design_styles": [
            "social realism", "constructivist typography", 
            "modernist labor poster design", "union propaganda graphics",
            "photomontage techniques", "silkscreen prints for advocacy",
            "hand-drawn illustrations"
        ],
        "cross_cultural_influences": [
            "mexican muralism and labor art", "cold war propaganda and labor",
            "global anti-colonial labor movements", "solidarity with south african workers"
        ],
        "key_events": [
            "may day", "world war i post-war strikes", 
            "world war ii wartime labor policies", "cold war labor activism",
            "civil rights movement labor support", "1970s union negotiations"
        ],
        "emerging_contexts": [
            "industrial automation backlash", "climate justice and labor alliances",
            "racial equity campaigns in unions", "healthcare union movements",
            "living wage struggles"
        ]
    }

def encode_categories(categories, model, device):
    """Encode historical context categories."""
    category_embeddings = {}
    for category, labels in categories.items():
        text_inputs = [f"{category}: {label}" for label in labels]
        with torch.no_grad():
            text_tokens = clip.tokenize(text_inputs).to(device)
            category_embeddings[category] = model.encode_text(text_tokens).cpu().numpy()
    return category_embeddings

def zero_shot_classification(image_embeddings, category_embeddings, historical_contexts):
    """Perform zero-shot classification for historical context."""
    results = {}
    for category, label_embeddings in category_embeddings.items():
        similarities = cosine_similarity(image_embeddings, label_embeddings)
        mean_similarities = np.mean(similarities, axis=0)  # Aggregate similarities
        top_3_indices = mean_similarities.argsort()[-3:][::-1]
        top_3_labels = [historical_contexts[category][idx] for idx in top_3_indices]
        top_3_scores = mean_similarities[top_3_indices]
        results[category] = list(zip(top_3_labels, top_3_scores))
    return results




# ------------------------------
# Combined Insights
# ------------------------------

def summarize_insights(poster_data_top_n):
    """Summarize insights based on metadata of top N posters."""
    if 'all_lincoln_themes' and 'decade' in poster_data_top_n:
        theme_counts = poster_data_top_n['all_lincoln_themes'].explode().value_counts()
        summary = ""
        for theme, count in theme_counts.items():
            summary += f"- {theme}: {count} occurrences\n"
        return summary
    else:
        return "### No Themes Found in the Data ###"


def summarize_historical_classifications(aggregated_results):
    """Summarize top-3 historical context classifications for the posters."""
    summary = ""
    for category, top_3 in aggregated_results.items():
        summary += f"- {category.capitalize()}:\n"
        for label, score in top_3:
            summary += f"    - {label}: {score:.3f}\n"
        summary += f"\n"
    return summary


