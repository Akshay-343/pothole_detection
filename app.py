import os
from flask import Flask, render_template, request, send_from_directory, url_for
import cv2
import numpy as np
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.util import img_as_ubyte
from skimage.measure import label, regionprops

app = Flask(__name__)
app.secret_key = "supersecretkey"  # needed for sessions
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

def detect_cracks_potholes_boxes_only(image_path, output_folder,
                                      disk_radius=5,
                                      min_area_deep=300,
                                      min_area_medium=500,
                                      deep_percentile=88,
                                      medium_percentile=78,
                                      deep_dilate=30,
                                      max_boxes=15,
                                      roi_margin=(0.0,0.0,0.08,0.08),
                                      lane_intensity_thresh=180,
                                      lane_aspect_ratio=4.0):
    import cv2, numpy as np
    from skimage.filters.rank import entropy
    from skimage.morphology import disk
    from skimage.util import img_as_ubyte
    from skimage.measure import label, regionprops

    # Load
    image = cv2.imread(image_path)
    if image is None: raise ValueError("Image not found")
    h, w = image.shape[:2]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5,5), 0)
    gray_uint8 = img_as_ubyte(gray_blur)

    # Entropy
    ent_img = entropy(gray_uint8, disk(disk_radius))
    ent_img_norm = ent_img / ent_img.max()
    ent_img_uint8 = img_as_ubyte(ent_img_norm)

    # Masks based on entropy + intensity
    deep_thresh = np.percentile(ent_img_uint8, deep_percentile)
    medium_thresh = np.percentile(ent_img_uint8, medium_percentile)

    deep_mask = (ent_img_uint8 > deep_thresh) | (gray_uint8 < 90)
    medium_mask = ((ent_img_uint8 > medium_thresh) & (ent_img_uint8 <= deep_thresh)) | ((gray_uint8 < 120) & (gray_uint8 >= 90))

    # Morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    deep_mask = cv2.morphologyEx(deep_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    deep_mask = cv2.morphologyEx(deep_mask, cv2.MORPH_OPEN, kernel)
    medium_mask = cv2.morphologyEx(medium_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    medium_mask = cv2.morphologyEx(medium_mask, cv2.MORPH_OPEN, kernel)

    # Dilate
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (deep_dilate, deep_dilate))
    deep_mask = cv2.dilate(deep_mask, dilate_kernel, iterations=1)

    # Filter regions
    def filter_regions(mask, min_area):
        labeled = label(mask)
        result = np.zeros_like(mask, dtype=np.uint8)
        for region in regionprops(labeled):
            if region.area < min_area: continue
            minr, minc, maxr, maxc = region.bbox
            width = maxc - minc
            height = maxr - minr

            # Ignore left/right only
            if minc < int(roi_margin[2]*w) or maxc > int((1-roi_margin[3])*w): continue

            # Ignore lanes
            if width/height > lane_aspect_ratio or height/width > lane_aspect_ratio: continue
            if gray[minr:maxr, minc:maxc].mean() > lane_intensity_thresh: continue

            result[minr:maxr, minc:maxc] = mask[minr:maxr, minc:maxc]
        return result

    deep_mask = filter_regions(deep_mask, min_area_deep)
    medium_mask = filter_regions(medium_mask, min_area_medium)

    # Draw boxes
    result_img = image.copy()
    deep_count = 0
    medium_count = 0
    boxes_drawn = 0
    for mask, min_area, color, severity in [
        (deep_mask, min_area_deep, (0,0,255), "Deep"),
        (medium_mask, min_area_medium, (0,165,255), "Medium")
    ]:
        labeled_img = label(mask)
        regions = sorted(regionprops(labeled_img), key=lambda r: r.area, reverse=True)
        for region in regions:
            if region.area < min_area: continue
            if max_boxes and boxes_drawn >= max_boxes: break
            minr, minc, maxr, maxc = region.bbox
            cv2.rectangle(result_img, (minc, minr), (maxc, maxr), color, 2)
            cv2.putText(result_img, severity, (minc, max(minr-5,15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color,2)
            boxes_drawn += 1
            if severity=="Deep": deep_count+=1
            else: medium_count+=1

    output_filename = f"advanced_{os.path.basename(image_path)}"
    cv2.imwrite(os.path.join(output_folder, output_filename), result_img)

    return output_filename, {'deep_count': deep_count, 'medium_count': medium_count}


# ---- Flask Routes ----
@app.route("/", methods=["GET", "POST"])
def index():
    uploaded_filename = None

    if request.method == "POST":
        # Check if a new file was uploaded
        if "image" in request.files and request.files["image"].filename != "":
            file = request.files["image"]
            uploaded_filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_filename)
            file.save(file_path)
        else:
            # No new file uploaded, try to reuse the previous file name
            uploaded_filename = request.form.get("uploaded_filename")
            if not uploaded_filename:
                return "No file provided.", 400
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_filename)

        # Read sliders/inputs from form
        deep_percentile = int(request.form.get("deep_percentile", 95))
        medium_percentile = int(request.form.get("medium_percentile", 85))
        min_area_deep = int(request.form.get("min_area_deep", 150))
        min_area_medium = int(request.form.get("min_area_medium", 250))
        deep_dilate = int(request.form.get("deep_dilate", 15))
        max_boxes = int(request.form.get("max_boxes", 10))

        try:
            overlay_filename, counts = detect_cracks_potholes_boxes_only(
                file_path, app.config['OUTPUT_FOLDER'],
                deep_percentile=deep_percentile,
                medium_percentile=medium_percentile,
                min_area_deep=min_area_deep,
                min_area_medium=min_area_medium,
                deep_dilate=deep_dilate,
                max_boxes=max_boxes
            )
        except Exception as e:
            return f"Error processing image: {e}", 500

        return render_template(
            "index.html",
            overlay_image=url_for('output_file', filename=overlay_filename),
            original_image=url_for('uploaded_file', filename=uploaded_filename),
            deep_count=counts['deep_count'],
            medium_count=counts['medium_count'],
            deep_percentile=deep_percentile,
            medium_percentile=medium_percentile,
            min_area_deep=min_area_deep,
            min_area_medium=min_area_medium,
            deep_dilate=deep_dilate,
            max_boxes=max_boxes,
            uploaded_filename=uploaded_filename
        )

    return render_template(
        "index.html",
        deep_percentile=95,
        medium_percentile=85,
        min_area_deep=150,
        min_area_medium=250,
        deep_dilate=15,
        max_boxes=10
    )

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/output/<filename>')
def output_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == "__main__":
    app.run(debug=True)
