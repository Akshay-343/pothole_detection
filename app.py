import os
from flask import Flask, render_template, request, send_from_directory, url_for
import cv2
import numpy as np
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.util import img_as_ubyte
from skimage.measure import label, regionprops

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# ---- Boxes-only detection function (configurable) ----

def detect_cracks_potholes_boxes_only(
    image_path,
    output_folder,
    disk_radius=5,
    min_area_deep=150,
    min_area_medium=250,
    deep_percentile=95,
    medium_percentile=85,
    deep_dilate=15,
    max_boxes=10,
    roi_margin=(0.1,0.1,0.05,0.05),  # top, bottom, left, right
    lane_intensity_thresh=500,
    lane_max_width_ratio=0.02  # max width/height ratio to consider lane marking
):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or cannot be read.")

    h, w = image.shape[:2]
    top, bottom, left, right = roi_margin
    roi = image[int(top*h):int((1-bottom)*h), int(left*w):int((1-right)*w)]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_uint8 = img_as_ubyte(gray)

    # Local entropy
    ent_img = entropy(gray_uint8, disk(disk_radius))
    ent_img_norm = ent_img / ent_img.max()
    ent_img_uint8 = img_as_ubyte(ent_img_norm)

    # Thresholds
    deep_thresh = np.percentile(ent_img_uint8, deep_percentile)
    medium_thresh = np.percentile(ent_img_uint8, medium_percentile)
    deep_mask = ent_img_uint8 > deep_thresh
    medium_mask = (ent_img_uint8 > medium_thresh) & (ent_img_uint8 <= deep_thresh)

    # Morphological cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    deep_mask = cv2.morphologyEx(deep_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    medium_mask = cv2.morphologyEx(medium_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)

    # Dilate Deep mask to merge nearby dots
    if deep_dilate > 0:
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (deep_dilate, deep_dilate))
        deep_mask = cv2.dilate(deep_mask, dilate_kernel, iterations=1)

    # Remove small areas
    def remove_small(mask, min_area):
        labeled = label(mask)
        result = np.zeros_like(mask)
        for region in regionprops(labeled):
            if region.area >= min_area:
                minr, minc, maxr, maxc = region.bbox
                # Lane ignoring: remove very thin horizontal regions
                width = maxc - minc
                height = maxr - minr
                if height > 0 and width/height < lane_max_width_ratio:
                    continue  # likely a lane marking
                # Lane intensity ignoring
                region_patch = gray[minr:maxr, minc:maxc]
                if region_patch.mean() > lane_intensity_thresh:
                    continue
                result[minr:maxr, minc:maxc] = mask[minr:maxr, minc:maxc]
        return result

    deep_mask = remove_small(deep_mask, min_area_deep)
    medium_mask = remove_small(medium_mask, min_area_medium)

    # Draw bounding boxes only
    result_img = roi.copy()
    deep_count = 0
    medium_count = 0
    boxes_drawn = 0

    for mask, min_area, color, severity in [(deep_mask, min_area_deep, (0,0,255), "Deep"),
                                            (medium_mask, min_area_medium, (0,165,255), "Medium")]:
        labeled_img = label(mask)
        regions = sorted(regionprops(labeled_img), key=lambda r: r.area, reverse=True)
        for region in regions:
            if region.area < min_area:
                continue
            if max_boxes is not None and boxes_drawn >= max_boxes:
                break
            minr, minc, maxr, maxc = region.bbox
            cv2.rectangle(result_img, (minc, minr), (maxc, maxr), color, 2)
            cv2.putText(result_img, severity, (minc, minr-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            boxes_drawn += 1
            if severity=="Deep":
                deep_count += 1
            else:
                medium_count += 1

    base_filename = os.path.basename(image_path)
    output_filename = f"advanced_{base_filename}"
    cv2.imwrite(os.path.join(output_folder, output_filename), result_img)

    return output_filename, {'deep_count': deep_count, 'medium_count': medium_count}
# ---- Flask Routes ----
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "image" not in request.files:
            return "No file part", 400
        file = request.files["image"]
        if file.filename == "":
            return "No selected file", 400

        # Read sliders/inputs from form
        deep_percentile = int(request.form.get("deep_percentile", 95))
        medium_percentile = int(request.form.get("medium_percentile", 85))
        min_area_deep = int(request.form.get("min_area_deep", 150))
        min_area_medium = int(request.form.get("min_area_medium", 250))
        deep_dilate = int(request.form.get("deep_dilate", 15))
        max_boxes = int(request.form.get("max_boxes", 10))

        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

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

        return render_template("index.html",
                               overlay_image=url_for('output_file', filename=overlay_filename),
                               deep_count=counts['deep_count'],
                               medium_count=counts['medium_count'],
                               deep_percentile=deep_percentile,
                               medium_percentile=medium_percentile,
                               min_area_deep=min_area_deep,
                               min_area_medium=min_area_medium,
                               deep_dilate=deep_dilate,
                               max_boxes=max_boxes
                               )

    return render_template("index.html",
                           deep_percentile=95,
                           medium_percentile=85,
                           min_area_deep=150,
                           min_area_medium=250,
                           deep_dilate=15,
                           max_boxes=10)

@app.route('/output/<filename>')
def output_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)
