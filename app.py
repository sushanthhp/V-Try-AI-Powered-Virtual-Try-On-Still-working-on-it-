import cv2
import numpy as np
import mediapipe as mp
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from bs4 import BeautifulSoup
import sys
import os
from urllib.parse import urljoin
from rembg import new_session, remove as rembg_remove
import math

# ---------------- Config ----------------
app = Flask(__name__)
CORS(app)

# Behavior
KEEP_FACE_UPRIGHT = True
MAX_ROLL_CORRECTION_DEG = 50        # clamp roll fix
SCALE_CLAMP = (0.55, 2.2)           # avoid extreme scaling
TOP_BAND_PX = 50                    # rows scanned just below shirt top
CENTER_BAND_FRAC = 0.35             # fraction of shoulder width around face center
NECK_PAD_PX = 8                     # keep a few pixels under collar from user
NECK_W_MIN_PX = 36                  # minimum collar width
DEBUG = False

# -------------- MediaPipe --------------
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_selfie = mp.solutions.selfie_segmentation

# -------------- Globals ----------------
_model_images = {}
_rembg_session = new_session()

# ------------- Utilities ---------------
def cv2_to_base64(image):
    ok, buffer = cv2.imencode('.png', image)
    if not ok:
        raise ValueError("Failed to encode image.")
    return base64.b64encode(buffer).decode('utf-8')

def get_face_landmarks(image_bgr):
    h, w = image_bgr.shape[:2]
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as fm:
        res = fm.process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            return None
        pts = []
        for lm in res.multi_face_landmarks[0].landmark:
            pts.append([int(lm.x * w), int(lm.y * h)])
        return pts

def get_pose_landmarks(image_bgr):
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    with mp_pose.Pose(static_image_mode=True, model_complexity=1) as pose:
        res = pose.process(rgb)
        return res.pose_landmarks

def shoulder_stats(image_bgr):
    h, w = image_bgr.shape[:2]
    pl = get_pose_landmarks(image_bgr)
    if not pl:
        return None, None, None, None
    lm = pl.landmark
    x_ls = int(lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w)
    x_rs = int(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w)
    y_ls = int(lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h)
    y_rs = int(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h)
    width = abs(x_rs - x_ls)
    y_avg = int(0.5 * (y_ls + y_rs))
    return (x_ls, y_ls), (x_rs, y_rs), width, y_avg

# ------------ Upright face (roll) -------------
def compute_roll_deg(face_pts) -> float:
    if not face_pts or len(face_pts) <= 263:
        return 0.0
    L = np.array(face_pts[33], dtype=np.float32)
    R = np.array(face_pts[263], dtype=np.float32)
    dy = R[1] - L[1]; dx = R[0] - L[0]
    return math.degrees(math.atan2(dy, dx))

def rotate_bound(image, angle_deg, border_color=(255, 255, 255)):
    (h, w) = image.shape[:2]
    cX, cY = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D((cX, cY), angle_deg, 1.0)
    cos = abs(M[0, 0]); sin = abs(M[0, 1])
    nW = int((h * sin) + (w * cos)); nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2.0) - cX; M[1, 2] += (nH / 2.0) - cY
    return cv2.warpAffine(image, M, (nW, nH), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=border_color)

def straighten_bgr(image_bgr):
    face_pts = get_face_landmarks(image_bgr)
    if not face_pts:
        return image_bgr, 0.0, None
    roll = compute_roll_deg(face_pts)
    applied = float(np.clip(-roll, -MAX_ROLL_CORRECTION_DEG, MAX_ROLL_CORRECTION_DEG))
    if abs(applied) < 0.5:
        return image_bgr, 0.0, face_pts
    rot = rotate_bound(image_bgr, applied, border_color=(255, 255, 255))
    return rot, applied, None

# ------------- Model decomposition -------------
def person_mask_selfie(img_bgr: np.ndarray) -> np.ndarray:
    with mp_selfie.SelfieSegmentation(model_selection=1) as seg:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        res = seg.process(img_rgb)
    mask = res.segmentation_mask
    if mask is None:
        mask = np.zeros(img_bgr.shape[:2], dtype=np.float32)
    mask = cv2.GaussianBlur(mask, (21, 21), 0)
    return (mask > 0.2).astype(np.float32)

def shoulder_cut_y(img_bgr: np.ndarray) -> int:
    h = img_bgr.shape[0]
    Ls, Rs, _, _ = shoulder_stats(img_bgr)
    if Ls and Rs:
        y = min(Ls[1], Rs[1]) - max(6, h // 100)
        return max(0, min(h - 1, y))
    # fallback to upper third
    return h // 3

def extract_shirt_rgba(img_bgr: np.ndarray):
    """Return shirt RGBA and alpha (shirt only, unchanged pixels)."""
    h, w = img_bgr.shape[:2]
    person = person_mask_selfie(img_bgr)
    y_cut = shoulder_cut_y(img_bgr)
    below = np.zeros_like(person); below[y_cut:h, :] = 1.0
    mask = (person * below)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (11, 11), 0)
    alpha = (np.clip(mask, 0, 1) * 255).astype(np.uint8)
    rgba = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = alpha
    rgba[:y_cut, :, 3] = 0  # force zero above shoulders
    return rgba, alpha

def background_without_person(img_bgr: np.ndarray) -> np.ndarray:
    """Inpaint entire person away to get clean background."""
    mask = (person_mask_selfie(img_bgr) * 255).astype(np.uint8)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)), 1)
    bg = cv2.inpaint(img_bgr, mask, 7, cv2.INPAINT_TELEA)
    return bg

# ------------- Collar width from shirt alpha -------------
def measure_collar_from_shirt_alpha(shirt_alpha: np.ndarray, model_bgr: np.ndarray):
    """
    Use the shirt alpha to read the neckline width from the top band.
    Returns ((lx,ly),(rx,ry)) and cx.
    """
    h, w = shirt_alpha.shape[:2]
    ys = np.where(shirt_alpha.sum(axis=1) > 0)[0]
    if ys.size == 0:
        # As a last resort, use shoulder-based guess
        Ls, Rs, sh_w, sh_y = shoulder_stats(model_bgr)
        if sh_w is None:
            cx, cy = w // 2, int(0.55 * h)
            return (cx - NECK_W_MIN_PX // 2, cy), (cx + NECK_W_MIN_PX // 2, cy), cx
        cx = (Ls[0] + Rs[0]) // 2
        y = sh_y - 10
        return (cx - max(NECK_W_MIN_PX, int(0.35 * sh_w)) // 2, y), (cx + max(NECK_W_MIN_PX, int(0.35 * sh_w)) // 2, y), cx

    y_top = int(ys.min())
    y0 = y_top; y1 = min(h, y_top + TOP_BAND_PX)

    face = get_face_landmarks(model_bgr)
    nose_x = face[1][0] if face else w // 2
    _, _, sh_w, _ = shoulder_stats(model_bgr)
    half_band = int(max(40, (sh_w or int(0.35 * w)) * CENTER_BAND_FRAC))

    best_rows = []
    for y in range(y0, y1):
        row = shirt_alpha[y] > 0
        xs = np.where(row)[0]
        if xs.size < 2:  # need left/right
            continue
        L = max(0, nose_x - half_band); R = min(w - 1, nose_x + half_band)
        xs = xs[(xs >= L) & (xs <= R)]
        if xs.size < 2:
            continue
        left = int(xs.min()); right = int(xs.max()); width = right - left
        best_rows.append((width, left, right, y))

    if not best_rows:
        # fallback to full-row extremes
        for y in range(y0, y1):
            xs = np.where(shirt_alpha[y] > 0)[0]
            if xs.size >= 2:
                left = int(xs.min()); right = int(xs.max())
                best_rows.append((right - left, left, right, y))

    if not best_rows:
        # absolute fallback
        cx, cy = w // 2, int((y0 + y1) / 2)
        return (cx - NECK_W_MIN_PX // 2, cy), (cx + NECK_W_MIN_PX // 2, cy), cx

    # Choose a stable early row: 25th percentile (avoids apex zero and shoulder width)
    best_rows.sort(key=lambda t: t[0])
    idx = max(0, int(0.25 * (len(best_rows) - 1)))
    width, left, right, y = best_rows[idx]
    if width < NECK_W_MIN_PX:
        cx = (left + right) // 2
        left = cx - NECK_W_MIN_PX // 2; right = cx + NECK_W_MIN_PX // 2; width = NECK_W_MIN_PX
    cx = (left + right) // 2
    return (left, y), (right, y), cx

# ------------- User head-only cut (no shirt) -------------
def user_head_rgba_and_alpha(user_bgr, neck_y):
    ok, buf = cv2.imencode('.png', user_bgr)
    if not ok:
        raise ValueError("Failed to encode user image.")
    bytes_ = rembg_remove(buf.tobytes(), session=_rembg_session)
    user_rgba = cv2.imdecode(np.frombuffer(bytes_, np.uint8), cv2.IMREAD_UNCHANGED)
    if user_rgba is None or user_rgba.shape[2] != 4:
        raise ValueError("Background removal failed.")

    h, w = user_bgr.shape[:2]
    a = user_rgba[:, :, 3].astype(np.float32) / 255.0

    # keep above neck line + small pad
    y_limit = min(h - 1, max(0, neck_y + NECK_PAD_PX))
    clamp = np.zeros((h, w), np.float32); clamp[:y_limit, :] = 1.0

    # ensure face oval is included
    face_pts = get_face_landmarks(user_bgr)
    face_mask = np.zeros((h, w), np.uint8)
    if face_pts:
        idx = sorted({i for (a_, b_) in mp_face_mesh.FACEMESH_FACE_OVAL for i in (a_, b_)})
        pts = np.array([face_pts[i] for i in idx], np.int32)
        hull = cv2.convexHull(pts)
        cv2.fillConvexPoly(face_mask, hull, 255)
    face_mask_f = face_mask.astype(np.float32) / 255.0

    final_a = np.clip(np.maximum(a, 0.8 * face_mask_f) * clamp, 0, 1)
    final_a = cv2.GaussianBlur(final_a, (7, 7), 0)
    return user_rgba, (final_a * 255).astype(np.uint8), face_pts

def user_neck_y_from_face_shoulders(user_bgr):
    h = user_bgr.shape[0]
    face = get_face_landmarks(user_bgr)
    _, _, _, sh_y = shoulder_stats(user_bgr)
    chin_y = face[152][1] if (face and len(face) > 152) else int(0.45 * h)
    if sh_y is None:
        sh_y = min(h - 1, chin_y + int(0.22 * h))
    return int(0.7 * chin_y + 0.3 * sh_y), face

# ------------- Scale+translate by width -------------
def scale_translate_by_width(model_lr, user_lr):
    (mlx, mly), (mrx, mry) = model_lr
    (ulx, uly), (urx, ury) = user_lr
    mw = float(abs(mrx - mlx)); uw = float(abs(urx - ulx))
    if uw < 1.0: uw = 1.0
    s = np.clip(mw / uw, SCALE_CLAMP[0], SCALE_CLAMP[1])
    mcx, mcy = (0.5 * (mlx + mrx), 0.5 * (mly + mry))
    ucx, ucy = (0.5 * (ulx + urx), 0.5 * (uly + ury))
    tx = mcx - s * ucx
    ty = mcy - s * ucy
    M = np.array([[s, 0.0, tx], [0.0, s, ty]], dtype=np.float32)
    return M, s

# ---------------- Core overlay ----------------
def tryon_pipeline(model_bgr: np.ndarray, user_bgr: np.ndarray, return_debug=False):
    # 0) Upright user
    if KEEP_FACE_UPRIGHT:
        user_bgr, _, _ = straighten_bgr(user_bgr)

    # 1) Decompose model into background (no person) and shirt overlay (unchanged pixels)
    shirt_rgba, shirt_alpha = extract_shirt_rgba(model_bgr)     # top overlay
    bg_clean = background_without_person(model_bgr)             # base layer

    # 2) Measure model collar width from shirt alpha (top band)
    mL, mR, cx_model = measure_collar_from_shirt_alpha(shirt_alpha, model_bgr)

    # 3) Prepare user head-only (no shirt), and estimate a user neck line
    u_neck_y, face_pts = user_neck_y_from_face_shoulders(user_bgr)
    user_rgba, user_alpha_u8, _ = user_head_rgba_and_alpha(user_bgr, u_neck_y)

    # Build a synthetic user neck pair around face center (it will be scaled to model width)
    h_u, w_u = user_bgr.shape[:2]
    nose_x = face_pts[1][0] if face_pts else w_u // 2
    # estimate starting width from face width; later scaled to model width
    if face_pts:
        Lf = face_pts[234] if len(face_pts) > 234 else face_pts[33]
        Rf = face_pts[454] if len(face_pts) > 454 else face_pts[263]
        est_w = max(40, int(0.9 * abs(Rf[0] - Lf[0])))
    else:
        est_w = 80
    uL = (max(0, nose_x - est_w // 2), u_neck_y)
    uR = (min(w_u - 1, nose_x + est_w // 2), u_neck_y)

    # 4) Scale by collar width and align at collar center (no rotation)
    M, s = scale_translate_by_width((mL, mR), (uL, uR))

    # 5) Composite: base background -> user head -> shirt overlay
    dsize = (model_bgr.shape[1], model_bgr.shape[0])
    warped_user_rgb = cv2.warpAffine(user_rgba[:, :, :3], M, dsize, flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    warped_user_a = cv2.warpAffine(user_alpha_u8, M, dsize, flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(np.float32) / 255.0
    warped_user_a = cv2.GaussianBlur(warped_user_a, (5, 5), 0)
    A3 = cv2.merge([warped_user_a, warped_user_a, warped_user_a])

    comp = bg_clean.astype(np.float32)
    comp = comp * (1.0 - A3) + warped_user_rgb.astype(np.float32) * A3
    comp = np.clip(comp, 0, 255).astype(np.uint8)

    # Shirt overlay on top (unchanged pixels)
    shirt_rgb = shirt_rgba[:, :, :3].astype(np.float32)
    shirt_a = (shirt_rgba[:, :, 3].astype(np.float32) / 255.0)
    S3 = cv2.merge([shirt_a, shirt_a, shirt_a])
    comp = comp * (1.0 - S3) + shirt_rgb * S3
    comp = np.clip(comp, 0, 255).astype(np.uint8)

    if return_debug or DEBUG:
        dbg = model_bgr.copy()
        cv2.line(dbg, mL, mR, (0, 255, 0), 3)
        cv2.circle(dbg, mL, 4, (0, 255, 0), -1)
        cv2.circle(dbg, mR, 4, (0, 255, 0), -1)
        # projected user neck after transform
        pL = (int(M[0,0]*uL[0] + M[0,2]), int(M[1,1]*uL[1] + M[1,2]))
        pR = (int(M[0,0]*uR[0] + M[0,2]), int(M[1,1]*uR[1] + M[1,2]))
        cv2.line(dbg, pL, pR, (0, 0, 255), 3)
        cv2.circle(dbg, pL, 4, (0, 0, 255), -1)
        cv2.circle(dbg, pR, 4, (0, 0, 255), -1)
        return comp, dbg
    return comp

# -------------- Networking helpers --------------
def scrape_image_from_url(page_url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(page_url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        og_image = soup.find('meta', property='og:image')
        if og_image and og_image.get('content'):
            return og_image['content']
        largest_image_url, max_size = None, 0
        for img in soup.find_all('img'):
            src = img.get('src')
            if not src or src.startswith('data:'): continue
            full_src = urljoin(page_url, src)
            try:
                img_res = requests.head(full_src, timeout=3, headers=headers, allow_redirects=True)
                if img_res.status_code >= 400: continue
                if 'image' not in img_res.headers.get('content-type', ''): continue
                size = int(img_res.headers.get('content-length', 0))
                if size > max_size:
                    max_size, largest_image_url = size, full_src
            except requests.exceptions.RequestException:
                continue
        return largest_image_url
    except Exception as e:
        print(f"Error scraping {page_url}: {e}", file=sys.stderr)
        return None

# -------------- Boot helpers --------------
def load_default_images():
    print("Loading default model images...")
    try:
        if not os.path.exists('female_model.jpg') or not os.path.exists('male_model.jpg'):
            print("\nFATAL ERROR: 'female_model.jpg' or 'male_model.jpg' not found.")
            sys.exit(1)
        _model_images['female'] = cv2.imread('female_model.jpg')
        _model_images['male'] = cv2.imread('male_model.jpg')
        if _model_images['female'] is None or _model_images['male'] is None:
            raise ValueError("Files are present but could not be read by OpenCV.")
        print("Default model images loaded successfully.")
    except Exception as e:
        print(f"FATAL: Could not load local model images: {e}", file=sys.stderr)
        sys.exit(1)

# ---------------- Endpoint ----------------
@app.route('/process-image', methods=['POST'])
def process_image_endpoint():
    try:
        user_image_file = request.files.get('user_image')
        if not user_image_file:
            return jsonify({'error': 'User image not provided'}), 400
        user_img = cv2.imdecode(np.frombuffer(user_image_file.read(), np.uint8), cv2.IMREAD_COLOR)
        if user_img is None:
            return jsonify({'error': 'Could not decode user image.'}), 400

        source_type = request.form.get('source_type')
        model_img = None

        if source_type == 'default':
            gender = request.form.get('gender', 'female')
            model = _model_images.get(gender)
            if model is None:
                return jsonify({'error': f"Default model image for gender '{gender}' not available."}), 400
            model_img = model.copy()

        elif source_type == 'upload':
            model_file = request.files.get('model_image_upload')
            if not model_file:
                return jsonify({'error': 'Clothing image file not provided.'}), 400
            model_img = cv2.imdecode(np.frombuffer(model_file.read(), np.uint8), cv2.IMREAD_COLOR)

        elif source_type == 'imageUrl':
            url = request.form.get('model_image_url')
            if not url:
                return jsonify({'error': 'Image URL not provided.'}), 400
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            if 'image' not in response.headers.get('content-type', ''):
                return jsonify({'error': 'Provided URL is not an image.'}), 400
            model_img = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)

        elif source_type == 'pageUrl':
            url = request.form.get('product_page_url')
            if not url:
                return jsonify({'error': 'Product page URL not provided.'}), 400
            image_url = scrape_image_from_url(url)
            if not image_url:
                return jsonify({'error': "Could not find a suitable image on that page."}), 400
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            if 'image' not in response.headers.get('content-type', ''):
                return jsonify({'error': 'Scraped URL is not an image.'}), 400
            model_img = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)

        else:
            return jsonify({'error': f"Invalid source_type provided: {source_type}"}), 400

        if model_img is None:
            return jsonify({'error': 'Failed to load or decode the clothing/model image.'}), 400

        return_debug = request.form.get('return_debug', 'false').lower() == 'true'
        result, dbg = tryon_pipeline(model_img, user_img, return_debug=return_debug)
        payload = {'image': cv2_to_base64(result)}
        if dbg is not None:
            payload['debug'] = cv2_to_base64(dbg)
        return jsonify(payload)

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except requests.exceptions.RequestException:
        return jsonify({'error': "Network error when fetching image. Check the URL."}), 500
    except Exception as e:
        print(f"Unexpected server error: {e}", file=sys.stderr)
        return jsonify({'error': 'Internal server error.'}), 500

if __name__ == '__main__':
    load_default_images()
    app.run(host='0.0.0.0', port=5000)