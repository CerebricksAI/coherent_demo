"""
Detector wrappers for the QSAID face-anonymisation model comparison test.

Three models, three roles:
  - SCRFD-10GF   (InsightFace buffalo_l, det_10g.onnx)      -> primary face detector
  - YOLOv11n-face (AdamCodd community weights, NMS-baked)    -> second face detector
  - YOLOX-s       (Megvii, COCO, person class only)          -> person safety net

Detection only. No recognition, no landmarks, no attributes, no re-identification.
"""
import os

import cv2
import numpy as np
import onnxruntime as ort

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


class ScrfdDetector:
    """Primary face detector. Thin wrapper over insightface's own SCRFD decode
    (anchor generation + NMS is nontrivial; reuse the vetted implementation
    rather than re-derive it)."""

    name = "SCRFD-10GF"

    def __init__(self):
        os.environ.setdefault(
            "INSIGHTFACE_HOME", os.path.abspath(os.path.join(MODELS_DIR, ".insightface"))
        )
        from insightface.app import FaceAnalysis

        self.app = FaceAnalysis(
            name="buffalo_l",
            root=os.environ["INSIGHTFACE_HOME"],
            allowed_modules=["detection"],
            providers=["CPUExecutionProvider"],
        )
        self.app.prepare(ctx_id=-1, det_size=(640, 640))

    def detect(self, frame_bgr):
        """Returns list of (x1, y1, x2, y2, score)."""
        faces = self.app.get(frame_bgr)
        out = []
        for f in faces:
            x1, y1, x2, y2 = f.bbox
            out.append((float(x1), float(y1), float(x2), float(y2), float(f.det_score)))
        return out


class YoloFaceDetector:
    """Second face detector: YOLOv11n-face, NMS baked into the exported graph."""

    name = "YOLOv11n-face"

    def __init__(self, conf_threshold=0.25):
        self.session = ort.InferenceSession(
            os.path.join(MODELS_DIR, "yolov11n_face_nms.onnx"),
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.conf_threshold = conf_threshold
        self.input_size = 640

    def _letterbox(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        scale = self.input_size / max(h, w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        resized = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        canvas[:nh, :nw] = resized
        return canvas, scale

    def detect(self, frame_bgr):
        canvas, scale = self._letterbox(frame_bgr)
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        chw = np.transpose(rgb, (2, 0, 1))[None]

        out = self.session.run(None, {self.input_name: chw})[0][0]  # (300, 6)

        results = []
        for x1, y1, x2, y2, conf, cls in out:
            if conf < self.conf_threshold:
                continue
            results.append((x1 / scale, y1 / scale, x2 / scale, y2 / scale, float(conf)))
        return results


class YoloxPersonDetector:
    """Safety net: YOLOX-s, COCO-pretrained, person class (id 0) only.
    Any tracked/detected person with no overlapping face box is the signal
    that a face detector alone would silently miss."""

    name = "YOLOX-s (person)"
    _COCO_PERSON_CLASS = 0

    def __init__(self, conf_threshold=0.3, nms_threshold=0.45):
        self.session = ort.InferenceSession(
            os.path.join(MODELS_DIR, "yolox_s.onnx"),
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.input_size = 640
        self._grids, self._strides = self._build_grids()

    def _build_grids(self):
        strides = [8, 16, 32]
        grids, expanded_strides = [], []
        for stride in strides:
            hsize = wsize = self.input_size // stride
            xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            expanded_strides.append(np.full((1, grid.shape[1], 1), stride))
        return np.concatenate(grids, 1), np.concatenate(expanded_strides, 1)

    def _letterbox(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        scale = self.input_size / max(h, w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        resized = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        canvas[:nh, :nw] = resized
        return canvas, scale

    def detect(self, frame_bgr):
        canvas, scale = self._letterbox(frame_bgr)
        # YOLOX's official demo keeps BGR and does NOT normalize to [0,1] -
        # the exported graph expects raw 0-255 float values.
        chw = np.transpose(canvas, (2, 0, 1)).astype(np.float32)[None]

        raw = self.session.run(None, {self.input_name: chw})[0]  # (1, 8400, 85)

        preds = raw.copy()
        preds[..., :2] = (preds[..., :2] + self._grids) * self._strides
        preds[..., 2:4] = np.exp(preds[..., 2:4]) * self._strides
        preds = preds[0]

        boxes_cxcywh = preds[:, :4]
        obj_conf = preds[:, 4]
        cls_scores = preds[:, 5:]
        cls_id = np.argmax(cls_scores, axis=1)
        cls_conf = cls_scores[np.arange(len(cls_scores)), cls_id]
        scores = obj_conf * cls_conf

        mask = (cls_id == self._COCO_PERSON_CLASS) & (scores > self.conf_threshold)
        if not mask.any():
            return []

        cx, cy, bw, bh = boxes_cxcywh[mask].T
        x1, y1 = (cx - bw / 2) / scale, (cy - bh / 2) / scale
        x2, y2 = (cx + bw / 2) / scale, (cy + bh / 2) / scale
        sc = scores[mask]

        boxes_xywh = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
        idxs = cv2.dnn.NMSBoxes(boxes_xywh, sc.tolist(), self.conf_threshold, self.nms_threshold)
        idxs = np.array(idxs).flatten() if len(idxs) else []

        return [(float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i]), float(sc[i])) for i in idxs]


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a[:4]
    bx1, by1, bx2, by2 = box_b[:4]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def box_center_inside(inner_box, outer_box):
    cx = (inner_box[0] + inner_box[2]) / 2
    cy = (inner_box[1] + inner_box[3]) / 2
    return outer_box[0] <= cx <= outer_box[2] and outer_box[1] <= cy <= outer_box[3]


def dilate_box(box, frac, frame_w, frame_h):
    x1, y1, x2, y2 = box[:4]
    w, h = x2 - x1, y2 - y1
    dx, dy = w * frac, h * frac
    return (
        max(0, x1 - dx),
        max(0, y1 - dy),
        min(frame_w, x2 + dx),
        min(frame_h, y2 + dy),
    )


def head_region_from_person(person_box, top_frac=0.28):
    """Estimate a head box as the top slice of a person box (used only when
    no face detector fired on this person at all)."""
    x1, y1, x2, y2 = person_box[:4]
    h = y2 - y1
    return (x1, y1, x2, y1 + h * top_frac)
