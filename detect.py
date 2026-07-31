from dataclasses import dataclass

from ultralytics import YOLO

PERSON_CLASS = 0
CAT_CLASS = 15

_model: YOLO | None = None


@dataclass
class Detection:
    cls: int
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2


def get_model() -> YOLO:
    global _model
    if _model is None:
        _model = YOLO("yolov8n.pt")
    return _model


def detect(frame, conf_threshold: float = 0.4) -> list[Detection]:
    model = get_model()
    results = model.predict(frame, classes=[PERSON_CLASS, CAT_CLASS], conf=conf_threshold, verbose=False)
    detections: list[Detection] = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = "person" if cls == PERSON_CLASS else "cat"
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
            detections.append(Detection(cls=cls, label=label, confidence=float(box.conf[0]), bbox=(x1, y1, x2, y2)))
    return detections
