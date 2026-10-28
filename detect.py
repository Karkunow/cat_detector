from dataclasses import dataclass

from ultralytics import YOLO

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
    results = model.predict(frame, classes=[CAT_CLASS], conf=conf_threshold, verbose=False)
    detections: list[Detection] = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
            detections.append(Detection(cls=CAT_CLASS, label="cat", confidence=float(box.conf[0]), bbox=(x1, y1, x2, y2)))
    return detections


PERSON_CLASS = 0


def person_present(frames, conf_threshold: float = 0.25, step: int = 2) -> bool:
    """Людина в кадрах події (Слава міняє пакет у лотку): >=2 кадрів із conf>=0.25 або один із conf>=0.4. Кадри — сирі зображення."""
    model = get_model()
    hits = 0
    for f in frames[::step]:
        results = model.predict(f, classes=[PERSON_CLASS], conf=conf_threshold, verbose=False)
        confs = [float(b.conf[0]) for r in results for b in r.boxes]
        if confs:
            if max(confs) >= 0.4:
                return True
            hits += 1
            if hits >= 2:
                return True
    return False
