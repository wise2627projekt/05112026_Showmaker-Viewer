"""The AI helper. It runs in its OWN Python process (started by photovec.ai_client), so that PhotoVec itself stays small and keeps
working without any AI libraries, and so that a crash or a full graphics card can never take the main program down.

It only uses models that already exist - nothing is trained or invented here:

  * SAM 3 (Meta, "Segment Anything with Concepts"): finds EVERY instance of a concept described in words ("graffiti", "ornament",
    "window") - the semantic recognition. Needs the gated weights file `sam3.pt` (see README) and an NVIDIA graphics card.
  * YOLOE (Ultralytics, open-vocabulary segmentation): the same idea with a small model that downloads itself (~30 MB) and also
    runs on a normal CPU. Less precise than SAM 3; its boxes are sharpened by SAM 2.
  * SAM 2.1 (Meta) / MobileSAM: "click on the object" - from a point, a stroke or a box to a precise mask. Small, quick.

Protocol: one JSON object per line on stdin, one JSON answer per line on stdout. Images and masks travel as PNG files in a folder
that the caller names, so there is no size limit and nothing is encoded twice.
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from typing import Dict, List, Optional

# This file sits in PhotoVec's own package folder, and Python puts a script's folder first on the import path. PhotoVec has modules
# (clip.py, ...) that share a name with libraries the AI models import - so that folder must not be on the path in this process.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

# stdout is the protocol channel: whatever the libraries print must go to stderr
_PROTO = sys.stdout
sys.stdout = sys.stderr

SAM3_FILES = ("sam3.pt", "sam3.1_multiplex.pt")
TEXT_FALLBACK = "yoloe-26s-seg.pt"
# click / box models, best first. The larger SAM 2 models are clearly better on thin strokes (a tag with drips comes out whole where the
# tiny one takes the wall); the first one in this order that is already in the models folder is used, otherwise sam2.1_b.pt is
# downloaded once (160 MB).
CLICK_MODELS = ("sam2.1_b.pt", "sam2.1_s.pt", "sam2.1_t.pt", "mobile_sam.pt", "sam_b.pt")


def _reply(obj: dict) -> None:
    _PROTO.write(json.dumps(obj) + "\n")
    _PROTO.flush()


class Engines:
    def __init__(self, models_dir: str):
        self.models_dir = os.path.abspath(models_dir)
        os.makedirs(self.models_dir, exist_ok=True)
        os.chdir(self.models_dir)                       # ultralytics downloads weights by name into the current folder
        self._models: Dict[str, object] = {}
        self._img_key: Optional[str] = None
        self._predictor = None
        self._predictor_name = ""
        self.device = "cpu"
        try:
            import torch
            if torch.cuda.is_available():
                self.device = "cuda"
        except Exception:
            pass

    # ------------------------------------------------------------------ what is there
    def sam3_path(self) -> Optional[str]:
        for d in (self.models_dir, os.getcwd()):
            for n in SAM3_FILES:
                p = os.path.join(d, n)
                if os.path.exists(p):
                    return p
        return None

    def status(self) -> dict:
        out = {"device": self.device, "models_dir": self.models_dir, "sam3": self.sam3_path() is not None, "errors": []}
        try:
            import torch
            out["torch"] = torch.__version__
            out["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
        except Exception as exc:
            out["torch"] = ""
            out["errors"].append(f"PyTorch is not installed: {exc}")
        try:
            import ultralytics
            out["ultralytics"] = ultralytics.__version__
        except Exception as exc:
            out["ultralytics"] = ""
            out["errors"].append(f"ultralytics is not installed: {exc}")
        out["text_engines"] = (["sam3"] if out["sam3"] else []) + (["yoloe"] if out["ultralytics"] else [])
        out["click_engines"] = ["sam2"] if out["ultralytics"] else []
        return out

    # ------------------------------------------------------------------ helpers
    def _load(self, name: str, cls_name: str = "YOLOE"):
        key = f"{cls_name}:{name}"
        if key not in self._models:
            import ultralytics
            self._models[key] = getattr(ultralytics, cls_name)(name)
        return self._models[key]

    @staticmethod
    def _masks_to_png(results, shape, folder: str, tag: str) -> List[dict]:
        import cv2
        import numpy as np
        h, w = shape
        out: List[dict] = []
        r = results[0] if isinstance(results, (list, tuple)) else results
        if r is None or r.masks is None:
            return out
        data = r.masks.data.cpu().numpy()
        conf = r.boxes.conf.cpu().numpy() if getattr(r, "boxes", None) is not None and r.boxes is not None else None
        boxes = r.boxes.xyxy.cpu().numpy() if getattr(r, "boxes", None) is not None and r.boxes is not None else None
        for i, m in enumerate(data):
            m = (m > 0.5).astype(np.uint8) * 255
            if m.shape[:2] != (h, w):                   # models answer on their own working size
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
            p = os.path.join(folder, f"{tag}_{i}.png")
            cv2.imwrite(p, m)
            out.append({"path": p, "score": float(conf[i]) if conf is not None and i < len(conf) else 1.0,
                        "box": [float(v) for v in boxes[i]] if boxes is not None and i < len(boxes) else None})
        return out

    # ------------------------------------------------------------------ semantic: words -> masks
    def text(self, image: str, prompts: List[str], engine: str, conf: float, folder: str) -> dict:
        import cv2
        shape = cv2.imread(image).shape[:2]
        sam3 = self.sam3_path()
        if engine == "auto":
            engine = "sam3" if sam3 else "yoloe"
        if engine == "sam3":
            if not sam3:
                raise RuntimeError("SAM 3 weights not found: put sam3.pt into " + self.models_dir + " (see the README).")
            from ultralytics.models.sam import SAM3SemanticPredictor
            if self._predictor_name != "sam3":
                overrides = dict(conf=conf, task="segment", mode="predict", model=sam3,
                                 save=False, verbose=False, device=self.device)
                self._predictor = SAM3SemanticPredictor(overrides=overrides)
                self._predictor_name, self._img_key = "sam3", None
            if self._img_key != image + "|sam3":
                self._predictor.set_image(image)
                self._img_key = image + "|sam3"
            results = self._predictor(text=list(prompts))
        elif engine == "yoloe":
            model = self._load(TEXT_FALLBACK, "YOLOE")
            model.set_classes(list(prompts))
            results = model.predict(image, conf=conf, verbose=False, device=self.device, retina_masks=True)
        else:
            raise RuntimeError("unknown engine: " + engine)
        found = self._masks_to_png(results, shape, folder, engine)
        found.sort(key=lambda d: -d["score"])
        return {"engine": engine, "masks": found}

    # ------------------------------------------------------------------ click / stroke / box -> one precise mask
    def points(self, image: str, points: List[List[float]], labels: List[int], box: Optional[List[float]], folder: str,
               model_name: str) -> dict:
        import cv2
        shape = cv2.imread(image).shape[:2]
        name = model_name or next((n for n in CLICK_MODELS if os.path.exists(os.path.join(self.models_dir, n))), CLICK_MODELS[0])
        from ultralytics.models.sam import Predictor, SAM2Predictor
        key = image + "|" + name
        if self._predictor_name != name:
            cls = SAM2Predictor if name.startswith("sam2") else Predictor
            overrides = dict(conf=0.25, task="segment", mode="predict", imgsz=1024, model=name, save=False, verbose=False,
                             device=self.device)
            self._predictor = cls(overrides=overrides)
            self._predictor_name, self._img_key = name, None
        if self._img_key != key:                        # the picture is encoded once; every further click only runs the light decoder
            self._predictor.set_image(image)
            self._img_key = key
        kw = {}
        if points:
            kw["points"] = [list(p) for p in points]
            kw["labels"] = list(labels) if labels else [1] * len(points)
        if box:
            kw["bboxes"] = list(box)
        results = self._predictor(**kw)
        return {"engine": name, "masks": self._masks_to_png(results, shape, folder, "click")}


def prefetch(eng: "Engines") -> int:
    """Download every model once (called by the install script, so the first use in PhotoVec does not have to wait)."""
    failed = 0
    steps = [("SAM 2 (click / box)", lambda: eng._load(CLICK_MODELS[0], "SAM")),
             ("YOLOE (words)", lambda: eng._load(TEXT_FALLBACK, "YOLOE").set_classes(["graffiti"]))]
    for label, fn in steps:
        try:
            fn()
            print(f"OK    {label}", file=sys.stderr)
        except Exception as exc:
            failed += 1
            print(f"FAILED {label}: {type(exc).__name__}: {exc}", file=sys.stderr)
    print("SAM 3 (best word search): " + ("found" if eng.sam3_path() else "not installed - see the README, it needs a free access request"), file=sys.stderr)
    return failed


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    models_dir = args[0] if args else os.path.join(os.path.expanduser("~"), ".photovec", "ai", "models")
    eng = Engines(models_dir)
    if "--prefetch" in sys.argv:
        sys.exit(1 if prefetch(eng) else 0)
    _reply({"id": 0, "ok": True, "result": {"ready": True}})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except ValueError:
            continue
        rid, cmd = req.get("id"), req.get("cmd")
        try:
            if cmd == "quit":
                _reply({"id": rid, "ok": True, "result": {}})
                return
            if cmd == "status":
                res = eng.status()
            elif cmd == "text":
                res = eng.text(req["image"], req["prompts"], req.get("engine", "auto"), float(req.get("conf", 0.25)), req["folder"])
            elif cmd == "points":
                res = eng.points(req["image"], req.get("points") or [], req.get("labels") or [], req.get("box"), req["folder"],
                                 req.get("model", ""))
            else:
                raise RuntimeError("unknown command: " + str(cmd))
            _reply({"id": rid, "ok": True, "result": res})
        except Exception as exc:                        # one failed request must not stop the helper
            traceback.print_exc()
            _reply({"id": rid, "ok": False, "error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    main()
