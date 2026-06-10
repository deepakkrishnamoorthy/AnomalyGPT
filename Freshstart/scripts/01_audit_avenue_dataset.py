"""Audit the fresh Avenue dataset layout without heavy processing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import scipy.io


def video_info(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"path": str(path), "error": "could_not_open"}
    info = {
        "path": str(path),
        "video_id": path.stem,
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
    }
    cap.release()
    return info


def mat_info(path: Path) -> dict:
    mat = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
    payload = {"path": str(path), "video_id": path.stem.replace("vol", ""), "keys": []}
    for key, value in mat.items():
        if key.startswith("__"):
            continue
        payload["keys"].append(
            {
                "name": key,
                "shape": list(getattr(value, "shape", [])),
                "dtype": str(getattr(value, "dtype", "")),
            }
        )
    return payload


def audit(root: Path) -> dict:
    splits = {
        "training": {
            "videos": root / "training_videos",
            "volumes": root / "training_vol",
        },
        "testing": {
            "videos": root / "testing_videos",
            "volumes": root / "testing_vol",
        },
    }
    report = {
        "dataset_root": str(root),
        "splits": {},
        "annotation_status": {
            "found": False,
            "note": "No ground-truth annotation file was found in the fresh Avenue Dataset folder. Keep labels separate if supplied later.",
        },
    }
    for split, paths in splits.items():
        videos = [video_info(path) for path in sorted(paths["videos"].glob("*.avi"))]
        volumes = [mat_info(path) for path in sorted(paths["volumes"].glob("*.mat"))]
        report["splits"][split] = {
            "video_count": len(videos),
            "volume_count": len(volumes),
            "videos": videos,
            "volumes": volumes,
        }

        volume_frames = {
            item["video_id"]: item["keys"][0]["shape"][2]
            for item in volumes
            if item["keys"] and len(item["keys"][0]["shape"]) == 3
        }
        for video in videos:
            vol_count = volume_frames.get(video["video_id"])
            video["volume_frame_count_match"] = vol_count == video.get("frame_count")
            video["volume_frame_count"] = vol_count
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("Avenue Dataset"))
    parser.add_argument("--out", type=Path, default=Path("reports/avenue_fresh_audit.json"))
    args = parser.parse_args()

    report = audit(args.dataset_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Wrote {args.out}")
    for split, payload in report["splits"].items():
        print(f"{split}: videos={payload['video_count']} volumes={payload['volume_count']}")
    print(report["annotation_status"]["note"])


if __name__ == "__main__":
    main()

