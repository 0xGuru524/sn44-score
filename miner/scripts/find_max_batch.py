import torch

def max_batch_size(model, imgsz=(640,640), device='cuda'):
    lo, hi = 1, 512  # start with a reasonable upper bound
    best = 1
    while lo <= hi:
        mid = (lo + hi) // 2
        try:
            torch.cuda.empty_cache()
            _ = model.predict(source=torch.zeros(mid, 3, *imgsz, device=device),
                              batch=mid)
            best = mid
            lo = mid + 1
        except RuntimeError as e:
            if 'out of memory' in str(e):
                hi = mid - 1
            else:
                raise
    return best

# Example:
from ultralytics import YOLO
model = YOLO('../data/football-pitch-detection.pt').to('cuda')
print("Max batch:", max_batch_size(model, imgsz=(736,1280)))