# generate_fakes.py -- Medium-Strong +25 % (final tuned)
import cv2, os, numpy as np, random
from tqdm import tqdm

real_path = "./custom_dataset/real"
output_path = "./custom_dataset/fake"
os.makedirs(output_path, exist_ok=True)

def tuned_fake(img):
    h, w = img.shape[:2]
    out = img.copy()

    # 1️⃣ Slightly stronger warp
    pts1 = np.float32([[0,0],[w,0],[0,h],[w,h]])
    pts2 = pts1 + np.random.normal(0, 15, pts1.shape).astype(np.float32)
    M = cv2.getPerspectiveTransform(pts1, pts2)
    out = cv2.warpPerspective(out, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # 2️⃣ 25 % stronger color variation
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[...,0] += random.uniform(-12, 12)
    hsv[...,1] *= random.uniform(0.65, 1.5)
    hsv[...,2] *= random.uniform(0.7, 1.35)
    hsv = np.clip(hsv, 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # 3️⃣ Stronger edge-blending halo
    mask = np.zeros((h,w), np.uint8)
    cx, cy = w//2, h//2
    axes = (int(w*0.4), int(h*0.5))
    cv2.ellipse(mask,(cx,cy),axes,0,0,360,255,-1)
    mask = cv2.GaussianBlur(mask,(91,91),30)/255.0
    mask = mask[:,:,None]
    blur = cv2.GaussianBlur(out,(9,9),6)
    out = (out*mask + blur*(1-mask)).astype(np.uint8)

    # 4️⃣ Patchy blur regions (+25 %)
    for _ in range(random.randint(2,5)):
        x1 = random.randint(0,w-40)
        y1 = random.randint(0,h-40)
        x2 = min(w, x1+random.randint(50,130))
        y2 = min(h, y1+random.randint(50,130))
        patch = out[y1:y2, x1:x2]
        patch = cv2.GaussianBlur(patch,(random.choice([7,9]),random.choice([7,9])),4)
        out[y1:y2, x1:x2] = patch

    # 5️⃣ Heavier compression
    q = random.randint(25,45)
    _, enc = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    out = cv2.imdecode(enc, 1)

    # 6️⃣ Slightly stronger noise
    noise = np.random.normal(0, random.uniform(6,12), out.shape).astype(np.float32)
    out = np.clip(out + noise, 0, 255).astype(np.uint8)

    # 7️⃣ Sharpening retained for clarity
    if random.random() < 0.8:
        kernel = np.array([[0,-1,0],[-1,5.5,-1],[0,-1,0]])
        out = cv2.filter2D(out,-1,kernel)

    # 8️⃣ Bigger random drift
    if random.random() < 0.7:
        dx, dy = random.randint(-6,6), random.randint(-6,6)
        M2 = np.float32([[1,0,dx],[0,1,dy]])
        out = cv2.warpAffine(out,M2,(w,h),borderMode=cv2.BORDER_REFLECT)

    return out

# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------
images = [f for f in os.listdir(real_path) if f.lower().endswith(('.jpg','.jpeg','.png'))]
print(f"Found {len(images)} real images in {real_path}")
if not images:
    print("❌ No images found in real folder."); exit(1)

for i,fname in enumerate(tqdm(images,desc="Generating tuned fakes")):
    img = cv2.imread(os.path.join(real_path,fname))
    if img is None: continue
    fake = tuned_fake(img)
    cv2.imwrite(os.path.join(output_path,f"fake_{i:04d}.jpg"), fake)

print(f"\n✅ Generated {len(images)} tuned fake images in {output_path}")
