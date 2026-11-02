import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
os.chdir("E://김민서//대학원//컴퓨터비전//HW//HW2_김민서//")
print(os.getcwd())


'''
Problem #1 Feature Descriptor and Feature Matching (30 pts)
'''
# Problem 1-1
# 이미지를 읽고, 파라미터를 설정하세요.
image = cv2.imread('paired_images/annapurna_left_01.png')
cv2.imshow("annapurna_left_01", image) # 이미지 확인
cv2.waitKey(0)
cv2.destroyAllWindows()


width, height, stride = 32, 32, 32
save_dir = 'result_1-1' # 패치 저장 폴더 생성
os.makedirs(save_dir, exist_ok=True) # 저장 폴더 생성
def make_patches(image, width, height, stride):
    patches = []
    keypoints = []
    H, W, _ = image.shape
    for h_i in range(0, H - height + 1, stride):
        for w_i in range(0, W - width + 1, stride):
            patch = image[h_i:h_i+height, w_i:w_i+width]
            center_x = w_i + width / 2.0
            center_y = h_i + height / 2.0
            kp = cv2.KeyPoint(center_x, center_y, (width + height) / 2.0)
            patches.append(patch)
            keypoints.append(kp)
            cv2.imwrite(
                os.path.join(save_dir, f'patch_{h_i}_{w_i}.png'),
                patch
            )
    return patches, keypoints
patches, keypoints = make_patches(image, width, height, stride)

# Problem 1-2
# 그레이디언트·히스토그램 추출 함수
def extract_grad_mag(patch):
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    # display 용으로 0–255 정규화
    return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def extract_color_hist(patch, bins):
    chans = cv2.split(patch)  # B, G, R 순
    hists = [cv2.calcHist([c], [0], None, [bins], [0,256]).flatten() for c in chans]
    return hists  # [hist_B, hist_G, hist_R]

out_dir = 'result_1-2'
os.makedirs(out_dir, exist_ok=True)
# 모든 패치에 대해 시각화 및 저장
for idx, patch in enumerate(patches):
    grad_mag = extract_grad_mag(patch)
    hist_B, hist_G, hist_R = extract_color_hist(patch, bins=32)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    # 원본 패치
    axes[0].imshow(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Patch')
    axes[0].axis('off')
    # Gradient magnitude
    axes[1].imshow(grad_mag, cmap='gray')
    axes[1].set_title('Gradient Magnitude')
    axes[1].axis('off')
    # Color histogram
    axes[2].plot(hist_R, color='r')
    axes[2].plot(hist_G, color='g')
    axes[2].plot(hist_B, color='b')
    axes[2].set_xlim([0, 31])
    axes[2].set_title('Color Histogram')
    axes[2].set_xlabel('Bin')
    axes[2].set_ylabel('Frequency')

    plt.tight_layout()
    save_path = os.path.join(out_dir, f'result_1-2_patch_{idx:03d}.png')
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

# Problem 1-3
# — 파라미터 설정 —
width, height, stride = 32, 32, 32
bins = 32
img_dir = 'paired_images'
out_dir = 'result_1-3'
os.makedirs(out_dir, exist_ok=True)

# — Brute-Force Matcher 생성 —
matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# — 매칭할 페어 지정 —
pairs = [
    ('scottsdale_left_01.png',    'scottsdale_right_01.png'),
    ('grand_canyon_left_01.png',  'grand_canyon_left_02.png')
]

for f1, f2 in pairs:
    img1 = cv2.imread(os.path.join(img_dir, f1))
    img2 = cv2.imread(os.path.join(img_dir, f2))

    # 1) 패치와 KeyPoint 생성 (문제 1-1 함수)
    patches1, kpts1 = make_patches(img1, width, height, stride)
    patches2, kpts2 = make_patches(img2, width, height, stride)

    # 2) 디스크립터 계산 (문제 1-2 함수)
    descs = {
        'patch_features':    ([p.flatten().astype(np.float32) for p in patches1],
                              [p.flatten().astype(np.float32) for p in patches2]),
        'gradient_features': ([extract_grad_mag(p).flatten().astype(np.float32) for p in patches1],
                              [extract_grad_mag(p).flatten().astype(np.float32) for p in patches2]),
        'histogram_features':([np.concatenate(extract_color_hist(p, bins)).astype(np.float32) for p in patches1],
                              [np.concatenate(extract_color_hist(p, bins)).astype(np.float32) for p in patches2]),
    }

    # 3) 각각의 디스크립터로 매칭 → drawMatches → 리스트에 저장
    draws = []
    for title, (d1, d2) in descs.items():
        matches = matcher.match(np.array(d1), np.array(d2))
        img_matches = cv2.drawMatches(
            img1, kpts1,
            img2, kpts2,
            matches[:20], None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        draws.append((title, cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB)))

    # 4) matplotlib으로 3행 1열 subplot 구성, 저장
    fig, axes = plt.subplots(3, 1, figsize=(12, 18))
    for ax, (title, im) in zip(axes, draws):
        ax.imshow(im)
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    out_path = os.path.join(
        out_dir,
        f"{os.path.splitext(f1)[0]}__{os.path.splitext(f2)[0]}__combined.png"
    )
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# Problem 1-4
#— 설정 파라미터 —
img_dir = 'paired_images'
out_dir = 'result_1-4'
os.makedirs(out_dir, exist_ok=True)

# — SIFT 디텍터 및 BFMatcher 생성 —
sift = cv2.xfeatures2d.SIFT_create()
matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# — 처리할 이미지 페어 —
pairs = [
    ('scottsdale_left_01.png',    'scottsdale_right_01.png'),
    ('grand_canyon_left_01.png',  'grand_canyon_left_02.png')
]

for f1, f2 in pairs:
    # 1) 이미지 로드
    img1 = cv2.imread(os.path.join(img_dir, f1))
    img2 = cv2.imread(os.path.join(img_dir, f2))

    # 2) SIFT 키포인트 & 디스크립터 계산
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # 3) 매칭 수행
    matches = matcher.match(des1, des2)

    # 4) 매칭 결과 그리기
    result = cv2.drawMatches(
        img1, kp1,
        img2, kp2,
        matches[:20], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    # 5) 결과 저장
    base1 = os.path.splitext(f1)[0]
    base2 = os.path.splitext(f2)[0]
    out_path = os.path.join(out_dir, f'{base1}__{base2}__sift.png')
    cv2.imwrite(out_path, result)

'''
Problem #2 Homography and Stitching (30 pts)
'''
# Problem 2-1
image_01 = cv2.imread('paired_images/scottsdale_left_01.png')
image_02 = cv2.imread('paired_images/scottsdale_right_01.png')

descriptor = cv2.xfeatures2d.SIFT_create()

keypoints_01, features_01 = descriptor.detectAndCompute(image_01, None)
keypoints_02, features_02 = descriptor.detectAndCompute(image_02, None)

matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
raw_matches = matcher.match(features_01, features_02)

pts1 = np.float32([ keypoints_01[m.queryIdx].pt for m in raw_matches ]).reshape(-1,1,2)
pts2 = np.float32([ keypoints_02[m.trainIdx].pt  for m in raw_matches ]).reshape(-1,1,2)
# 위 결과로 Homoraphy 행렬을 구합니다.
H, ok = cv2.findHomography(pts1, pts2, method=0)

print("Homography H:\n", H)

# Problem 2-2
H_ransac, mask = cv2.findHomography(pts1, pts2,method=cv2.RANSAC,ransacReprojThreshold=3.0)
print("RANSAC Homography H_ransac:\n", H_ransac)

# Problem 2-3
# 파라미터
img_dir = 'paired_images'
out_dir = 'result_2'
os.makedirs(out_dir, exist_ok=True)

# — SIFT 디텍터 & BFMatcher —
sift    = cv2.xfeatures2d.SIFT_create()
matcher = cv2.BFMatcher()

# — 처리할 이미지 페어 —
pairs = [
    ('grand_canyon_left_01.png',  'grand_canyon_right_01.png'),
    ('sedona_left_01.png',        'sedona_right_01.png'),
    ('annapurna_left_01.png',     'annapurna_right_01.png')
]

def stitch(img1, img2, H):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    canvas_w = w1 + w2
    canvas_h = max(h1, h2)
    # img2 워핑
    warped = cv2.warpPerspective(img2, H, (canvas_w, canvas_h))
    # img1 복사
    warped[0:h1, 0:w1] = img1
    return warped

for f1, f2 in pairs:
    # 1) 이미지 로드
    img1 = cv2.imread(os.path.join(img_dir, f1))
    img2 = cv2.imread(os.path.join(img_dir, f2))

    # 2) SIFT 키포인트 & 디스크립터
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # 3) 매칭 (ratio test)
    raw = matcher.knnMatch(des1, des2, k=2)
    good = [m for m,n in raw if m.distance < 0.75 * n.distance]

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

    # 4) Homography 공통 SVD 방식
    H_common, _ = cv2.findHomography(pts1, pts2, method=0)
    #    Homography RANSAC
    H_ransac, _ = cv2.findHomography(pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=3.0)

    # 5) 스티칭 & 저장
    base1 = os.path.splitext(f1)[0]
    base2 = os.path.splitext(f2)[0]

    result_common = stitch(img1, img2, H_common)
    cv2.imwrite(
        os.path.join(out_dir, f'{base1}__{base2}__common.png'),
        result_common
    )

    result_ransac = stitch(img1, img2, H_ransac)
    cv2.imwrite(
        os.path.join(out_dir, f'{base1}__{base2}__ransac.png'),
        result_ransac
    )



'''
Problem #3 Homography with RANSAC 직접 구현 (40 pts)
'''
# Problem 3-1
# — 파라미터 —
img_dir  = 'paired_images'
out_dir  = 'result_3'
os.makedirs(out_dir, exist_ok=True)
pairs = [
    ('grand_canyon_left_01.png', 'grand_canyon_right_01.png'),
    ('sedona_left_01.png',       'sedona_right_01.png')
]

# DLT 로 Homography 계산
def compute_homography_dlt(pts_src, pts_dst):
    A = []
    for (x,y), (xp,yp) in zip(pts_src, pts_dst):
        A.append([-x, -y, -1,  0,  0,  0, x*xp, y*xp, xp])
        A.append([ 0,  0,  0, -x, -y, -1, x*yp, y*yp, yp])
    A = np.asarray(A, dtype=np.float64)
    _,_,Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3,3)
    return H / H[2,2]

# RANSAC 으로 robust H 구하기
def ransac_homography(pts_src, pts_dst, thresh=4.0, max_iters=1500):
    best_H, best_inliers = None, []
    N = len(pts_src)
    src_h = np.hstack([pts_src, np.ones((N,1))])
    for _ in range(max_iters):
        idx = np.random.choice(N, 4, replace=False)
        Hc = compute_homography_dlt(pts_src[idx], pts_dst[idx])
        proj = (Hc @ src_h.T).T
        proj = proj[:, :2] / proj[:, 2:3]
        d = np.linalg.norm(proj - pts_dst, axis=1)
        inliers = np.where(d < thresh)[0]
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_H = Hc
    if best_H is None:
        raise RuntimeError("RANSAC failed")
    # 최종 H 재계산
    best_H = compute_homography_dlt(pts_src[best_inliers], pts_dst[best_inliers])
    mask = np.zeros(N, bool)
    mask[best_inliers] = True
    return best_H, mask

#3-2 문제
# 동적 캔버스 스티칭
def stitch_images(img1, img2, H):
    h1,w1 = img1.shape[:2]
    h2,w2 = img2.shape[:2]
    # img2 모서리 warp
    corners2 = np.array([[0,0],[0,h2],[w2,h2],[w2,0]],float).reshape(-1,1,2)
    warped2_c = cv2.perspectiveTransform(corners2, H)
    # img1 모서리
    corners1 = np.array([[0,0],[0,h1],[w1,h1],[w1,0]],float).reshape(-1,1,2)
    all_c = np.vstack((warped2_c, corners1))
    x_min,y_min = np.int32(all_c.min(axis=0).ravel() - .5)
    x_max,y_max = np.int32(all_c.max(axis=0).ravel() + .5)
    # translation
    T = np.array([[1,0,-x_min],[0,1,-y_min],[0,0,1]],float)
    pano_w, pano_h = x_max - x_min, y_max - y_min
    pano = cv2.warpPerspective(img2, T.dot(H), (pano_w, pano_h))
    pano[-y_min:h1 - y_min, -x_min:w1 - x_min] = img1
    return pano

# SIFT + 매칭
sift    = cv2.xfeatures2d.SIFT_create()
matcher = cv2.BFMatcher()

for f1,f2 in pairs:
    img1 = cv2.imread(os.path.join(img_dir, f1))
    img2 = cv2.imread(os.path.join(img_dir, f2))

    kp1,des1 = sift.detectAndCompute(img1, None)
    kp2,des2 = sift.detectAndCompute(img2, None)

    raw = matcher.knnMatch(des1, des2, k=2)
    good = [m for m,n in raw if m.distance < 0.75 * n.distance]

    # **순서를 바꿔서** img2→img1 로 H 추정
    pts2 = np.float64([kp2[m.trainIdx].pt for m in good])
    pts1 = np.float64([kp1[m.queryIdx].pt for m in good])

    H_manual, mask = ransac_homography(pts2, pts1, thresh=4.0, max_iters=1500)
    pano = stitch_images(img1, img2, H_manual)

    out_name = f"{os.path.splitext(f1)[0]}__{os.path.splitext(f2)[0]}__ransac_fixed.png"
    cv2.imwrite(os.path.join(out_dir, out_name), pano)