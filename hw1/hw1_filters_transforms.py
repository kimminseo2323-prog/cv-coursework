import cv2
import numpy as np
import random
import os
#os.chdir("E://김민서//대학원//컴퓨터비전//HW//HW1_김민서//")
print(os.getcwd())

# 이미지 로드
image = cv2.imread('lenna/lenna.png')
cv2.imshow("Lenna_org", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ===========================
# 문제 1. Image Transformations
# ===========================
# 1-1
def translate(image, x, y):
    translation_matrix = np.float32([[1, 0, x],[0, 1, y]])# 이동 변환 행렬 생성
    transformed_image = cv2.warpAffine(image, translation_matrix, (image.shape[1], image.shape[0])) # 변환 적용
    return transformed_image

def rotate(image, angle):
    center = (image.shape[1] // 2, image.shape[0] // 2) # 중심점 설정
    transform_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0) # 회전 변환 행렬 생성
    transformed_image = cv2.warpAffine(image, transform_matrix, (image.shape[1], image.shape[0])) # 변환 적용
    return transformed_image

def AffineTransformation(image, source_match_point, target_match_point):
    transform_matrix = cv2.getAffineTransform(np.float32(source_match_point), np.float32(target_match_point)) # 어파인 변환 행렬 생성
    transformed_image = cv2.warpAffine(image, transform_matrix, (image.shape[1], image.shape[0])) # 변환 적용
    return transformed_image

def PerspectiveTransformation(image, source_match_point, target_match_point):
    transform_matrix = cv2.getPerspectiveTransform(np.float32(source_match_point), np.float32(target_match_point))# 원근 변환 행렬 생성
    transformed_image = cv2.warpPerspective(image, transform_matrix, (image.shape[1], image.shape[0])) # 변환 적용
    return transformed_image

# 테스트
translated = translate(image, 50, 30)
rotated = rotate(image, 45)
affine = AffineTransformation(image,source_match_point=[[50, 50], [200, 50], [50, 200]],target_match_point=[[10, 100], [200, 50], [100, 250]])
perspective = PerspectiveTransformation(image,source_match_point=[[50, 50], [200, 50], [50, 200], [200, 200]],target_match_point=[[10, 100], [200, 50], [100, 250], [220, 220]])

# 결과 확인 (cv2.imshow 사용)
cv2.imshow('Translated', translated)
cv2.imshow('Rotated', rotated)
cv2.imshow('Affine', affine)
cv2.imshow('Perspective', perspective)

cv2.waitKey(0)
cv2.destroyAllWindows()

# 이미지 저장
cv2.imwrite('Translated.png', translated)
cv2.imwrite('Rotated.png', rotated)
cv2.imwrite('Affine.png', affine)
cv2.imwrite('Perspective.png', perspective)



# ===========================
# 문제 2. Filters
# ===========================

# 2-1 Gaussian Filter
def Gaussian_filter(image):
    return cv2.GaussianBlur(image, (5, 5), sigmaX=1.0)

gaussian_result = Gaussian_filter(image)
cv2.imshow('Gaussian Filter', gaussian_result)
cv2.imwrite("gaussian_result.png", gaussian_result)

# 2-1 Sobel Filter
def Sobel(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    result = cv2.magnitude(sobel_x, sobel_y)
    return np.uint8(np.clip(result, 0, 255))

sobel_result = Sobel(image)
cv2.imshow('Sobel Filter', sobel_result)
cv2.imwrite("sobel_result.png", sobel_result)

# 2-1 Laplacian Filter
def Laplacian(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return np.uint8(np.clip(laplacian, 0, 255))

laplacian_result = Laplacian(image)
cv2.imshow('Laplacian Filter', laplacian_result)
cv2.imwrite("laplacian_result.png", laplacian_result)

# 2-2 Add Salt & Pepper Noise
def add_salt_pepper_noise(image, prob):
    result = np.copy(image)
    th = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rand = random.random()
            if rand < prob:
                result[i][j] = 0
            elif rand > th:
                result[i][j] = 255
    return result

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
noisy_image = add_salt_pepper_noise(gray_image, 0.05)
cv2.imshow('Salt & Pepper Noise', noisy_image)
cv2.imwrite("noisy_image.png", noisy_image)

# 2-2 Gaussian Filter on Noisy Image
gaussian_on_noise = Gaussian_filter(noisy_image)
cv2.imshow('Gaussian on Noisy', gaussian_on_noise)
cv2.imwrite("gaussian_on_noise.png", gaussian_on_noise)

# 2-3 Median Filter
def median_blur(image):
    return cv2.medianBlur(image, 5)

median_result = median_blur(noisy_image)
cv2.imshow('Median Filter', median_result)
cv2.imwrite("median_result.png", median_result)

# 창이 열려있는 상태에서 대기
cv2.waitKey(0)
cv2.destroyAllWindows()


# ===========================
# 문제 3. Image Pyramids
# ===========================

# 3-1 Resize with interpolation
def resize_interpolations(image):
    rows, cols = image.shape[:2]
    down1 = cv2.resize(image, (cols // 2, rows // 2), interpolation=cv2.INTER_LINEAR)
    down2 = cv2.resize(image, (cols // 2, rows // 2), interpolation=cv2.INTER_CUBIC)
    down3 = cv2.resize(image, (cols // 2, rows // 2), interpolation=cv2.INTER_AREA)

    up1 = cv2.resize(image, (cols * 2, rows * 2), interpolation=cv2.INTER_LINEAR)
    up2 = cv2.resize(image, (cols * 2, rows * 2), interpolation=cv2.INTER_CUBIC)
    up3 = cv2.resize(image, (cols * 2, rows * 2), interpolation=cv2.INTER_NEAREST)

    return down1, down2, down3, up1, up2, up3

down1, down2, down3, up1, up2, up3 = resize_interpolations(image)

cv2.imshow("Down - Linear", down1)
cv2.imwrite("resize_down_linear.png", down1)

cv2.imshow("Down - Cubic", down2)
cv2.imwrite("resize_down_cubic.png", down2)

cv2.imshow("Down - Area", down3)
cv2.imwrite("resize_down_area.png", down3)

cv2.imshow("Up - Linear", up1)
cv2.imwrite("resize_up_linear.png", up1)

cv2.imshow("Up - Cubic", up2)
cv2.imwrite("resize_up_cubic.png", up2)

cv2.imshow("Up - Nearest", up3)
cv2.imwrite("resize_up_nearest.png", up3)

# 모든 창 열어두기
cv2.waitKey(0)
cv2.destroyAllWindows()

# 3-2 Gaussian Pyramid
def gaussian_pyramid(image, levels=3):
    pyramid = [image]
    for i in range(1, levels + 1):
        image = cv2.pyrDown(image)
        pyramid.append(image)
        cv2.imshow(f"Gaussian Level {i}", image)
        cv2.imwrite(f"gaussian_level_{i}.png", image)
    return pyramid

gaussian_pyr = gaussian_pyramid(image, levels=3)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 3-3 Laplacian Pyramid + 복원
# 이미지 로딩 및 사이즈 고정 (2ⁿ x 2ⁿ 형태로 맞춤)
image_copy_org = image.copy()
image_copy_org_resize = cv2.resize(image_copy_org, (512, 512))  # 512 = 2^9

cv2.imshow("Original_resize", image_copy_org_resize)
cv2.imwrite("original_resize.png", image_copy_org_resize)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Gaussian Pyramid 생성
def gaussian_pyramid(image, levels=3):
    g_pyramid = [image]
    for i in range(levels):
        image = cv2.pyrDown(image)
        g_pyramid.append(image)
        cv2.imshow(f"Gaussian Level {i+1}", image)
        cv2.imwrite(f"gaussian_level_{i+1}.png", image)
    return g_pyramid

# 실행
g_pyr = gaussian_pyramid(image_copy_org_resize, levels=3)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Laplacian Pyramid 생성
def laplacian_pyramid(g_pyramid):
    l_pyramid = []
    for i in range(len(g_pyramid)-1, 0, -1):
        size = (g_pyramid[i-1].shape[1], g_pyramid[i-1].shape[0])
        gaussian_expanded = cv2.pyrUp(g_pyramid[i], dstsize=size)
        laplacian = cv2.subtract(g_pyramid[i-1], gaussian_expanded)
        l_pyramid.append(laplacian)
        cv2.imshow(f"Laplacian Level {len(g_pyramid)-1 - i + 1}", laplacian)
        cv2.imwrite(f"laplacian_level_{len(g_pyramid)-1 - i + 1}.png", laplacian)
    return l_pyramid

# Laplacian Pyramid 복원
def reconstruct_from_laplacian(l_pyramid, base_image):
    image = base_image
    for i in range(len(l_pyramid)-1, -1, -1):
        # pyrUp으로 업샘플링
        image = cv2.pyrUp(image)
        # Laplacian 레벨과 정확히 같은 크기로 보정
        image = cv2.resize(image, (l_pyramid[i].shape[1], l_pyramid[i].shape[0]))
        # 복원
        image = cv2.add(image, l_pyramid[i])
    return image

#실행
l_pyr = laplacian_pyramid(g_pyr)
# 복원 실행
reconstructed = reconstruct_from_laplacian(l_pyr, g_pyr[-1])
cv2.imshow("Reconstructed", reconstructed)
cv2.imwrite("reconstructed.png", reconstructed)

cv2.waitKey(0)
cv2.destroyAllWindows()

# 복원 이미지 사이즈 출력
# 크기 출력 (퀄리티 비교용)
print("크기 비교:")
print(f"원본: {image_copy_org_resize.shape}")
for i, g in enumerate(g_pyr):
    print(f"Gaussian Level {i}: {g.shape}")
for i, l in enumerate(l_pyr):
    print(f"Laplacian Level {i+1}: {l.shape}")
print(f"복원 이미지: {reconstructed.shape}")



# ===========================
# 문제 4. Median Blur 직접 구현
# ===========================
image = cv2.imread('lenna/lenna.png', cv2.IMREAD_GRAYSCALE)
cv2.imshow("Original", image)
cv2.imwrite("original_gray.png", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 2. Median Blur 직접 구현
def my_median_blur(image, size):
    pad = size // 2
    padded_image = np.pad(image, ((pad, pad), (pad, pad)), mode='constant', constant_values=0)
    result = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            window = padded_image[i:i + size, j:j + size]
            sorted_vals = np.sort(window, axis=None)
            result[i, j] = sorted_vals[len(sorted_vals) // 2]
    return result

# 3. 직접 구현한 Median Blur 적용
my_blurred = my_median_blur(image, size=3)

# 4. OpenCV 내장 medianBlur 적용 (검증용)
opencv_blurred = cv2.medianBlur(image, ksize=3)

# 5. 결과 출력 및 저장
cv2.imshow("My Median Blur (3x3)", my_blurred)
cv2.imwrite("my_median_blur_3x3.png", my_blurred)

cv2.imshow("OpenCV Median Blur (3x3)", opencv_blurred)
cv2.imwrite("opencv_median_blur_3x3.png", opencv_blurred)

# 6. 차이 이미지 확인 (절댓값 차이)
diff = cv2.absdiff(my_blurred, opencv_blurred)
cv2.imshow("Difference", diff)
cv2.imwrite("diff_my_vs_opencv.png", diff)

# 7. 수치 분석 (MSE)
mse = np.mean((my_blurred.astype(np.float32) - opencv_blurred.astype(np.float32)) ** 2)
print(f"MSE (Mean Squared Error) between my blur and OpenCV blur: {mse:.4f}")

cv2.waitKey(0)
cv2.destroyAllWindows()
