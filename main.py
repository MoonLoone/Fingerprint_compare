import cv2
import numpy as np
from PIL import Image, ImageFilter


def binarize_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    return binary_image


def skeletonize_image(binary_image):
    skeleton = cv2.ximgproc.thinning(binary_image, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    return skeleton


def find_special_points(skeleton):
    # Примерный алгоритм для поиска ключевых точек
    # Здесь может быть реализован более сложный анализ
    points = []
    for i in range(skeleton.shape[0]):
        for j in range(skeleton.shape[1]):
            if skeleton[i, j] == 255:
                points.append((i, j))
    return points


def compare_fingerprints(points1, points2):
    match_score = 0
    for point in points1:
        if point in points2:
            match_score += 1
    return match_score


def test_image_yourself():
    binary_image1 = binarize_image('res/print_leha.png')
    skeleton1 = skeletonize_image(binary_image1)
    points1 = find_special_points(skeleton1)

    binary_image2 = binarize_image('res/print_leha.png')
    skeleton2 = skeletonize_image(binary_image2)
    points2 = find_special_points(skeleton2)

    match_score = compare_fingerprints(points1, points2)
    print(f'Степень соответствия: {match_score}')


def test_different():
    binary_image1 = binarize_image('res/print_leha.png')
    skeleton1 = skeletonize_image(binary_image1)
    points1 = find_special_points(skeleton1)

    binary_image2 = binarize_image('res/print_oleg.png')
    skeleton2 = skeletonize_image(binary_image2)
    points2 = find_special_points(skeleton2)

    match_score = compare_fingerprints(points1, points2)
    print(f'Степень соответствия: {match_score}')


def test_blurred_yourself():
    binary_image1 = binarize_image('res/print_leha.png')
    skeleton1 = skeletonize_image(binary_image1)
    points1 = find_special_points(skeleton1)

    blur_image("res/print_leha.png", "res/blurred.png")

    binary_image2 = binarize_image("res/blurred.png")
    skeleton2 = skeletonize_image(binary_image2)
    points2 = find_special_points(skeleton2)

    match_score = compare_fingerprints(points1, points2)
    print(f'Степень соответствия: {match_score}')


def blur_image(input_path, output_path):
    img = Image.open(input_path)
    blurred_img = img.filter(ImageFilter.GaussianBlur(radius=2))
    blurred_img.save(output_path)


def test_lighten_yourself():
    binary_image1 = binarize_image('res/print_leha.png')
    skeleton1 = skeletonize_image(binary_image1)
    points1 = find_special_points(skeleton1)

    img = Image.open('res/print_leha.png')
    img_array = np.array(img)
    lighting_image(img_array, 0.5)

    binary_image2 = binarize_image('res/lighten.png')
    skeleton2 = skeletonize_image(binary_image2)
    points2 = find_special_points(skeleton2)

    match_score = compare_fingerprints(points1, points2)
    print(f'Степень соответствия: {match_score}')


def lighting_image(img, coef):
    img = np.array(img) / 255  # Нормализация значений пикселей к диапазону [0, 1]
    img = img + img * coef  # Увеличение яркости на коэффициент
    img[img > 1] = 1  # Ограничение значений пикселей до 1
    lighted_img = Image.fromarray((img * 255).astype(np.uint8))
    lighted_img.save("res/lighten.png")


def main():
    test_image_yourself()
    test_different()
    test_blurred_yourself()
    test_lighten_yourself()


if __name__ == "__main__":
    main()
