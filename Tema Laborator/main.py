import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import misc
from scipy.fft import dctn, idctn
from dahuffman import HuffmanCodec


def rgb_to_ycbcr(X):
    transformation = np.array([[0.299, 0.587, 0.114],
                               [-0.1687, -0.3313, 0.5],
                               [0.5, -0.4187, -0.0813]])
    ycbcr = X.dot(transformation.T)
    ycbcr[:, :, [1, 2]] += 128
    return ycbcr


def ycbcr_to_rgb(X):
    transformation = np.array([[1.0, 0.0, 1.402],
                               [1.0, -0.3441, -0.7141],
                               [1.0, 1.772, 0.0]])
    rgb = X.copy()
    rgb[:, :, [1, 2]] -= 128
    rgb = rgb.dot(transformation.T)
    return rgb

# We want to make the matrix such that it can be divided into full 8x8 blocks


def change_shape(X):
    n = np.shape(X)[0]
    m = np.shape(X)[1]
    horizontal_blocks = n // 8
    vertical_blocks = m // 8

    if n % 8 != 0:
        horizontal_blocks += 1
    if m % 8 != 0:
        vertical_blocks += 1

    new_X = []
    if len(np.shape(X)) == 3:
        new_X = np.zeros((8*horizontal_blocks, 8*vertical_blocks, 3))
    else:
        new_X = np.zeros((8*horizontal_blocks, 8*vertical_blocks))

    for i in range(n):
        for j in range(m):
            new_X[i][j] = X[i][j]

    return new_X


def apply_dct_and_quantize(X, Q):
    X_jpeg = X

    horizontal_blocks = np.shape(X)[0] // 8
    vertical_blocks = np.shape(X)[1] // 8

    y_nnz = 0
    y_jpeg_nnz = 0

    for i in range(horizontal_blocks):
        for j in range(vertical_blocks):
            # Encoding
            x = X[8*i:8*(i+1), 8*j:8*(j+1)]
            y = dctn(x)
            y_jpeg = np.round(y/Q)

            # Decoding
            x_jpeg = idctn(y_jpeg)

            # Results
            y_nnz += np.count_nonzero(y)
            y_jpeg_nnz += np.count_nonzero(y_jpeg)

            X_jpeg[8*i:8*(i+1), 8*j:8*(j+1)] = x_jpeg

    return (X_jpeg, y_nnz, y_jpeg_nnz)


def zig_zag_vec(matrix):
    n = len(matrix)
    m = len(matrix[0])
    no_elements = 0
    ans = []

    i = 0
    j = 0
    direction = 1  # 1 going up-right, 0 going down-left
    while len(ans) < n*m:
        ans.append(matrix[i][j])

        if direction == 1:
            if i == 0:
                j += 1
                direction = 0
            elif j == m-1:
                i += 1
                direction = 0
            else:
                i -= 1
                j += 1
        else:
            if i == n-1:
                j += 1
                direction = 1
            elif j == 0:
                i += 1
                direction = 1
            else:
                i += 1
                j -= 1

    return ans


def huffman_encoding(data):
    codec = HuffmanCodec.from_data(data)
    encoded_data = codec.encode(data)

    original_size = len(bytearray(data))
    encoded_size = len(encoded_data)
    compression_ratio = encoded_size / original_size

    print(f"- Dimensiunea mesajului initial: {original_size}")
    print(f"- Dimensiunea mesajului encodat: {encoded_size}")
    print(f"- Rata de compresie: {compression_ratio}\n")


def ex1(X, Q):
    Q = Q.copy()
    X_copy = X.copy()

    X = change_shape(X)
    X, y_nnz, y_jpeg_nnz = apply_dct_and_quantize(X, Q)
    X = np.clip(X, 0, 255).astype(np.uint8)

    plt.subplot(121).imshow(X_copy, cmap=plt.cm.gray)
    plt.title('Original')
    plt.axis('off')

    plt.subplot(122).imshow(X, cmap=plt.cm.gray)
    plt.title('JPEG')
    plt.axis('off')

    plt.savefig('Plot_ex1.pdf')
    plt.clf()

    print('Resultate exercitiul 1:')
    print(f'- Componente în frecvență: {y_nnz}' +
          f'\n- Componente în frecvență după cuantizare: {y_jpeg_nnz}')
    huffman_encoding(zig_zag_vec(X))


def ex2(X, Q_luminance, Q_chrominance):
    Q_luminance = Q_luminance.copy()
    Q_chrominance = Q_chrominance.copy()
    X_copy = X.copy()

    X = change_shape(X)
    X = rgb_to_ycbcr(X)

    # Get each component
    Y = X[:, :, 0]
    Cb = X[:, :, 1]
    Cr = X[:, :, 2]

    # Apply DCT and quantize for each component
    Y, y_nnz_y, y_jpeg_nnz_y = apply_dct_and_quantize(Y, Q_luminance)
    Cb, y_nnz_cb, y_jpeg_nnz_cb = apply_dct_and_quantize(Cb, Q_chrominance)
    Cr, y_nnz_cr, y_jpeg_nnz_cr = apply_dct_and_quantize(Cr, Q_chrominance)

    y_nnz = y_nnz_y + y_nnz_cb + y_nnz_cr
    y_jpeg_nnz = y_jpeg_nnz_y + y_jpeg_nnz_cb + y_jpeg_nnz_cr

    X = np.stack((Y, Cb, Cr), axis=-1)
    X = ycbcr_to_rgb(X)
    X = np.clip(X, 0, 255).astype(np.uint8)

    plt.subplot(121).imshow(X_copy, cmap=plt.cm.gray)
    plt.title('Original')
    plt.axis('off')

    plt.subplot(122).imshow(X, cmap=plt.cm.gray)
    plt.title('JPEG')
    plt.axis('off')

    plt.savefig('Plot_ex2.pdf')
    plt.clf()

    print('Resultate exercitiul 2:')
    print(f'- Componente în frecvență: {y_nnz}' +
          f'\n- Componente în frecvență după cuantizare: {y_jpeg_nnz}')
    huffman_encoding(zig_zag_vec(X))


def ex3_black_and_white(X, mse_threshold, Q):
    Q = Q.copy()
    X = change_shape(X)
    original_X = X.copy()

    mse = float('inf')
    iteration = 0

    while mse > mse_threshold:
        X, y_nnz, y_jpeg_nnz = apply_dct_and_quantize(X, Q)
        X = np.clip(X, 0, 255).astype(np.uint8)

        if mse == np.mean((original_X-X)**2):
            break
        mse = np.mean((original_X-X)**2)
        Q = np.round(Q * 1.1).astype(int)
        iteration += 1

        print(f'Rezultate exercitiul 3 - alb-negru, etapa {iteration}:')
        print(f'- Componente în frecvență: {y_nnz}')
        print(f'- Componente în frecvență după cuantizare: {y_jpeg_nnz}')
        print(f'- MSE: {mse}\n')

    plt.subplot(121).imshow(original_X, cmap=plt.cm.gray)
    plt.title('Original')
    plt.axis('off')

    plt.subplot(122).imshow(X, cmap=plt.cm.gray)
    plt.title('Compressed')
    plt.axis('off')

    plt.savefig('Plot_ex3_black_and_white.pdf')
    plt.clf()


def ex3_color(X, mse_threshold, Q_luminance, Q_chrominance):
    Q_luminance = Q_luminance.copy()
    Q_chrominance = Q_chrominance.copy()

    X = change_shape(X)
    original_X = np.clip(X, 0, 255).astype(np.uint8)

    mse = float('inf')
    iteration = 0

    while mse > mse_threshold:
        X = rgb_to_ycbcr(X)
        Y = X[:, :, 0]
        Cb = X[:, :, 1]
        Cr = X[:, :, 2]

        Y, y_nnz_y, y_jpeg_nnz_y = apply_dct_and_quantize(Y, Q_luminance)
        Cb, y_nnz_cb, y_jpeg_nnz_cb = apply_dct_and_quantize(Cb, Q_chrominance)
        Cr, y_nnz_cr, y_jpeg_nnz_cr = apply_dct_and_quantize(Cr, Q_chrominance)

        y_nnz = y_nnz_y + y_nnz_cb + y_nnz_cr
        y_jpeg_nnz = y_jpeg_nnz_y + y_jpeg_nnz_cb + y_jpeg_nnz_cr

        X = np.stack((Y, Cb, Cr), axis=-1)
        X = ycbcr_to_rgb(X)
        X = np.clip(X, 0, 255).astype(np.uint8)

        if mse == np.mean((original_X-X)**2):
            break
        mse = np.mean((original_X-X)**2)
        Q_luminance = np.round(Q_luminance * 1.1).astype(int)
        Q_chrominance = np.round(Q_chrominance * 1.1).astype(int)
        iteration += 1

        print(f'Rezultate exercitiul 3 - color, etapa {iteration}:')
        print(f'- Componente în frecvență: {y_nnz}')
        print(f'- Componente în frecvență după cuantizare: {y_jpeg_nnz}')
        print(f'- MSE: {mse}\n')

    plt.subplot(121).imshow(original_X, cmap=plt.cm.gray)
    plt.title('Original')
    plt.axis('off')

    plt.subplot(122).imshow(X, cmap=plt.cm.gray)
    plt.title('Compressed')
    plt.axis('off')

    plt.savefig('Plot_ex3_color.pdf')
    plt.clf()


def compress_frame(X, Q_luminance, Q_chrominance):
    Q_luminance = Q_luminance.copy()
    Q_chrominance = Q_chrominance.copy()
    X_copy = X.copy()

    X = change_shape(X)
    X = rgb_to_ycbcr(X)

    # Get each component
    Y = X[:, :, 0]
    Cb = X[:, :, 1]
    Cr = X[:, :, 2]

    # Apply DCT and quantize for each component
    Y, _, _ = apply_dct_and_quantize(Y, Q_luminance)
    Cb, _, _ = apply_dct_and_quantize(Cb, Q_chrominance)
    Cr, _, _ = apply_dct_and_quantize(Cr, Q_chrominance)

    X = np.stack((Y, Cb, Cr), axis=-1)
    X = ycbcr_to_rgb(X)
    X = np.clip(X, 0, 255).astype(np.uint8)

    return X


def ex4(input_path, output_path, Q_luminance, Q_chrominance):
    input = cv2.VideoCapture(input_path)

    width = int(input.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(input.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = input.get(cv2.CAP_PROP_FPS)

    output = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

    while input.isOpened():
        ret, frame = input.read()
        if not ret:
            break
        output.write(compress_frame(frame, Q_luminance, Q_chrominance))

    input.release()
    output.release()
    cv2.destroyAllWindows()


Q_luminance = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                        [12, 12, 14, 19, 26, 28, 60, 55],
                        [14, 13, 16, 24, 40, 57, 69, 56],
                        [14, 17, 22, 29, 51, 87, 80, 62],
                        [18, 22, 37, 56, 68, 109, 103, 77],
                        [24, 35, 55, 64, 81, 104, 113, 92],
                        [49, 64, 78, 87, 103, 121, 120, 101],
                        [72, 92, 95, 98, 112, 100, 103, 99]])
Q_chrominance = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
                          [18, 21, 26, 66, 99, 99, 99, 99],
                          [24, 26, 56, 99, 99, 99, 99, 99],
                          [47, 66, 99, 99, 99, 99, 99, 99],
                          [99, 99, 99, 99, 99, 99, 99, 99],
                          [99, 99, 99, 99, 99, 99, 99, 99],
                          [99, 99, 99, 99, 99, 99, 99, 99],
                          [99, 99, 99, 99, 99, 99, 99, 99]])

# ex1(misc.ascent(), Q_luminance)
ex2(misc.face(), Q_luminance, Q_chrominance)
# ex3_black_and_white(misc.ascent(), 9000.8, Q_luminance)
# ex3_color(misc.face(), 107.2, Q_luminance, Q_chrominance)
# ex4('input_ex4.mp4', 'output_ex4.avi', Q_luminance, Q_chrominance)
