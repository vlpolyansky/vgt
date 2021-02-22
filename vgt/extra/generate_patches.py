import numpy as np
# from matplotlib import pyplot as plt
from tqdm import trange


def main():
    # Parameters
    npatches_per_image = 5000  # how many patches to take initially
    npatches_per_image_picked = 1000   # how many patches to take after filtering by d_norm
    sigma = 0.05
    images_filename = 'vanhateren_images.npy'
    # output_filename = f'data/vanhateren_patches.npy'
    output_filename = f'vanhateren_patches_p{npatches_per_image // npatches_per_image_picked}_s{sigma}.npy'

    images = np.load(images_filename)

    # plt.imshow(np.log(1 + images[0]), cmap='gray')
    # plt.show()

    # Make patches
    ncorners = 1022 * 1534    # number of possible patches from an image
    D = np.array([
        [2, -1, 0, -1, 0, 0, 0, 0, 0],
        [-1, 3, -1, 0, -1, 0, 0, 0, 0],
        [0, -1, 2, 0, 0, -1, 0, 0, 0],
        [-1, 0, 0, 3, -1, 0, -1, 0, 0],
        [0, -1, 0, -1, 4, -1, 0, -1, 0],
        [0, 0, -1, 0, -1, 3, 0, 0, -1],
        [0, 0, 0, -1, 0, 0, 2, -1, 0],
        [0, 0, 0, 0, -1, 0, -1, 3, -1],
        [0, 0, 0, 0, 0, -1, 0, -1, 2]
    ])
    all_patches = np.zeros((images.shape[0] * npatches_per_image_picked, 9))
    for i in trange(images.shape[0]):
        tmp = np.random.choice(ncorners, npatches_per_image, replace=False)
        xs = tmp % 1534
        ys = tmp // 1534
        patches = np.zeros((npatches_per_image, 3, 3), dtype=np.float32)
        for j in range(npatches_per_image):
            patches[j] = images[i, ys[j]:ys[j] + 3, xs[j]:xs[j] + 3]

        patches = np.log(1 + patches)
        patches = patches - np.mean(patches, axis=(1, 2), keepdims=True)
        patches = patches.reshape((patches.shape[0], 9))

        d_norms = np.sqrt(np.sum(np.dot(patches, D) * patches, axis=1))
        indices = np.argsort(-d_norms)

        indices = indices[:npatches_per_image_picked]
        patches = patches[indices] / d_norms[indices, None]

        all_patches[i * npatches_per_image_picked: (i + 1) * npatches_per_image_picked] = patches

    all_patches += np.random.normal(0, sigma, all_patches.shape)

    np.save(output_filename, all_patches.astype(np.float32))


if __name__ == '__main__':
    main()
