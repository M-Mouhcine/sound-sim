from skimage.draw import random_shapes


def generate_random_map(map_size, random_seed):
    """Randomly generate an array of zeros (free media) and ones (obstacles).
    The obstacles have basic geometric shapes.

    Parameters:
        map_size(tuple): shape of the map to generate
        random_seed (int): random seed for random generation of obstacles
    Returns:
        random_map (np.array): array shaped as map_size, containing random
                               obstacles.
    """
    result = random_shapes(
        map_size,
        intensity_range=(0, 1),
        min_size=8,
        max_size=15,
        min_shapes=2,
        max_shapes=10,
        rng=random_seed,
        num_channels=1,
        allow_overlap=False,
    )
    # result is a tuple consisting of
    # # (1) the image with the generated shapes
    # # (2) a list of label tuples with the kind of shape
    # (e.g. circle, rectangle) and ((r0, r1), (c0, c1)) coordinates.
    obstacle_map, labels = result
    # Force free media in a square of 20x20 at the center of the map
    width_center = map_size[0] // 2
    length_center = map_size[1] // 2
    obstacle_map[
        width_center - 20 : width_center + 21,
        length_center - 20 : length_center + 21,
    ] = 255
    free_media = obstacle_map == 255
    # Obstacles = 1, free media = 0
    obstacles = obstacle_map == 0
    obstacle_map[free_media] = 0
    obstacle_map[obstacles] = 1
    return obstacle_map[..., 0]
