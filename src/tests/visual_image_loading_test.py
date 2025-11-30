import os
import site
import sys
import numpy as np
import re
import pathlib

# Add the main path for module resolution
full_path = str(pathlib.Path(__file__).absolute())
base_path = re.sub("/src/tests/visual_image_loading_test.py$", "", full_path)
site.addsitedir(os.path.join(base_path, 'src'))

if __name__ == "__main__":
    from Utils.ImageTools import add_points_2_image, debug_plot_org_and_transformed_points, \
        get_fixed_transformed_image, get_random_transformed_image, image_backend, \
        load_image, fix_dimension_and_normalize

    print(f'Run image tests with backend {image_backend}')
    im, success = load_image(
        os.path.join(
            f'{base_path}',
            'test_data/imgs/Wrist/1st_export/Export 2006 to 2009/',
            '10072--23924--2009-10-13-09.04.19--left--Handledfrontalsin.png',
        ))

    if not success:
        print('Failed to load image in test_data')
        sys.exit(1)

    points = np.array([[0.05, 0.8], [0.95, 0.6], [0.9, 0.9], [0.1, 0.1]])
    # Add yellow large points so that we can see the true origins
    im = add_points_2_image(im=im, points=points, radius=5, color=(198, 198, 0))
    im, _ = fix_dimension_and_normalize(im=im, keep_aspect=True, scale_to=512, colorspace='Gray')

    def _transform_plot(im, ts_fn, title: str, no_examples: int, skip_pad: bool, skip_resize: bool):
        for j in range(no_examples):
            tr_im, spatial_transform = ts_fn(im, (256, 256), j, skip_pad=skip_pad, skip_resize=skip_resize)
            images_with_meta = []

            # Convert to the target image and plot points
            tr_points = spatial_transform.convert_original_points_to_processed_image_space(points.copy())
            images_with_meta.append({'name': 'transformed', 'im': tr_im, 'points': tr_points})

            # Revers the process back to the original image and plot points
            back_points = spatial_transform.convert_processed_image_points_to_original_space(tr_points)
            images_with_meta.append({'name': 'back 2 org', 'im': im, 'points': back_points})

            debug_plot_org_and_transformed_points(org_image=im, images_with_meta=images_with_meta, main_title=title)

    _transform_plot(im=im,
                    ts_fn=get_fixed_transformed_image,
                    title="Fixed - with padding",
                    no_examples=2,
                    skip_pad=False,
                    skip_resize=False)
    _transform_plot(im=im,
                    ts_fn=get_random_transformed_image,
                    title="Random - with padding",
                    no_examples=2,
                    skip_pad=False,
                    skip_resize=False)
    _transform_plot(im=im,
                    ts_fn=get_random_transformed_image,
                    title="Random - without padding",
                    no_examples=2,
                    skip_pad=True,
                    skip_resize=False)
    _transform_plot(im=im,
                    ts_fn=get_random_transformed_image,
                    title="Random - without padding and resize",
                    no_examples=2,
                    skip_pad=True,
                    skip_resize=True)
