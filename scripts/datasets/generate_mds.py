from multi_dsprites import dataset
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import os
from PIL import Image

def pad(id, num_digits):
    return str(id).zfill(num_digits)

def filter_class_1(df):
    # A red square and a heart
    heart_image_ids = df[df['shape'] == 3]['image_id'].unique()

    red_square_image_ids = df[
        (df['shape'] == 1) &
        (df['color_r'] > 0.5) &
        (df['color_g'] < 0.25) &
        (df['color_b'] < 0.25)
    ]['image_id'].unique()

    common_ids = np.intersect1d(heart_image_ids, red_square_image_ids)

    return df[df['image_id'].isin(common_ids)]

def filter_class_2(df):
    # Two hearts on the left.
    left_hearts = df[(df['shape'] == 3) & (df['x'] < 0.5)]

    heart_counts = left_hearts.groupby('image_id').size()

    image_ids_with_2_left_hearts = heart_counts[heart_counts >= 2].index

    return df[df['image_id'].isin(image_ids_with_2_left_hearts)]

def filter_class_3(df):
    # An ellipse and two squares
    ellipse_image_ids = df[df['shape'] == 2]['image_id'].unique()

    square_df = df[df['shape'] == 1]
    square_counts = square_df.groupby('image_id').size()
    square_image_ids = square_counts[square_counts >= 2].index

    common_ids = np.intersect1d(ellipse_image_ids, square_image_ids)

    return df[df['image_id'].isin(common_ids)]


def filter_class_4(df, bg_thresh=0.3, bright_thresh=0.6):
    # A bright object infront of a dark background 
    bg_df = df[df['shape'] == 0]
    dark_bg_image_ids = bg_df[
        (bg_df['color_r'] < bg_thresh) &
        (bg_df['color_g'] < bg_thresh) &
        (bg_df['color_b'] < bg_thresh)
    ]['image_id'].unique()

    bright_fg_df = df[
        (df['shape'] != 0) & (
            (df['color_r'] > bright_thresh) |
            (df['color_g'] > bright_thresh) |
            (df['color_b'] > bright_thresh)
        )
    ]
    bright_fg_image_ids = bright_fg_df['image_id'].unique()

    matching_ids = np.intersect1d(dark_bg_image_ids, bright_fg_image_ids)

    return df[df['image_id'].isin(matching_ids)]

def filter_class_5(df):
    # 3 different shapes on the right side
    def has_three_shapes_right(group):
        right_shapes = group[(group['shape'] != 0) & (group['x'] >= 0.5)]
        return right_shapes['shape'].nunique() >= 3

    grouped = df.groupby('image_id')
    valid_ids = [image_id for image_id, group in grouped if has_three_shapes_right(group)]

    return df[df['image_id'].isin(valid_ids)]

def split_ids(ids_list, train_frac=0.7, val_frac=0.15, shuffle=True):
    ids = np.array(ids_list)
    if shuffle:
        np.random.shuffle(ids)

    n = len(ids)
    train_end = int(train_frac * n)
    val_end = train_end + int(val_frac * n)

    train_ids = ids[:train_end]
    val_ids = ids[train_end:val_end]
    test_ids = ids[val_end:]

    return train_ids, val_ids, test_ids

if __name__ == "__main__":
    tf.disable_eager_execution()
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Generate MDS for multi-dsprites dataset.")
    parser.add_argument("--record_path", type=str, help="Path to the dataset.")
    parser.add_argument("--save_path", type=str, default="./data/mds", help="Path to save the MDS results.")
    parser.add_argument("--count", type=int, default=280000, help="Number of images to process.")
    args = parser.parse_args()

    # Pull in all data
    path = args.record_path
    data = dataset(path, "colored_on_colored")
    iterator = tf.data.make_one_shot_iterator(data)

    rows = []
    images = []

    task = iterator.get_next()

    with tf.Session() as sess:
        image_counter = 0
        for _ in range(args.count):
            try:
                sample = sess.run(task)
                num_entities = len(sample['shape'])

                for i in range(num_entities):
                    row = {
                        'image_id': image_counter,
                        'entity_id': i,
                        'shape': int(round(sample['shape'][i])),
                        'x': sample['x'][i],
                        'visibility': sample['visibility'][i],
                    }

                    color = sample['color'][i]
                    if len(color) == 3:
                        row.update({
                            'color_r': color[0],
                            'color_g': color[1],
                            'color_b': color[2],
                        })
                    else:
                        row['color_gray'] = color[0]

                    rows.append(row)

                # Optionally store image (decoded as uint8)
                images.append(sample['image'])

                image_counter += 1

            except tf.errors.OutOfRangeError:
                break

    df = pd.DataFrame(rows)

    visible_df = df[df['visibility'] == 1]

    # Filter out
    df1 = filter_class_1(visible_df)
    df2 = filter_class_2(visible_df)
    df3 = filter_class_3(visible_df)
    df4 = filter_class_4(visible_df)
    df5 = filter_class_5(visible_df)

    # Get image_id sets
    ids_1 = set(df1['image_id'].unique())
    ids_2 = set(df2['image_id'].unique())
    ids_3 = set(df3['image_id'].unique())
    ids_4 = set(df4['image_id'].unique())
    ids_5 = set(df5['image_id'].unique())

    # Remove overlaps
    ids_2 -= ids_1
    ids_3 -= (ids_1 | ids_2)
    ids_4 -= (ids_1 | ids_2 | ids_3)
    ids_5 -= (ids_1 | ids_2 | ids_3 | ids_4)

    count = min(len(ids_1), len(ids_2), len(ids_3), len(ids_4), len(ids_5))
    ids_1_list = list(ids_1)[:count]
    ids_2_list = list(ids_2)[:count]
    ids_3_list = list(ids_3)[:count]
    ids_4_list = list(ids_4)[:count]
    ids_5_list = list(ids_5)[:count]

    set_1 = split_ids(ids_1_list)
    set_2 = split_ids(ids_2_list)
    set_3 = split_ids(ids_3_list)
    set_4 = split_ids(ids_4_list)
    set_5 = split_ids(ids_5_list)

    # Save images
    for i, split in enumerate(["train", "val", "test"]):
        print(f"Processing split: {split}")

        output_base_dir = os.path.join(args.save_path, split)
        if not os.path.exists(output_base_dir):
            os.makedirs(output_base_dir)
        
        class_to_ids = {
            'class_0': set_1[i],
            'class_1': set_2[i],
            'class_2': set_3[i],
            'class_3': set_4[i],
            'class_4': set_5[i],
        }

        for class_name, id_list in class_to_ids.items():
            for i, image_id in enumerate(id_list):
                img_array = images[image_id]
                
                img = Image.fromarray(img_array)
                
                filename = f"{class_name}_{pad(i, 4)}.png"
                filepath = os.path.join(output_base_dir, filename)

                img.save(filepath)

    print(f"MDS generation completed and saved to: {args.save_path}")
