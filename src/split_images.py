from pathlib import Path
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

import exifread

RAW_IMAGE_DIR = Path("data/all_images")

OUT_DIR = Path("data")

TEST_SIZE = 0.20
RANDOM_SEED = 42

GRID_SIZE = 4  

FILENAME_COL = "file_name"
LAT_COL = "Latitude"
LON_COL = "Longitude"

VALID_EXTENSIONS = {".jpg", ".jpeg", ".JPG", ".JPEG"}

def get_exif_data(image_path):
    with open(image_path, "rb") as image_file:
        tags = exifread.process_file(image_file)
    return tags


def convert_to_decimal_degrees(value):
    d, m, s = value.values

    degrees = d.num / d.den
    minutes = m.num / m.den
    seconds = s.num / s.den

    return degrees + minutes / 60 + seconds / 3600


def extract_gps_from_image(image_path):
    exif_data = get_exif_data(image_path)

    gps_latitude = exif_data.get("GPS GPSLatitude")
    gps_latitude_ref = exif_data.get("GPS GPSLatitudeRef")
    gps_longitude = exif_data.get("GPS GPSLongitude")
    gps_longitude_ref = exif_data.get("GPS GPSLongitudeRef")

    if gps_latitude is None or gps_longitude is None:
        return None

    latitude = convert_to_decimal_degrees(gps_latitude)
    longitude = convert_to_decimal_degrees(gps_longitude)

    if gps_latitude_ref is not None and gps_latitude_ref.values[0] == "S":
        latitude = -latitude

    if gps_longitude_ref is not None and gps_longitude_ref.values[0] == "W":
        longitude = -longitude

    return latitude, longitude


def make_spatial_bins(df, grid_size=GRID_SIZE):
    lat_bins = pd.qcut(
        df[LAT_COL],
        q=grid_size,
        labels=False,
        duplicates="drop"
    )

    lon_bins = pd.qcut(
        df[LON_COL],
        q=grid_size,
        labels=False,
        duplicates="drop"
    )

    spatial_labels = lat_bins.astype(str) + "_" + lon_bins.astype(str)
    return spatial_labels


def safe_spatial_train_test_split(df):

    for grid_size in [GRID_SIZE, 3, 2]:
        try:
            spatial_labels = make_spatial_bins(df, grid_size=grid_size)
            counts = spatial_labels.value_counts()

            print(f"\nTrying grid_size={grid_size}")
            print("Smallest bin count:", counts.min())
            print("Number of spatial bins:", counts.shape[0])
            if counts.min() < 2:
                print("Trying smaller grid")
                continue

            train_df, test_df = train_test_split(
                df,
                test_size=TEST_SIZE,
                random_state=RANDOM_SEED,
                shuffle=True,
                stratify=spatial_labels,
            )
            return train_df, test_df, spatial_labels

        except Exception as e:
            print("Trying smaller grid...")
    print("Falling back to plain random split.")

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        shuffle=True,
    )

    return train_df, test_df, None

if not RAW_IMAGE_DIR.exists():
    raise FileNotFoundError(f"RAW_IMAGE_DIR does not exist: {RAW_IMAGE_DIR.resolve()}")

image_files = sorted([p for p in RAW_IMAGE_DIR.iterdir() if p.is_file()])

print(f"Found {len(image_files)} files in {RAW_IMAGE_DIR.resolve()}")

rows = []
skipped_bad_ext = []
skipped_no_gps = []

for image_path in image_files:
    if image_path.suffix not in VALID_EXTENSIONS:
        skipped_bad_ext.append(image_path.name)
        continue

    try:
        gps = extract_gps_from_image(image_path)
    except Exception as e:
        print(f"Could not read EXIF from {image_path.name}: {e}")
        skipped_no_gps.append(image_path.name)
        continue

    if gps is None:
        skipped_no_gps.append(image_path.name)
        continue

    latitude, longitude = gps

    rows.append({
        FILENAME_COL: image_path.name,
        LAT_COL: latitude,
        LON_COL: longitude,
    })

metadata_df = pd.DataFrame(rows)

print("\nMetadata creation complete.")
print(f"Images with usable GPS: {len(metadata_df)}")
print(f"Skipped unsupported extension: {len(skipped_bad_ext)}")
print(f"Skipped missing/unreadable GPS: {len(skipped_no_gps)}")

if len(metadata_df) == 0:
    raise ValueError(
        "No usable GPS metadata found. "
    )

OUT_DIR.mkdir(parents=True, exist_ok=True)

metadata_all_path = OUT_DIR / "metadata_all.csv"
metadata_df.to_csv(metadata_all_path, index=False)

print(f"\nFull metadata saved to: {metadata_all_path.resolve()}")

train_df, test_df, spatial_labels = safe_spatial_train_test_split(metadata_df)

train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

print("\nSplit sizes:")
print(f"train: {len(train_df)}")
print(f"test:  {len(test_df)}")

print("\nLocation range check:")

print("\nFull data:")
print(metadata_df[[LAT_COL, LON_COL]].describe())

print("\nTrain data:")
print(train_df[[LAT_COL, LON_COL]].describe())

print("\nTest data:")
print(test_df[[LAT_COL, LON_COL]].describe())

if spatial_labels is not None:
    temp_df = metadata_df.copy()
    temp_df["spatial_bin"] = spatial_labels

    train_names = set(train_df[FILENAME_COL])
    test_names = set(test_df[FILENAME_COL])

    temp_df["split"] = temp_df[FILENAME_COL].apply(
        lambda name: "train" if name in train_names else "test"
    )

    bin_counts = (
        temp_df
        .groupby(["spatial_bin", "split"])
        .size()
        .unstack(fill_value=0)
    )

    bin_counts["total"] = bin_counts.sum(axis=1)

    if "train" not in bin_counts.columns:
        bin_counts["train"] = 0

    if "test" not in bin_counts.columns:
        bin_counts["test"] = 0

    bin_counts["test_fraction"] = bin_counts["test"] / bin_counts["total"]

    split_check_path = OUT_DIR / "spatial_split_check.csv"
    bin_counts.to_csv(split_check_path)

    print(f"\nSpatial split check saved to: {split_check_path.resolve()}")
    print("\nSpatial bin counts:")
    print(bin_counts)

def write_split(split_name, split_df):
    split_dir = OUT_DIR / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    metadata_out = split_dir / "metadata.csv"
    split_df.to_csv(metadata_out, index=False)

    copied = 0
    skipped_existing = 0

    for _, row in split_df.iterrows():
        filename = str(row[FILENAME_COL])

        src = RAW_IMAGE_DIR / filename
        dst = split_dir / filename

        if not src.exists():
            print(f"WARNING: missing source image, skipping: {src}")
            continue

        if dst.exists():
            skipped_existing += 1
            continue

        shutil.copy2(src, dst)
        copied += 1

    print(f"\n{split_name}:")
    print(f"metadata saved to: {metadata_out.resolve()}")
    print(f"images copied: {copied}")
    print(f"images skipped because already existed: {skipped_existing}")


write_split("train", train_df)
write_split("test", test_df)

print("\nDone.")
print(f"Output folder: {OUT_DIR.resolve()}")
print(f"Train metadata: {(OUT_DIR / 'train' / 'metadata.csv').resolve()}")
print(f"Test metadata:  {(OUT_DIR / 'test' / 'metadata.csv').resolve()}")

print("\nFinal folder structure:")
print("data/")
print("  metadata_all.csv")
print("  spatial_split_check.csv")
print("  train/")
print("    metadata.csv")
print("    images...")
print("  test/")
print("    metadata.csv")
print("    images...")