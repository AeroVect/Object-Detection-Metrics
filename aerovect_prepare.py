import os
import glob
import json


def load_json(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)
    return data


def prepare_groundtruth(labels_map, label_paths, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    label_files = sorted(glob.glob(label_paths))
    assert len(label_files) > 0, f"No label files found in {label_paths}"

    for label_file in label_files:
        out_file = f"{output_dir}/{label_file.split('/')[-1]}"

        with open(label_file, "r") as f:
            labels_text = f.readlines()

        out_lines = []
        for line in labels_text:
            line = line.split()
            class_id, x1, y1, x2, y2 = (
                str(line[0]),
                int(line[1]),
                int(line[2]),
                int(line[3]),
                int(line[4]),
            )
            if class_id not in labels_map.keys():
                continue

            w = x2 - x1
            h = y2 - y1

            out_lines.append([labels_map[class_id], x1, y1, w, h])

        with open(out_file, "w") as f:
            for line in out_lines:
                f.write(" ".join([str(l) for l in line]) + "\n")


def prepare_predictions(image_path, preds_file, image_ids_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    images = sorted(glob.glob(image_path))

    preds_text = []
    with open(preds_file, "r") as f:
        pred_lines = f.readlines()
        preds_per_image = []
        for line in pred_lines:
            if line == "---\n":
                preds_text.append(preds_per_image)
                preds_per_image = []
            elif line == "\n":
                continue
            else:
                preds_per_image.append(line)

    with open(image_ids_file, "r") as f:
        image_ids = [int(id) for id in f.readlines()]

    assert (
        len(images) == len(image_ids) == len(preds_text)
    ), f"Size mismatch {len(images)} == {len(image_ids)} == {len(preds_text)}"
    assert image_ids == sorted(image_ids), "image_ids are unsorted"

    for image_id in image_ids:
        out_file = (
            f"{output_dir}/{images[image_id].split('/')[-1].replace('.jpg', '.txt')}"
        )

        out_lines = []
        for pred in preds_text[image_id]:
            pred = pred.split()
            class_id, score, x1, y1, x2, y2 = (
                int(pred[0]),
                float(pred[1]),
                (float(pred[2])),
                (float(pred[3])),
                (float(pred[4])),
                (float(pred[5])),
            )

            w = x2 - x1
            h = y2 - y1

            out_lines.append([class_id, score, int(x1), int(y1), int(w), int(h)])

        with open(out_file, "w") as f:
            for line in out_lines:
                f.write(" ".join([str(l) for l in line]) + "\n")


def main():
    # CHANGE THESE PATHS IF NEEDED
    # Dataset and RT-DETR output paths
    dataset_path = (
        "/home/mei/repos/data/datasets/aerovect/aerovect_image_dataset/processed"
    )
    rtdetr_output_path = "/home/mei/repos/RT-DETR/rtdetrv2_pytorch/output/aerovect"

    labels_path = f"{dataset_path}/**/labels/*.txt"
    image_path = f"{dataset_path}/**/images/*.jpg"

    # Outputs of RT-DETR eval
    labels_map_path = f"{rtdetr_output_path}/labels_map.json"
    preds_file = f"{rtdetr_output_path}/predictions.txt"
    image_ids_file = f"{rtdetr_output_path}/image_ids.txt"

    # Output directories to store formatted groundtruth and prediction files
    output_gt_dir = f"{os.path.dirname(__file__)}/groundtruths"
    output_pred_dir = f"{os.path.dirname(__file__)}/detections"

    labels_map = load_json(labels_map_path)
    prepare_groundtruth(labels_map, labels_path, output_gt_dir)
    prepare_predictions(image_path, preds_file, image_ids_file, output_pred_dir)


if __name__ == "__main__":
    main()
