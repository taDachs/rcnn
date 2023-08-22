#!/usr/bin/env python3

import tensorflow as tf
import matplotlib  # type: ignore
import argparse

from rcnn.faster_rcnn import FasterRCNN
from rcnn.data import pascal_voc, kitti
from rcnn.vis import vis_single_image, visualize_model_output


def load_dataset(args):
    if args.dataset == "voc":
        return pascal_voc(args.img_size)
    elif args.dataset == "kitti":
        return kitti(args.img_size)
    else:
        print(f"Invalid dataset name: {args.dataset}")
        exit(1)


def train(args):
    model = FasterRCNN(
        anc_sizes=args.anc_sizes,
        anc_ratios=args.anc_ratios,
        num_classes=args.num_classes,
        roi_size=args.roi_size,
        l2=args.l2,
        rpn_foreground_iou_thresh=args.rpn_pos_thresh,
        rpn_background_iou_thresh=args.rpn_neg_thresh,
        detector_foreground_iou_thresh=args.detector_pos_thresh,
        detector_background_iou_thresh=args.detector_neg_thresh,
        rpn_batch_size=args.rpn_batch_size,
        detector_batch_size=args.detector_batch_size,
        backbone_type=args.backbone,
    )

    if args.weights_path:
        model.load_weights(args.weights_path)

    ds_train, _, _ = load_dataset(args)

    callbacks = [
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.ModelCheckpoint(
            args.checkpoint_path, "loss", save_best_only=True, save_weights_only=True
        ),
    ]

    if args.optimizer == "adam":
        optimizer = tf.keras.optimizers.Adam(args.lr)
    elif args.optimizer == "sgd":
        optimizer = tf.keras.optimizers.SGD(
            args.lr, momentum=args.momentum, weight_decay=args.weight_decay
        )

    model.compile(optimizer)
    model.fit(ds_train, epochs=args.epochs, workers=args.workers, callbacks=callbacks)

    model.save_weights(args.save_weights_path)


def eval_on_dataset(args):
    model = FasterRCNN(
        anc_sizes=args.anc_sizes,
        anc_ratios=args.anc_ratios,
        num_classes=args.num_classes,
        roi_size=args.roi_size,
    )
    model.load_weights(args.weights_path)
    _, ds_val, label_mapping = load_dataset(args)
    visualize_model_output(ds_val, model, label_mapping)


def eval_single_image(args):
    model = FasterRCNN(
        anc_sizes=args.anc_sizes,
        anc_ratios=args.anc_ratios,
        num_classes=args.num_classes,
        roi_size=args.roi_size,
    )
    _, _, label_mapping = load_dataset(args)
    model.load_weights(args.weights_path)
    img = tf.keras.utils.load_img(args.image_path)
    img = tf.keras.utils.img_to_array(img) / 255
    vis_single_image(img, model, label_mapping)


def main():
    parser = argparse.ArgumentParser(description="Faster R-CNN Toolset")
    subparsers = parser.add_subparsers(title="modes", dest="mode")

    # Common arguments
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument("--anc_ratios", type=float, nargs="+", default=(0.5, 1.0, 2.0))
    common_parser.add_argument("--anc_sizes", type=int, nargs="+", default=(128, 256, 512))
    common_parser.add_argument("--img_size", type=int, default=600)
    common_parser.add_argument("--num_classes", type=int, default=21)
    common_parser.add_argument("--roi_size", type=int, default=7)
    common_parser.add_argument("--dataset", type=str, choices=["kitti", "voc"], default="voc")
    common_parser.add_argument(
        "--backbone", type=str, choices=["vgg", "resnet", "mobilenet"], default="vgg"
    )

    # Training subparser
    train_parser = subparsers.add_parser("train", parents=[common_parser], help="Train the model")
    train_parser.add_argument("--optimizer", type=str, choices=["adam", "sgd"], default="sgd")
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--momentum", type=float, default=0.9)
    train_parser.add_argument("--weight_decay", type=float, default=5e-4)
    train_parser.add_argument("--epochs", type=int, default=12)
    train_parser.add_argument("--workers", type=int, default=14)
    train_parser.add_argument("--checkpoint_path", type=str, default="./checkpoints/chkpt")
    train_parser.add_argument("--save_weights_path", type=str, default="./test_new_anchors/weights")
    train_parser.add_argument(
        "--weights-path", type=str, required=False, help="Path to model weights"
    )

    # Evaluation subparser
    eval_parser = subparsers.add_parser(
        "eval", parents=[common_parser], help="Evaluate the model on test data"
    )
    eval_parser.add_argument(
        "--weights-path", type=str, required=True, help="Path to model weights"
    )

    # Single Image Evaluation subparser
    eval_img_parser = subparsers.add_parser(
        "eval_single_image", parents=[common_parser], help="Evaluate the model on a single image"
    )
    eval_img_parser.add_argument(
        "--image-path", type=str, required=True, help="Path to the image for evaluation"
    )
    eval_img_parser.add_argument(
        "--weights-path", type=str, required=True, help="Path to model weights"
    )

    args = parser.parse_args()
    tf.random.set_seed(1337)
    matplotlib.use("GTK3Agg")  # Or any other X11 back-end

    if args.mode == "train":
        train(args)
    elif args.mode == "eval":
        eval_on_dataset(args)
    elif args.mode == "eval_single_image":
        eval_single_image(args)
    else:
        print("Please select a valid mode: train, eval, or eval_single_image")
        exit(1)


if __name__ == "__main__":
    main()
