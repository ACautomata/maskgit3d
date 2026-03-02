"""
Main CLI entry point for maskgit3d.

Usage:
    maskgit3d --help
    maskgit3d train --help
    maskgit3d test --help
"""

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def train_command(args):
    """Run training command."""
    from maskgit3d.cli.train import main as train_main

    # Convert argparse args to sys.argv for Hydra
    sys.argv = ["train"] + args.config_overrides
    train_main()


def test_command(args):
    """Run testing command."""
    from maskgit3d.cli.test import main as test_main

    # Convert argparse args to sys.argv for Hydra
    sys.argv = ["test"] + args.config_overrides
    test_main()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="maskgit3d",
        description="MaskGIT 3D - Deep Learning Framework for Medical Image Segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Training
  maskgit3d train model=maskgit dataset=medmnist3d
  maskgit3d train model=vqgan dataset=brats training.num_epochs=50

  # Testing  
  maskgit3d test model=maskgit dataset=medmnist3d checkpoint.load_from=./checkpoints/best.ckpt
  maskgit3d test model=vqgan dataset=brats output.save_predictions=true

  # Using specific config directory
  maskgit3d --config-dir=/path/to/configs train model=custom

For more information, visit: https://github.com/yourusername/maskgit-3d
        """,
    )

    parser.add_argument(
        "--config-dir",
        type=str,
        default=None,
        help="Path to configuration directory (default: ./conf)",
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default="config",
        help="Name of the main configuration file (default: config)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser(
        "train",
        help="Run training",
        description="Train a model using Hydra configuration.",
    )
    train_parser.add_argument(
        "config_overrides",
        nargs="*",
        help="Hydra config overrides (e.g., model=maskgit training.num_epochs=100)",
    )

    # Test command
    test_parser = subparsers.add_parser(
        "test",
        help="Run testing/inference",
        description="Run testing or inference using Hydra configuration.",
    )
    test_parser.add_argument(
        "config_overrides",
        nargs="*",
        help="Hydra config overrides (e.g., model=vqgan checkpoint.load_from=./best.ckpt)",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Handle config directory override
    if args.config_dir:
        import os

        os.environ["HYDRA_CONFIG_PATH"] = args.config_dir

    if args.command == "train":
        train_command(args)
    elif args.command == "test":
        test_command(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
