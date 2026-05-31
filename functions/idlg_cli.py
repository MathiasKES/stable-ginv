import argparse


def build_parser():
    """Build the iDLG experiment CLI parser."""
    parser = argparse.ArgumentParser()

    parser.add_argument("--mask_mode", type=str, default="gradsize_topfrac", help=(
        "Determines which gradient parameters are used during masked iDLG reconstruction. "
        "Options include gradsize_topk, gradsize_topfrac, gradsize_topk_entries, "
        "gradsize_topfrac_entries, gradsize_topfrac_entries_layer, gradsize_topk_entries_layer, "
        "prefix, and prefix_* combinations."
    ))
    parser.add_argument("--prefixes", type=str, default="conv1:1.0,layer1:1.0,layer2:1.0,layer3:1.0,fc:1.0", help=(
        "Comma-separated layer-name prefixes. Prefixes may include fractions, e.g. "
        "'conv1:1.0,layer1:0.5,fc:1.0'."
    ))
    parser.add_argument("--gradsize_topk", type=int, default=20, help=(
        "Number of parameters, tensors, or entries retained by top-k mask modes."
    ))
    parser.add_argument("--gradsize_topfrac", type=float, default=0.5, help=(
        "Fraction in (0, 1] retained by top-fraction mask modes."
    ))
    parser.add_argument("--gradsize_metric", type=str, default="l2", help=(
        "Metric used to rank gradient tensor size: l2, mean_abs, or sum_abs."
    ))
    parser.add_argument("--lr", type=float, default=1, help="Learning rate for reconstruction.")
    parser.add_argument("--grad_loss", type=str, default="cos", choices=["cos", "l2"], help=(
        "Gradient matching loss: cos or l2."
    ))
    parser.add_argument("--num_dummy", type=int, default=1, help=(
        "Number of dummy data samples optimized simultaneously."
    ))
    parser.add_argument("--iteration", type=int, default=1000, help=(
        "Maximum optimization iterations per reconstruction attack."
    ))
    parser.add_argument("--num_exp", type=int, default=10, help=(
        "Number of independent gradient-inversion experiments to run."
    ))
    parser.add_argument("--network", type=str, default="resnet18", help=(
        "Network architecture to attack, e.g. LeNet, MediumCNN, or resnet variants."
    ))
    parser.add_argument("--dataset", type=str, default="cifar100", help=(
        "Dataset to sample from: MNIST, cifar10, cifar100, or lfw."
    ))
    parser.add_argument("--run_id", type=int, default=0, help=(
        "Start offset for this experiment batch. Use the previous accumulated sample count "
        "to append a contiguous batch to the matching registry entry."
    ))
    parser.add_argument("--methods", type=str, default="idlg", choices=["idlg", "masked", "both"], help=(
        "Which reconstruction method(s) to run."
    ))
    parser.add_argument("--compute_jacobian_rank", action="store_true", help=(
        "Compute the Jacobian rank of observed gradients with respect to the input."
    ))
    parser.add_argument("--jacobian_max_entries", type=int, default=4000, help=(
        "Maximum observed gradient entries used for Jacobian-rank computation."
    ))
    parser.add_argument("--jacobian_select_mode", type=str, default="topk_abs",
                        choices=["topk_abs", "first", "random"], help=(
        "How observed gradient entries are selected for Jacobian-rank computation."
    ))
    parser.add_argument("--tv_weight", type=float, default=0.0, help=(
        "Total variation regularization weight."
    ))
    parser.add_argument("--optimizer", type=str, default="lbfgs",
                        choices=["lbfgs", "adam", "adamw", "signed_adam", "signed_adamw"], help=(
        "Optimizer used for dummy_data reconstruction."
    ))
    parser.add_argument("--num_restarts", type=int, default=1, help=(
        "Number of random restarts for each reconstruction."
    ))
    parser.add_argument("--max_iteration", type=int, default=20, help=(
        "Maximum iterations per LBFGS step. Only used with --optimizer lbfgs."
    ))
    parser.add_argument("--history_size", type=int, default=100, help=(
        "LBFGS history size. Only used with --optimizer lbfgs."
    ))
    parser.add_argument("--save_gif", action="store_true", help=(
        "Save an animated GIF showing reconstruction progress per experiment."
    ))
    parser.add_argument("--pretrained", action="store_true", help=(
        "Load ImageNet-pretrained torchvision weights for supported networks."
    ))
    parser.add_argument("--gamma", type=float, default=0.5, help="Gamma for learning-rate scheduler.")

    return parser


def parse_idlg_args(argv=None):
    """Parse and validate iDLG CLI args."""
    parser = build_parser()
    cli_args = argv[1:] if argv is not None else None
    args = parser.parse_args(cli_args)
    argv_for_checks = argv or []

    if args.optimizer != "lbfgs":
        if "--max_iteration" in argv_for_checks or "--history_size" in argv_for_checks:
            parser.error("--max_iteration and --history_size can only be used when --optimizer lbfgs")

    if not (0.0 < args.gradsize_topfrac <= 1.0):
        parser.error(f"--gradsize_topfrac must be in (0, 1], got {args.gradsize_topfrac}")

    methods_was_explicit = any(arg == "--methods" or arg.startswith("--methods=") for arg in argv_for_checks)
    if (
        args.mask_mode == "gradsize_topfrac_entries_layer"
        and args.methods == "idlg"
        and not methods_was_explicit
    ):
        args.methods = "masked"
        print("[INFO] --mask_mode gradsize_topfrac_entries_layer selected without --methods; running --methods masked.")

    return args
