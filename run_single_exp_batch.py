import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import consts

from masking import (get_keep_ids, get_keep_ids_by_gradsize, get_entry_masks_by_gradsize,
    get_prefix_keep_ids, get_entry_masks_by_prefix_group)
from metrics import compute_psnr_from_mse, compute_jacobian_rank, total_variation
from training_utils import build_network
from Network import weights_init


def run_single_experiment(idx_net, device_id, dst, dataset_name, config, result_queue):
    """Runs a single experiment on an assigned GPU."""

    torch.cuda.set_device(device_id)
    device = f"cuda:{device_id}"

    # -------- unpack config --------
    channel = config["channel"]
    num_classes = config["num_classes"]
    shape_img = config["shape_img"]
    lr = config["lr"]
    num_dummy = config["num_dummy"]
    Iteration = config["Iteration"]
    MASK_MODE = config["MASK_MODE"]
    PREFIXES = config.get("PREFIXES", ())
    GRADSIZE_TOPK = config["GRADSIZE_TOPK"]
    GRADSIZE_TOPFRAC = config["GRADSIZE_TOPFRAC"]
    GRADSIZE_THRESHOLD = config["GRADSIZE_THRESHOLD"]
    GRADSIZE_METRIC = config["GRADSIZE_METRIC"]
    NETWORK_NAME = config["NETWORK_NAME"]
    METHODS = config.get("METHODS", "both")
    COMPUTE_JACOBIAN_RANK = config.get("COMPUTE_JACOBIAN_RANK", False)
    JACOBIAN_MAX_ENTRIES = config.get("JACOBIAN_MAX_ENTRIES", 4000)
    JACOBIAN_SELECT_MODE = config.get("JACOBIAN_SELECT_MODE", "topk_abs")
    TV_WEIGHT = config.get("TV_WEIGHT", 0.0)
    OPTIMIZER = config.get("OPTIMIZER", "lbfgs")

    seed = config.get("run_id", 0) + idx_net + 1
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    net = build_network(
        NETWORK_NAME,
        channel=channel,
        num_classes=num_classes,
        input_size=shape_img,
    )
    if not NETWORK_NAME.startswith("resnet"):
        net.apply(weights_init)
    net = net.to(device)
    net.eval()

    print(f"[GPU {device_id}] Running experiment {idx_net}")

    idx_shuffle = np.random.permutation(len(dst))
    tt = transforms.Compose([transforms.ToTensor()])

    final_recon = {}
    early_stop_reason_dict = {}
    early_stop_iter_dict = {}

    best_loss_iDLG = None
    best_mse_iDLG = None
    best_psnr_iDLG = None

    best_loss_iDLG_masked = None
    best_mse_iDLG_masked = None
    best_psnr_iDLG_masked = None

    if METHODS == "idlg":
        methods_to_run = ["iDLG"]
    elif METHODS == "masked":
        methods_to_run = ["iDLG_masked"]
    else:
        methods_to_run = ["iDLG", "iDLG_masked"]

    jac_rank_iDLG = None
    jac_rank_iDLG_masked = None
    jac_shape_iDLG = None
    jac_shape_iDLG_masked = None

    # -------- helper metric --------
    def batch_metrics(x_rec, x_true):
        """
        Returns:
            mean_mse_over_images,
            mean_psnr_over_images
        """
        per_image_mse = ((x_rec - x_true) ** 2).view(x_true.size(0), -1).mean(dim=1)
        mean_mse = per_image_mse.mean().item()
        mean_psnr = float(
            np.mean([compute_psnr_from_mse(m.item(), max_val=1.0) for m in per_image_mse])
        )
        return mean_mse, mean_psnr

    for method in methods_to_run:
        print(f"[GPU {device_id}] {method}: trying to reconstruct {num_dummy} image(s)")

        criterion = nn.CrossEntropyLoss().to(device)
        imidx_list = []

        # -------- build GT batch --------
        gt_data_list = []
        gt_label_list = []

        for imidx in range(num_dummy):
            idx = idx_shuffle[imidx]
            imidx_list.append(idx)

            img = tt(dst[idx][0]).float().to(device).unsqueeze(0)
            lab = torch.tensor([dst[idx][1]], dtype=torch.long, device=device)

            gt_data_list.append(img)
            gt_label_list.append(lab)

        gt_data = torch.cat(gt_data_list, dim=0)
        gt_label = torch.cat(gt_label_list, dim=0)

        dm = torch.tensor(
            getattr(consts, f"{dataset_name.lower()}_mean"),
            device=device
        ).view(1, channel, 1, 1)

        ds = torch.tensor(
            getattr(consts, f"{dataset_name.lower()}_std"),
            device=device
        ).view(1, channel, 1, 1)

        # -------- compute original gradients --------
        gt_data_norm = (gt_data - dm) / ds
        out = net(gt_data_norm)
        y = criterion(out, gt_label)
        dy_dx = torch.autograd.grad(y, net.parameters())
        original_dy_dx = [g.detach() for g in dy_dx]

        # -------- dummy init --------
        dummy_data = torch.randn_like(gt_data, device=device, requires_grad=True)

        if OPTIMIZER == "lbfgs":
            optimizer = torch.optim.LBFGS([dummy_data], lr=lr)
        elif OPTIMIZER == "adam":
            optimizer = torch.optim.Adam([dummy_data], lr=lr)
        else:
            raise ValueError(f"Unknown optimizer: {OPTIMIZER}")

        # -------- labels --------
        # iDLG label inference is only valid in the single-sample case.
        if num_dummy == 1:
            label_pred = torch.argmin(
                torch.sum(original_dy_dx[-2], dim=-1), dim=-1
            ).detach().reshape((1,))
        else:
            label_pred = gt_label.detach().clone()

        candidate_ids = None
        keep_ids = None
        entry_masks = None
        ranked = None

        # -------- choose observed gradients --------
        if method == "iDLG":
            keep_ids = get_keep_ids("all", net=net)
        else:
            if MASK_MODE.startswith("prefix_"):
                candidate_ids = get_prefix_keep_ids(net, PREFIXES)

            if MASK_MODE == "gradsize_topk":
                keep_ids, ranked = get_keep_ids_by_gradsize(
                    original_dy_dx, mode="topk", topk=GRADSIZE_TOPK, metric=GRADSIZE_METRIC
                )

            elif MASK_MODE == "gradsize_topfrac":
                keep_ids, ranked = get_keep_ids_by_gradsize(
                    original_dy_dx, mode="topfrac", top_frac=GRADSIZE_TOPFRAC, metric=GRADSIZE_METRIC
                )

            elif MASK_MODE == "gradsize_topk_entries":
                entry_masks, observed_entries, total_entries = get_entry_masks_by_gradsize(
                    original_dy_dx, mode="topk_entries", topk=GRADSIZE_TOPK
                )

            elif MASK_MODE == "gradsize_topfrac_entries":
                entry_masks, observed_entries, total_entries = get_entry_masks_by_gradsize(
                    original_dy_dx, mode="topfrac_entries", top_frac=GRADSIZE_TOPFRAC
                )

            elif MASK_MODE == "prefix_topk":
                keep_ids, ranked = get_keep_ids_by_gradsize(
                    original_dy_dx,
                    mode="topk",
                    topk=GRADSIZE_TOPK,
                    metric=GRADSIZE_METRIC,
                    candidate_ids=candidate_ids,
                )

            elif MASK_MODE == "prefix_topfrac":
                keep_ids, ranked = get_keep_ids_by_gradsize(
                    original_dy_dx,
                    mode="topfrac",
                    top_frac=GRADSIZE_TOPFRAC,
                    metric=GRADSIZE_METRIC,
                    candidate_ids=candidate_ids,
                )

            elif MASK_MODE == "prefix_topk_entries":
                entry_masks, observed_entries, total_entries = get_entry_masks_by_gradsize(
                    original_dy_dx,
                    mode="topk_entries",
                    topk=GRADSIZE_TOPK,
                    candidate_ids=candidate_ids,
                )

            elif MASK_MODE == "prefix_topfrac_entries":
                entry_masks, observed_entries, total_entries = get_entry_masks_by_gradsize(
                    original_dy_dx,
                    mode="topfrac_entries",
                    top_frac=GRADSIZE_TOPFRAC,
                    candidate_ids=candidate_ids,
                )

            elif MASK_MODE == "prefix_topk_entries_layer":
                entry_masks, observed_entries, total_entries = get_entry_masks_by_prefix_group(
                    net=net,
                    original_dy_dx=original_dy_dx,
                    prefixes=PREFIXES,
                    mode="topk_entries",
                    topk=GRADSIZE_TOPK,
                )

            elif MASK_MODE == "prefix_topfrac_entries_layer":
                entry_masks, observed_entries, total_entries = get_entry_masks_by_prefix_group(
                    net=net,
                    original_dy_dx=original_dy_dx,
                    prefixes=PREFIXES,
                    mode="topfrac_entries",
                    top_frac=GRADSIZE_TOPFRAC,
                )

            elif MASK_MODE == "gradsize_threshold":
                keep_ids, ranked = get_keep_ids_by_gradsize(
                    original_dy_dx,
                    mode="threshold",
                    threshold=GRADSIZE_THRESHOLD,
                    metric=GRADSIZE_METRIC,
                )

            elif MASK_MODE == "prefix":
                keep_ids = get_keep_ids(mask_mode="prefix", net=net, prefixes=PREFIXES)

            else:
                keep_ids = get_keep_ids(MASK_MODE, net=net)

        # -------- counts --------
        unknowns = int(gt_data.numel())

        if entry_masks is not None:
            observed_entries = sum(int(m.sum().item()) for m in entry_masks if m is not None)
            total_entries = sum(g.numel() for g in original_dy_dx if g is not None)
        else:
            observed_entries = sum(
                g.numel()
                for i, g in enumerate(original_dy_dx)
                if g is not None and i in keep_ids
            )
            total_entries = sum(g.numel() for g in original_dy_dx if g is not None)

        kept_fraction = observed_entries / total_entries

        jacobian_rank = None
        jacobian_shape = None

        print(
            f"[GPU {device_id}] {method}: observed_entries={observed_entries}, "
            f"total_entries={total_entries}, kept_fraction={kept_fraction:.4f}, "
            f"unknowns={unknowns}"
        )

        if MASK_MODE in ["prefix_topfrac_entries_layer", "prefix_topk_entries_layer"] and entry_masks is not None:
            print(f"[GPU {device_id}] kept entries per prefix:")
            for prefix in PREFIXES:
                kept = 0
                total = 0
                for i, (name, _) in enumerate(net.named_parameters()):
                    if name.startswith(prefix) and entry_masks[i] is not None:
                        kept += int(entry_masks[i].sum().item())
                        total += entry_masks[i].numel()
                if total > 0:
                    print(f"  {prefix}: kept {kept}/{total} = {kept / total:.4f}")

        if COMPUTE_JACOBIAN_RANK:
            if num_dummy != 1:
                raise ValueError("Jacobian-rank computation currently assumes num_dummy=1.")

            jacobian_rank, jacobian_shape, jac_obs, jac_unknowns = compute_jacobian_rank(
                net=net,
                x_norm=gt_data_norm,
                y=gt_label,
                criterion=criterion,
                keep_ids=keep_ids,
                entry_masks=entry_masks,
                max_entries=JACOBIAN_MAX_ENTRIES,
                select_mode=JACOBIAN_SELECT_MODE,
                device_for_J="cpu",
            )
            print(
                f"[GPU {device_id}] {method}: jacobian_shape={jacobian_shape}, "
                f"jacobian_rank={jacobian_rank}, unknowns={jac_unknowns}"
            )

        if observed_entries < unknowns:
            print(
                f"[GPU {device_id}] {method}: too few gradients for reconstruction "
                f"({observed_entries} < {unknowns})"
            )

            final_recon[method] = torch.zeros_like(gt_data)

            if method == "iDLG":
                loss_iDLG = [float("inf")]
                label_iDLG = label_pred.detach().cpu().numpy()
                mse_iDLG = [float("inf")]
                psnr_iDLG = [float("-inf")]
                best_loss_iDLG = float("inf")
                best_mse_iDLG = float("inf")
                best_psnr_iDLG = float("-inf")
            else:
                loss_iDLG_masked = [float("inf")]
                label_iDLG_masked = label_pred.detach().cpu().numpy()
                mse_iDLG_masked = [float("inf")]
                psnr_iDLG_masked = [float("-inf")]
                best_loss_iDLG_masked = float("inf")
                best_mse_iDLG_masked = float("inf")
                best_psnr_iDLG_masked = float("-inf")

            early_stop_reason_dict[method] = "too_few_gradients"
            early_stop_iter_dict[method] = 0
            continue

        # -------- selected parameters --------
        all_params = list(net.parameters())

        if entry_masks is not None:
            selected_ids = [i for i, m in enumerate(entry_masks) if m is not None and m.any()]
        else:
            selected_ids = sorted(list(keep_ids))

        selected_params = [all_params[i] for i in selected_ids]
        selected_original = [original_dy_dx[i] for i in selected_ids]
        selected_entry_masks = [entry_masks[i] for i in selected_ids] if entry_masks is not None else None

        # -------- optimize --------
        losses = []
        mses = []
        psnrs = []

        es = config.get("EarlyStop", {})
        best_loss = float("inf")
        no_improve = 0

        patience = int(es.get("patience", 50))
        min_rel_improve = float(es.get("min_rel_improve", 1e-4))
        explode_factor = float(es.get("explode_factor", 500.0))
        warmup = int(es.get("warmup", 100))
        loss_tol = float(es.get("loss_tol", 1e-6))
        max_nan = int(es.get("max_nan", 1))

        nan_count = 0
        early_stop_reason = None
        early_stop_iter = None

        best_loss_value = float("inf")
        best_dummy = None
        best_mse_value = None
        best_psnr_value = None

        for iters in range(Iteration):
            if OPTIMIZER == "lbfgs":

                def closure():
                    optimizer.zero_grad()

                    x = torch.sigmoid(dummy_data)
                    x_norm = (x - dm) / ds

                    pred = net(x_norm)
                    dummy_loss = criterion(pred, gt_label)
                    dummy_dy_dx = torch.autograd.grad(
                        dummy_loss, selected_params, create_graph=True
                    )

                    grad_diff = 0.0
                    if selected_entry_masks is not None:
                        for gx, gy, m in zip(dummy_dy_dx, selected_original, selected_entry_masks):
                            diff = gx[m] - gy[m]
                            grad_diff = grad_diff + (diff ** 2).sum()
                    else:
                        for gx, gy in zip(dummy_dy_dx, selected_original):
                            grad_diff = grad_diff + ((gx - gy) ** 2).sum()

                    tv_loss = total_variation(x)
                    total_loss = grad_diff + TV_WEIGHT * tv_loss
                    total_loss.backward()
                    return total_loss

                current_loss = optimizer.step(closure).item()

            elif OPTIMIZER == "adam":
                optimizer.zero_grad()

                x = torch.sigmoid(dummy_data)
                x_norm = (x - dm) / ds

                pred = net(x_norm)
                dummy_loss = criterion(pred, gt_label)
                dummy_dy_dx = torch.autograd.grad(dummy_loss, selected_params, create_graph=True)

                grad_diff = 0.0
                if selected_entry_masks is not None:
                    for gx, gy, m in zip(dummy_dy_dx, selected_original, selected_entry_masks):
                        diff = gx[m] - gy[m]
                        grad_diff = grad_diff + (diff ** 2).sum()
                else:
                    for gx, gy in zip(dummy_dy_dx, selected_original):
                        grad_diff = grad_diff + ((gx - gy) ** 2).sum()

                tv_loss = total_variation(x)
                total_loss = grad_diff + TV_WEIGHT * tv_loss
                total_loss.backward()
                optimizer.step()

                current_loss = total_loss.item()

            else:
                raise ValueError(f"Unknown optimizer: {OPTIMIZER}")

            if np.isfinite(current_loss):
                current_x = torch.sigmoid(dummy_data)
                current_mse, current_psnr = batch_metrics(current_x, gt_data)

                if current_loss < best_loss_value:
                    best_loss_value = current_loss
                    best_dummy = current_x.detach().clone()
                    best_mse_value = current_mse
                    best_psnr_value = current_psnr

            if not np.isfinite(current_loss):
                nan_count += 1
                early_stop_reason = "nan_or_inf"
                early_stop_iter = iters
                print(f"[GPU {device_id}] Early stop ({method}): NaN/Inf at iter {iters}")
                if nan_count >= max_nan:
                    break
            else:
                if best_loss == float("inf"):
                    best_loss = current_loss
                    no_improve = 0
                else:
                    rel_improve = (best_loss - current_loss) / max(abs(best_loss), 1e-12)
                    if rel_improve > min_rel_improve:
                        best_loss = current_loss
                        no_improve = 0
                    elif iters >= warmup:
                        no_improve += 1

            if iters >= warmup and current_loss < loss_tol:
                early_stop_reason = "loss_tol"
                early_stop_iter = iters
                print(
                    f"[GPU {device_id}] Early stop ({method}): "
                    f"loss_tol reached at iter {iters} (loss={current_loss:.3e})"
                )
                break

            if iters >= warmup and best_loss < float("inf") and current_loss > explode_factor * best_loss:
                early_stop_reason = "explosion"
                early_stop_iter = iters
                print(
                    f"[GPU {device_id}] Early stop ({method}): "
                    f"exploded at iter {iters} (loss={current_loss:.3e}, best={best_loss:.3e})"
                )
                break

            if iters >= warmup and no_improve >= patience:
                early_stop_reason = "plateau"
                early_stop_iter = iters
                print(
                    f"[GPU {device_id}] Early stop ({method}): "
                    f"plateau at iter {iters} (best={best_loss:.3e})"
                )
                break

            losses.append(current_loss)

            current_x = torch.sigmoid(dummy_data)
            current_mse, current_psnr = batch_metrics(current_x, gt_data)
            mses.append(current_mse)
            psnrs.append(current_psnr)

            if iters % 100 == 0:
                print(
                    f"[GPU {device_id}] {OPTIMIZER} iters {iters}, "
                    f"loss = {current_loss:.8f}, mse = {current_mse:.8f}, psnr = {current_psnr:.4f}"
                )

        # -------- final recon --------
        if best_dummy is not None:
            final_recon[method] = best_dummy
        else:
            final_recon[method] = torch.sigmoid(dummy_data).detach().clone()
            fallback_mse, fallback_psnr = batch_metrics(final_recon[method], gt_data)
            if best_mse_value is None:
                best_mse_value = fallback_mse
            if best_psnr_value is None:
                best_psnr_value = fallback_psnr
            if best_loss_value == float("inf"):
                best_loss_value = current_loss if np.isfinite(current_loss) else float("inf")

        if method == "iDLG":
            loss_iDLG = losses
            label_iDLG = label_pred.detach().cpu().numpy()
            mse_iDLG = mses
            psnr_iDLG = psnrs
            best_loss_iDLG = best_loss_value
            best_mse_iDLG = best_mse_value
            best_psnr_iDLG = best_psnr_value
            jac_rank_iDLG = jacobian_rank
            jac_shape_iDLG = jacobian_shape
        else:
            loss_iDLG_masked = losses
            label_iDLG_masked = label_pred.detach().cpu().numpy()
            mse_iDLG_masked = mses
            psnr_iDLG_masked = psnrs
            best_loss_iDLG_masked = best_loss_value
            best_mse_iDLG_masked = best_mse_value
            best_psnr_iDLG_masked = best_psnr_value
            jac_rank_iDLG_masked = jacobian_rank
            jac_shape_iDLG_masked = jacobian_shape

        early_stop_reason_dict[method] = early_stop_reason
        early_stop_iter_dict[method] = early_stop_iter

    # -------- package results --------
    result = {
        "idx_net": idx_net,
        "device_id": device_id,
        "gt_data": gt_data.detach().cpu().numpy(),
        "final_recon": {k: v.detach().cpu().numpy() for k, v in final_recon.items()},

        "last_psnr_idlg": psnr_iDLG[-1] if "iDLG" in final_recon and len(psnr_iDLG) > 0 else None,
        "last_psnr_masked": psnr_iDLG_masked[-1] if "iDLG_masked" in final_recon and len(psnr_iDLG_masked) > 0 else None,

        "last_loss_iDLG": loss_iDLG[-1] if "iDLG" in final_recon and len(loss_iDLG) > 0 else None,
        "last_mse_iDLG": mse_iDLG[-1] if "iDLG" in final_recon and len(mse_iDLG) > 0 else None,

        "last_loss_iDLG_masked": loss_iDLG_masked[-1] if "iDLG_masked" in final_recon and len(loss_iDLG_masked) > 0 else None,
        "last_mse_iDLG_masked": mse_iDLG_masked[-1] if "iDLG_masked" in final_recon and len(mse_iDLG_masked) > 0 else None,

        "best_psnr_idlg": best_psnr_iDLG if best_psnr_iDLG is not None else None,
        "best_psnr_masked": best_psnr_iDLG_masked if best_psnr_iDLG_masked is not None else None,

        "best_loss_iDLG": best_loss_iDLG,
        "best_mse_iDLG": best_mse_iDLG,
        "best_loss_iDLG_masked": best_loss_iDLG_masked,
        "best_mse_iDLG_masked": best_mse_iDLG_masked,

        "label_iDLG": label_iDLG if "iDLG" in final_recon else None,
        "label_iDLG_masked": label_iDLG_masked if "iDLG_masked" in final_recon else None,

        "jac_rank_iDLG": jac_rank_iDLG if "iDLG" in final_recon else None,
        "jac_shape_iDLG": jac_shape_iDLG if "iDLG" in final_recon else None,
        "jac_rank_iDLG_masked": jac_rank_iDLG_masked if "iDLG_masked" in final_recon else None,
        "jac_shape_iDLG_masked": jac_shape_iDLG_masked if "iDLG_masked" in final_recon else None,

        "gt_label": gt_label.detach().cpu().numpy(),
        "imidx_list": imidx_list,
        "early_stop_reason": early_stop_reason_dict,
        "early_stop_iter": early_stop_iter_dict,
    }

    print(f"[GPU {device_id}] putting result for experiment {idx_net}", flush=True)
    result_queue.put(result)
    print(f"[GPU {device_id}] finished put for experiment {idx_net}", flush=True)