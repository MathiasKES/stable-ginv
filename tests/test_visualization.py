from stable_ginv.viz import panels


def test_flush_recon_panel_sorts_columns_by_experiment_index(monkeypatch, tmp_path):
    buffers = panels.create_panel_buffers()
    for idx in (2, 0, 1):
        buffers["exp_idx"].append(idx)
        buffers["gt"].append(f"gt-{idx}")
        buffers["idlg"].append(f"idlg-{idx}")
        buffers["masked"].append(f"masked-{idx}")
        buffers["psnr_idlg"].append(idx + 0.1)
        buffers["ssim_idlg"].append(idx + 0.2)
        buffers["mse_idlg"].append(idx + 0.3)
        buffers["psnr_masked"].append(idx + 0.4)
        buffers["ssim_masked"].append(idx + 0.5)
        buffers["mse_masked"].append(idx + 0.6)

    captured = {}

    def fake_save_recon_panel(params, gt, idlg, masked, *args, **kwargs):
        captured["gt"] = gt
        captured["masked"] = masked
        captured["exp_indices"] = kwargs["exp_indices"]
        captured["psnr_masked"] = kwargs["psnr_masked"]
        return str(tmp_path / "panel.png")

    monkeypatch.setattr(panels, "save_recon_panel", fake_save_recon_panel)

    panels.flush_recon_panel(
        {}, buffers, [], str(tmp_path), 0, "cifar100", "none", "timestamp", "both"
    )

    assert captured["exp_indices"] == [0, 1, 2]
    assert captured["gt"] == ["gt-0", "gt-1", "gt-2"]
    assert captured["masked"] == ["masked-0", "masked-1", "masked-2"]
    assert captured["psnr_masked"] == [0.4, 1.4, 2.4]
