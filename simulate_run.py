"""
Simulation script for a 1-minute ML-Server run.

This script demonstrates how to use the ``mlserverpy`` client package to
report live metrics from two methods (``DLG`` and ``iDLG``) over the
course of sixty seconds.  It logs a simple loss metric every second,
flushes the results to the server, generates a plot of the loss curves
using matplotlib, and uploads the plot image along with a log file.

To execute the script, ensure that an ML-Server instance is running and
that the ``mlserverpy`` package is installed in the Python environment.
Update the ``host``, ``username`` and ``password`` variables as
appropriate for your deployment.
"""

import time
import random
import matplotlib.pyplot as plt

import mlserverpy

def simulate_run() -> None:
    """Run a 60-second simulation with live metric posting and upload artefacts."""
    # Configure the client with your server address and credentials
    client = mlserverpy.Client(
        host="https://mlserver.mkes.dk",
        username="admin",
        password="MLServerBachelorStable-GINV",
        offline_mode="queue",  # queue events if the server is temporarily unreachable
        flush_interval=2.0,
    )

    # Create a new run with two methods.  Adjust dataset/methods/iterations as needed.
    run_id = client.run(
        name="simulation-run",
        dataset="simulated",
        methods=["DLG", "iDLG"],
        iterations=60,
    )

    # Prepare storage for our synthetic metrics and a text log
    dlg_loss: list[float] = []
    idlg_loss: list[float] = []
    log_lines: list[str] = []

    # Simulate 60 seconds of metric updates
    for step in range(60):
        print("Simulating step", step)
        # Generate synthetic loss values that decay over time with noise
        loss_dlg = 1.0 / (step + 1) + random.uniform(-0.05, 0.05)
        loss_idlg = 1.2 / (step + 1) + random.uniform(-0.05, 0.05)

        # Buffer the metrics using log_metric; these are sent on flush
        client.log_metric(method="DLG", metric="loss", value=loss_dlg, step=step)
        client.log_metric(method="iDLG", metric="loss", value=loss_idlg, step=step)

        # Append to our local storage for plotting
        dlg_loss.append(loss_dlg)
        idlg_loss.append(loss_idlg)

        # Append a line to the run log
        log_lines.append(f"Step {step}: DLG loss={loss_dlg:.4f}, iDLG loss={loss_idlg:.4f}")

        # Sleep to approximate real time; reduce or remove for faster testing
        time.sleep(1)

    # Flush any remaining buffered metrics to the server
    client.flush(run_id=run_id)

    # Generate a simple loss curve plot and save it
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(dlg_loss, label="DLG Loss")
    ax.plot(idlg_loss, label="iDLG Loss")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Loss")
    ax.set_title("Simulated Loss Curves")
    ax.legend()
    client.post_image(run_id=run_id, figure=fig, filename="loss_plot.png")
    plt.close(fig)

    # Write the log file to disk and upload it
    log_path = "run.log"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")
    client.post_log(run_id=run_id, text="\n".join(log_lines), filename="run.log")


if __name__ == "__main__":
    simulate_run()