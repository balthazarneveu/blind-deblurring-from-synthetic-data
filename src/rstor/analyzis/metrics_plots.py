import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd


def snr_to_sigma(snr):
    return 10**(-snr/20.)*255.


def sigma_to_snr(sigma):
    return -20.*np.log10(sigma/255.)


def plot_results(selected_paths, title=None, diff=True):
    # plt.figure(figsize=(10, 10))
    fig, ax = plt.subplots(layout='constrained', figsize=(10, 10))
    for selected_path, selected_regex in selected_paths:
        selected_path = Path(selected_path)
        assert selected_path.exists()
        results_path = sorted(list(selected_path.glob(selected_regex)))
        stats = []
        for result_path in results_path:
            df = pd.read_csv(result_path)
            in_psnr = df["in_PSNR"].mean()
            out_psnr = df["out_PSNR"].mean()
            stats.append({
                "in_psnr": in_psnr,
                "out_psnr": out_psnr,
            })
            label = selected_path.name + " " + df["size"][0]
        stats_array = pd.DataFrame(stats)
        x_data = stats_array["in_psnr"].copy()
        x_data = snr_to_sigma(x_data)

        ax.plot(
            x_data,
            stats_array["out_psnr"]-stats_array["in_psnr"] if diff else stats_array["out_psnr"],
            "-o",
            label=label
        )
        # label=selected_path.name)
    if not diff:
        neutral_sigma = np.linspace(1, 80, 80)
        ax.plot(neutral_sigma, sigma_to_snr(neutral_sigma), "k--", alpha=0.1, label="Neutral")
    secax = ax.secondary_xaxis('top', functions=(sigma_to_snr, snr_to_sigma))
    secax.set_xlabel('PSNR [db]')

    ax.set_xlabel("sigma 255")
    ax.set_ylabel("PSNR improvement" if diff else "PSNR out")
    plt.xlim(1., 50.)
    if diff:
        plt.ylim(0, 15)
    if title is not None:
        plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()
