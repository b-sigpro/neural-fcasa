from typing import Any

import matplotlib.pyplot as plt
import numpy as np

import lightning as lt


class VisualizerCallback(lt.Callback):
    def on_validation_start(self, trainer: lt.Trainer, pl_module: Any, tag: str = "training"):
        if not hasattr(pl_module, "dump"):
            return

        dump = pl_module.dump

        # numpyize dumped variables
        B, F, N, T = dump.lm.shape
        b = np.random.choice(B)

        logx = dump.logx[b].cpu().numpy()
        loglm = dump.lm[b].log().cpu().numpy()
        z = dump.z[b].cpu().numpy()
        w = dump.w[b].cpu().numpy()
        act = dump.act[b].cpu().numpy()

        xt = dump.xt[b].cpu().numpy()
        _, M, _ = xt.shape

        # plot observation and PSDs
        gridspec_kw = dict(height_ratios=[2] + N * [2, 0.5, 1])
        fig, axs = plt.subplots(1 + (3 * N), 1, sharex=True, gridspec_kw=gridspec_kw, figsize=[8, 2 + 3 * N])

        axs[0].imshow(logx, origin="lower", aspect="auto")

        norder = np.argsort(act.sum(axis=-1))[::-1]

        lmmin, lmmax = loglm.min(), loglm.max()
        zmin, zmax = z.min(), z.max()
        for n_, (ax1, ax2, ax3) in enumerate(axs[1 : 1 + 3 * N].reshape(-1, 3)):
            n = norder[n_]

            ax1.imshow(loglm[..., n, :], origin="lower", aspect="auto", vmin=lmmin, vmax=lmmax)

            ax2.plot(act[n])
            ax2.plot(w[..., n, :].T)
            ax2.set_xlim(0, T - 1)
            ax2.set_ylim(-0.1, 1.1)

            ax3.plot(z[..., n, :].T)
            ax3.set_xlim(0, T - 1)
            ax3.set_ylim(zmin, zmax)

        fig.tight_layout(pad=0.1)
        pl_module.logger.experiment.add_figure(f"{tag}/dump", fig, global_step=trainer.current_epoch)
        plt.close(fig)

        # plot xt
        fig, axs = plt.subplots(M, 1, sharex=True, figsize=[8, 2 * M])
        logxt = np.log(xt.clip(1e-6))
        vmin, vmax = logxt.min(), logxt.max()
        for m, ax in enumerate(axs):
            ax.imshow(logxt[..., m, :], origin="lower", aspect="auto", vmin=vmin, vmax=vmax)

        fig.tight_layout(pad=0.1)
        pl_module.logger.experiment.add_figure(f"{tag}/xt", fig, global_step=trainer.current_epoch)
        plt.close(fig)

    def on_validation_end(self, trainer: lt.Trainer, pl_module: Any):
        self.on_validation_start(trainer, pl_module, tag="validation")
