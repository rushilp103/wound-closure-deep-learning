"""
Terminal UI for the wound-closure pipeline (Textual).

Plots are written to a temporary file for preview; use Save to copy into
your chosen path (results dir or elsewhere). Open preview uses the OS viewer.

Run from project root:
  python textual_app.py

A single µm/px value is used for assign-layers and for speed/size plots (omit
physical scaling by setting µm/px to 0).
"""
from __future__ import annotations

import asyncio
import importlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from textual import on
from textual.app import App, ComposeResult
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.widgets import Button, Footer, Header, Input, Label, RichLog, Select, Static

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def build_child_env(
    base_name: str,
    data_dir: str,
    results_dir: str,
    btrack_config: str,
) -> dict[str, str]:
    env = os.environ.copy()
    env["WOUND_BASE_NAME"] = (base_name or "ctrl-1").strip() or "ctrl-1"
    env["WOUND_DATA_DIR"] = (data_dir or "").strip() or str(ROOT / "Data Sets")
    env["WOUND_RESULTS_DIR"] = (results_dir or "").strip() or str(ROOT / "Results")
    env["WOUND_BTRACK_CONFIG"] = (btrack_config or "").strip() or str(ROOT / "btrack_config.json")
    env.setdefault("MPLBACKEND", "Agg")
    return env


def reload_pipeline_config(env: dict[str, str]):
    os.environ["WOUND_BASE_NAME"] = env["WOUND_BASE_NAME"]
    os.environ["WOUND_DATA_DIR"] = env["WOUND_DATA_DIR"]
    os.environ["WOUND_RESULTS_DIR"] = env["WOUND_RESULTS_DIR"]
    os.environ["WOUND_BTRACK_CONFIG"] = env["WOUND_BTRACK_CONFIG"]
    import pipeline_config as pc

    importlib.reload(pc)
    return pc


def suggested_plot_path(pc, plot_kind: str) -> Path:
    return Path(pc.RESULTS_DIR) / f"{pc.BASE_NAME}_plot_{plot_kind}.png"


def open_path_in_viewer(path: Path) -> None:
    path = path.resolve()
    try:
        if sys.platform == "darwin":
            subprocess.Popen(["open", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif os.name == "nt":
            os.startfile(str(path))  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError as e:
        raise RuntimeError(f"Could not open viewer: {e}") from e


class PipelineApp(App[None]):
    CSS = """
    Screen { align: center middle; }
    #main { width: 100%; height: 100%; padding: 0 1; }
    
    .section-title { text-style: bold; color: #ffa529; padding-top: 1; padding-bottom: 0; }
    
    #cfg { height: auto; max-height: 35%; padding: 0 1; border-bottom: solid $primary 50%; }
    .input-row { height: auto; padding-bottom: 1; }
    .input-row Label { margin-top: 1; min-width: 12; }
    .input-row Input { width: 1fr; margin-right: 2; }
    
    .param-row { height: auto; padding-bottom: 1; }
    .param-row Label { margin-top: 1; margin-right: 1; }
    .small-input { width: 12; margin-right: 2; }
    
    #paths_container {
        height: auto;
        min-height: 4;
        padding: 1;
        border: round $primary;
        margin-bottom: 1;
    }
    .paths_row { height: auto; }
    .paths_row Static { width: 1fr; min-height: 2; }
    #paths_warn { height: auto; padding-top: 1; }
    
    #btns { height: auto; padding: 1 0 1 0; }
    Button { margin-right: 1; }
    #plot_kind { width: 16; margin-right: 1; }
    #ball { margin-left: 2; }
    
    #plot_path_row { height: auto; padding: 1 0 1 0; }
    #plot_path_row Label { margin-top: 0; min-width: 8; height: 3; content-align: left middle; }
    #save_plot_path { width: 1fr; max-width: 95; margin-right: 1; }
    #plot_path_row Button { margin-right: 1; }
    
    #log { height: 1fr; border: round $success; min-height: 8; }
    """

    BINDINGS = [("q", "quit", "Quit")]

    def __init__(self) -> None:
        super().__init__()
        self._preview_file: Path | None = None
        self._preview_dir: Path | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical(id="main"):
            
            with ScrollableContainer(id="cfg"):
                yield Label("Data Inputs (Paste absolute paths or Drag & Drop into terminal)", classes="section-title")
                with Horizontal(classes="input-row"):
                    yield Label("Base Name")
                    yield Input(value="ctrl-1", id="base_name")
                    yield Label("Data Dir")
                    yield Input(value=str(ROOT / "Data Sets"), id="data_dir")
                with Horizontal(classes="input-row"):
                    yield Label("BTrack Cfg")
                    yield Input(value=str(ROOT / "btrack_config.json"), id="btrack_config")
                    yield Label("Results Dir")
                    yield Input(value=str(ROOT / "Results"), id="results_dir")

                yield Label("Analysis Parameters", classes="section-title")
                with Horizontal(classes="param-row"):
                    yield Label("µm/px (layers + plots)")
                    yield Input(value="1.0", id="um_per_pixel", classes="small-input")
                    yield Label("layer µm")
                    yield Input(value="49.0", id="layer_width_um", classes="small-input")
                    yield Label("# layers")
                    yield Input(value="10", id="num_layers", classes="small-input")
                    yield Label("min/frame")
                    yield Input(value="20.0", id="minutes_per_frame", classes="small-input")

            yield Label("File Generation Status", classes="section-title")
            with Vertical(id="paths_container"):
                with Horizontal(classes="paths_row"):
                    yield Static(id="path_cell_0", markup=True)
                    yield Static(id="path_cell_1", markup=True)
                with Horizontal(classes="paths_row"):
                    yield Static(id="path_cell_2", markup=True)
                    yield Static(id="path_cell_3", markup=True)
                with Horizontal(classes="paths_row"):
                    yield Static(id="path_cell_4", markup=True)
                    yield Static(id="path_cell_5", markup=True)
                yield Static(id="paths_warn", markup=True)
            
            with Horizontal(id="btns"):
                yield Button("1 Cellpose", id="b1", variant="primary")
                yield Button("2 Masks→obj", id="b2")
                yield Button("3 Layers", id="b3")
                yield Button("4 Track", id="b4")
                yield Button("5 H5→CSV", id="b5")
                yield Button("6 Plot", id="b6")
                yield Select(
                    (("aspect", "aspect"), ("speed", "speed"), ("size", "size")),
                    id="plot_kind",
                    value="aspect",
                )
                yield Button("Run All", id="ball", variant="warning")
                
            with Horizontal(id="plot_path_row"):
                yield Label("Save to ")
                yield Input(placeholder="path/to/plot.png", id="save_plot_path")
                yield Button("Save plot", id="save_plot", variant="success")
                yield Button("Open preview", id="open_preview")
                
            yield RichLog(id="log", highlight=False, markup=False, wrap=True, auto_scroll=True)
            
        yield Footer()

    def on_mount(self) -> None:
        if hasattr(self, "theme"):
            self.theme = "textual-dark"
        elif hasattr(self, "dark"):
            self.dark = True
        self._busy = False
        self.refresh_paths()
        self._update_save_controls()

    def _read_env(self) -> dict[str, str]:
        def gv(wid: str, default: str = "") -> str:
            try:
                return self.query_one(f"#{wid}", Input).value
            except Exception:
                return default

        return build_child_env(
            gv("base_name", "ctrl-1"),
            gv("data_dir", str(ROOT / "Data Sets")),
            gv("results_dir", str(ROOT / "Results")),
            gv("btrack_config", str(ROOT / "btrack_config.json")),
        )

    def _clear_preview_state(self) -> None:
        self._preview_file = None
        self._preview_dir = None

    def _update_save_controls(self) -> None:
        has_preview = self._preview_file is not None or self._preview_dir is not None
        self.query_one("#save_plot", Button).disabled = not has_preview
        self.query_one("#open_preview", Button).disabled = not has_preview

    def refresh_paths(self) -> None:
        pc = reload_pipeline_config(self._read_env())

        def status(path_str: str) -> str:
            p = Path(path_str)
            filename = p.name or "Unknown"
            if p.is_file():
                return f"[bold green][✓][/bold green] {filename}"
            return f"[bold yellow][·][/bold yellow] [dim]{filename}[/dim]"

        cells = [
            pc.input_tif_path,
            pc.masks_tracking_path,
            pc.objects_csv_path,
            pc.tracks_h5_path,
            pc.converted_tracks_csv_path,
            pc.objects_with_layers_csv_path,
        ]
        for i, path_str in enumerate(cells):
            self.query_one(f"#path_cell_{i}", Static).update(status(path_str))
        warn = (
            ""
            if Path(pc.input_tif_path).is_file()
            else "[red](Input TIFF missing)[/red]"
        )
        self.query_one("#paths_warn", Static).update(warn)

    @on(Input.Changed)
    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id in {"base_name", "data_dir", "results_dir", "btrack_config"}:
            self.refresh_paths()

    def log_line(self, msg: str) -> None:
        self.query_one("#log", RichLog).write(msg)

    def set_busy(self, busy: bool) -> None:
        self._busy = busy
        for bid in ("b1", "b2", "b3", "b4", "b5", "b6", "ball"):
            self.query_one(f"#{bid}", Button).disabled = busy
        self.query_one("#plot_kind", Select).disabled = busy
        if not busy:
            self._update_save_controls()
        else:
            for bid in ("save_plot", "open_preview"):
                self.query_one(f"#{bid}", Button).disabled = True

    async def stream_subprocess(self, label: str, script: str, args: list[str]) -> int:
        env = self._read_env()
        cmd = [sys.executable, str(ROOT / script), *args]
        self.log_line(f"\n--- {label} ---")
        self.log_line(" ".join(cmd))
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(ROOT),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        assert proc.stdout is not None
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            self.log_line(line.decode(errors="replace").rstrip("\n\r"))
        code = await proc.wait()
        self.log_line(f"--- exit {code} ---")
        return code

    def _float(self, wid: str, default: float) -> float:
        try:
            return float(self.query_one(f"#{wid}", Input).value)
        except Exception:
            return default

    def _int(self, wid: str, default: int) -> int:
        try:
            return int(float(self.query_one(f"#{wid}", Input).value))
        except Exception:
            return default

    def _final_plots_args(self, plot_kind: str, out: Path) -> list[str]:
        cmd_args = [
            "--plot",
            plot_kind,
            "--minutes-per-frame",
            str(self._float("minutes_per_frame", 20.0)),
            "--o",
            str(out),
            "--x-axis",
            "hours",
        ]
        um = self._float("um_per_pixel", 1.0)
        if plot_kind in ("speed", "size") and um > 0:
            cmd_args.extend(["--um-per-pixel", str(um)])
        return cmd_args

    async def _run_guarded(self, label: str, script: str, args: list[str]) -> None:
        if self._busy:
            return
        self.set_busy(True)
        try:
            await self.stream_subprocess(label, script, args)
        finally:
            self.set_busy(False)
            self.refresh_paths()

    @on(Button.Pressed, "#b1")
    async def on_b1(self) -> None:
        await self._run_guarded("Cellpose", "cellpose_inference.py", [])

    @on(Button.Pressed, "#b2")
    async def on_b2(self) -> None:
        await self._run_guarded("Masks → objects", "masks_to_objects.py", [])

    @on(Button.Pressed, "#b3")
    async def on_b3(self) -> None:
        await self._run_guarded(
            "Assign layers",
            "assign_layers.py",
            [
                "--um-per-pixel",
                str(self._float("um_per_pixel", 1.0)),
                "--layer-width",
                str(self._float("layer_width_um", 49.0)),
                "--num-layers",
                str(self._int("num_layers", 10)),
            ],
        )

    @on(Button.Pressed, "#b4")
    async def on_b4(self) -> None:
        await self._run_guarded("Tracking", "run_tracking.py", [])

    @on(Button.Pressed, "#b5")
    async def on_b5(self) -> None:
        await self._run_guarded("Convert H5", "convert_h5_results.py", [])

    @on(Button.Pressed, "#b6")
    async def on_b6(self) -> None:
        if self._busy:
            return
        self.set_busy(True)
        try:
            pc = reload_pipeline_config(self._read_env())
            plot_kind = str(self.query_one("#plot_kind", Select).value)
            fd, tmp_name = tempfile.mkstemp(suffix=".png", prefix="wound_plot_")
            os.close(fd)
            tmp_path = Path(tmp_name)
            self._clear_preview_state()
            self._preview_file = tmp_path
            code = await self.stream_subprocess(
                "Final plots",
                "final_plots.py",
                self._final_plots_args(plot_kind, tmp_path),
            )
            if code == 0 and tmp_path.is_file():
                dest = suggested_plot_path(pc, plot_kind)
                self.query_one("#save_plot_path", Input).value = str(dest)
                self.log_line(f"Preview (temp, not saved until you click Save): {tmp_path}")
                try:
                    open_path_in_viewer(tmp_path)
                    self.log_line("Opened preview in system viewer.")
                except RuntimeError as e:
                    self.log_line(str(e))
            else:
                self._preview_file = None
                tmp_path.unlink(missing_ok=True)
        finally:
            self.set_busy(False)
            self._update_save_controls()

    @on(Button.Pressed, "#ball")
    async def on_ball(self) -> None:
        if self._busy:
            return
        self.set_busy(True)
        try:
            steps = [
                ("Cellpose", "cellpose_inference.py", []),
                ("Masks → objects", "masks_to_objects.py", []),
                (
                    "Assign layers",
                    "assign_layers.py",
                    [
                        "--um-per-pixel",
                        str(self._float("um_per_pixel", 1.0)),
                        "--layer-width",
                        str(self._float("layer_width_um", 49.0)),
                        "--num-layers",
                        str(self._int("num_layers", 10)),
                    ],
                ),
                ("Tracking", "run_tracking.py", []),
                ("Convert H5", "convert_h5_results.py", []),
            ]
            for label, script, args in steps:
                code = await self.stream_subprocess(label, script, args)
                if code != 0:
                    return
            pc = reload_pipeline_config(self._read_env())
            tdir = Path(tempfile.mkdtemp(prefix="wound_plots_"))
            self._clear_preview_state()
            self._preview_dir = tdir
            for pk in ("aspect", "speed", "size"):
                out = tdir / f"{pc.BASE_NAME}_plot_{pk}.png"
                cmd_args = self._final_plots_args(pk, out)
                code = await self.stream_subprocess(f"Final plot · {pk}", "final_plots.py", cmd_args)
                if code != 0:
                    self.log_line(f"Plot step failed; previews in {tdir}")
                    return
            self.log_line(f"All plots preview folder (temp, not in results): {tdir}")
            self.query_one("#save_plot_path", Input).value = str(Path(pc.RESULTS_DIR))
            try:
                open_path_in_viewer(tdir)
                self.log_line("Opened preview folder in system file manager.")
            except RuntimeError as e:
                self.log_line(str(e))
        finally:
            self.set_busy(False)
            self.refresh_paths()
            self._update_save_controls()

    @on(Button.Pressed, "#save_plot")
    def on_save_plot(self) -> None:
        raw = self.query_one("#save_plot_path", Input).value.strip()
        if not raw:
            self.log_line("Save: enter a destination path first.")
            return
        dest = Path(raw).expanduser()
        try:
            if self._preview_file is not None and self._preview_file.is_file():
                pc = reload_pipeline_config(self._read_env())
                plot_kind = str(self.query_one("#plot_kind", Select).value)
                default_name = suggested_plot_path(pc, plot_kind).name
                if raw.endswith(("/", "\\")) or (dest.exists() and dest.is_dir()):
                    dest = dest / default_name
                elif dest.suffix.lower() != ".png":
                    dest = dest.with_name(default_name) if dest.suffix else dest.with_suffix(".png")
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(self._preview_file, dest)
                self.log_line(f"Saved plot to {dest}")
            elif self._preview_dir is not None and self._preview_dir.is_dir():
                dest.mkdir(parents=True, exist_ok=True)
                copied = 0
                for src in sorted(self._preview_dir.glob("*.png")):
                    shutil.copy2(src, dest / src.name)
                    copied += 1
                self.log_line(f"Saved {copied} plot(s) into {dest}")
            else:
                self.log_line("Nothing to save; run step 6 or Run all first.")
        except OSError as e:
            self.log_line(f"Save failed: {e}")

    @on(Button.Pressed, "#open_preview")
    def on_open_preview(self) -> None:
        try:
            if self._preview_file and self._preview_file.is_file():
                open_path_in_viewer(self._preview_file)
            elif self._preview_dir and self._preview_dir.is_dir():
                open_path_in_viewer(self._preview_dir)
            else:
                self.log_line("No preview available.")
        except RuntimeError as e:
            self.log_line(str(e))

def main() -> None:
    PipelineApp().run()


if __name__ == "__main__":
    main()