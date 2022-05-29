"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/plugins/stable/guides.html#widgets

Replace code below according to your needs.

05/29/2022 - added widget access from console (see https://forum.image.sc/t/access-plugin-dockwidget-instance-through-console/64201/5)

"""

from pathlib import Path

import numpy as np
import WEA
from napari.qt.threading import thread_worker
from napari_plugin_engine import napari_hook_implementation
from qtpy.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

FILE_FORMATS = ["*.mrc", "*.dv", "*.nd2", "*.tif", "*.tiff"]


def argsort(seq):
    return sorted(range(len(seq) - 1, -1, -1), key=seq.__getitem__)


class WEAWidget(QWidget):
    # for getting access of the widget from console, keep track of widget
    _instance = None
    # your QWidget.__init__ can optionally request the napari viewer instance
    # in one of two ways:
    # 1. use a parameter called `napari_viewer`, as done here
    # 2. use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, napari_viewer):
        super().__init__()
        
        WEAWidget._instance = self

        self.viewer = napari_viewer

        self.current_img = None
        self.fov = None

        # main layout
        self.layout = QVBoxLayout()

        # file input interface
        self.choose_folder_btn = QPushButton("Choose a folder")
        self.current_folder_label = QLabel("")
        self.flist_widget = QListWidget()
        self.flist_groupbox = QGroupBox("Input files")
        self.flist_vbox = QVBoxLayout()
        self.flist_vbox.addWidget(self.choose_folder_btn)
        self.flist_vbox.addWidget(self.current_folder_label)
        self.flist_vbox.addWidget(self.flist_widget)
        self.flist_groupbox.setLayout(self.flist_vbox)
        self.layout.addWidget(self.flist_groupbox)

        # program option interface
        self.ch_groupbox = QGroupBox("Segmentation channels")
        self.ch_vbox = QVBoxLayout()
        self.cytogroup = QComboBox(self)
        self.nucgroup = QComboBox(self)
        self.tubgroup = QComboBox(self)
        self.assign_channels_btn = QPushButton("Assign channels")

        self.ch_vbox.addWidget(QLabel("Cytoplasm channel"))
        self.ch_vbox.addWidget(self.cytogroup)
        self.ch_vbox.addWidget(QLabel("Nucleus channel"))
        self.ch_vbox.addWidget(self.nucgroup)
        self.ch_vbox.addWidget(QLabel("Tubulin channel"))
        self.ch_vbox.addWidget(self.tubgroup)
        self.ch_vbox.addWidget(self.assign_channels_btn)
        # set channel vbox (and its widgets) to channel groupbox
        self.ch_groupbox.setLayout(self.ch_vbox)
        self.layout.addWidget(self.ch_groupbox)

        # WEA option interface
        self.wea_groupbox = QGroupBox("Segmentation parameters")
        self.wea_vbox = QVBoxLayout()
        nuc_field_widget = QWidget()
        nuc_field_layout = QHBoxLayout()
        cyto_field_widget = QWidget()
        cyto_field_layout = QHBoxLayout()
        nuc_field_widget.setLayout(nuc_field_layout)
        cyto_field_widget.setLayout(cyto_field_layout)
        nuc_label = QLabel("Nucleus diam. (µm)")
        cyto_label = QLabel("Cell diam. (µm)")
        self.cell_size_field = QDoubleSpinBox()
        # set default values for cell size
        self.cell_size_field.setValue(75.0)
        self.nucleus_size_field = QDoubleSpinBox()
        # set default value for nucleus size
        self.nucleus_size_field.setValue(15.0)
        self.cell_size_field.setRange(1, 1000)
        self.nucleus_size_field.setRange(1, 1000)
        cyto_field_layout.addWidget(cyto_label)
        cyto_field_layout.addWidget(self.cell_size_field)
        nuc_field_layout.addWidget(nuc_label)
        nuc_field_layout.addWidget(self.nucleus_size_field)
        self.sketch_cell_size = QPushButton("Sketch sizes")
        self.run_singlerun_btn = QPushButton("Do it!")
        self.wea_vbox.addWidget(cyto_field_widget)
        self.wea_vbox.addWidget(nuc_field_widget)
        self.wea_vbox.addWidget(self.sketch_cell_size)
        self.wea_vbox.addWidget(self.run_singlerun_btn)
        self.wea_groupbox.setLayout(self.wea_vbox)
        self.layout.addWidget(self.wea_groupbox)

        # fill the rest of the vertical space so preceding widgets stack
        # from top-to-bottom
        self.setLayout(self.layout)
        self.layout.addStretch()

        # gui behavior
        self.choose_folder_btn.clicked.connect(self._open_file_dialog)
        self.assign_channels_btn.clicked.connect(self._assign_channels)
        self.sketch_cell_size.clicked.connect(self._sketch_cell)
        self.run_singlerun_btn.clicked.connect(self._run_wea_single)

    @classmethod
    def instance(cls):
        """return current/last widget"""
        return cls._instance

    def _open_file_dialog(self):
        self.flist_widget.clear()
        self.img_folder = Path(
            QFileDialog.getExistingDirectory(self, "Choose a folder", "~")
        )

        imgpath = Path(self.img_folder)
        flist = []

        for ext in FILE_FORMATS:
            flist.extend([f for f in imgpath.glob(ext)])

        nfiles = len(flist)
        # construct abbreviated path
        abbr_path = f"...{self.img_folder.parent.name}/{self.img_folder.name}"
        # update current folder label
        self.current_folder_label.setText(f"{abbr_path:s} ({nfiles:d})")

        for f in sorted(flist):
            self.flist_widget.addItem(f.name)

        self.flist_widget.currentItemChanged.connect(self._fetch_img_info)

    def _fetch_img_info(self, key):
        # save the number of channels available before clearing it up
        prev_nch = self.cytogroup.count()
        prev_chnames = [self.cytogroup.itemText(i) for i in range(prev_nch)]

        # clear all current layers
        self.viewer.layers.clear()

        if key:
            fname = key.text()
            fpath = self.img_folder / fname
            self.current_img = WEA.io.CanonizedImage(fpath)

            current_nch = self.current_img.data.shape[-1]

            if current_nch != prev_nch:
                ch_names = [f"ch={i}" for i in range(current_nch)]

                self.viewer.add_image(
                    self.current_img.data, name=ch_names, channel_axis=-1
                )

                # update the channel groupboxes
                available_layers = [layer.name for layer in self.viewer.layers]

                # clear current channel groups
                self.cytogroup.clear()
                self.nucgroup.clear()
                self.tubgroup.clear()

                # add new ones
                self.cytogroup.addItems(available_layers)
                self.nucgroup.addItems(available_layers)
                self.tubgroup.addItems(available_layers)

            else:
                self.viewer.add_image(
                    self.current_img.data, name=prev_chnames, channel_axis=-1
                )

    def _assign_channels(self):

        if self.current_img is None:
            return

        def ch_from_text(s):
            substr = s.split(",")[0]
            chstr = substr.split("=")[1]
            return int(chstr)

        cytoplasm_choice = ch_from_text(self.cytogroup.currentText())
        nucleus_choice = ch_from_text(self.nucgroup.currentText())
        tubulin_choice = ch_from_text(self.tubgroup.currentText())

        if self.current_img.data.ndim == 4:
            # reduce by finding focus in tubulin channel
            img2d = self.current_img.get_focused_plane(tubulin_choice)
        else:
            img2d = self.current_img.data

        self.fov = WEA.core.ImageField(
            img2d,
            self.current_img.dxy,
            nucleus_ch=nucleus_choice,
            cyto_channel=cytoplasm_choice,
            tubulin_ch=tubulin_choice,
        )

        choice_id = [cytoplasm_choice, nucleus_choice, tubulin_choice]
        choice_str = ["cyto", "dapi", "tubulin"]
        cmaps = {"cyto": "green", "dapi": "cyan", "tubulin": "magenta"}

        self.viewer.layers.clear()

        fileItem = self.flist_widget.currentItem()
        fname = fileItem.text()

        # re-add image with the chosen channels
        self.viewer.add_image(
            self.fov.data,
            channel_axis=-1,
            name=[f"{choice_str[i]}:{fname}" for i in choice_id],
            colormap=[cmaps[choice_str[i]] for i in choice_id],
        )

        # change contrast limit to 1-99% percentile
        for layer in self.viewer.layers:
            lo = np.percentile(layer.data, 0.5)
            hi = np.percentile(layer.data, 99.5)
            layer.contrast_limits = [lo, hi]

    def _sketch_cell(self):

        if self.current_img is None:
            return

        cell_diam = self.cell_size_field.value()
        nuc_diam = self.nucleus_size_field.value()

        Ny, Nx = self.current_img.data.shape[:2]
        imgcenter = (Ny // 2, Nx // 2)
        cellrad_px = 0.5 * cell_diam / self.current_img.dxy
        nucrad_px = 0.5 * nuc_diam / self.current_img.dxy
        sketch_data = [
            np.array([imgcenter, (cellrad_px, cellrad_px)]),
            np.array([imgcenter, (nucrad_px, nucrad_px)]),
        ]

        # clear current sketch (if any)
        if "sketch" in self.viewer.layers:
            self.viewer.layers["sketch"].data = []
            self.viewer.layers["sketch"].data = sketch_data
            self.viewer.layers["sketch"].shape_type = "ellipse"
            self.viewer.layers["sketch"].face_color = ["green", "blue"]

        else:
            self.viewer.add_shapes(
                sketch_data,
                name="sketch",
                shape_type="ellipse",
                face_color=["green", "blue"],
            )

    def _run_wea_single(self):

        if self.fov is None:
            return

        celldiam = self.cell_size_field.value()
        nucdiam = self.nucleus_size_field.value()

        if self.fov:
            self.run_singlerun_btn.setText("Segmenting ...")
            worker = self.__run_wea_task(celldiam, nucdiam)
            worker.returned.connect(self.__display_result)
            worker.finished.connect(self.__change_run_btn_status)
            worker.start()

    @thread_worker
    def __run_wea_task(self, celldiam, nucdiam):
        self.fov.segment_cells(celldiam=celldiam, nucdiam=nucdiam)
        self.fov.run_detection(cell_diam=celldiam, nuc_diam=nucdiam)
        cellpose_output, woundedge = self.fov._detection_result()
        mtoc_df, cell_df = self.fov.run_analysis(
            self.current_img.filename.name
        )
        self.run_singlerun_btn.setText("Analyzing ...")
        return {
            "labcells": cellpose_output,
            "woundedge": woundedge,
            "mtoc_df": mtoc_df,
            "cell_df": cell_df,
        }

    def __check_args(*args):
        print(args)

    def __change_run_btn_status(self):
        self.run_singlerun_btn.setText("Do it!")

    def __display_result(self, segresult):

        self.viewer.add_labels(segresult["labcells"], name="detected cells")
        self.viewer.add_labels(
            segresult["woundedge"],
            name="wound edge",
            num_colors=1,
            color={1: "red"},
        )
        WEA.vis.add_to_napari(
            self.viewer, segresult["mtoc_df"], segresult["cell_df"]
        )


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    # you can return either a single widget, or a sequence of widgets
    return [WEAWidget]