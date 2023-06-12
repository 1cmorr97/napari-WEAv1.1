"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/plugins/stable/guides.html#widgets

Replace code below according to your needs.

05/29/2022 - added widget access from console
            (see https://forum.image.sc/t/access-plugin-dockwidget-instance-through-console/64201/5)

            from napari_wea._widget import WEAWidget

            w = WEAWidget.instance()
            # now you can access your current widget via variable 'w'

"""

from pathlib import Path
import cv2
import WoundRUs
import numpy as np
import pandas as pd
from napari.qt.threading import thread_worker
from napari_plugin_engine import napari_hook_implementation
from qtpy.QtCore import QProcess
from qtpy.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QCheckBox,
    QTabWidget,
    QFormLayout,
)
from qtpy.QtCore import Qt
from tifffile import imwrite

FILE_FORMATS = ["*.mrc", "*.dv", "*.nd2", "*.tif", "*.tiff"]


def argsort(seq):
    return sorted(range(len(seq) - 1, -1, -1), key=seq.__getitem__)


def ch_from_text(s):
    substr = s.split(":")[0]
    chstr = substr.split("=")[1]
    return int(chstr)


class InputFileList(QListWidget):
    def __init__(self):
        super().__init__()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Delete:
            current_index = self.currentRow()
            self.takeItem(current_index)

        if e.key() == Qt.Key_Up:
            current_index = self.currentRow()
            moved_index = max(current_index - 1, 0)
            self.setCurrentRow(moved_index)

        if e.key() == Qt.Key_Down:
            n_items = self.count()
            current_index = self.currentRow()
            moved_index = min(current_index + 1, n_items - 1)
            self.setCurrentRow(moved_index)


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
        self.current_img_metadata = None
        self.current_filename = None
        # img2d is the composite image
        self.img2d = None

        # main layout
        self.layout = QVBoxLayout()

        # file input interface
        self.choose_folder_btn = QPushButton("Choose a folder")
        self.current_folder_label = QLabel("")
        self.flist_widget = InputFileList()
        self.flist_groupbox = QGroupBox("Input files")
        self.flist_vbox = QVBoxLayout()
        self.flist_vbox.addWidget(self.choose_folder_btn)
        self.flist_vbox.addWidget(self.current_folder_label)
        self.flist_vbox.addWidget(self.flist_widget)
        self.flist_groupbox.setLayout(self.flist_vbox)
        self.layout.addWidget(self.flist_groupbox)

        self.compute_label_increment_btn = QPushButton("Get unique label")
        self.layout.addWidget(self.compute_label_increment_btn)

        main_tab = QTabWidget()

        # program option interface
        # self.ch_groupbox = QGroupBox("Segmentation channels")
        self.ch_tab_widget = QWidget()
        self.ch_vbox = QVBoxLayout()
        self.cytogroup = QComboBox(self)
        self.nucgroup = QComboBox(self)
        self.tubgroup = QComboBox(self)
        self.do_maxproj_checkbox = QCheckBox("Use max-projection")
        self.use_tubulin_for_cyto_checkbox = QCheckBox("Use tubulin as cyto")
        self.assign_channels_btn = QPushButton("Assign channels")

        # add option to enforce pixel size
        px_size_widget = QWidget()
        px_size_layout = QHBoxLayout()
        px_size_widget.setLayout(px_size_layout)
        self.enforce_px_size_checkbox = QCheckBox("Force pixel size")
        # set pixel size entry form (adaptive step type is 0.1 of current value, int = 1)
        self.px_size_entry = QDoubleSpinBox(decimals=5, stepType=1)
        self.px_size_label = QLabel("um/px")
        px_size_layout.addWidget(self.px_size_entry)
        px_size_layout.addWidget(self.px_size_label)

        self.ch_vbox.addWidget(QLabel("Cytoplasm channel"))
        self.ch_vbox.addWidget(self.cytogroup)
        self.ch_vbox.addWidget(QLabel("Nucleus channel"))
        self.ch_vbox.addWidget(self.nucgroup)
        self.ch_vbox.addWidget(QLabel("Tubulin channel"))
        self.ch_vbox.addWidget(self.tubgroup)
        self.ch_vbox.addWidget(self.do_maxproj_checkbox)
        self.ch_vbox.addWidget(self.use_tubulin_for_cyto_checkbox)
        self.ch_vbox.addWidget(self.enforce_px_size_checkbox)
        self.ch_vbox.addWidget(px_size_widget)

        # set channel vbox (and its widgets) to channel groupbox
        self.ch_tab_widget.setLayout(self.ch_vbox)
        main_tab.addTab(self.ch_tab_widget, "Segmentation channels")

        # WEA option interface (Segmentation parameters)
        self.wea_tab_widget = QWidget()
        _wea_form_layout = QFormLayout()
        self.wea_tab_widget.setLayout(_wea_form_layout)

        self.nucleus_size_field = QDoubleSpinBox()
        self.nucleus_size_field.setRange(1, 1000)
        self.nucleus_size_field.setValue(130.0)

        self.cell_size_field = QDoubleSpinBox()
        self.cell_size_field.setRange(1, 1000)
        self.cell_size_field.setValue(400)

        self.slope_threshold = QDoubleSpinBox()
        self.slope_threshold.setRange(1, 1000)
        self.slope_threshold.setValue(50)

        self.nucleus_sigma = QDoubleSpinBox()
        self.nucleus_sigma.setRange(1, 1000)
        self.nucleus_sigma.setValue(50)

        self.mtoc_log_sigma = QDoubleSpinBox()
        self.mtoc_log_sigma.setRange(1, 20)
        self.mtoc_log_sigma.setValue(3.0)

        _wea_form_layout.addRow("Cell diam. (px)", self.cell_size_field)
        _wea_form_layout.addRow("Nucleus diam. (px)", self.nucleus_size_field)
        _wea_form_layout.addRow("mtoc LoG sigma", self.mtoc_log_sigma)
        _wea_form_layout.addRow("slope thres.", self.slope_threshold)
        _wea_form_layout.addRow("nucleus weight sigma", self.nucleus_sigma)

        main_tab.addTab(self.wea_tab_widget, "Segmentation parameters")

        self.sketch_cell_size = QPushButton("Sketch sizes")
        self.process_image_btn = QPushButton("Process image")
        self.apply_manual_changes_btn = QPushButton("apply changes")

        self.layout.addWidget(main_tab)
        self.layout.addWidget(self.assign_channels_btn)
        self.layout.addWidget(self.sketch_cell_size)
        self.layout.addWidget(self.process_image_btn)
        self.layout.addWidget(self.apply_manual_changes_btn)

        # fill the rest of the vertical space so preceding widgets stack
        # from top-to-bottom
        self.setLayout(self.layout)
        self.layout.addStretch()

        # gui behavior
        self.choose_folder_btn.clicked.connect(self._open_file_dialog)
        self.assign_channels_btn.clicked.connect(self._assign_channels)
        self.compute_label_increment_btn.clicked.connect(self._get_unique_label)
        self.sketch_cell_size.clicked.connect(self._sketch_cell)
        self.process_image_btn.clicked.connect(self._process_image)
        self.apply_manual_changes_btn.clicked.connect(
            self._apply_manual_changes
        )

    @classmethod
    def instance(cls):
        """return current/last widget"""
        return cls._instance

    def _open_file_dialog(self):
        self.img_folder = Path(
            QFileDialog.getExistingDirectory(self, "Choose a folder", "~")
        )

        if self.img_folder:
            self.flist_widget.clear()
        else:
            return

        self.imgpath = Path(self.img_folder)
        flist = []

        excluded_suffixes = ["_cp_masks", "_cp", "_nucmasks", "_wedge"]

        for ext in FILE_FORMATS:
            # filter output tif files
            if ext == "*.tif" or ext == "*.tiff":
                _flist = [
                    file
                    for file in self.imgpath.glob(ext)
                    if not any(
                        suffix in file.stem for suffix in excluded_suffixes
                    )
                ]
            else:
                _flist = [f for f in self.imgpath.glob(ext)]

            flist.extend(_flist)

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
            (
                self.current_img,
                self.current_img_metadata,
            ) = WoundRUs.IO.read_image(fpath)

            channels = self.current_img_metadata["channels"]
            current_nch = len(channels)

            if current_nch != prev_nch:
                ch_names = [f"ch={i:d}:{ch:s}" for i, ch in enumerate(channels)]

                self.viewer.add_image(
                    self.current_img, name=ch_names, channel_axis=-1
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
                    self.current_img, name=prev_chnames, channel_axis=-1
                )

            # if we dont enforce pixel size
            # retrieve pixel size information if available and display in GUI
            if not self.enforce_px_size_checkbox.isChecked():
                self.px_size_entry.setValue(
                    self.current_img_metadata["pixel_size"]
                )

    def _assign_channels(self):
        fileItem = self.flist_widget.currentItem()
        fname = fileItem.text()

        do_max_projection = self.do_maxproj_checkbox.isChecked()

        self.current_filename = fname

        if self.current_img is None:
            return

        cytoplasm_choice = ch_from_text(self.cytogroup.currentText())
        nucleus_choice = ch_from_text(self.nucgroup.currentText())
        tubulin_choice = ch_from_text(self.tubgroup.currentText())

        if self.current_img.ndim == 4:
            # ensure a variable named 'img2d' is created
            # reduce by finding focus in tubulin channel
            if do_max_projection:
                # do max projection
                NotImplementedError("reduction from 3D to 2D not implemented.")
            else:
                # find focused plane
                NotImplementedError("reduction from 3D to 2D not implemented.")
        else:
            img2d = self.current_img

        force_pixel_size = self.enforce_px_size_checkbox.isChecked()
        self.img2d = img2d

        if force_pixel_size:
            dxy = self.px_size_entry.value()
        else:
            dxy = self.current_img_metadata["pixel_size"]

        use_tubulin_for_cyto = self.use_tubulin_for_cyto_checkbox.isChecked()

        cmaps = {
            "cyto": "green",
            "dapi": "cyan",
            "tubulin": "magenta",
            "extra": "gray",
        }

        choice_dict = {
            cytoplasm_choice: "cyto",
            nucleus_choice: "dapi",
            tubulin_choice: "tubulin",
        }

        # sort choice_dict by key to get corresponding channels in order 0,1,2
        sorted_dict = dict(sorted(choice_dict.items()))
        ch_names = [f"{s}:{fname}" for s in list(sorted_dict.values())]
        ch_luts = [cmaps[s] for s in list(sorted_dict.values())]

        self.viewer.layers.clear()

        # re-add image with the chosen channels
        ch_ids = tuple(sorted_dict.keys())
        self.viewer.add_image(
            self.img2d[:, :, ch_ids],
            channel_axis=-1,
            name=ch_names,
            colormap=ch_luts,
        )

        # change contrast limit to 1-99% percentile
        for layer in self.viewer.layers:
            lo = np.percentile(layer.data, 0.5)
            hi = np.percentile(layer.data, 99.8)
            layer.contrast_limits = [lo, hi]

    def _get_unique_label(self):
        if "cells" in self.viewer.layers:
            # get current labels and remove 0 (background)
            current_labels = set(np.unique(self.viewer.layers["cells"].data))
            current_labels.remove(0)
            intact_sequence = set(range(1, max(current_labels) + 2))
            # get unique available id using bitwise XOR
            available_ids = intact_sequence ^ current_labels
            next_id = min(available_ids)
            # assign id to the current selected label
            self.viewer.layers["cells"].selected_label = next_id

    def _sketch_cell(self):
        if self.current_img is None:
            return

        cell_diam = self.cell_size_field.value()
        nuc_diam = self.nucleus_size_field.value()

        Ny, Nx = self.current_img.data.shape[:2]
        imgcenter = (Ny // 2, Nx // 2)
        cellrad_px = 0.5 * cell_diam
        nucrad_px = 0.5 * nuc_diam
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

    def _process_image(self):
        celldiam = self.cell_size_field.value()
        nucdiam = self.nucleus_size_field.value()
        slope_thres = self.slope_threshold.value()
        log_sigma = self.mtoc_log_sigma.value()
        weight_sigma = self.nucleus_sigma.value()

        if self.current_img is not None:
            self.process_image_btn.setText("Segmenting ...")
            worker = self.__run_processing_task(
                celldiam, nucdiam, log_sigma, weight_sigma, slope_thres
            )
            worker.returned.connect(self.__display_result)
            # change the text on the button back to 'normal'
            worker.finished.connect(self.__reset_run_btn_status)
            worker.start()

    @thread_worker
    def __run_processing_task(
        self, celldiam, nucdiam, log_sigma, weight_sigma, slope_threshold
    ):
        if self.use_tubulin_for_cyto_checkbox.isChecked():
            cyto_channels = [1, 3]
        else:
            cyto_channels = [2, 3]

        cell_mask, nuc_mask = WoundRUs.Segmenter.segment_cells_and_nuclei(
            self.img2d,
            cell_diameter=celldiam,
            nucleus_diameter=nucdiam,
        )
        wound_mask = WoundRUs.Segmenter.compute_wound_edge(cell_mask)
        cone_mask, props_df = WoundRUs.Segmenter.process_edge_cells(
            cell_mask, nuc_mask, wound_mask
        )

        # get segmentation parameters

        mtoc_df = WoundRUs.Segmenter.find_mtocs(
            self.img2d[:, :, 1],
            self.img2d[:, :, 0],
            cell_mask,
            nuc_mask,
            cone_mask,
            props_df,
            weight_sigma=weight_sigma,
            slope_threshold=slope_threshold,
            log_sigma=log_sigma,
        )

        return {
            "cells": cell_mask,
            "nuclei": nuc_mask,
            "cone": cone_mask,
            "wound": wound_mask,
            "props_df": props_df,
            "mtoc_df": mtoc_df,
        }

    @thread_worker
    def __apply_current_changes(self):
        # fetch data from napari to do re-processing
        cell_mask = self.viewer.layers["cells"].data
        nuc_mask = self.viewer.layers["nuclei"].data
        cone_mask = self.viewer.layers["orientation wedge"].data

        # recompute wound-edge
        wound_mask = WoundRUs.Segmenter.compute_wound_edge(cell_mask)
        cone_mask, props_df = WoundRUs.Segmenter.process_edge_cells(
            cell_mask, nuc_mask, wound_mask
        )

        (
            c_mother_mtoc_arr,
            c_daughter_mtoc_arr,
        ) = WoundRUs.utilfuncs.get_napari_mtocs(self.viewer)

        c_mother_df = WoundRUs.utilfuncs.get_tubulin_intensity(
            self.img2d[:, :, 0], c_mother_mtoc_arr
        )
        c_daughter_df = WoundRUs.utilfuncs.get_tubulin_intensity(
            self.img2d[:, :, 0], c_daughter_mtoc_arr
        )

        c_mother_df["mtoc_id"] = "mother"
        c_daughter_df["mtoc_id"] = "daughter"

        mtoc_df = pd.concat(
            [c_mother_df, c_daughter_df], axis=0, ignore_index=True
        )

        mtoc_df = WoundRUs.utilfuncs.det_classic_ori(
            mtoc_df, cone_mask, nuc_mask
        )

        mtoc_locs = tuple(
            [
                mtoc_df["mtoc_y"].values.astype(int),
                mtoc_df["mtoc_x"].values.astype(int),
            ]
        )

        mtoc_df["cell_id"] = cell_mask[mtoc_locs]

        return {
            "cells": cell_mask,
            "nuclei": nuc_mask,
            "cone": cone_mask,
            "wound": wound_mask,
            "props_df": props_df,
            "mtoc_df": mtoc_df,
        }

    def __check_args(*args):
        print(args)

    def __reset_run_btn_status(self):
        self.process_image_btn.setText("Process image")

    def __reset_apply_changes_btn_status(self):
        self.apply_manual_changes_btn.setText("Apply changes")

    def __display_result(self, segresult):
        if "cells" in self.viewer.layers:
            self.viewer.layers["cells"].data = segresult["cells"]
        else:
            cellmask_layer = self.viewer.add_labels(
                segresult["cells"], name="cells"
            )
            cellmask_layer.contour = 2

        if "nuclei" in self.viewer.layers:
            self.viewer.layers["nuclei"].data = segresult["nuclei"]

        else:
            nucmask_layer = self.viewer.add_labels(
                segresult["nuclei"], name="nuclei"
            )
            nucmask_layer.contour = 2

        if "orientation wedge" in self.viewer.layers:
            self.viewer.layers["orientation wedge"].data = segresult["cone"]
        else:
            self.viewer.add_labels(segresult["cone"], name="orientation wedge")

        if "wound" in self.viewer.layers:
            self.viewer.layers["wound"].data = segresult["wound"]
        else:
            self.viewer.add_labels(
                segresult["wound"],
                name="wound",
                num_colors=1,
                color={1: "red"},
            )

        # draw cell axis
        cell_axis_coordinates = []

        for i, row in segresult["props_df"].iterrows():
            start_coord = (row["back_y"], row["back_x"])
            end_coord = (row["migration_axis_y"], row["migration_axis_x"])
            cell_axis_coordinates.append([start_coord, end_coord])

        if "cell axis" in self.viewer.layers:
            self.viewer.layers["cell axis"].data = cell_axis_coordinates
        else:
            self.viewer.add_shapes(
                cell_axis_coordinates,
                name="cell axis",
                shape_type="line",
                edge_width=5,
                edge_color="#ff8ba0",
            )

        nuc_axis_coordinates = []

        for i, row in segresult["props_df"].iterrows():
            start_coord = (row["nucleus_centroid_y"], row["nucleus_centroid_x"])
            end_coord = (row["nucleus_major_y"], row["nucleus_major_x"])
            nuc_axis_coordinates.append([start_coord, end_coord])

        if "nucleus axis" in self.viewer.layers:
            self.viewer.layers["nucleus axis"].data = nuc_axis_coordinates
        else:
            self.viewer.add_shapes(
                nuc_axis_coordinates,
                name="nucleus axis",
                shape_type="line",
                edge_width=5,
                edge_color="#fa8128",
            )

        # display mtoc as points
        _mtoc = segresult["mtoc_df"]
        _mother_mtoc = _mtoc[_mtoc["mtoc_id"] == "mother"]
        _daughter_mtoc = _mtoc[_mtoc["mtoc_id"] == "daughter"]

        if "daughter mtoc" in self.viewer.layers:
            _mtoc_yx = _daughter_mtoc[["mtoc_y", "mtoc_x"]].values
            self.viewer.layers["daughter mtoc"].data = _mtoc_yx
            self.viewer.layers["daughter mtoc"].features = _daughter_mtoc

        else:
            _mtoc_yx = _daughter_mtoc[["mtoc_y", "mtoc_x"]].values
            _text = {
                "string": "{tubulin_intensity:0.0f}",
                "size": 12,
                "color": "white",
                "translation": np.array([-7, 0]),
            }

            self.viewer.add_points(
                _mtoc_yx,
                face_color="white",
                name="daughter mtoc",
                features=_daughter_mtoc,
                text=_text,
                size=25,
            )

        if "mother mtoc" in self.viewer.layers:
            _mtoc_yx = _mother_mtoc[["mtoc_y", "mtoc_x"]].values
            self.viewer.layers["mother mtoc"].data = _mtoc_yx
            self.viewer.layers["mother mtoc"].features = _mother_mtoc
        else:
            _tub_int_vals = _mother_mtoc["tubulin_intensity"].values
            _mtoc_yx = _mother_mtoc[["mtoc_y", "mtoc_x"]].values
            _text = {
                "string": "{tubulin_intensity:0.0f}",
                "size": 12,
                "color": "green",
                "translation": np.array([-7, 0]),
            }
            self.viewer.add_points(
                _mtoc_yx,
                face_color="green",
                name="mother mtoc",
                features=_mother_mtoc,
                text=_text,
                size=25,
            )

        # also save the data to current folder
        current_path = self.imgpath / self.current_filename
        df_fn_prefix = f"{current_path.stem}_props.csv"
        img_fn_prefix = f"{current_path.stem}_cp.tif"
        wedge_fn_prefix = f"{current_path.stem}_wedge.tif"
        mask_fn_prefix = f"{current_path.stem}_cp_masks.tif"
        nucmask_fn_prefix = f"{current_path.stem}_nucmasks.tif"
        mtoc_fn_prefix = f"{current_path.stem}_mtoc.csv"

        segresult["props_df"].to_csv(self.imgpath / df_fn_prefix, index=False)
        segresult["mtoc_df"].to_csv(self.imgpath / mtoc_fn_prefix, index=False)

        # save input image and its mask
        imwrite(self.imgpath / img_fn_prefix, self.img2d)
        imwrite(self.imgpath / mask_fn_prefix, segresult["cells"])
        imwrite(self.imgpath / wedge_fn_prefix, segresult["cone"])
        imwrite(self.imgpath / nucmask_fn_prefix, segresult["nuclei"])

    def _apply_manual_changes(self):
        self.apply_manual_changes_btn.setText(" ... ")
        worker = self.__apply_current_changes()
        worker.returned.connect(self.__display_result)
        # change the text on the button back to 'normal'
        worker.finished.connect(self.__reset_apply_changes_btn_status)
        worker.start()


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    # you can return either a single widget, or a sequence of widgets
    return [
        WEAWidget,
    ]
