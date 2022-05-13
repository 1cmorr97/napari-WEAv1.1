"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/plugins/stable/guides.html#widgets

Replace code below according to your needs.
"""

from napari_plugin_engine import napari_hook_implementation
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QGroupBox,
    QPushButton,
    QLabel,
    QComboBox,
)
import WEA
import numpy as np


class WEAWidget(QWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # in one of two ways:
    # 1. use a parameter called `napari_viewer`, as done here
    # 2. use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        self.layout = QVBoxLayout()

        self.ch_groupbox = QGroupBox("Segmentation channels")
        self.ch_vbox = QVBoxLayout()

        self.setLayout(self.layout)

        available_layers = [layer.name for layer in self.viewer.layers]

        self.cytogroup = QComboBox(self)
        self.nucgroup = QComboBox(self)
        self.cytogroup.addItems(available_layers)
        self.nucgroup.addItems(available_layers)
        self.ch_vbox.addWidget(QLabel("Cytoplasm channel"))
        self.ch_vbox.addWidget(self.cytogroup)
        self.ch_vbox.addWidget(QLabel("Nucleus channel"))
        self.ch_vbox.addWidget(self.nucgroup)
        self.run_current_frame_btn = QPushButton("Run current frame")
        self.ch_vbox.addWidget(self.run_current_frame_btn)

        # set channel vbox (and its widgets) to channel groupbox
        self.ch_groupbox.setLayout(self.ch_vbox)
        self.layout.addWidget(self.ch_groupbox)

        # fill the rest of the vertical space so preceding widgets stack
        # from top-to-bottom
        self.layout.addStretch()

        # gui behavior
        self.run_current_frame_btn.clicked.connect(self._run_current_frame)

    def _run_current_frame(self):
        pass
        # get currently viewed z index
        # current_z = int(self.viewer.layers.selection.active.position[0])
        # cyto_layer = self.cytogroup.currentText()
        # nuc_layer = self.nucgroup.currentText()
        # cyto_img = self.viewer.layers[cyto_layer].data[current_z, :, :]
        # nuc_img = self.viewer.layers[nuc_layer].data[current_z, :, :]
        # rgbcomp = np.zeros(cyto_img.shape + (3,), dtype=np.uint8)
        # rgbcomp[:, :, 1] = norm28bit(np.log(cyto_img))
        # rgbcomp[:, :, 2] = norm28bit(nuc_img)

        # cell_label = segmentCell(rgbcomp)

        # img_shape = self.viewer.layers[cyto_layer].data.shape

        # if "Cellpose result" in [layer.name for layer in self.viewer.layers]:
        #     labels = self.viewer.layers["Cellpose result"].data
        #     self.viewer.layers["Cellpose result"].data[
        #         current_z, :, :
        #     ] = cell_label
        # else:
        #     labels = np.zeros(img_shape, dtype=np.uint8)
        #     labels[current_z, :, :] = cell_label
        #     self.viewer.add_labels(labels, name="Cellpose result")


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    # you can return either a single widget, or a sequence of widgets
    return [WEAWidget]
