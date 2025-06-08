import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal
from skimage import morphology, measure
from skimage.morphology import (
    remove_small_objects,
    skeletonize,
    disk,
    closing
)
import scipy.ndimage as ndi
from skan.csr import make_degree_image
from skan import Skeleton, summarize
import math
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
from collections import OrderedDict


# -----------------------------------
# Utility functions (strictly classes 1–7, 0 = NoData)
# -----------------------------------
def read_raster(path):
    ds = gdal.Open(path)
    if ds is None:
        raise IOError(f"Cannot open {path}")
    arr = ds.GetRasterBand(1).ReadAsArray()
    arr[arr == 127] = 0
    return ds, (arr > 0).astype(np.uint8)


def write_classified_raster(path, array, ref_ds, cls_map):
    drv = gdal.GetDriverByName('GTiff')
    out = drv.Create(
        path,
        ref_ds.RasterXSize,
        ref_ds.RasterYSize,
        1,
        gdal.GDT_Byte
    )
    out.SetGeoTransform(ref_ds.GetGeoTransform())
    out.SetProjection(ref_ds.GetProjection())
    band = out.GetRasterBand(1)
    band.WriteArray(array.astype(np.uint8))
    band.SetNoDataValue(0)

    rat = gdal.RasterAttributeTable()
    rat.CreateColumn('Value', gdal.GFT_Integer, gdal.GFU_MinMax)
    rat.CreateColumn('ClassName', gdal.GFT_String, gdal.GFU_Name)
    rat.SetRowCount(len(cls_map))
    for row, (name, val) in enumerate(cls_map.items()):
        rat.SetValueAsInt(row, 0, val)
        rat.SetValueAsString(row, 1, name.capitalize())
    band.SetDefaultRAT(rat)
    band = None
    out = None


# -----------------------------------
# MSPA Analysis Class
# -----------------------------------
class MSPAAnalysis:
    def __init__(self, file_paths, edge_width=1, core_method='cellular', core_expand=1):
        self.file_paths = file_paths
        self.edge_width = edge_width
        self.core_method = core_method
        self.core_expand = core_expand
        self.results = {}

        self.cls_map = OrderedDict([
            ('core', 1),
            ('edge', 2),
            ('perforation', 3),
            ('islet', 4),
            ('bridge', 5),
            ('branch', 6),
            ('loop', 7),
        ])
        self.cls_colors = {
            'core': (1, 0, 0, 0.6),
            'edge': (0, 1, 0, 0.6),
            'perforation': (0, 0, 1, 0.5),
            'islet': (1, 0.5, 0, 0.7),
            'bridge': (0, 0, 0.5, 0.8),
            'branch': (0.6, 0.6, 0.6, 0.8),
            'loop': (1, 1, 0, 0.8),
        }

    def load_data(self):
        for year, path in self.file_paths.items():
            ds, mask = read_raster(path)
            self.results[year] = {'binary': mask.astype(bool), 'ref_ds': ds}

    def _detect_core(self, bi):
        """Core: the densest contiguous urban areas."""
        if self.core_method == 'cellular':
            kern = np.ones((3, 3), np.uint8)
            core = bi.copy()
            for _ in range(self.edge_width):
                nbr = ndi.convolve(core.astype(np.uint8), kern, mode='constant')
                core &= (nbr == 9)
            # lower the min_size to include more small cores
            core = remove_small_objects(core, min_size=10)
        else:
            dist = ndi.distance_transform_edt(bi)
            thr = self.edge_width * dist.max() / (self.edge_width + 1)
            core = (dist > thr) & bi
            core = remove_small_objects(core, min_size=10)
        if self.core_expand > 0:
            core = morphology.binary_dilation(core, disk(self.core_expand))
            core = remove_small_objects(core, min_size=10)
        return core

    def _detect_core_2015(self, bi):
        """Specialized Core detection for 2015 combining cellular and geodesic methods."""
        core_std = self._detect_core(bi)
        dist = ndi.distance_transform_edt(bi)
        norm = dist / (dist.max() + 1e-9)
        geo = (norm > 0.2) & bi
        geo = remove_small_objects(geo, min_size=10)
        merged = closing(core_std | geo, disk(2))
        if self.core_expand > 0:
            merged = morphology.binary_dilation(merged, disk(self.core_expand))
            merged = remove_small_objects(merged, min_size=10)
        return merged

    def _detect_edge(self, bi, core):
        """
        Edge: forms a band around each Core, capturing the transition zone
        where urban and non-urban meet.
        """
        dil = morphology.dilation(bi, disk(self.edge_width))
        ero = morphology.erosion(bi, disk(self.edge_width))
        return (dil ^ ero) & ~core

    def _detect_perforations(self, core):
        """
        Perforation: non-urban “holes” entirely surrounded by Core.
        """
        filled = ndi.binary_fill_holes(core)
        holes = filled & ~core
        lbl = measure.label(holes, connectivity=2)
        perf = np.zeros_like(core, dtype=bool)
        for reg in measure.regionprops(lbl):
            # lower the min_area and solidity threshold
            if reg.area >= 3 and reg.solidity > 0.3:
                perf[tuple(zip(*reg.coords))] = True
        return perf

    def _detect_islets(self, bi, core):
        """
        Islet: small, isolated urban patches that do not connect to any Core area.
        """
        lbl = measure.label(bi, connectivity=2)
        iso = np.zeros_like(bi, dtype=bool)
        core_buffer = morphology.binary_dilation(core, disk(self.edge_width + 1))
        for reg in measure.regionprops(lbl):
            coords = tuple(zip(*reg.coords))
            if not core[coords].any() and not core_buffer[coords].any():
                # include patches up to 200 pixels
                if reg.area <= 200:
                    iso[coords] = True
        return iso

    def _skeleton_elements(self, bi, core):
        """
        Bridge: narrow urban corridors that connect two separate Core areas.
        Branch: skeleton endpoints—narrow spurs extending off a Core or Bridge.
        """
        skel = skeletonize(bi)
        # lower skeleton prune size to keep finer details
        pruned = remove_small_objects(skel, min_size=5)
        skeleton_obj = Skeleton(pruned)
        branch_data = summarize(skeleton_obj)

        bridge_mask = np.zeros_like(pruned, dtype=bool)
        branch_mask = np.zeros_like(pruned, dtype=bool)

        for idx, row in branch_data.iterrows():
            btype = row['branch-type']
            coords = skeleton_obj.path_coordinates(idx)
            mask = bridge_mask if btype == 'junction_to_junction' else branch_mask
            for y, x in coords:
                mask[int(y), int(x)] = True

        deg = make_degree_image(pruned)
        endpoints = (deg == 1)
        branch_mask |= endpoints

        bridge = morphology.binary_dilation(bridge_mask, disk(1))
        branch = morphology.binary_dilation(branch_mask, disk(1))

        core_lbl = measure.label(core, connectivity=2)
        bridge_lbl = measure.label(bridge, connectivity=2)
        refined_bridge = np.zeros_like(bridge, dtype=bool)
        for reg in measure.regionprops(bridge_lbl):
            coords = tuple(zip(*reg.coords))
            touched = np.unique(core_lbl[coords])
            touched = touched[touched > 0]
            if len(touched) >= 2:
                refined_bridge[coords] = True
        bridge = refined_bridge

        return bridge, branch

    def _detect_loops(self, edge):
        """
        Loop: closed edge contours, identified as holes in the edge mask
        with sufficient size and circularity.
        """
        filled = ndi.binary_fill_holes(edge)
        holes = filled & ~edge
        lbl_holes = measure.label(holes, connectivity=2)
        loop = np.zeros_like(edge, dtype=bool)
        for reg in measure.regionprops(lbl_holes):
            # lower loop area and circularity thresholds
            if reg.area >= 10:
                circ = 4 * math.pi * reg.area / (reg.perimeter ** 2 + 1e-9)
                if circ >= 0.4:
                    loop[tuple(zip(*reg.coords))] = True
        loop = morphology.binary_dilation(loop, disk(1))
        return loop

    def analyze(self):
        for year, rec in self.results.items():
            bi = rec['binary']
            core = (self._detect_core_2015(bi) if year == 2015 else self._detect_core(bi))
            raw_edge = self._detect_edge(bi, core)

            lbl_edge = measure.label(raw_edge, connectivity=2)
            tiny_edge = np.zeros_like(raw_edge, dtype=bool)

            # allow larger tiny-edge components
            for reg in measure.regionprops(lbl_edge):
                if reg.area <= 10:
                    tiny_edge[tuple(zip(*reg.coords))] = True

            edge = raw_edge & ~tiny_edge
            isle = self._detect_islets(bi, core) | tiny_edge
            perf = self._detect_perforations(core)
            bridge, branch = self._skeleton_elements(bi, core)
            branch &= ~core
            loop = self._detect_loops(edge)

            rec.update({
                'core': core,
                'edge': edge,
                'perforation': perf,
                'islet': isle,
                'bridge': bridge,
                'branch': branch,
                'loop': loop
            })

            for name in self.cls_map:
                rec[name] &= bi

    def visualize(self):
        for year, rec in self.results.items():
            fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
            for name in self.cls_map:
                ax.imshow(
                    rec[name],
                    cmap=ListedColormap([(0, 0, 0, 0), self.cls_colors[name]]),
                    vmin=0, vmax=1
                )
            patches = [
                Patch(facecolor=self.cls_colors[k][:3],
                      alpha=self.cls_colors[k][3],
                      label=k.capitalize())
                for k in self.cls_map
            ]
            ax.legend(handles=patches,
                      loc='lower center',
                      bbox_to_anchor=(0.5, -0.05),
                      ncol=4, frameon=False, fontsize='small')
            ax.set_title(f"MSPA classes — {year}")
            ax.axis('off')
            plt.tight_layout()
            plt.savefig(f"MSPA_{year}.png", dpi=300, bbox_inches='tight')
            plt.close()

    def compute_metrics(self):
        for year, rec in self.results.items():
            ds = rec['ref_ds']
            gt = ds.GetGeoTransform()
            pa = gt[1] * abs(gt[5])
            print(f"\n--- {year} area (km²) ---")
            for k in self.cls_map:
                cnt = np.count_nonzero(rec[k])
                area_km2 = cnt * pa / 1e6
                print(f"{k.capitalize():12s}: {area_km2:8.3f}")

    def export_rasters(self):
        for year, rec in self.results.items():
            cls = np.zeros_like(rec['binary'], dtype=np.uint8)
            for name, val in self.cls_map.items():
                cls[rec[name]] = val
            write_classified_raster(
                f"MSPA_classes_{year}.tif",
                cls,
                rec['ref_ds'],
                self.cls_map
            )


if __name__ == "__main__":
    file_paths = {
        2015: "Hong Kong DATAnew/2005hk.tif",
        2020: "Hong Kong DATAnew/2015hk.tif",
        2025: "Hong Kong DATAnew/2025hk.tif"
    }
    mspa = MSPAAnalysis(file_paths, edge_width=1, core_method='cellular', core_expand=1)
    print("Loading…")
    mspa.load_data()
    print("Analyzing…")
    mspa.analyze()
    print("Visualizing…")
    mspa.visualize()
    print("Metrics…")
    mspa.compute_metrics()
    print("Exporting…")
    mspa.export_rasters()
    print("Done.")
