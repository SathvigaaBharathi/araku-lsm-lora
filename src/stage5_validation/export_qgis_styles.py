import xml.etree.ElementTree as ET
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.utils import OUTPUTS_DIR, get_logger

logger = get_logger("export_qgis_styles")


def generate_lsm_qml():
    """Generate QGIS QML style file for Susceptibility Map (Spectral color ramp: Blue->Yellow->Red)."""
    qml_content = """<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.28.0" hasScaleBasedVisibilityFlag="0">
  <pipe>
    <provider>
      <resampling maxOversampling="2" zoomedOutResamplingMethod="nearestNeighbour" zoomedInResamplingMethod="nearestNeighbour"/>
    </provider>
    <rasterrenderer type="singlebandpseudocolor" opacity="1" classificationMin="0" classificationMax="1" band="1">
      <rasterTransparency/>
      <rastershader>
        <colorrampshader colorRampType="INTERPOLATED" clip="0">
          <colorramp type="gradient" name="[custom]">
            <item value="0.0" color="#2c7bb6" alpha="255" label="Very Low (0.0)"/>
            <item value="0.25" color="#abd9e9" alpha="255" label="Low (0.25)"/>
            <item value="0.50" color="#ffffbf" alpha="255" label="Moderate (0.50)"/>
            <item value="0.75" color="#fdae61" alpha="255" label="High (0.75)"/>
            <item value="1.0" color="#d7191c" alpha="255" label="Very High (1.0)"/>
          </colorramp>
        </colorrampshader>
      </rastershader>
    </rasterrenderer>
  </pipe>
  <blendMode>0</blendMode>
</qgis>
"""
    qml_path = OUTPUTS_DIR / "araku_lsm_10m.qml"
    with open(qml_path, "w", encoding="utf-8") as f:
        f.write(qml_content.strip())
    logger.info(f"Generated QGIS style file for Susceptibility Map at {qml_path}")
    return qml_path


def generate_uncertainty_qml():
    """Generate QGIS QML style file for Epistemic Uncertainty Map (Viridis color ramp)."""
    qml_content = """<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.28.0" hasScaleBasedVisibilityFlag="0">
  <pipe>
    <provider>
      <resampling maxOversampling="2" zoomedOutResamplingMethod="nearestNeighbour" zoomedInResamplingMethod="nearestNeighbour"/>
    </provider>
    <rasterrenderer type="singlebandpseudocolor" opacity="1" classificationMin="0" classificationMax="1" band="1">
      <rasterTransparency/>
      <rastershader>
        <colorrampshader colorRampType="INTERPOLATED" clip="0">
          <colorramp type="gradient" name="Viridis">
            <item value="0.0" color="#440154" alpha="255" label="Certain / Low Variance (0.0)"/>
            <item value="0.25" color="#3b528b" alpha="255" label="Low-Mod Variance (0.25)"/>
            <item value="0.50" color="#21918c" alpha="255" label="Moderate Variance (0.50)"/>
            <item value="0.75" color="#5ec962" alpha="255" label="High Variance (0.75)"/>
            <item value="1.0" color="#fde725" alpha="255" label="High Uncertainty (1.0)"/>
          </colorramp>
        </colorrampshader>
      </rastershader>
    </rasterrenderer>
  </pipe>
  <blendMode>0</blendMode>
</qgis>
"""
    qml_path = OUTPUTS_DIR / "araku_uncertainty_10m.qml"
    with open(qml_path, "w", encoding="utf-8") as f:
        f.write(qml_content.strip())
    logger.info(f"Generated QGIS style file for Uncertainty Map at {qml_path}")
    return qml_path


if __name__ == "__main__":
    generate_lsm_qml()
    generate_uncertainty_qml()
