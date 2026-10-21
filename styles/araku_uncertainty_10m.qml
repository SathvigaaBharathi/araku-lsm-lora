<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
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