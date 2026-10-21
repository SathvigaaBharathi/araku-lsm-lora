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