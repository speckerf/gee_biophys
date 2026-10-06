/*******************************************************************************
 * Model
 ******************************************************************************/
var m = {fullImgc: null, images: [], traits: {}, chartCollection: null, scale: 20};
var b = {};

/*******************************************************************************
 * Styling + configuration
 ******************************************************************************/
var s = {};
s.imgcPath = '';
s.panelLeft = {width: '400px', padding: '8px'};
s.title = {fontSize: '22px', fontWeight: 'bold'};
s.description = {fontSize: '12px', color: '#555555'};
s.bandDescriptions = {
  mean: 'Average trait value for the export period.',
  stdDev: 'Total uncertainty, combining within-image uncertainty and variability across observations.',
  stdDev_within: 'Average predictive uncertainty within individual images.',
  stdDev_across: 'Variability of predictions across observations in the export period.',
  count: 'Number of valid observations per pixel in the export period.'
};
s.selectors = {shown: false};
s.panelRight = {width: '440px', padding: '10px', position: 'top-right', shown: false,
  backgroundColor: '#f5f7f8'};
s.section = {stretch: 'horizontal', padding: '10px', margin: '0 0 8px 0',
  backgroundColor: 'white', border: '1px solid #dce2e5'};
s.sectionTitle = {fontSize: '15px', fontWeight: 'bold', color: '#263238', margin: '0 0 6px 0'};
s.timeLabel = {fontSize: '13px', fontWeight: 'bold', color: '#196b73', margin: '8px 0'};
s.slider = {stretch: 'horizontal', margin: '4px 0'};
s.chartPanel = {stretch: 'horizontal', height: '280px', margin: '8px 0 0 0'};
s.legendTitle = {fontWeight: 'bold', fontSize: '12px'};
s.legendBar = {height: '10px', width: '220px', margin: '0 0 4px 0'};
s.legendTicks = {width: '220px'};
s.tick = {fontSize: '10px'};
s.palettes = {
  fcover: ['#f7fcf5', '#c7e9c0', '#74c476', '#238b45', '#00441b'],
  fapar: ['#ffffdd', '#e6ad12', '#c53859', '#3a26a1', '#000000'],
  lai: ['#fffdcd', '#e1cd73', '#aaac20', '#5f920c', '#187328', '#144b2a', '#172313'],
  stdDev: ['#440154', '#433982', '#30678D', '#218F8B', '#36B677', '#8ED542', '#FDE725'],
  count: ['#0C0A09', '#FAFAF9']
};
s.palettes.laie = s.palettes.lai;
s.ranges = {fapar: [0, 1], fcover: [0, 1], lai: [0, 5], laie: [0, 5]};
s.stdRanges = {fapar: [0, 0.3], fcover: [0, 0.3], lai: [0, 2], laie: [0, 2]};
s.variableNames = {fapar: 'FAPAR', fcover: 'FCOVER', lai: 'Leaf Area Index', laie: 'Effective Leaf Area Index'};
s.chartOptions = {
  height: 270, chartArea: {left: 60, right: 18, top: 40, bottom: 55},
  hAxis: {title: 'Date'}, vAxis: {title: 'Value', minValue: 0},
  lineWidth: 2, pointSize: 3, legend: {position: 'none'}, interpolateNulls: true,
  series: {0: {color: 'grey', lineDashStyle: [4, 4]},
    1: {color: 'black'}, 2: {color: 'grey', lineDashStyle: [4, 4]}}
};

/*******************************************************************************
 * Components
 ******************************************************************************/
var c = {};
c.path = ui.Textbox({placeholder: 'projects/your-project/assets/your-collection', value: s.imgcPath});
c.load = ui.Button({label: 'Load collection', onClick: function() { b.loadCollection(); }});
c.status = ui.Label('Load a collection, then select a trait and band.');
c.trait = ui.Select({placeholder: 'Select trait', onChange: function() { b.updateBands(); }});
c.band = ui.Select({placeholder: 'Select band', onChange: function() { b.updateMap(); }});
c.legend = ui.Panel();
c.bandDescription = ui.Label('', s.description);
c.selectors = ui.Panel({widgets: [
  ui.Label('Trait:'), c.trait, ui.Label('Band:'), c.band, c.bandDescription, ui.Label('Legend:'), c.legend
], style: s.selectors});
c.leftPanel = ui.Panel({widgets: [
  ui.Label('Visualize gee-biophys exports', s.title),
  ui.Label('gee-biophys exports Sentinel-2 vegetation traits for custom regions and time periods.', s.description),
  ui.Label('LAIe: effective leaf area index; FAPAR: fraction of absorbed photosynthetically active radiation; FCOVER: fractional vegetation cover.', s.description),
  ui.Label('Load a collection, choose a trait and band, then drag the slider through export periods. Click the map for the trait’s mean time series and, when available, mean ± standard deviation.', s.description),
  ui.Label('For the published app, export with --public. View private assets by running this script in your own Earth Engine Code Editor.', s.description),
  ui.Label('Input ImageCollection asset ID:'), c.path, c.load, c.status, c.selectors,
  ui.Label({value: 'gee-biophys documentation', targetUrl: 'https://pypi.org/project/gee-biophys/2.1.3/', style: s.description}),
  ui.Label('Developed by Felix Specker · WSL · felix.specker@wsl.ch', s.description),
  ui.Label({value: 'Source code · MIT license', targetUrl: 'https://github.com/speckerf/gee_biophys', style: s.description}),
  ui.Label({value: 'Citation: Specker, Schweiger, Féret et al. (2026). Advancing Ecosystem Monitoring with Global High-Resolution Maps of Vegetation Biophysical Properties. Preprint, version 3.',
    targetUrl: 'https://doi.org/10.21203/rs.3.rs-6343364/v3', style: s.description}),
  ui.Label('Part of the Open-Earth-Monitor Cyberinfrastructure (OEMC) project, funded by the European Union’s Horizon Europe programme (grant No. 101059548).', s.description)
], style: s.panelLeft});
c.timeLabel = ui.Label('', s.timeLabel);
c.slider = ui.Slider({min: 0, max: 1, value: 0, step: 1, style: s.slider,
  onChange: function(value) { b.showImage(Math.round(value)); }});
// Keep updating layers while the mouse is held down and the slider moves.
c.slider.onSlide(function(value) { b.showImage(Math.round(value)); });
c.timeline = ui.Panel({widgets: [
  ui.Label('Export period', s.sectionTitle),
  ui.Label('Drag to browse images; the map updates as you move.', s.description),
  c.timeLabel, c.slider
], style: s.section});
c.timeline.style().set('shown', false);
c.chartPanel = ui.Panel({style: s.chartPanel});
c.pointLabel = ui.Label('Click a location on the map.', s.description);
c.timeSeriesSection = ui.Panel({widgets: [
  ui.Label('Point time series', s.sectionTitle),
  ui.Label('Mean over all export periods, with ± standard deviation when available.', s.description),
  c.pointLabel, c.chartPanel
], style: s.section});
c.rightPanel = ui.Panel({widgets: [c.timeline, c.timeSeriesSection],
  layout: ui.Panel.Layout.flow('vertical'), style: s.panelRight});
c.pointLayer = ui.Map.Layer(ee.FeatureCollection([]), {color: 'yellow'}, 'Clicked point');
c.layers = [];

/*******************************************************************************
 * Behaviour
 ******************************************************************************/
b.visParams = function(trait, band) {
  var uncertainty = band.indexOf('stdDev') === 0;
  var range = band === 'count' ? [0, 30] :
    (uncertainty ? s.stdRanges[trait] : s.ranges[trait]) || [0, 1];
  return {min: range[0], max: range[1],
    palette: band === 'count' ? s.palettes.count :
      uncertainty ? s.palettes.stdDev : s.palettes[trait] || s.palettes.fapar};
};

b.clearMap = function() {
  c.layers.forEach(function(layer) { Map.layers().remove(layer); });
  c.layers = [];
  c.timeline.style().set('shown', false);
  c.legend.clear();
  c.chartPanel.clear();
  c.pointLabel.setValue('Click a location on the map.');
  c.pointLayer.setEeObject(ee.FeatureCollection([]));
  m.chartCollection = null;
};

b.loadCollection = function() {
  b.clearMap();
  c.selectors.style().set('shown', false);
  c.rightPanel.style().set('shown', false);
  m.images = [];
  m.traits = {};
  var path = c.path.getValue().trim();
  if (!path) { c.status.setValue('Enter an ImageCollection asset ID.'); return; }
  c.load.setDisabled(true);
  c.status.setValue('Loading collection...');
  m.fullImgc = ee.ImageCollection(path).sort('system:time_start');
  // Fetch metadata once; keep image pixels on the server.
  m.fullImgc.toList(m.fullImgc.size()).map(function(item) {
    var image = ee.Image(item);
    return ee.Dictionary({index: image.get('system:index'), bands: image.bandNames(),
      start: image.get('system:time_start'), end: image.get('system:time_end'),
      scale: image.get('export_scale')});
  }).evaluate(function(images, error) {
    c.load.setDisabled(false);
    if (error || !images || !images.length) {
      c.status.setValue(error ? 'Could not load collection: ' + error : 'Collection is empty.');
      return;
    }
    m.images = images;
    images.forEach(function(image) {
      image.bands.forEach(function(name) {
        var split = name.indexOf('_');
        if (split < 1) { return; }
        var trait = name.slice(0, split);
        var band = name.slice(split + 1);
        if (!m.traits[trait]) { m.traits[trait] = []; }
        if (m.traits[trait].indexOf(band) < 0) { m.traits[trait].push(band); }
      });
    });
    var traits = Object.keys(m.traits).sort();
    if (!traits.length) { c.status.setValue('No trait_band bands found.'); return; }
    c.trait.items().reset(traits);
    c.trait.setValue(traits[0], false);
    b.updateBands();
    c.selectors.style().set('shown', true);
    c.rightPanel.style().set('shown', true);
    c.status.setValue(images.length + ' images loaded.');
    Map.centerObject(m.fullImgc.first());
  });
};

b.getImage = function(metadata) {
  return ee.Image(m.fullImgc.filter(ee.Filter.eq('system:index', metadata.index)).first());
};

b.updateBands = function() {
  var bands = m.traits[c.trait.getValue()] || [];
  c.band.items().reset(bands);
  c.band.setValue(bands.indexOf('mean') >= 0 ? 'mean' : bands[0], false);
  b.updateMap();
};

b.updateLegend = function(name, vis) {
  c.legend.add(ui.Label((s.variableNames[c.trait.getValue()] || c.trait.getValue()) + ' (' + name + ')', s.legendTitle));
  c.legend.add(ui.Thumbnail({image: ee.Image.pixelLonLat().select(0),
    params: {bbox: [0, 0, 1, 0.1], dimensions: '220x10', format: 'png',
      min: 0, max: 1, palette: vis.palette}, style: s.legendBar}));
  c.legend.add(ui.Panel({widgets: [ui.Label(vis.min.toFixed(2), s.tick),
    ui.Label('', {stretch: 'horizontal'}), ui.Label(vis.max.toFixed(2), s.tick)],
    layout: ui.Panel.Layout.flow('horizontal'), style: s.legendTicks}));
};

b.showImage = function(index) {
  c.layers.forEach(function(layer, i) { layer.setOpacity(i === index ? 1 : 0); });
  if (c.layers[index]) { c.timeLabel.setValue(c.layers[index].getName()); }
};

b.updateMap = function() {
  b.clearMap();
  var trait = c.trait.getValue();
  var band = c.band.getValue();
  if (!trait || !band) { return; }
  c.bandDescription.setValue(s.bandDescriptions[band] || 'Exported band: ' + band);
  var name = trait + '_' + band;
  var vis = b.visParams(trait, band);
  var images = m.images.filter(function(image) { return image.bands.indexOf(name) >= 0; });
  images.forEach(function(metadata, index) {
    var label = metadata.start != null ? new Date(metadata.start).toISOString().slice(0, 10) : metadata.index;
    if (metadata.end != null) { label += ' – ' + new Date(metadata.end).toISOString().slice(0, 10); }
    var layer = ui.Map.Layer(b.getImage(metadata).select(name), vis, label, true, index === 0 ? 1 : 0);
    c.layers.push(layer);
    Map.layers().add(layer);
  });
  b.updateLegend(name, vis);
  c.slider.setMax(Math.max(1, images.length - 1));
  c.slider.setValue(0, false);
  c.slider.style().set('shown', images.length > 1);
  c.timeline.style().set('shown', images.length > 0);
  b.showImage(0);
  b.prepareChart(trait);
};

b.prepareChart = function(trait) {
  var mean = trait + '_mean';
  var std = trait + '_stdDev';
  var images = m.images.filter(function(image) {
    return image.bands.indexOf(mean) >= 0 && image.start != null;
  });
  if (!images.length) {
    c.chartPanel.add(ui.Label('Point time series requires mean bands and timestamps.'));
    return;
  }
  m.scale = images[0].scale || 20;
  // Use uncertainty limits only when every mean image has a standard deviation.
  var hasStd = images.every(function(image) { return image.bands.indexOf(std) >= 0; });
  m.chartCollection = ee.ImageCollection.fromImages(images.map(function(metadata) {
    var image = b.getImage(metadata);
    var result = image.select(mean);
    if (hasStd) {
      result = image.select(mean).subtract(image.select(std)).max(0).rename(trait + '_lower')
        .addBands(result)
        .addBands(image.select(mean).add(image.select(std)).rename(trait + '_upper'));
    }
    return result.copyProperties(image, ['system:time_start', 'system:time_end']);
  }));
  m.hasStd = hasStd;
  c.chartPanel.add(ui.Label('Click the map to extract a point time series.'));
};

b.displayPointTimeSeries = function(coords) {
  if (!m.chartCollection) { return; }
  var point = ee.Geometry.Point([coords.lon, coords.lat]);
  c.pointLayer.setEeObject(point);
  c.pointLabel.setValue('Longitude ' + coords.lon.toFixed(5) + ' · Latitude ' + coords.lat.toFixed(5));
  s.chartOptions.title = c.trait.getValue() + ' at (' + coords.lon.toFixed(3) + ', ' + coords.lat.toFixed(3) + ')';
  s.chartOptions.vAxis.title = (s.variableNames[c.trait.getValue()] || c.trait.getValue()) + (m.hasStd ? ' (mean ± std)' : ' (mean)');
  s.chartOptions.series = m.hasStd ? {0: {color: 'grey', lineDashStyle: [4, 4]},
    1: {color: 'black'}, 2: {color: 'grey', lineDashStyle: [4, 4]}} : {0: {color: 'black'}};
  var chart = ui.Chart.image.series({imageCollection: m.chartCollection, region: point,
    reducer: ee.Reducer.mean(), scale: m.scale, xProperty: 'system:time_start'})
    .setChartType('LineChart').setOptions(s.chartOptions);
  c.chartPanel.widgets().reset([chart]);
};

/*******************************************************************************
 * Initialize
 ******************************************************************************/
Map.setOptions('SATELLITE');
Map.style().set({cursor: 'crosshair'});
Map.drawingTools().setShown(false);
ui.root.widgets().insert(0, c.leftPanel);
Map.add(c.rightPanel);
Map.layers().add(c.pointLayer);
Map.onClick(b.displayPointTimeSeries);
