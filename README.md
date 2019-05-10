# Smelter

Overhead-free ONNX graph inference engine with Metal under the hood.

⚠️ ⚠️ ⚠️ IMPORTANT: Currently the library is highly experimental. Not all the layers of ONNX format are supported, pull requests are always welcome.

## Key Features
* Native support for ONNX file format
* Seamless raw GPU integration
* Easy to use API

## Usage examples:

* ##### Graph Initialization:
  ```swift
  // Read ONNX model data.
  let modelURL = Bundle.main.url(forResource: "yourAwesomeModel",
                                withExtension: "onnx")!
  let modelData = try Data(contentsOf: modelURL)
  // Specify ONNXGraph configuration.
  let configuration = ONNXGraph.Configuration(inputConstraint: .forceInputScale(scale: .bilinear))
  // Initialize the MPSNNGraph.
  let nnGraph = try ONNXGraph(data: modelData).metalGraph(device: context.device,
                                                          configuration: configuration)
  ```

* ##### Graph Inference:
  Due to the fact that `Smelter` uses [`Alloy`](https://github.com/s1ddok/Alloy) as a dependency, we   are free to use its handy utils for texture creation and graph result reading.
  ```swift
  /// Create texture from CGImage.
  guard
      let inputTexture = try self.metalContext.texture(from: cgImage)
  else { throw Errors.textureCreationFailed }
  /// Create MPSImage from MTLTexture.
  let inputMPSImage = MPSImage(texture: inputTexture,
                               featureChannels: 3)
  /// Encode the MPSNNGraph.
  guard
      let modelGraphResult = nnGraph.encode(to: commandBuffer,
                                            sourceImages: [inputMPSImage])
  else { throw Errors.graphEncodingFailed }
  /// Read the results.
  guard
      let output = modelGraphResult.toFloatArray()
  else { throw Errors.graphResultReadingFailed }
  ```

* ##### Model optmization for MPS:
  Project repo contains `ONNX2MPS.py` script that allows to optimize model's layers for `MPS`.
  ```python
  python ONNX2MPS.py --input ulr_to_your_model.onnx --output ulr_to_your_optimized_model.onnx
  ```

## Known Problems
* As `MPSNNGraph` does not have analogs of all layers provided by ONNX, not all models are supported.
* `Non-MPS-Optimized` models might not work correctly enough.

## Installation

##### CocoaPods

[CocoaPods](https://cocoapods.org) is a dependency manager for Cocoa projects. For usage and installation instructions, visit their website. To integrate Smelter into your Xcode project using CocoaPods, specify it in your `Podfile`:

```ruby
# Optionally add version, i.e. '~> 1.0.0'
pod 'Smelter'
```

## License

MIT
