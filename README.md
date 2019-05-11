# Smelter

Overhead-free ONNX graph inference engine with Metal under the hood.

⚠️  IMPORTANT: Even though we run all neural networks using this library in production, currently it is in a highly experimental state: APIs are subject to change, not all the layers of ONNX format are supported. Pull requests are always welcome!

## Key Features
* Native support for ONNX file format with different storage modes 
* ONNX graph optimizing preprocessing steps
* Seamless raw GPU integration
* Easy to use API

## Usage examples:

* ##### Graph Initialization:
  ```swift
  // Read ONNX model data.
  let context = MTLContext(device: Metal.device)
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
  Due to the fact that `Smelter` uses [`Alloy`](https://github.com/s1ddok/Alloy) as a dependency, we are free to use its handy utils for texture creation and graph result reading.
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

* ##### Preprocessing steps:
  ONNX storage weight's storage convention differs from the one used in Metal Performance Shaders. Smelter supports transposing them at run-time, but it can take a while for certain graphs. We recommned preprocess your graphs with our script. It has some bonuses of fusing certain layers (i.e. Batch Normalization into Convolution), converting your weights into fp16 and more. 
  
  ```python
  python3 ONNX2MPS.py --input ulr_to_your_model.onnx --output ulr_to_your_optimized_model.onnx [--half]
  ```

## Known Problems
* As `MPSNNGraph` does not have analogs of all layers provided by ONNX, not all models are supported
* Some Metal implementations are buggy:
  - Concat works only when `both operands have a multiple of 4 feature channels`
  - Concat doesn't work for some layers when they are the second operand (i.e. `Upsample + Conv` works and `Conv + Upsample` doesn't)
  - Certain nodes are only available on higher iOS versions (i.e. Reshape is available only after 12.1, but usually you can cut it alltogether from the model)
  - Certain Metal convolution implementations produce `NaN`s
  
## Installation

##### CocoaPods

[CocoaPods](https://cocoapods.org) is a dependency manager for Cocoa projects. For usage and installation instructions, visit their website. To integrate Smelter into your Xcode project using CocoaPods, specify it in your `Podfile`:

```ruby
# Optionally add version, i.e. '~> 1.0.0'
pod 'Smelter'
```

## License

MIT

## Authors

Oleg Poyaganov ([@opedge](https://github.com/opedge)), Andrey Volodin ([@s1ddok](https://github.com/s1ddok)), Konstantin Semyanov ([@ksemianov](https://github.com/ksemianov)), Eugene Bokhan ([@eugenebokhan](https://github.com/eugenebokhan)) Anton Lebedev ([@antoleb](https://github.com/antoleb))
