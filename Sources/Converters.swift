import Alloy
import MetalPerformanceShaders

import Accelerate

enum ConvWeightArray {
    case float32([Float])
    case float16([Float16])
    case invalid
}

// MARK: Convolutions

@objc private final class ConvDataSource: NSObject, MPSCNNConvolutionDataSource {
    func copy(with zone: NSZone? = nil) -> Any {
        return self.mutableCopy()
    }

    private var weight: ConvWeightArray
    private var bias: [Float]?
    public var outputChannels: Int = 0

    private let desc: MPSCNNConvolutionDescriptor

    private let isDepthwise: Bool

    init(weight: Onnx_TensorProto,
         bias: Onnx_TensorProto?,
         dilations: (Int, Int),
         strides: (Int, Int),
         groups: Int,
         isTranspose: Bool,
         isONNX2MPS: Bool) {
        var outputChannels: Int
        var kernelHeight: Int
        var kernelWidth: Int
        var inputChannels: Int

        if isONNX2MPS {
            outputChannels = Int(weight.dims[0])
            kernelHeight = Int(weight.dims[1])
            kernelWidth = Int(weight.dims[2])
            inputChannels = Int(weight.dims[3])
        } else {
            kernelHeight = Int(weight.dims[2])
            kernelWidth = Int(weight.dims[3])
            if isTranspose {
                outputChannels = Int(weight.dims[1])
                inputChannels = Int(weight.dims[0])
            } else {
                outputChannels = Int(weight.dims[0])
                inputChannels = Int(weight.dims[1])
            }
        }

        self.isDepthwise = (groups != 1) && (groups == outputChannels)

        if isDepthwise {
            self.desc = MPSCNNDepthWiseConvolutionDescriptor(
                kernelWidth: kernelWidth,
                kernelHeight: kernelHeight,
                inputFeatureChannels: outputChannels,
                outputFeatureChannels: outputChannels)
            self.desc.groups = 1
        } else {
            self.desc = MPSCNNConvolutionDescriptor(
                kernelWidth: kernelWidth,
                kernelHeight: kernelHeight,
                inputFeatureChannels: inputChannels,
                outputFeatureChannels: outputChannels)
            self.desc.groups = groups
        }

        self.desc.dilationRateY = dilations.0
        self.desc.dilationRateX = dilations.1
        self.desc.strideInPixelsY = strides.0
        self.desc.strideInPixelsX = strides.1

        switch Int(weight.dataType) {
        case Onnx_TensorProto.DataType.float.rawValue:
            self.weight = .float32(weight.floats)
        case Onnx_TensorProto.DataType.float16.rawValue:
            self.weight = .float16(weight.rawData.withUnsafeBytes {
                [UInt16](
                    UnsafeBufferPointer<UInt16>(
                        start: $0,
                        count: outputChannels * inputChannels * kernelHeight * kernelWidth
                    )
                )
            })
        default:
            self.weight = .invalid
        }

        if !isONNX2MPS {
            switch self.weight {
            case let .float32(array):
                self.weight = .float32(array.reformatConvWeight(outputChannels: outputChannels,
                                                                inputChannels: inputChannels,
                                                                kernelHeight: kernelHeight,
                                                                kernelWidth: kernelWidth,
                                                                isTranspose: isTranspose))
            case let .float16(array):
                self.weight = .float16(array.reformatConvWeight(outputChannels: outputChannels,
                                                                inputChannels: inputChannels,
                                                                kernelHeight: kernelHeight,
                                                                kernelWidth: kernelWidth,
                                                                isTranspose: isTranspose))
            case .invalid:
                break
            }
        }

        if let bias = bias {
            switch Int(bias.dataType) {
            case Onnx_TensorProto.DataType.float.rawValue:
                self.bias = bias.rawData.withUnsafeBytes { [Float](UnsafeBufferPointer<Float>(start: $0, count: outputChannels)) }
            case Onnx_TensorProto.DataType.float16.rawValue:
                self.bias = bias.rawData.withUnsafeBytes { float16to32(UnsafeMutableRawPointer(mutating: $0), count: outputChannels) }
            default:
                break
            }
        }

        self.outputChannels = outputChannels
    }

    func dataType() -> MPSDataType {
        switch self.weight {
        case .float32:
            return .float32
        case .float16:
            return .float16
        case .invalid:
            return .invalid
        }
    }

    func descriptor() -> MPSCNNConvolutionDescriptor {
        return self.desc
    }

    func weights() -> UnsafeMutableRawPointer {
        switch self.weight {
        case let .float32(array):
            return UnsafeMutableRawPointer(mutating: array)
        case let .float16(array):
            return UnsafeMutableRawPointer(mutating: array)
        case .invalid:
            return UnsafeMutableRawPointer(mutating: [])
        }
    }

    func biasTerms() -> UnsafeMutablePointer<Float>? {
        guard self.bias != nil else {
            return nil
        }

        return UnsafeMutablePointer(mutating: self.bias)
    }

    func load() -> Bool {
        return true
    }

    func purge() {
        //no-op
    }

    func label() -> String? {
        return nil
    }
}

final class ConvolutionConverter: NodeConverter {
    func convert(in graph: ONNXGraph, node: Onnx_NodeProto) throws {
        guard
            let input = graph.output(name: node.input[0]),
            let inputShape = graph.shape(output: node.input[0])
        else { throw ONNXGraph.Errors.noSuchOutput }

        guard
            let weight = graph.tensor(name: node.input[1])
        else { throw ONNXGraph.Errors.insufficientInputs }

        var bias: Onnx_TensorProto?
        if node.input.count > 2 {
            bias = graph.tensor(name: node.input[2])
        }

        var kernel: Kernel?
        var dilations: Dilations?
        var strides: Strides?
        var groups: Int = 1
        var pads: Pads = (0, 0, 0, 0)
        var outputPadding = Padding(height: 0, width: 0)

        for attr in node.attribute {
            switch attr.name {
            case "dilations":
                dilations = (Int(attr.ints[0]), Int(attr.ints[1]))
            case "strides":
                strides = (Int(attr.ints[0]), Int(attr.ints[1]))
            case "group":
                groups = Int(attr.i)
            case "pads":
                pads = (Int(attr.ints[0]), Int(attr.ints[1]), Int(attr.ints[2]), Int(attr.ints[3]))
            case "kernel_shape":
                kernel = (Int(attr.ints[0]), Int(attr.ints[1]))
            case "output_padding":
                outputPadding = (Int(attr.ints[0]), Int(attr.ints[1]))
            default:
                break
            }
        }

        guard
            let k = kernel,
            let d = dilations,
            let s = strides
        else { throw ONNXGraph.Errors.notEnoughAttributes }

        var conv: MPSCNNConvolutionNode!
        var convDataSource: ConvDataSource!
        switch node.opType {
        case "Conv":
            convDataSource = ConvDataSource(weight: weight,
                                            bias: bias,
                                            dilations: d,
                                            strides: s,
                                            groups: groups,
                                            isTranspose: false,
                                            isONNX2MPS: graph.modelFormat == .mpsFlavor)
            conv = MPSCNNConvolutionNode(source: input,
                                         weights: convDataSource)
            conv.paddingPolicy = ONNXConvolutionPadding(kernel: k,
                                                        strides: s,
                                                        dilations: d,
                                                        pads: pads,
                                                        outputPadding: outputPadding,
                                                        isTranspose: false)
        case "ConvTranspose":
            convDataSource = ConvDataSource(weight: weight,
                                            bias: bias,
                                            dilations: d,
                                            strides: s,
                                            groups: groups,
                                            isTranspose: true,
                                            isONNX2MPS: graph.modelFormat == .mpsFlavor)
            conv = MPSCNNConvolutionTransposeNode(source: input,
                                                  weights: convDataSource)
            conv.paddingPolicy = ONNXConvolutionPadding(kernel: k,
                                                        strides: s,
                                                        dilations: d,
                                                        pads: pads,
                                                        outputPadding: outputPadding,
                                                        isTranspose: true)
        default:
            throw ONNXGraph.Errors.inconsistentState
        }
        if #available(iOS 12.0, OSX 10.14, *) {
            conv.accumulatorPrecision = weight.dataType == Onnx_TensorProto.DataType.float16.rawValue ? .half : .float
        }

        let paddingPolicy = (conv.paddingPolicy as! ONNXConvolutionPadding)
        let paddedSize = paddingPolicy.paddedSize(inputWidth: inputShape.width,
                                                  inputHeight: inputShape.height)
        let outputShape = Shape(channels: 1,
                                width: paddedSize.width,
                                height: paddedSize.height,
                                depth: convDataSource.outputChannels)
        
        graph.addFilter(conv, outputShape: outputShape, withOutputs: node.output)
    }
}

// MARK: Activations

final class ReluConverter: NodeConverter {
    func convert(in graph: ONNXGraph, node: Onnx_NodeProto) throws {
        guard
            let input = graph.output(name: node.input[0]),
            let inputShape = graph.shape(output: node.input[0])
        else { throw ONNXGraph.Errors.noSuchOutput }
        
        let relu = MPSCNNNeuronReLUNode(source: input)
        graph.addFilter(relu, outputShape: inputShape, withOutputs: node.output)
    }
}

final class EluConverter: NodeConverter {
    func convert(in graph: ONNXGraph, node: Onnx_NodeProto) throws {
        guard
            let input = graph.output(name: node.input[0]),
            let inputShape = graph.shape(output: node.input[0])
        else { throw ONNXGraph.Errors.noSuchOutput }

        let attr = node.attribute[0]
        #if DEBUG
            precondition(attr.name == "alpha")
        #endif

        let alpha = attr.f

        let elu = MPSCNNNeuronELUNode(source: input, a: alpha)
        graph.addFilter(elu, outputShape: inputShape, withOutputs: node.output)
    }
}

final class ExpConverter: NodeConverter {
    func convert(in graph: ONNXGraph, node: Onnx_NodeProto) throws {
        guard
            let input = graph.output(name: node.input[0]),
            let inputShape = graph.shape(output: node.input[0])
        else { throw ONNXGraph.Errors.noSuchOutput }

        if #available(iOS 11.3, macOS 10.13.4, *) {
            let sigmoid = MPSCNNNeuronExponentialNode(source: input)

            graph.addFilter(sigmoid, outputShape: inputShape, withOutputs: node.output)
        } else {
            throw ONNXGraph.Errors.unknownNodeOpType(opType: "Exp")
        }
    }
}

final class AddConverter: NodeConverter {
    func convert(in graph: ONNXGraph, node: Onnx_NodeProto) throws {
        guard
            let input1 = graph.output(name: node.input[0]),
            let input2 = graph.output(name: node.input[1]),
            let inputShape = graph.shape(output: node.input[0]),
            node.input.count == 2
        else { throw ONNXGraph.Errors.noSuchOutput }

        let add = MPSNNAdditionNode(leftSource: input1, rightSource: input2)
        graph.addFilter(add, outputShape: inputShape, withOutputs: node.output)
    }
}

@available(iOS 11.0, macOS 10.13.0, tvOS 11.0, *)
final class SubConverter: NodeConverter {
    func convert(in graph: ONNXGraph, node: Onnx_NodeProto) throws {
        guard
            let input1 = graph.output(name: node.input[0]),
            let input2 = graph.output(name: node.input[1]),
            let inputShape = graph.shape(output: node.input[0]),
            node.input.count == 2
        else { throw ONNXGraph.Errors.noSuchOutput }

        let sub = MPSNNSubtractionNode(leftSource: input1, rightSource: input2)
        sub.label = "Sub"
        graph.addFilter(sub, outputShape: inputShape, withOutputs: node.output)
    }
}

final class MulConverter: NodeConverter {
    func convert(in graph: ONNXGraph, node: Onnx_NodeProto) throws {
        guard
            let input1 = graph.output(name: node.input[0]),
            let input2 = graph.output(name: node.input[1]),
            let inputShape = graph.shape(output: node.input[0]),
            node.input.count == 2
        else { throw ONNXGraph.Errors.noSuchOutput }

        let add = MPSNNMultiplicationNode(leftSource: input1, rightSource: input2)
        graph.addFilter(add, outputShape: inputShape, withOutputs: node.output)
    }
}

@available(iOS 11.0, macOS 10.13.0, tvOS 11.0, *)
final class DivConverter: NodeConverter {
    func convert(in graph: ONNXGraph, node: Onnx_NodeProto) throws {
        guard
            let input1 = graph.output(name: node.input[0]),
            let input2 = graph.output(name: node.input[1]),
            let inputShape = graph.shape(output: node.input[0]),
            node.input.count == 2
        else { throw ONNXGraph.Errors.noSuchOutput }

        let div = MPSNNDivisionNode(leftSource: input1, rightSource: input2)
        div.label = "Div"
        graph.addFilter(div, outputShape: inputShape, withOutputs: node.output)
    }
}

final class SigmoidConverter: NodeConverter {
    func convert(in graph: ONNXGraph, node: Onnx_NodeProto) throws {
        guard
            let input = graph.output(name: node.input[0]),
            let inputShape = graph.shape(output: node.input[0])
        else { throw ONNXGraph.Errors.noSuchOutput }

        let sigmoid = MPSCNNNeuronSigmoidNode(source: input)
        graph.addFilter(sigmoid, outputShape: inputShape, withOutputs: node.output)
    }
}


class UpsampleConverter: NodeConverter {

    let alignCorners: Bool

    init(alignCorners: Bool) {
        self.alignCorners = alignCorners
    }

    func convert(in graph: ONNXGraph, node: Onnx_NodeProto) throws {
        guard
            let input = graph.output(name: node.input[0]),
            let inputShape = graph.shape(output: node.input[0])
        else { throw ONNXGraph.Errors.noSuchOutput }

        var mode: String?
        var scales: Scales?

        for attr in node.attribute {
            switch attr.name {
            case "mode":
                mode = String(data: attr.s, encoding: .utf8)
            case "scales":
                scales = (Int(attr.floats[2]), Int(attr.floats[3]))
            default:
                break
            }
        }

        if scales == nil {
            guard
                let scales_tensor = graph.tensor(name: node.input[1])
            else { throw ONNXGraph.Errors.notEnoughAttributes }
            let scales_count = Int(scales_tensor.dims[0])

            #if DEBUG
            precondition(scales_count == 4)
            #endif

            let scales_values = scales_tensor.integers
            scales = (scales_values[2], scales_values[3])
        }

        guard let _ = mode, let s = scales else {
            throw ONNXGraph.Errors.notEnoughAttributes
        }

        var upsample: MPSNNFilterNode!

        switch mode {
        case "nearest":
            upsample = MPSCNNUpsamplingNearestNode(source: input,
                                                   integerScaleFactorX: s.width,
                                                   integerScaleFactorY: s.height)
        case "bilinear", "linear":
            upsample = MPSCNNUpsamplingBilinearNode(source: input,
                                                    integerScaleFactorX: s.width,
                                                    integerScaleFactorY: s.height,
                                                    alignCorners: self.alignCorners)
        default:
            throw ONNXGraph.Errors.unknownNodeOpType(opType: node.opType)
        }

        let outputShape = (inputShape.channels,
                           s.width,
                           s.height,
                           inputShape.depth)
        graph.addFilter(upsample, outputShape: outputShape, withOutputs: node.output)
    }
}

final class ConcatConverter: NodeConverter {
    func convert(in graph: ONNXGraph, node: Onnx_NodeProto) throws {
        let inputs = try (0..<node.input.count).map( { (idx: Int) -> MPSNNImageNode in
            guard
                let input = graph.output(name: node.input[idx])
            else { throw ONNXGraph.Errors.noSuchOutput }
            return input
        })

        guard
            let inputShape = graph.shape(output: node.input[0])
        else { throw ONNXGraph.Errors.noSuchOutput }

        let concat = MPSNNConcatenationNode(sources: inputs)
        var outputShape = inputShape
        outputShape.depth *= 2
        graph.addFilter(concat, outputShape: outputShape, withOutputs: node.output)
    }
}

// MARK: Poolings

final class GlobalAveragePoolConverter: NodeConverter {
    func convert(in graph: ONNXGraph, node: Onnx_NodeProto) throws {
        guard
            let input = graph.output(name: node.input[0]),
            let inputShape = graph.shape(output: node.input[0])
        else { throw ONNXGraph.Errors.noSuchOutput }

        let avgPool = MPSCNNPoolingAverageNode(source: input,
                                               kernelWidth: inputShape.width,
                                               kernelHeight: inputShape.height,
                                               strideInPixelsX: 1,
                                               strideInPixelsY: 1)
        avgPool.paddingPolicy = GlobalPoolPadding()

        let outputShape = Shape(inputShape.channels, 1, 1, inputShape.depth)
        graph.addFilter(avgPool, outputShape: outputShape, withOutputs: node.output)
    }
}

final class AveragePoolConverter: NodeConverter {
    func convert(in graph: ONNXGraph, node: Onnx_NodeProto) throws {
        guard
            let input = graph.output(name: node.input[0]),
            let inputShape = graph.shape(output: node.input[0])
        else { throw ONNXGraph.Errors.noSuchOutput }

        let kernel_shape = node.attribute[0]
        let pads = node.attribute[1]
        let strides = node.attribute[2]

        #if DEBUG
        precondition(kernel_shape.name == "kernel_shape")
        precondition(pads.name == "pads")
        precondition(strides.name == "strides")
        #endif

        let avgPool = MPSCNNPoolingAverageNode(source: input,
                                               kernelWidth: Int(kernel_shape.ints[1]),
                                               kernelHeight: Int(kernel_shape.ints[0]),
                                               strideInPixelsX: Int(strides.ints[1]),
                                               strideInPixelsY: Int(strides.ints[0]))
        let paddingPolicy = PyTorchPoolPadding(kernelWidth: Int(kernel_shape.ints[1]),
                                               kernelHeight: Int(kernel_shape.ints[0]),
                                               paddingWidth: Int(pads.ints[1]),
                                               paddingHeight: Int(pads.ints[0]),
                                               strideInPixelsX: Int(strides.ints[1]),
                                               strideInPixelsY: Int(strides.ints[0]))
        avgPool.paddingPolicy = paddingPolicy

        let paddedSize = paddingPolicy.paddedSize(inputWidth: inputShape.width,
                                                  inputHeight: inputShape.height)
        let outputShape = (inputShape.channels,
                           paddedSize.width,
                           paddedSize.height,
                           inputShape.depth)
        graph.addFilter(avgPool, outputShape: outputShape, withOutputs: node.output)
    }
}

final class MaxPoolConverter: NodeConverter {
    func convert(in graph: ONNXGraph, node: Onnx_NodeProto) throws {
        guard
            let input = graph.output(name: node.input[0]),
            let inputShape = graph.shape(output: node.input[0])
        else { throw ONNXGraph.Errors.noSuchOutput }

        let kernel_shape = node.attribute[0]
        let pads = node.attribute[1]
        let strides = node.attribute[2]

        #if DEBUG
        precondition(kernel_shape.name == "kernel_shape")
        precondition(pads.name == "pads")
        precondition(strides.name == "strides")
        #endif

        let maxPool = MPSCNNPoolingMaxNode(source: input,
                                           kernelWidth: Int(kernel_shape.ints[1]),
                                           kernelHeight: Int(kernel_shape.ints[0]),
                                           strideInPixelsX: Int(strides.ints[1]),
                                           strideInPixelsY: Int(strides.ints[0]))
        let paddingPolicy = PyTorchPoolPadding(kernelWidth: Int(kernel_shape.ints[1]),
                                               kernelHeight: Int(kernel_shape.ints[0]),
                                               paddingWidth: Int(pads.ints[1]),
                                               paddingHeight: Int(pads.ints[0]),
                                               strideInPixelsX: Int(strides.ints[1]),
                                               strideInPixelsY: Int(strides.ints[0]))
        maxPool.paddingPolicy = paddingPolicy

        let paddedSize = paddingPolicy.paddedSize(inputWidth: inputShape.width,
                                                  inputHeight: inputShape.height)
        let outputShape = (inputShape.channels,
                           paddedSize.width,
                           paddedSize.height,
                           inputShape.depth)

        graph.addFilter(maxPool, outputShape: outputShape, withOutputs: node.output)
    }
}


final class SoftmaxConverter: NodeConverter {
    func convert(in graph: ONNXGraph, node: Onnx_NodeProto) throws {
        guard
            let input = graph.output(name: node.input[0]),
            let inputShape = graph.shape(output: node.input[0])
        else { throw ONNXGraph.Errors.noSuchOutput }
        let axis = node.attribute[0]

        #if DEBUG
        precondition(axis.name == "axis")
        precondition(axis.i == 1)
        #endif

        let softmax = MPSCNNSoftMaxNode(source: input)
        graph.addFilter(softmax, outputShape: inputShape, withOutputs: node.output)
    }
}


final class ConstantConverter: NodeConverter {
    func convert(in graph: ONNXGraph, node: Onnx_NodeProto) throws {
        let name = node.output[0]
        let value = node.attribute[0]

        #if DEBUG
        precondition(value.name == "value")
        #endif

        graph.initTensor(name, data: value.t)
    }
}

@objc private final class BNDataSource: NSObject, MPSCNNBatchNormalizationDataSource {
    required init?(coder aDecoder: NSCoder) {
        guard let data = aDecoder.decodeData(),
            let other = NSKeyedUnarchiver.unarchiveObject(with: data) as? BNDataSource else {
                return nil
        }
        self.nFeatureChannels = other.nFeatureChannels
    }

    private var meanW: [Float] = []
    private var varianceW: [Float] = []
    private var gammaW: [Float] = []
    private var betaW: [Float] = []
    private let nFeatureChannels: Int

    func mean() -> UnsafeMutablePointer<Float>? {
        return UnsafeMutablePointer(mutating: self.meanW)
    }
    func variance() -> UnsafeMutablePointer<Float>? {
        return UnsafeMutablePointer(mutating: self.varianceW)
    }
    func gamma() -> UnsafeMutablePointer<Float>? {
        return UnsafeMutablePointer(mutating: self.gammaW)
    }
    func beta() -> UnsafeMutablePointer<Float>? {
        return UnsafeMutablePointer(mutating: self.betaW)
    }

    init(mean: Onnx_TensorProto, variance: Onnx_TensorProto, gamma: Onnx_TensorProto, beta: Onnx_TensorProto) {
        let nFeatureChannels = Int(mean.dims[0])
        self.nFeatureChannels = nFeatureChannels

        self.meanW = mean.floats
        self.varianceW = variance.floats
        self.gammaW = gamma.floats
        self.betaW = beta.floats
    }

    func numberOfFeatureChannels() -> Int {
        return self.nFeatureChannels
    }

    func load() -> Bool {
        return true
    }

    func purge() {
        // no-op
    }

    func label() -> String? {
        return nil
    }

    func copy(with zone: NSZone? = nil) -> Any {
        return self.mutableCopy()
    }
}

@available(iOS 11.3, tvOS 11.3, macOS 10.13.4, *)
final class BatchNormalizationConverter:NodeConverter {
    func convert(in graph: ONNXGraph, node: Onnx_NodeProto) throws {
        guard
            let input = graph.output(name: node.input[0]),
            let inputShape = graph.shape(output: node.input[0])
        else { throw ONNXGraph.Errors.noSuchOutput }

        guard
            let gamma = graph.tensor(name: node.input[1]),
            let beta = graph.tensor(name: node.input[2]),
            let mean = graph.tensor(name: node.input[3]),
            let variance = graph.tensor(name: node.input[4])
        else { throw ONNXGraph.Errors.insufficientInputs }

        let dataSource = BNDataSource(mean: mean,
                                      variance: variance,
                                      gamma: gamma,
                                      beta: beta)
        let batchNormalization = MPSCNNBatchNormalizationNode(source: input,
                                                              dataSource: dataSource)
        graph.addFilter(batchNormalization, outputShape: inputShape, withOutputs: node.output)
    }
}

@available(iOS 12.1, tvOS 12.1, macOS 10.14.1, *)
final class ReshapeConverter: NodeConverter {

    func convert(in graph: ONNXGraph, node: Onnx_NodeProto) throws {
        guard
            let input = graph.output(name: node.input[0]),
            let inputShape = graph.shape(output: node.input[0]),
            let shapeTensor = graph.tensor(name: node.input[1])
        else { throw ONNXGraph.Errors.noSuchOutput }

        let shape = shapeTensor.integers
        let totalDims = inputShape.channels * inputShape.depth * inputShape.height * inputShape.width

        var normalizedDims = (0...2).map { Int(shape[safe: $0] ?? 1) }

        normalizedDims = (0...2).map {
            let dim = normalizedDims[$0]
            return dim == 0 ? self.dim(of: inputShape, at: $0) : dim
        }

        if let inferredIndex = normalizedDims.index(of: -1) {
            let inferredDim = (0...2).reduce(totalDims) {
                $1 == inferredIndex ? $0 : $0 / Int(normalizedDims[$1])
            }

            normalizedDims[inferredIndex] = inferredDim
        }

        let reshape = MPSNNReshapeNode(source: input,
                                       resultWidth: normalizedDims[0],
                                       resultHeight: normalizedDims[1],
                                       resultFeatureChannels: normalizedDims[2])

        let outputShape = (inputShape.channels,
                           normalizedDims[0],
                           normalizedDims[1],
                           normalizedDims[2])
        graph.addFilter(reshape, outputShape: outputShape, withOutputs: node.output)
    }

    private func dim(of shape: Shape, at index: Int) -> Int {
        switch index {
        case 0:
            return shape.0
        case 1:
            return shape.1
        case 2:
            return shape.2
        case 3:
            return shape.3
        default: return 1
        }
    }

}

@available(iOS 11.3, tvOS 11.3, macOS 10.13.4, *)
final class DropoutConverter: NodeConverter {
    func convert(in graph: ONNXGraph, node: Onnx_NodeProto) throws {
        guard
            let input = graph.output(name: node.input[0]),
            let inputShape = graph.shape(output: node.input[0]),
            node.attribute.count > 0
        else { throw ONNXGraph.Errors.noSuchOutput }

        let ratioAttribuge = node.attribute[0]

        #if DEBUG
        precondition(ratioAttribuge.name == "ratio")
        #endif

        let ratio = ratioAttribuge.f

        let dropout = MPSCNNDropoutNode(source: input, keepProbability: ratio)
        dropout.label = "Dropout \(ratio)"

        graph.addFilter(dropout, outputShape: inputShape, withOutputs: node.output)

    }
}

@available(iOS 12.1, tvOS 12.1, macOS 10.14.1, *)
class PaddingConverter: NodeConverter {
    func convert(in graph: ONNXGraph, node: Onnx_NodeProto) throws {
        guard
            let input = graph.output(name: node.input[0]),
            let inputShape = graph.shape(output: node.input[0]),
            let pads = node.attribute.first { $0.name == "pads" }?.ints.map(Int.init)
        else { throw ONNXGraph.Errors.noSuchOutput }
        
        // can be constant(default), reflect, edge
        var mode = "constant"
        if let modeAttribute = node.attribute.first { $0.name == "mode" }?.s {
            mode = String(data: modeAttribute, encoding: .utf8) ?? mode
        }
        
        let edgeMode: MPSImageEdgeMode
        
        switch mode {
        case "constant":
            edgeMode = .constant
        case "reflect":
            edgeMode = .mirror
        case "edge":
            edgeMode = .clamp
        default:
            throw ONNXGraph.Errors.inconsistentState
        }
        
        let pad = MPSNNPadNode(source: input,
                               paddingSizeBefore: .init(x: pads[3], y: pads[2], channel: pads[1]),
                               paddingSizeAfter: .init(x: pads[7], y: pads[6], channel: pads[5]),
                               edgeMode: edgeMode)
        let outputShape: Shape = (inputShape.channels + pads[1] + pads[5],
                                  inputShape.width + pads[3] + pads[7],
                                  inputShape.height + pads[2] + pads[6],
                                  inputShape.depth)
        graph.addFilter(pad, outputShape: outputShape, withOutputs: node.output)
    }
}

class InstanceNormConverter: NodeConverter {
    func convert(in graph: ONNXGraph, node: Onnx_NodeProto) throws {
        guard
            let input = graph.output(name: node.input[0]),
            let inputShape = graph.shape(output: node.input[0]),
            let gamma = graph.tensor(name: node.input[1])?.floats,
            let beta = graph.tensor(name: node.input[2])?.floats
        else { throw ONNXGraph.Errors.noSuchOutput }

        let dataSource = InstanceNormDataSource(channels: gamma.count,
                                                gammas: gamma,
                                                betas: beta)
        
        let instanceNorm = MPSCNNInstanceNormalizationNode(source: input,
                                                           dataSource: dataSource)
        graph.addFilter(instanceNorm,
                        outputShape: inputShape,
                        withOutputs: node.output)
    }
    
}

@objc class InstanceNormDataSource: NSObject, MPSCNNInstanceNormalizationDataSource {
    
    let numberOfFeatureChannels: Int
    private(set) var gammas: [Float]
    private(set) var betas: [Float]
    
    public init(channels: Int, gammas: [Float], betas: [Float]) {
        self.numberOfFeatureChannels = channels
        self.gammas = gammas
        self.betas = betas
    }
    
    func gamma() -> UnsafeMutablePointer<Float>? {
        return UnsafeMutablePointer(mutating: self.gammas)
    }

    func beta() -> UnsafeMutablePointer<Float>? {
        return UnsafeMutablePointer(mutating: self.betas)
    }

    func label() -> String? {
        return "Instance norm data source"
    }

    required init?(coder aDecoder: NSCoder) {
        fatalError()
    }

    func copy(with zone: NSZone? = nil) -> Any {
        return self.mutableCopy()
    }

}
