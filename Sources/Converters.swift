//
//  DefaultConverters.swift
//  Smelter
//
//  Created by Andrey Volodin on 16/04/2019.
//

import Alloy
import MetalPerformanceShaders

import Accelerate

enum ConvWeightArray {
    case float32([Float])
    case float16([Float16])
    case invalid
}

// MARK: - Constant

class ConstantConverter: NodeConverter {
    func convert(in graph: ONNXGraph, node: Onnx_NodeProto) throws {
        let name = node.output[0]
        let value = node.attribute[0]

        #if DEBUG
        precondition(value.name == "value")
        #endif

        graph.initTensor(name, data: value.t)
    }
}

// MARK: - Arithmetic Layer Nodes

@available(iOS 11.0, macOS 10.13.0, tvOS 11.0, *)
class AddConverter: NodeConverter {
    func convert(in graph: ONNXGraph, node: Onnx_NodeProto) throws {
        guard
            let input1 = graph.output(name: node.input[0]),
            let input2 = graph.output(name: node.input[1]),
            let inputShape = graph.shape(output: node.input[0]),
            node.input.count == 2
        else { throw ONNXGraph.Errors.noSuchOutput }

        let add = MPSNNAdditionNode(leftSource: input1, rightSource: input2)
        add.label = "Add"
        graph.addFilter(add, outputShape: inputShape, withOutputs: node.output)
    }
}

@available(iOS 11.0, macOS 10.13.0, tvOS 11.0, *)
class SubConverter: NodeConverter {
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

@available(iOS 11.0, macOS 10.13.0, tvOS 11.0, *)
class MulConverter: NodeConverter {
    func convert(in graph: ONNXGraph, node: Onnx_NodeProto) throws {
        guard
            let input1 = graph.output(name: node.input[0]),
            let input2 = graph.output(name: node.input[1]),
            let inputShape = graph.shape(output: node.input[0]),
            node.input.count == 2
        else { throw ONNXGraph.Errors.noSuchOutput }

        let add = MPSNNMultiplicationNode(leftSource: input1, rightSource: input2)
        add.label = "Add"
        graph.addFilter(add, outputShape: inputShape, withOutputs: node.output)
    }
}

@available(iOS 11.0, macOS 10.13.0, tvOS 11.0, *)
class DivConverter: NodeConverter {
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

// MARK: - Convolution Layer Nodes

@available(iOS 11.0, macOS 10.13.0, tvOS 11.0, *)
@objc private class ConvDataSource: NSObject, MPSCNNConvolutionDataSource {
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
                self.bias = bias.rawData.withUnsafeBytes {
                    [Float](UnsafeBufferPointer<Float>(start: $0,
                                                       count: outputChannels))
                }
            case Onnx_TensorProto.DataType.float16.rawValue:
                self.bias = bias.rawData.withUnsafeBytes {
                    float16to32(UnsafeMutableRawPointer(mutating: $0),
                                count: outputChannels)
                }
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

@available(iOS 11.0, macOS 10.13.0, tvOS 11.0, *)
class ConvolutionConverter: NodeConverter {
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
        var groups: Int?
        var pads: Pads?
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
            let s = strides,
            let g = groups,
            let p = pads
        else { throw ONNXGraph.Errors.notEnoughAttributes }

        var conv: MPSCNNConvolutionNode!
        var convDataSource: ConvDataSource!
        switch node.opType {
        case "Conv":
            convDataSource = ConvDataSource(weight: weight,
                                            bias: bias,
                                            dilations: d,
                                            strides: s,
                                            groups: g,
                                            isTranspose: false,
                                            isONNX2MPS: graph.modelFormat == .mpsFlavor)
            conv = MPSCNNConvolutionNode(source: input,
                                         weights: convDataSource)
            conv.paddingPolicy = ONNXConvolutionPadding(kernel: k,
                                                        strides: s,
                                                        dilations: d,
                                                        pads: p,
                                                        outputPadding: outputPadding,
                                                        isTranspose: false)
        case "ConvTranspose":
            convDataSource = ConvDataSource(weight: weight,
                                            bias: bias,
                                            dilations: d,
                                            strides: s,
                                            groups: g,
                                            isTranspose: true,
                                            isONNX2MPS: graph.modelFormat == .mpsFlavor)
            conv = MPSCNNConvolutionTransposeNode(source: input,
                                                  weights: convDataSource)
            conv.label = "Conv"
            conv.paddingPolicy = ONNXConvolutionPadding(kernel: k,
                                                        strides: s,
                                                        dilations: d,
                                                        pads: p,
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

// MARK: - Pooling Layer Nodes

@available(iOS 11.0, macOS 10.13.0, tvOS 11.0, *)
class AveragePoolConverter: NodeConverter {
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

        let averagePool = MPSCNNPoolingAverageNode(source: input,
                                                   kernelWidth: Int(kernel_shape.ints[1]),
                                                   kernelHeight: Int(kernel_shape.ints[0]),
                                                   strideInPixelsX: Int(strides.ints[1]),
                                                   strideInPixelsY: Int(strides.ints[0]))
        averagePool.label = "AveragePool"
        let paddingPolicy = PyTorchPoolPadding(kernelWidth: Int(kernel_shape.ints[1]),
                                               kernelHeight: Int(kernel_shape.ints[0]),
                                               paddingWidth: Int(pads.ints[1]),
                                               paddingHeight: Int(pads.ints[0]),
                                               strideInPixelsX: Int(strides.ints[1]),
                                               strideInPixelsY: Int(strides.ints[0]))
        averagePool.paddingPolicy = paddingPolicy

        let paddedSize = paddingPolicy.paddedSize(inputWidth: inputShape.width,
                                                  inputHeight: inputShape.height)
        let outputShape = (inputShape.channels,
                           paddedSize.width,
                           paddedSize.height,
                           inputShape.depth)
        graph.addFilter(averagePool, outputShape: outputShape, withOutputs: node.output)
    }
}

@available(iOS 11.0, macOS 10.13.0, tvOS 11.0, *)
class GlobalAveragePoolConverter: NodeConverter {
    func convert(in graph: ONNXGraph, node: Onnx_NodeProto) throws {
        guard
            let input = graph.output(name: node.input[0]),
            let inputShape = graph.shape(output: node.input[0])
        else { throw ONNXGraph.Errors.noSuchOutput }

        let globalAveragePool = MPSCNNPoolingAverageNode(source: input,
                                                         kernelWidth: inputShape.width,
                                                         kernelHeight: inputShape.height,
                                                         strideInPixelsX: 1,
                                                         strideInPixelsY: 1)
        globalAveragePool.label = "GlobalAveragePool"
        globalAveragePool.paddingPolicy = GlobalPoolPadding()

        let outputShape = Shape(inputShape.channels, 1, 1, inputShape.depth)
        graph.addFilter(globalAveragePool, outputShape: outputShape, withOutputs: node.output)
    }
}

@available(iOS 11.0, macOS 10.13.0, tvOS 11.0, *)
class MaxPoolConverter: NodeConverter {
    func convert(in graph: ONNXGraph, node: Onnx_NodeProto) throws {
        guard
            let input = graph.output(name: node.input[0]),
            let inputShape = graph.shape(output: node.input[0])
        else { throw ONNXGraph.Errors.noSuchOutput }

        let maxPool = MPSCNNPoolingMaxNode(source: input,
                                           kernelWidth: inputShape.width,
                                           kernelHeight: inputShape.height,
                                           strideInPixelsX: 1,
                                           strideInPixelsY: 1)
        maxPool.label = "MaxPool"
        maxPool.paddingPolicy = GlobalPoolPadding()

        let outputShape = Shape(inputShape.channels, 1, 1, inputShape.depth)
        graph.addFilter(maxPool, outputShape: outputShape, withOutputs: node.output)
    }
}

// MARK: - Normalization Layer Nodes

@available(iOS 11.3, macOS 10.13.4, tvOS 11.3, *)
@objc private class BNDataSource: NSObject, MPSCNNBatchNormalizationDataSource {
    required init?(coder aDecoder: NSCoder) {
        guard
            let data = aDecoder.decodeData(),
            let other = NSKeyedUnarchiver.unarchiveObject(with: data) as? BNDataSource
        else { return nil }
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

@available(iOS 11.3, macOS 10.13.4, tvOS 11.3, *)
class BatchNormalizationConverter: NodeConverter {
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
        batchNormalization.label = "BatchNormalization"
        graph.addFilter(batchNormalization, outputShape: inputShape, withOutputs: node.output)
    }
}

// MARK: - Neuron Layer Nodes

@available(iOS 11.0, macOS 10.13.0, tvOS 11.0, *)
class AbsConverter: NodeConverter {
    func convert(in graph: ONNXGraph, node: Onnx_NodeProto) throws {
        guard
            let input = graph.output(name: node.input[0]),
            let inputShape = graph.shape(output: node.input[0])
        else { throw ONNXGraph.Errors.noSuchOutput }

        let abs = MPSCNNNeuronAbsoluteNode(source: input)
        abs.label = "Abs"
        graph.addFilter(abs, outputShape: inputShape, withOutputs: node.output)
    }
}

@available(iOS 11.0, macOS 10.13.0, tvOS 11.0, *)
class EluConverter: NodeConverter {
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
        elu.label = "Elu"
        graph.addFilter(elu, outputShape: inputShape, withOutputs: node.output)
    }
}

@available(iOS 11.0, macOS 10.13.0, tvOS 11.0, *)
class ReluConverter: NodeConverter {
    func convert(in graph: ONNXGraph, node: Onnx_NodeProto) throws {
        guard
            let input = graph.output(name: node.input[0]),
            let inputShape = graph.shape(output: node.input[0])
        else { throw ONNXGraph.Errors.noSuchOutput }
        
        let relu = MPSCNNNeuronReLUNode(source: input)
        relu.label = "Relu"
        graph.addFilter(relu, outputShape: inputShape, withOutputs: node.output)
    }
}

@available(iOS 11.0, macOS 10.13.0, tvOS 11.0, *)
class SigmoidConverter: NodeConverter {
    func convert(in graph: ONNXGraph, node: Onnx_NodeProto) throws {
        guard
            let input = graph.output(name: node.input[0]),
            let inputShape = graph.shape(output: node.input[0])
        else { throw ONNXGraph.Errors.noSuchOutput }

        let sigmoid = MPSCNNNeuronSigmoidNode(source: input)
        sigmoid.label = "Sigmoid"
        graph.addFilter(sigmoid, outputShape: inputShape, withOutputs: node.output)
    }
}

@available(iOS 11.0, macOS 10.13.0, tvOS 11.0, *)
class HardSigmoidConverter: NodeConverter {
    func convert(in graph: ONNXGraph, node: Onnx_NodeProto) throws {
        guard
            let input = graph.output(name: node.input[0]),
            let inputShape = graph.shape(output: node.input[0])
        else { throw ONNXGraph.Errors.noSuchOutput }

        let hardSigmoid = MPSCNNNeuronHardSigmoidNode(source: input)
        hardSigmoid.label = "HardSigmoid"
        graph.addFilter(hardSigmoid, outputShape: inputShape, withOutputs: node.output)
    }
}

@available(iOS 11.0, macOS 10.13.0, tvOS 11.0, *)
class SoftplusConverter: NodeConverter {
    func convert(in graph: ONNXGraph, node: Onnx_NodeProto) throws {
        guard
            let input = graph.output(name: node.input[0]),
            let inputShape = graph.shape(output: node.input[0])
        else { throw ONNXGraph.Errors.noSuchOutput }


        let softplus = MPSCNNNeuronSoftPlusNode(source: input)
        softplus.label = "Softplus"
        graph.addFilter(softplus, outputShape: inputShape, withOutputs: node.output)
    }
}

@available(iOS 11.0, macOS 10.13.0, tvOS 11.0, *)
class SoftsignConverter: NodeConverter {
    func convert(in graph: ONNXGraph, node: Onnx_NodeProto) throws {
        guard
            let input = graph.output(name: node.input[0]),
            let inputShape = graph.shape(output: node.input[0])
        else { throw ONNXGraph.Errors.noSuchOutput }

        let softsign = MPSCNNNeuronSoftSignNode(source: input)
        softsign.label = "Softsign"
        graph.addFilter(softsign, outputShape: inputShape, withOutputs: node.output)
    }
}

@available(iOS 11.0, macOS 10.13.0, tvOS 11.0, *)
class TanhConverter: NodeConverter {
    func convert(in graph: ONNXGraph, node: Onnx_NodeProto) throws {
        guard
            let input = graph.output(name: node.input[0]),
            let inputShape = graph.shape(output: node.input[0])
        else { throw ONNXGraph.Errors.noSuchOutput }

        let tanh = MPSCNNNeuronTanHNode(source: input)
        tanh.label = "Tanh"
        graph.addFilter(tanh, outputShape: inputShape, withOutputs: node.output)
    }
}

@available(iOS 11.3, macOS 10.13.4, tvOS 11.3, *)
class ExpConverter: NodeConverter {
    func convert(in graph: ONNXGraph, node: Onnx_NodeProto) throws {
        guard
            let input = graph.output(name: node.input[0]),
            let inputShape = graph.shape(output: node.input[0])
        else { throw ONNXGraph.Errors.noSuchOutput }

        let exp = MPSCNNNeuronExponentialNode(source: input)
        exp.label = "Exp"
        graph.addFilter(exp, outputShape: inputShape, withOutputs: node.output)
    }
}

@available(iOS 11.3, macOS 10.13.4, tvOS 11.3, *)
class LogConverter: NodeConverter {
    func convert(in graph: ONNXGraph, node: Onnx_NodeProto) throws {
        guard
            let input = graph.output(name: node.input[0]),
            let inputShape = graph.shape(output: node.input[0])
        else { throw ONNXGraph.Errors.noSuchOutput }

        let log = MPSCNNNeuronLogarithmNode(source: input)
        log.label = "Log"
        graph.addFilter(log, outputShape: inputShape, withOutputs: node.output)
    }
}

@available(iOS 11.3, macOS 10.13.4, tvOS 11.3, *)
class PowConverter: NodeConverter {
    func convert(in graph: ONNXGraph, node: Onnx_NodeProto) throws {
        guard
            let input = graph.output(name: node.input[0]),
            let inputShape = graph.shape(output: node.input[0])
        else { throw ONNXGraph.Errors.noSuchOutput }

        let pow = MPSCNNNeuronPowerNode(source: input)
        pow.label = "Pow"
        graph.addFilter(pow, outputShape: inputShape, withOutputs: node.output)
    }
}

// MARK: - Softmax Layer Nodes

@available(iOS 11.0, macOS 10.13.0, tvOS 11.0, *)
class SoftmaxConverter: NodeConverter {
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
        softmax.label = "Softmax"
        graph.addFilter(softmax, outputShape: inputShape, withOutputs: node.output)
    }
}

@available(iOS 11.0, macOS 10.13.0, tvOS 11.0, *)
class LogSoftmaxConverter: NodeConverter {
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

        let logSoftmax = MPSCNNLogSoftMaxNode(source: input)
        logSoftmax.label = "LogSoftmax"
        graph.addFilter(logSoftmax, outputShape: inputShape, withOutputs: node.output)
    }
}

// MARK: - Upsampling Layer Nodes

@available(iOS 11.0, macOS 10.13.0, tvOS 11.0, *)
class UpsampleConverter: NodeConverter {
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
            upsample = MPSCNNUpsamplingNearestNode(
                source: input, integerScaleFactorX: s.width, integerScaleFactorY: s.height)
        case "bilinear", "linear":
            upsample = MPSCNNUpsamplingBilinearNode(
                source: input, integerScaleFactorX: s.width, integerScaleFactorY: s.height)
        default:
            throw ONNXGraph.Errors.unknownNodeOpType(opType: node.opType)
        }
        upsample.label = "Upsample"

        let outputShape = (inputShape.channels,
                           s.width,
                           s.height,
                           inputShape.depth)
        graph.addFilter(upsample, outputShape: outputShape, withOutputs: node.output)
    }
}

// MARK: - Dropout Layer Nodes

@available(iOS 11.3, macOS 10.13.4, tvOS 11.3, *)
class DropoutConverter: NodeConverter {
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

// MARK: - Kernel Concatenation Nodes

@available(iOS 11.0, macOS 10.13.0, tvOS 11.0, *)
class ConcatConverter: NodeConverter {
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
        concat.label = "Concat"
        var outputShape = inputShape
        outputShape.depth *= 2
        graph.addFilter(concat, outputShape: outputShape, withOutputs: node.output)
    }
}

// MARK: - Reshape Nodes

@available(iOS 12.1, macOS 10.14.1, tvOS 12.1, *)
class ReshapeConverter: NodeConverter {
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
        reshape.label = "Reshape"

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
