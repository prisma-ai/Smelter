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

class ConstantConverter: NodeConverter {
    func convert(in graph: ONNXGraph, node: Onnx_NodeProto) throws {
        let name = node.output[0]
        let value = node.attribute[0]
        assert(value.name == "value")
        graph.initTensor(name, data: value.t)
    }
}

// MARK: - Arithmetic Layer Nodes

class AddConverter: NodeConverter {
    func convert(in graph: ONNXGraph, node: Onnx_NodeProto) throws {
        guard
            let input1 = graph.output(name: node.input[0]),
            let input2 = graph.output(name: node.input[1]),
            node.input.count == 2
        else { throw ONNXGraph.Errors.noSuchOutput }

        let add = MPSNNAdditionNode(leftSource: input1, rightSource: input2)
        graph.addFilter(add, withOutputs: node.output)
    }
}

class SubConverter: NodeConverter {
    func convert(in graph: ONNXGraph, node: Onnx_NodeProto) throws {
        guard
            let input1 = graph.output(name: node.input[0]),
            let input2 = graph.output(name: node.input[1]),
            node.input.count == 2
        else { throw ONNXGraph.Errors.noSuchOutput }

        let sub = MPSNNSubtractionNode(leftSource: input1, rightSource: input2)
        graph.addFilter(sub, withOutputs: node.output)
    }
}

class MulConverter: NodeConverter {
    func convert(in graph: ONNXGraph, node: Onnx_NodeProto) throws {
        guard
            let input1 = graph.output(name: node.input[0]),
            let input2 = graph.output(name: node.input[1]),
            node.input.count == 2
        else { throw ONNXGraph.Errors.noSuchOutput }

        let add = MPSNNMultiplicationNode(leftSource: input1, rightSource: input2)
        graph.addFilter(add, withOutputs: node.output)
    }
}

class DivConverter: NodeConverter {
    func convert(in graph: ONNXGraph, node: Onnx_NodeProto) throws {
        guard
            let input1 = graph.output(name: node.input[0]),
            let input2 = graph.output(name: node.input[1]),
            node.input.count == 2
        else { throw ONNXGraph.Errors.noSuchOutput }

        let div = MPSNNDivisionNode(leftSource: input1, rightSource: input2)
        graph.addFilter(div, withOutputs: node.output)
    }
}

// MARK: - Convolution Layer Nodes

@objc private class ConvDataSource: NSObject, MPSCNNConvolutionDataSource {
    func copy(with zone: NSZone? = nil) -> Any {
        return self.mutableCopy()
    }

    private var weight: ConvWeightArray
    private var bias: [Float]?

    private let desc: MPSCNNConvolutionDescriptor

    private let isDepthwise: Bool

    init(weight: Onnx_TensorProto, bias: Onnx_TensorProto?, dilations: (Int, Int), strides: (Int, Int), groups: Int, isTranspose: Bool, isONNX2MPS: Bool) {
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
            self.weight = .float32(weight.rawData.withUnsafeBytes {
                [Float](
                    UnsafeBufferPointer<Float>(
                        start: $0,
                        count: outputChannels * inputChannels * kernelHeight * kernelWidth
                    )
                )
            })
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

class ConvolutionConverter: NodeConverter {
    func convert(in graph: ONNXGraph, node: Onnx_NodeProto) throws {
        guard let input = graph.output(name: node.input[0]) else {
            throw ONNXGraph.Errors.noSuchOutput
        }

        guard let weight = graph.tensor(node.input[1]) else {
            throw ONNXGraph.Errors.insufficientInputs
        }

        var bias: Onnx_TensorProto?
        if node.input.count > 2 {
            bias = graph.tensor(node.input[2])
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

        guard let k = kernel, let d = dilations, let s = strides, let g = groups, let p = pads else {
            throw ONNXGraph.Errors.notEnoughAttributes
        }

        var conv: MPSCNNConvolutionNode!
        switch node.opType {
        case "Conv":
            let dataSource = ConvDataSource(weight: weight,
                                            bias: bias,
                                            dilations: d,
                                            strides: s,
                                            groups: g,
                                            isTranspose: false,
                                            isONNX2MPS: graph.modelFormat == .mpsFlavor)
            conv = MPSCNNConvolutionNode(source: input,
                                         weights: dataSource)
            conv.paddingPolicy = ONNXConvolutionPadding(kernel: k,
                                                        strides: s,
                                                        dilations: d,
                                                        pads: p,
                                                        outputPadding: outputPadding,
                                                        isTranspose: false)
        case "ConvTranspose":
            let dataSource = ConvDataSource(weight: weight,
                                            bias: bias,
                                            dilations: d,
                                            strides: s,
                                            groups: g,
                                            isTranspose: true,
                                            isONNX2MPS: graph.modelFormat == .mpsFlavor)
            conv = MPSCNNConvolutionTransposeNode(source: input,
                                                  weights: dataSource)
            conv.paddingPolicy = ONNXConvolutionPadding(kernel: k,
                                                        strides: s,
                                                        dilations: d,
                                                        pads: p,
                                                        outputPadding: outputPadding,
                                                        isTranspose: true)
        default:
            throw ONNXGraph.Errors.inconsistentState
        }
        graph.addFilter(conv, withOutputs: node.output)
    }
}

// MARK: - Pooling Layer Nodes

class AveragePoolConverter: NodeConverter {
    func convert(in graph: ONNXGraph, node: Onnx_NodeProto) throws {
        guard let input = graph.output(name: node.input[0]) else {
            throw ONNXGraph.Errors.noSuchOutput
        }
        let kernel_shape = node.attribute[0]
        assert(kernel_shape.name == "kernel_shape")
        let pads = node.attribute[1]
        assert(pads.name == "pads")
        let strides = node.attribute[2]
        assert(strides.name == "strides")

        let avgPool = MPSCNNPoolingAverageNode(source: input,
                                               kernelWidth: Int(kernel_shape.ints[1]),
                                               kernelHeight: Int(kernel_shape.ints[0]),
                                               strideInPixelsX: Int(strides.ints[1]),
                                               strideInPixelsY: Int(strides.ints[0]))
        avgPool.paddingPolicy = PyTorchPoolPadding(kernelWidth: Int(kernel_shape.ints[1]),
                                                   kernelHeight: Int(kernel_shape.ints[0]),
                                                   paddingWidth: Int(pads.ints[1]),
                                                   paddingHeight: Int(pads.ints[0]),
                                                   strideInPixelsX: Int(strides.ints[1]),
                                                   strideInPixelsY: Int(strides.ints[0]))
        graph.addFilter(avgPool, withOutputs: node.output)
    }
}

class MaxPoolConverter: NodeConverter {
    func convert(in graph: ONNXGraph, node: Onnx_NodeProto) throws {
        guard let input = graph.output(name: node.input[0]) else {
            throw ONNXGraph.Errors.noSuchOutput
        }
        let kernel_shape = node.attribute[0]
        assert(kernel_shape.name == "kernel_shape")
        let pads = node.attribute[1]
        assert(pads.name == "pads")
        let strides = node.attribute[2]
        assert(strides.name == "strides")

        let maxPool = MPSCNNPoolingMaxNode(source: input,
                                           kernelWidth: Int(kernel_shape.ints[1]),
                                           kernelHeight: Int(kernel_shape.ints[0]),
                                           strideInPixelsX: Int(strides.ints[1]),
                                           strideInPixelsY: Int(strides.ints[0]))
        maxPool.paddingPolicy = PyTorchPoolPadding(kernelWidth: Int(kernel_shape.ints[1]),
                                                   kernelHeight: Int(kernel_shape.ints[0]),
                                                   paddingWidth: Int(pads.ints[1]),
                                                   paddingHeight: Int(pads.ints[0]),
                                                   strideInPixelsX: Int(strides.ints[1]),
                                                   strideInPixelsY: Int(strides.ints[0]))
        graph.addFilter(maxPool, withOutputs: node.output)
    }
}

// MARK: - Fully Connected Layer Nodes

class GemmConverter: NodeConverter {
    func convert(in graph: ONNXGraph, node: Onnx_NodeProto) throws {
        guard let input = graph.output(name: node.input[0]) else {
            throw ONNXGraph.Errors.noSuchOutput
        }

        guard let weight = graph.tensor(node.input[1]) else {
            throw ONNXGraph.Errors.insufficientInputs
        }

        var bias: Onnx_TensorProto?
        if node.input.count > 2 {
            bias = graph.tensor(node.input[2])
        }

        var dilations: Dilations?
        var strides: Strides?
        var groups: Int?

        for attr in node.attribute {
            switch attr.name {
            case "dilations":
                dilations = (Int(attr.ints[0]), Int(attr.ints[1]))
            case "strides":
                strides = (Int(attr.ints[0]), Int(attr.ints[1]))
            case "group":
                groups = Int(attr.i)
            default:
                break
            }
        }

        guard let d = dilations, let s = strides, let g = groups else {
            throw ONNXGraph.Errors.notEnoughAttributes
        }

        let dataSource = ConvDataSource(weight: weight,
                                        bias: bias,
                                        dilations: d,
                                        strides: s,
                                        groups: g,
                                        isTranspose: false,
                                        isONNX2MPS: graph.modelFormat == .mpsFlavor)

        let gemm = MPSCNNFullyConnectedNode(source: input, weights: dataSource)
        graph.addFilter(gemm, withOutputs: node.output)
    }
}

// MARK: - Normalization Layer Nodes

@objc private class BatchNormalizationDataSource: NSObject, MPSCNNBatchNormalizationDataSource {

    required init?(coder aDecoder: NSCoder) {
        guard let data = aDecoder.decodeData(),
            let other = NSKeyedUnarchiver.unarchiveObject(with: data) as? BatchNormalizationDataSource else {
                return nil
        }
        self.nFeatureChannels = other.nFeatureChannels
    }

    private var meanW: [Float]?
    private var varianceW: [Float]?
    private var gammaW: [Float]?
    private var betaW: [Float]?
    private let nFeatureChannels: Int

    func mean() -> UnsafeMutablePointer<Float>? {
        return UnsafeMutablePointer(OpaquePointer(UnsafeRawPointer(self.meanW)))
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

    private static func toFloatArray(weight: Onnx_TensorProto, nFeatureChannels: Int) -> [Float]? {
        switch Int(weight.dataType) {
        case Onnx_TensorProto.DataType.float.rawValue:
            return weight.rawData.withUnsafeBytes {
                [Float](UnsafeBufferPointer<Float>(start: $0, count: nFeatureChannels))
            }
        case Onnx_TensorProto.DataType.float16.rawValue:
            return weight.rawData.withUnsafeBytes {
                float16to32(UnsafeMutableRawPointer(mutating: $0), count: nFeatureChannels)
            }
        default:
            return nil
        }
    }

    init(mean: Onnx_TensorProto, variance: Onnx_TensorProto, gamma: Onnx_TensorProto, beta: Onnx_TensorProto) {

        let nFeatureChannels = Int(mean.dims[0])
        self.nFeatureChannels = nFeatureChannels

        self.meanW = BatchNormalizationDataSource.toFloatArray(weight: mean, nFeatureChannels: nFeatureChannels)
        self.varianceW = BatchNormalizationDataSource.toFloatArray(weight: variance, nFeatureChannels: nFeatureChannels)
        self.gammaW = BatchNormalizationDataSource.toFloatArray(weight: gamma, nFeatureChannels: nFeatureChannels)
        self.betaW = BatchNormalizationDataSource.toFloatArray(weight: beta, nFeatureChannels: nFeatureChannels)
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

class BatchNormalizationConverter: NodeConverter {
    func convert(in graph: ONNXGraph, node: Onnx_NodeProto) throws {
        guard let input = graph.output(name: node.input[0]) else {
            throw ONNXGraph.Errors.noSuchOutput
        }

        guard
            let gamma = graph.tensor(node.input[1]),
            let beta = graph.tensor(node.input[2]),
            let mean = graph.tensor(node.input[3]),
            let variance = graph.tensor(node.input[4])
        else { throw ONNXGraph.Errors.insufficientInputs }

        if #available(iOS 11.3, macOS 10.13.4, *) {
            let dataSource = BatchNormalizationDataSource(mean: mean,
                                                          variance: variance,
                                                          gamma: gamma,
                                                          beta: beta)

            let batchNormalization = MPSCNNBatchNormalizationNode(source: input,
                                                                  dataSource: dataSource)
            graph.addFilter(batchNormalization, withOutputs: node.output)
        } else {
            throw ONNXGraph.Errors.unknownNodeOpType(opType: "BatchNormalization")
        }
    }
}

// MARK: - Neuron Layer Nodes

class AbsConverter: NodeConverter {
    func convert(in graph: ONNXGraph, node: Onnx_NodeProto) throws {
        guard let input = graph.output(name: node.input[0]) else {
            throw ONNXGraph.Errors.noSuchOutput
        }

        let abs = MPSCNNNeuronAbsoluteNode(source: input)
        graph.addFilter(abs, withOutputs: node.output)
    }
}

class EluConverter: NodeConverter {
    func convert(in graph: ONNXGraph, node: Onnx_NodeProto) throws {
        guard let input = graph.output(name: node.input[0]) else {
            throw ONNXGraph.Errors.noSuchOutput
        }

        let attr = node.attribute[0]
        #if DEBUG
        precondition(attr.name == "alpha")
        #endif

        let alpha = attr.f

        let elu = MPSCNNNeuronELUNode(source: input, a: alpha)
        graph.addFilter(elu, withOutputs: node.output)
    }
}

class HardSigmoidConverter: NodeConverter {
    func convert(in graph: ONNXGraph, node: Onnx_NodeProto) throws {
        guard let input = graph.output(name: node.input[0]) else {
            throw ONNXGraph.Errors.noSuchOutput
        }

        let hardSigmoid = MPSCNNNeuronHardSigmoidNode(source: input)
        graph.addFilter(hardSigmoid, withOutputs: node.output)
    }
}

class ReluConverter: NodeConverter {
    func convert(in graph: ONNXGraph, node: Onnx_NodeProto) throws {
        guard let input = graph.output(name: node.input[0]) else {
            throw ONNXGraph.Errors.noSuchOutput
        }

        let relu = MPSCNNNeuronReLUNode(source: input)
        graph.addFilter(relu, withOutputs: node.output)
    }
}

class SigmoidConverter: NodeConverter {
    func convert(in graph: ONNXGraph, node: Onnx_NodeProto) throws {
        guard let input = graph.output(name: node.input[0]) else {
            throw ONNXGraph.Errors.noSuchOutput
        }

        let sigmoid = MPSCNNNeuronSigmoidNode(source: input)
        graph.addFilter(sigmoid, withOutputs: node.output)
    }
}

class SoftplusConverter: NodeConverter {
    func convert(in graph: ONNXGraph, node: Onnx_NodeProto) throws {
        guard let input = graph.output(name: node.input[0]) else {
            throw ONNXGraph.Errors.noSuchOutput
        }

        let softplus = MPSCNNNeuronSoftPlusNode(source: input)
        graph.addFilter(softplus, withOutputs: node.output)
    }
}

class SoftsignConverter: NodeConverter {
    func convert(in graph: ONNXGraph, node: Onnx_NodeProto) throws {
        guard let input = graph.output(name: node.input[0]) else {
            throw ONNXGraph.Errors.noSuchOutput
        }

        let Softsign = MPSCNNNeuronSoftSignNode(source: input)
        graph.addFilter(Softsign, withOutputs: node.output)
    }
}

class TanhConverter: NodeConverter {
    func convert(in graph: ONNXGraph, node: Onnx_NodeProto) throws {
        guard let input = graph.output(name: node.input[0]) else {
            throw ONNXGraph.Errors.noSuchOutput
        }

        let tanh = MPSCNNNeuronTanHNode(source: input)
        graph.addFilter(tanh, withOutputs: node.output)
    }
}

class ExpConverter: NodeConverter {
    func convert(in graph: ONNXGraph, node: Onnx_NodeProto) throws {
        guard let input = graph.output(name: node.input[0]) else {
            throw ONNXGraph.Errors.noSuchOutput
        }

        if #available(iOS 11.3, macOS 10.13.4, *) {
            let sigmoid = MPSCNNNeuronExponentialNode(source: input)

            graph.addFilter(sigmoid, withOutputs: node.output)
        } else {
            throw ONNXGraph.Errors.unknownNodeOpType(opType: "Exp")
        }
    }
}

class LogConverter: NodeConverter {
    func convert(in graph: ONNXGraph, node: Onnx_NodeProto) throws {
        guard let input = graph.output(name: node.input[0]) else {
            throw ONNXGraph.Errors.noSuchOutput
        }

        if #available(OSX 10.13.4, *) {
            let log = MPSCNNNeuronLogarithmNode(source: input)
            graph.addFilter(log, withOutputs: node.output)
        } else {
            throw ONNXGraph.Errors.unknownNodeOpType(opType: "Log")
        }
    }
}

class PowConverter: NodeConverter {
    func convert(in graph: ONNXGraph, node: Onnx_NodeProto) throws {
        guard let input = graph.output(name: node.input[0]) else {
            throw ONNXGraph.Errors.noSuchOutput
        }

        if #available(OSX 10.13.4, *) {
            let pow = MPSCNNNeuronPowerNode(source: input)
            graph.addFilter(pow, withOutputs: node.output)
        } else {
            throw ONNXGraph.Errors.unknownNodeOpType(opType: "Pow")
        }
    }
}

// MARK: - Softmax Layer Nodes

class SoftmaxConverter: NodeConverter {
    func convert(in graph: ONNXGraph, node: Onnx_NodeProto) throws {
        guard let input = graph.output(name: node.input[0]) else {
            throw ONNXGraph.Errors.noSuchOutput
        }
        let axis = node.attribute[0]
        assert(axis.name == "axis")
        assert(axis.i == 1)
        let softmax = MPSCNNSoftMaxNode(source: input)
        graph.addFilter(softmax, withOutputs: node.output)
    }
}

class LogSoftmaxConverter: NodeConverter {
    func convert(in graph: ONNXGraph, node: Onnx_NodeProto) throws {
        guard let input = graph.output(name: node.input[0]) else {
            throw ONNXGraph.Errors.noSuchOutput
        }
        let axis = node.attribute[0]
        assert(axis.name == "axis")
        assert(axis.i == 1)
        let logSoftmax = MPSCNNLogSoftMaxNode(source: input)
        graph.addFilter(logSoftmax, withOutputs: node.output)
    }
}

// MARK: - Upsampling Layer Nodes

class UpsampleConverter: NodeConverter {
    func convert(in graph: ONNXGraph, node: Onnx_NodeProto) throws {
        guard let input = graph.output(name: node.input[0]) else {
            throw ONNXGraph.Errors.noSuchOutput
        }

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
            guard let scales_tensor = graph.tensor(node.input[1]) else {
                throw ONNXGraph.Errors.notEnoughAttributes
            }
            let scales_count = Int(scales_tensor.dims[0])
            assert(scales_count == 4)
            let scales_values = scales_tensor.rawData.withUnsafeBytes {
                [Float](
                    UnsafeBufferPointer<Float>(
                        start: $0,
                        count: scales_count
                    )
                )
            }
            scales = (Int(scales_values[2]), Int(scales_values[3]))
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

        graph.addFilter(upsample, withOutputs: node.output)
    }
}

// MARK: - Dropout Layer Nodes

class DropoutConverter: NodeConverter {
    func convert(in graph: ONNXGraph, node: Onnx_NodeProto) throws {
        guard let input = graph.output(name: node.input[0]) else {
            throw ONNXGraph.Errors.noSuchOutput
        }

        if #available(OSX 10.13.4, *) {
            let dropout = MPSCNNDropoutNode(source: input)
            graph.addFilter(dropout, withOutputs: node.output)
        } else {
            throw ONNXGraph.Errors.unknownNodeOpType(opType: "Dropout")
        }
    }
}

// MARK: - Kernel Concatenation Nodes

class ConcatConverter: NodeConverter {
    func convert(in graph: ONNXGraph, node: Onnx_NodeProto) throws {
        let inputs = try (0..<node.input.count).map( { (idx: Int) -> MPSNNImageNode in
            guard let input = graph.output(name: node.input[idx]) else {
                throw ONNXGraph.Errors.noSuchOutput
            }
            return input
        })

        let concat = MPSNNConcatenationNode(sources: inputs)
        graph.addFilter(concat, withOutputs: node.output)
    }
}
