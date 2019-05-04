//
//  ONNXGraph.swift
//  Smelter
//
//  Created by Andrey Volodin on 16/04/2019.
//

import Alloy
import MetalPerformanceShaders

public final class ONNXGraph {
    public struct Configuration {
        public enum Scale {
            case lanczos
            case bilinear
            case none
        }

        public let inputScale: Scale

        public init(inputScale: Scale = .none) {
            self.inputScale = inputScale
        }
    }

    public enum Errors: Error {
        case unsupportedInput
        case unsupportedOutput
        case unknownNodeOpType(opType: String)
        case noSuchOutput
        case graphInternalError
        case insufficientInputs
        case inconsistentState
        case notEnoughAttributes
    }

    public enum Format {
        case onnx
        case mpsFlavor
    }

    private typealias Filter = AnyObject
    
    public var modelFormat: Format = .onnx

    private var converters: [String: NodeConverter] = [:]
    private var tensors: [String: Onnx_TensorProto] = [:]
    private var outputs: [String: MPSNNImageNode] = [:]

    private var filters = [Filter]()

    private var graphProto: Onnx_GraphProto!

    public init(data: Data) throws {
        let modelProto = try Onnx_ModelProto(serializedData: data)
        self.graphProto = modelProto.graph
        switch modelProto.producerName {
        case "ONNX2MPS":
            self.modelFormat = .mpsFlavor
        default:
            self.modelFormat = .onnx
        }

        self.tensors = self.graphProto.initializer.reduce(into: self.tensors) { (res, tensor) in
            res[tensor.name] = tensor
        }

        let _ = self
            .register(name: "Add", converter: AddConverter())
            .register(name: "Sub", converter: SubConverter())
            .register(name: "Mul", converter: MulConverter())
            .register(name: "Div", converter: DivConverter())

            .register(name: "Conv", converter: ConvolutionConverter())
            .register(name: "ConvTranspose", converter: ConvolutionConverter())

            .register(name: "AveragePool", converter: AveragePoolConverter())
            .register(name: "MaxPool", converter: MaxPoolConverter())

            .register(name: "Abs", converter: AbsConverter())
            .register(name: "Elu", converter: EluConverter())
            .register(name: "HardSigmoid", converter: HardSigmoidConverter())
            .register(name: "Relu", converter: ReluConverter())
            .register(name: "Sigmoid", converter: SigmoidConverter())
            .register(name: "Softplus", converter: SoftplusConverter())
            .register(name: "Softsign", converter: SoftsignConverter())
            .register(name: "Tanh", converter: TanhConverter())
            .register(name: "Exp", converter: ExpConverter())
            .register(name: "Log", converter: LogConverter())
            .register(name: "Pow", converter: PowConverter())

            .register(name: "Softmax", converter: SoftmaxConverter())
            .register(name: "LogSoftmax", converter: LogSoftmaxConverter())

            .register(name: "Upsample", converter: UpsampleConverter())

            .register(name: "Dropout", converter: DropoutConverter())

            .register(name: "Concat", converter: ConcatConverter())

            .register(name: "Constant", converter: ConstantConverter())

            .register(name: "Gemm", converter: GemmConverter())

            .register(name: "BatchNormalization", converter: BatchNormalizationConverter())
    }

    public convenience init(contentsOf url: URL) throws {
        let data = try Data(contentsOf: url)
        try self.init(data: data)
    }

    private func register(name: String, converter: NodeConverter) -> ONNXGraph {
        self.converters[name] = converter
        return self
    }

    private func initOutputs(configuration: Configuration) throws {
        self.outputs = try self.graphProto.input.reduce(into: [String:MPSNNImageNode]()) { (res, valueInfo) in
            if self.tensor(valueInfo.name) == nil {
                let shape = valueInfo.type.tensorType.shape.dim.map { Int($0.dimValue) }
                var channels, height, width: Int
                switch shape.count {
                case 3:
                    channels = shape[0]
                    height = shape[1]
                    width = shape[2]
                case 4:
                    channels = shape[1]
                    height = shape[2]
                    width = shape[3]
                default:
                    throw Errors.unsupportedInput
                }

                let imageNode = MPSNNImageNode(handle: nil)

                switch configuration.inputScale {
                case .none:
                    res[valueInfo.name] = imageNode
                case .lanczos:
                    res[valueInfo.name + "_input"] = imageNode
                    let scale = MPSNNLanczosScaleNode(
                        source: imageNode,
                        outputSize: MTLSize(width: width, height: height, depth: channels))
                    self.filters.append(scale)
                    res[valueInfo.name] = scale.resultImage
                case .bilinear:
                    res[valueInfo.name + "_input"] = imageNode
                    let scale = MPSNNBilinearScaleNode(
                        source: imageNode,
                        outputSize: MTLSize(width: width, height: height, depth: channels))
                    self.filters.append(scale)
                    res[valueInfo.name] = scale.resultImage
                }
            }
        }
    }

    public func metalGraph(device: MTLDevice, configuration: Configuration) throws -> MPSNNGraph {
        try self.initOutputs(configuration: configuration)

        for node in self.graphProto.node {
            guard let converter = self.converters[node.opType] else {
                print(node.opType)
                throw Errors.unknownNodeOpType(opType: node.opType)
            }

            try converter.convert(in: self, node: node)
        }

        if self.graphProto.output.count != 1 {
            throw Errors.unsupportedOutput
        }

        guard let output = self.output(name: self.graphProto.output[0].name) else {
            throw Errors.noSuchOutput
        }

        guard let graph = MPSNNGraph(device: device,
                                     resultImage: output,
                                     resultImageIsNeeded: true)
        else { throw Errors.graphInternalError }

        return graph
    }

    // TODO: Probably should be returning shapes paired with names
    public var outputShapes: [Shape] {
        return self.graphProto.output.compactMap { output in
            let shape = output.type.tensorType.shape.dim.map { Int($0.dimValue) }

            var channels, height, width, depth: Int
            switch shape.count {
            case 3:
                channels = shape[0]
                height = shape[1]
                width = shape[2]
                depth = 1

                return (channels, width, height, depth)
            case 4:
                channels = shape[1]
                height = shape[2]
                width = shape[3]
                depth = 1
                return (channels, width, height, depth)
            default: return nil
            }
        }
    }

    internal func addFilter(_ filter: MPSNNFilterNode, withOutputs outputs: [String]) {
        self.filters.append(filter)
        for output in outputs {
            self.outputs[output] = filter.resultImage
        }
    }

    public func output(name: String) -> MPSNNImageNode? {
        return self.outputs[name]
    }

    internal func tensor(_ name: String) -> Onnx_TensorProto? {
        return self.tensors[name]
    }

    internal func initTensor(_ name: String, data: Onnx_TensorProto) {
        self.tensors[name] = data
    }
}
