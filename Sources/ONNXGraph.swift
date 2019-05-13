//
//  ONNXGraph.swift
//  Smelter
//
//  Created by Andrey Volodin on 16/04/2019.
//

import Alloy
import MetalPerformanceShaders

public final class ONNXGraph {

    // MARK - Configuration

    public struct Configuration {
        public enum Scale {
            case lanczos
            case bilinear
        }

        public enum InputConstraint {
            case none
            case forceInputScale(scale: Scale)
        }

        public let inputConstraint: InputConstraint

        public init(inputConstraint: InputConstraint = .none) {
            self.inputConstraint = inputConstraint
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

    // MARK - Properties
    
    public var modelFormat: Format = .onnx

    private var converters: [String: NodeConverter] = [:]
    private var tensors: [String: Onnx_TensorProto] = [:]
    private var outputs: [String: MPSNNImageNode] = [:]
    private var nodeShapes: [String: Shape] = [:]
    private var filters = [Filter]()
    private var graphProto: Onnx_GraphProto!

    // TODO: Probably should be returning shapes paired with names
    internal var outputShapes: [Shape] {
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

    // MARK - Life Cycle

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

            // Constant
        self.register(name: "Constant", converter: ConstantConverter())
            // Arithmetic Layer Nodes
            .register(name: "Add", converter: AddConverter())
            .register(name: "Sub", converter: SubConverter())
            .register(name: "Mul", converter: MulConverter())
            .register(name: "Div", converter: DivConverter())
            // Convolution Layer Nodes
            .register(name: "Conv", converter: ConvolutionConverter())
            .register(name: "ConvTranspose", converter: ConvolutionConverter())
            // Pooling Layer Nodes
            .register(name: "AveragePool", converter: AveragePoolConverter())
            .register(name: "GlobalAveragePool", converter: GlobalAveragePoolConverter())
            .register(name: "MaxPool", converter: MaxPoolConverter())
            // Neuron Layer Nodes
            .register(name: "Abs", converter: AbsConverter())
            .register(name: "Elu", converter: EluConverter())
            .register(name: "Relu", converter: ReluConverter())
            .register(name: "Sigmoid", converter: SigmoidConverter())
            .register(name: "HardSigmoid", converter: HardSigmoidConverter())
            .register(name: "Softplus", converter: SoftplusConverter())
            .register(name: "Softsign", converter: SoftsignConverter())
            .register(name: "Tanh", converter: TanhConverter())
            // Softmax Layer Nodes
            .register(name: "Softmax", converter: SoftmaxConverter())
            .register(name: "LogSoftmax", converter: LogSoftmaxConverter())
            // Upsampling Layer Nodes
            .register(name: "Upsample", converter: UpsampleConverter())
            // Kernel Concatenation Nodes
            .register(name: "Concat", converter: ConcatConverter())

        if #available(iOS 11.3, tvOS 11.3, macOS 10.13.4, *) {
                // Normalization Layer Nodes
            self.register(name: "BatchNormalization", converter: BatchNormalizationConverter())
                // // Neuron Layer Nodes
                .register(name: "Exp", converter: ExpConverter())
                .register(name: "Log", converter: LogConverter())
                .register(name: "Pow", converter: PowConverter())
                // Dropout Layer Nodes
                .register(name: "Dropout", converter: DropoutConverter())

        }
        
        if #available(iOS 12.1, tvOS 12.1, macOS 10.14.1, *) {
            // Reshape Nodes
            self.register(name: "Reshape", converter: ReshapeConverter())
        }
    }

    public convenience init(contentsOf url: URL) throws {
        let data = try Data(contentsOf: url)
        try self.init(data: data)
    }

    public func metalGraph(device: MTLDevice, configuration: Configuration) throws -> MPSNNGraph {
        try self.initOutputs(configuration: configuration)

        for node in self.graphProto.node {
            guard let converter = self.converters[node.opType]
            else { throw Errors.unknownNodeOpType(opType: node.opType) }
            try converter.convert(in: self, node: node)
        }

        if self.graphProto.output.count != 1 {
            throw Errors.unsupportedOutput
        }

        guard let output = self.output(name: self.graphProto.output[0].name)
        else { throw Errors.noSuchOutput }

        guard let graph = MPSNNGraph(device: device,
                                     resultImage: output,
                                     resultImageIsNeeded: true)
        else { throw Errors.graphInternalError }

        return graph
    }

    // MARK - Private / Internal Methods

    private func initOutputs(configuration: Configuration) throws {
        self.outputs = try self.graphProto.input.reduce(into: [String:MPSNNImageNode]()) { (res, valueInfo) in
            if self.tensor(name: valueInfo.name) == nil {
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

                switch configuration.inputConstraint {
                case .none:
                    res[valueInfo.name] = imageNode
                case .forceInputScale(let scale):
                    switch scale {
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

                self.nodeShapes[valueInfo.name] = Shape(1, width, height, channels)
            }
        }
    }

    @discardableResult
    private func register(name: String, converter: NodeConverter) -> ONNXGraph {
        self.converters[name] = converter
        return self
    }

    internal func initTensor(_ name: String, data: Onnx_TensorProto) {
        self.tensors[name] = data
    }

    internal func addFilter(_ filter: MPSNNFilterNode, outputShape: Shape, withOutputs outputs: [String]) {
        self.filters.append(filter)
        for output in outputs {
            self.nodeShapes[output] = outputShape
            self.outputs[output] = filter.resultImage
        }
    }

    internal func output(name: String) -> MPSNNImageNode? {
        return self.outputs[name]
    }

    internal func tensor(name: String) -> Onnx_TensorProto? {
        return self.tensors[name]
    }

    internal func shape(output: String) -> Shape? {
        return self.nodeShapes[output]
    }
}
