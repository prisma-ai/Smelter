import MetalPerformanceShaders

public final class ONNXGraph {
    // MARK: - Configuration

    public struct Configuration {
        public enum Scale {
            case lanczos
            case bilinear
        }

        public enum InputConstraint {
            case none
            case forceInputScale(scale: Scale)
        }

        public struct BillinearUpsampling {
            public let alignCorners: Bool

            public static let `default` = BillinearUpsampling(alignCorners: true)
        }

        public let inputConstraint: InputConstraint
        public let billinearUpsamplingConfiguration: BillinearUpsampling
        public let dims: [Int: Int]

        public init(
            inputConstraint: InputConstraint = .none,
            billinearUpsamplingConfiguration: BillinearUpsampling = .default,
            dims: [Int: Int] = [:]
        ) {
            self.inputConstraint = inputConstraint
            self.billinearUpsamplingConfiguration = billinearUpsamplingConfiguration
            self.dims = dims
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

    // MARK: - Properties

    public var modelFormat: Format = .onnx
    public var configuration: Configuration

    private var converters: [String: NodeConverter] = [:]
    private var tensors: [String: Onnx_TensorProto] = [:]
    private var outputs: [String: MPSNNImageNode] = [:]
    private var nodeShapes: [String: Shape] = [:]
    private var filters = [Filter]()
    private var graphProto: Onnx_GraphProto!

    // TODO: Probably should be returning shapes paired with names
    public var outputShapes: [Shape] {
        self.graphProto.output.compactMap { output in
            let shape = output.type.tensorType.shape.dim.map { Int($0.dimValue) }

            switch shape.count {
            case 3:
                return .init(
                    channels: shape[0],
                    width: shape[1],
                    height: shape[2],
                    depth: 1
                )
            case 4:
                return .init(
                    channels: shape[1],
                    width: shape[2],
                    height: shape[3],
                    depth: 1
                )
            default: return nil
            }
        }
    }

    // MARK: - Life Cycle

    public init(data: Data, configuration: Configuration) throws {
        let modelProto = try Onnx_ModelProto(serializedData: data)
        self.graphProto = modelProto.graph
        switch modelProto.producerName {
        case "ONNX2MPS":
            self.modelFormat = .mpsFlavor
        default:
            self.modelFormat = .onnx
        }
        self.configuration = configuration

        self.tensors = self.graphProto.initializer.reduce(into: self.tensors) { res, tensor in
            res[tensor.name] = tensor
        }

        self.register(name: "Conv", converter: ConvolutionConverter())
            .register(name: "Gemm", converter: ConvolutionConverter())
            .register(name: "Relu", converter: ReluConverter())
            .register(name: "Elu", converter: EluConverter())
            .register(name: "Add", converter: AddConverter())
            .register(name: "Sub", converter: SubConverter())
            .register(name: "ConvTranspose", converter: ConvolutionConverter())
            .register(name: "Sigmoid", converter: SigmoidConverter())
            .register(name: "Upsample", converter: UpsampleConverter(alignCorners: self.configuration
                    .billinearUpsamplingConfiguration
                    .alignCorners))
            .register(name: "HardSigmoid", converter: HardSigmoidConverter())
            .register(name: "HardSigmoid", converter: HardSigmoidConverter())
            .register(name: "Concat", converter: ConcatConverter())
            .register(name: "AveragePool", converter: AveragePoolConverter())
            .register(name: "MaxPool", converter: MaxPoolConverter())
            .register(name: "Softmax", converter: SoftmaxConverter())
            .register(name: "LogSoftmax", converter: LogSoftmaxConverter())
            .register(name: "Constant", converter: ConstantConverter())
            .register(name: "Mul", converter: MulConverter())
            .register(name: "Div", converter: DivConverter())
            .register(name: "GlobalAveragePool", converter: GlobalAveragePoolConverter())
            .register(name: "Abs", converter: AbsConverter())
            .register(name: "Softplus", converter: SoftplusConverter())
            .register(name: "Softsign", converter: SoftsignConverter())
            .register(name: "Tanh", converter: TanhConverter())
            .register(name: "PRelu", converter: PReluConverter())

        if #available(iOS 11.3, tvOS 11.3, macOS 10.13.4, *) {
            self.register(name: "BatchNormalization", converter: BatchNormalizationConverter())
                .register(name: "Dropout", converter: DropoutConverter())
                .register(name: "InstanceNormalization", converter: InstanceNormConverter())
                .register(name: "Log", converter: LogConverter())
                .register(name: "Pow", converter: PowConverter())
                .register(name: "Exp", converter: ExpConverter())
        }

        if #available(iOS 12.1, tvOS 12.1, macOS 10.14.1, *) {
            self.register(name: "Reshape", converter: ReshapeConverter())
                .register(name: "Flatten", converter: FlattenConverter())
                .register(name: "Pad", converter: PaddingConverter())
        }

        if #available(iOS 13.0, tvOS 13.0, macOS 10.15.0, *) {
            self.register(name: "custom_group_norm", converter: GroupNormConverter())
        }
    }

    public convenience init(
        contentsOf url: URL,
        configuration: Configuration
    ) throws {
        let data = try Data(contentsOf: url)
        try self.init(
            data: data,
            configuration: configuration
        )
    }

    public func metalGraph(device: MTLDevice) throws -> MPSNNGraph {
        try self.initOutputs(configuration: self.configuration)

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

        guard let graph = MPSNNGraph(
            device: device,
            resultImage: output,
            resultImageIsNeeded: true
        )
        else { throw Errors.graphInternalError }

        return graph
    }

    // MARK: - Private / Internal Methods

    private func initOutputs(configuration: Configuration) throws {
        self.outputs = try self.graphProto.input.reduce(into: [String: MPSNNImageNode]()) { res, valueInfo in
            if self.tensor(name: valueInfo.name) == nil {
                let shape = valueInfo.type.tensorType.shape.dim.enumerated().map {
                    self.configuration.dims[$0.offset] ?? Int($0.element.dimValue)
                }
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
                case let .forceInputScale(scale):
                    switch scale {
                    case .lanczos:
                        res[valueInfo.name + "_input"] = imageNode
                        let scale = MPSNNLanczosScaleNode(
                            source: imageNode,
                            outputSize: MTLSize(width: width, height: height, depth: channels)
                        )
                        self.filters.append(scale)
                        res[valueInfo.name] = scale.resultImage
                    case .bilinear:
                        res[valueInfo.name + "_input"] = imageNode
                        let scale = MPSNNBilinearScaleNode(
                            source: imageNode,
                            outputSize: MTLSize(width: width, height: height, depth: channels)
                        )
                        self.filters.append(scale)
                        res[valueInfo.name] = scale.resultImage
                    }
                }

                self.nodeShapes[valueInfo.name] = .init(
                    channels: 1,
                    width: width,
                    height: height,
                    depth: channels
                )
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

    internal func addFilter(
        _ filter: MPSNNFilterNode,
        outputShape: Shape,
        withOutputs outputs: [String]
    ) {
        self.filters.append(filter)
        for output in outputs {
            self.nodeShapes[output] = outputShape
            self.outputs[output] = filter.resultImage
        }
    }

    internal func output(name: String) -> MPSNNImageNode? {
        self.outputs[name]
    }

    internal func tensor(name: String) -> Onnx_TensorProto? {
        self.tensors[name]
    }

    internal func shape(output: String) -> Shape? {
        self.nodeShapes[output]
    }
}
