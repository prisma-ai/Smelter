import Alloy
import Smelter

final class MobileNetEncoder {
    
    // MARK: - Type Definition
    
    enum Error: Swift.Error {
        case graphEncodingFailed
    }

    // MARK: - Propertires

    private let mobileNetGraph: MPSNNGraph
    private let normalize: Normalize

    // MARK: - Life Cycle

    init(context: MTLContext,
         modelData: Data,
         configuration: ONNXGraph.Configuration) throws {
        self.normalize = try .init(context: context)
        self.mobileNetGraph = try ONNXGraph(data: modelData,
                                            configuration: configuration).metalGraph(device: context.device)
    }
    
    // MARK: - Encode
    
    func encode(source: MTLTexture,
                in commandBuffer: MTLCommandBuffer) throws -> MPSImage {
        let descriptor = source.descriptor
        descriptor.usage = [.shaderRead, .shaderWrite]
        descriptor.storageMode = .private
        // we need a signed pixel format to store negative values after normalization
        descriptor.pixelFormat = .rgba16Float
        
        let normalizedImage = MPSTemporaryImage(commandBuffer: commandBuffer,
                                                textureDescriptor: descriptor)
        defer { normalizedImage.readCount = .zero }

        self.normalize(source: source,
                       mean: Self.normalizationMean,
                       std: Self.normalizationSTD,
                       destination: normalizedImage.texture,
                       in: commandBuffer)
        
        let inputMPSImage = MPSImage(texture: normalizedImage.texture,
                                     featureChannels: 3)
        guard let modelGraphResult = self.mobileNetGraph.encode(to: commandBuffer,
                                                                sourceImages: [inputMPSImage])
        else { throw Error.graphEncodingFailed }

        return modelGraphResult
    }

    private static let normalizationMean = vector_float3(0.485, 0.456, 0.406)
    private static let normalizationSTD = vector_float3(0.229, 0.224, 0.225)
}
