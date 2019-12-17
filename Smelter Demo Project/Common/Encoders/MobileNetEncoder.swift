//
//  MobileNetEncoder.swift
//  Smelter Demo
//
//  Created by Eugene Bokhan on 08/05/2019.
//  Copyright © 2019 Eugene Bokhan. All rights reserved.
//

import Alloy
import Smelter
import MetalPerformanceShaders

final internal class MobileNetEncoder {

    // MARK: - Errors

    internal enum Errors: Error {
        case graphEncodingFailed
    }

    // MARK: - Propertires

    private let mobileNetGraph: MPSNNGraph
    private let normalizeKernelEncoder: NormalizeKernelEncoder

    // MARK: - Life Cycle

    internal init(context: MTLContext,
                  modelData: Data,
                  configuration: ONNXGraph.Configuration) throws {
        self.normalizeKernelEncoder = try NormalizeKernelEncoder(context: context)
        self.mobileNetGraph = try ONNXGraph(data: modelData).metalGraph(device: context.device,
                                                                        configuration: configuration)
    }

    // MARK: - Encode

    internal func encode(inputTexture: MTLTexture,
                         in commandBuffer: MTLCommandBuffer) throws -> MPSImage {
        let descriptor = inputTexture.descriptor
        descriptor.usage = [.shaderRead, .shaderWrite]
        descriptor.storageMode = .private
        // we need a signed pixel format to store negative values after normalization
        descriptor.pixelFormat = .rgba16Float
        
        let normalizedImage = MPSTemporaryImage(commandBuffer: commandBuffer,
                                                textureDescriptor: descriptor)
        defer { normalizedImage.readCount = 0 }

        self.normalizeKernelEncoder.encode(sourceTexture: inputTexture,
                                           destinationTexture: normalizedImage.texture,
                                           mean: MobileNetEncoder.normalizationMean,
                                           std: MobileNetEncoder.normalizationSTD,
                                           in: commandBuffer)

        commandBuffer.pushDebugGroup("MobileNet Encoder")
        
        let inputMPSImage = MPSImage(texture: normalizedImage.texture,
                                     featureChannels: 3)
        guard
            let modelGraphResult = self.mobileNetGraph.encode(to: commandBuffer,
                                                              sourceImages: [inputMPSImage])
        else { throw Errors.graphEncodingFailed }

        commandBuffer.popDebugGroup()

        return modelGraphResult
    }

    private static let normalizationMean = vector_float3(0.485, 0.456, 0.406)
    private static let normalizationSTD = vector_float3(0.229, 0.224, 0.225)
}
