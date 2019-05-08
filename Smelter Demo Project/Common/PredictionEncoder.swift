//
//  PredictionEncoder.swift
//  Smelter Demo
//
//  Created by Eugene Bokhan on 08/05/2019.
//  Copyright © 2019 Eugene Bokhan. All rights reserved.
//

import Alloy
import Smelter
import MetalPerformanceShaders

final internal class PredictionEncoder {

    // MARK: - Errors

    internal enum Errors: Error {
        case graphEncodingFailed
    }

    // MARK: - Propertires

    private let mobileNetGraph: MPSNNGraph
    private let readModelGraphResultEncoder: ReadModelGraphResultEncoder

    // MARK: - Life Cycle

    internal init(context: MTLContext,
                  modelData: Data,
                  configuration: ONNXGraph.Configuration) throws {
        self.mobileNetGraph = try ONNXGraph(data: modelData)
            .metalGraph(device: context.device,
                        configuration: configuration)
        self.readModelGraphResultEncoder = try ReadModelGraphResultEncoder(context: context)
    }

    // MARK: - Encode

    internal func encode(inputTexture: MTLTexture,
                         resultBuffer: MTLBuffer,
                         in commandBuffer: MTLCommandBuffer) throws {
        let inputMPSImage = MPSImage(texture: inputTexture,
                                     featureChannels: 3)
        commandBuffer.pushDebugGroup("Read Model Graph Result")
        guard
            let modelGraphResult = self.mobileNetGraph.encode(to: commandBuffer,
                                                              sourceImages: [inputMPSImage])
        else { throw Errors.graphEncodingFailed }

        try self.readModelGraphResultEncoder.encode(modelGraphResult: modelGraphResult,
                                                    resultBuffer: resultBuffer,
                                                    in: commandBuffer)
        commandBuffer.popDebugGroup()
    }
}
