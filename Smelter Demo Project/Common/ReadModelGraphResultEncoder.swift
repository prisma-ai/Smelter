//
//  ReadModelGraphResultEncoder.swift
//  Smelter Demo
//
//  Created by Eugene Bokhan on 08/05/2019.
//  Copyright © 2019 Eugene Bokhan. All rights reserved.
//

import Alloy
import MetalPerformanceShaders

final internal class ReadModelGraphResultEncoder {

    // MARK: - Errors

    internal enum Errors: Error {
        case libraryCreationFailed
    }

    // MARK: - Propertires

    private let pipelineState: MTLComputePipelineState
    private let deviceSupportsNonuniformThreadgroups: Bool

    // MARK: - Life Cycle

    internal convenience init(context: MTLContext) throws {
        guard
            let library = context.shaderLibrary(for: PredictionEncoder.self)
        else { throw Errors.libraryCreationFailed }
        try self.init(library: library)
    }

    internal init(library: MTLLibrary) throws {
        #if os(iOS) || os(tvOS)
        self.deviceSupportsNonuniformThreadgroups = library.device.supportsFeatureSet(.iOS_GPUFamily4_v1)
        #elseif os(macOS)
        self.deviceSupportsNonuniformThreadgroups = library.device.supportsFeatureSet(.macOS_GPUFamily1_v3)
        #endif
        let constantValues = MTLFunctionConstantValues()
        var dispatchFlag = self.deviceSupportsNonuniformThreadgroups
        constantValues.setConstantValue(&dispatchFlag,
                                        type: .bool,
                                        index: 0)

        self.pipelineState = try library.computePipelineState(function: ReadModelGraphResultEncoder.functionName,
                                                              constants: constantValues)
    }

    // MARK: - Encode

    internal func encode(modelGraphResult: MPSImage,
                         resultBuffer: MTLBuffer,
                         in commandBuffer: MTLCommandBuffer) throws {
        commandBuffer.compute { encoder in
            encoder.pushDebugGroup("Prediction Encoder")
            
            encoder.setTexture(modelGraphResult.texture,
                               index: 0)
            encoder.setBuffer(resultBuffer,
                              offset: 0,
                              index: 0)

            if self.deviceSupportsNonuniformThreadgroups {
                encoder.dispatch2d(state: self.pipelineState,
                                   exactly: modelGraphResult.texture.size)
            } else {
                encoder.dispatch2d(state: self.pipelineState,
                                   covering: modelGraphResult.texture.size)
            }

            encoder.popDebugGroup()
        }
    }

    private static let functionName = "readModelGraphResult"
}
