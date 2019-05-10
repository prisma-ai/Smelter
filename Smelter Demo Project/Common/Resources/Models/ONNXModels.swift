//
//  ONNXModels.swift
//  Smelter Demo
//
//  Created by Eugene Bokhan on 04/05/2019.
//  Copyright © 2019 Eugene Bokhan. All rights reserved.
//

import Foundation

struct ONNXModels {
    private init () {}

    static let mobilenetv2_fused = Bundle.main.url(forResource: "mobilenetv2_fp16",
                                                   withExtension: "onnx")!
}
