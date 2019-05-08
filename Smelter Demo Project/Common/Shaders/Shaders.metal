//
//  Shaders.metal
//  Smelter Demo
//
//  Created by Eugene Bokhan on 08/05/2019.
//  Copyright © 2019 Eugene Bokhan. All rights reserved.
//

#include <metal_stdlib>
using namespace metal;

constant bool deviceSupportsNonuniformThreadgroups [[function_constant(0)]];

kernel void readModelGraphResult(texture2d<float, access::read> inputTexture [[ texture(0) ]],
                                 device float* result [[ buffer(0) ]],
                                 uint2 position [[thread_position_in_grid]]) {
    const ushort2 textureSize = ushort2(inputTexture.get_width(),
                                        inputTexture.get_height());
    if (!deviceSupportsNonuniformThreadgroups) {
        if (position.x >= textureSize.x || position.y >= textureSize.y) {
            return;
        }
    }
    // Read mpsnngraph result value.
    float value = (float)inputTexture.read(position).r;
    result[(int)position.y] = value;
}
