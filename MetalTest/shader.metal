//
//  shader.metal
//  MetalTest
//
//  Created by Tom on 2020/6/12.
//  Copyright © 2020 Tom. All rights reserved.
//

#include <metal_stdlib>
using namespace metal;

kernel void sigmoid(const device float *inVector [[ buffer(0) ]],
                    device float *outVector [[ buffer(1) ]],
                    uint id [[ thread_position_in_grid ]]) {
    // This calculates sigmoid for _one_ position (=id) in a vector per call on the GPU

    outVector[id] = 1.0 / (1.0 + exp(-inVector[id]));
}
