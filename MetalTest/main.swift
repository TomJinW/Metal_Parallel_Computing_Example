//
//  main.swift
//  MetalTest
//
//  Created by Tom on 2020/6/12.
//  Copyright © 2020 Tom. All rights reserved.
//

import Foundation
import Metal

print("Hello, World!")

func initMetal() -> (MTLDevice, MTLCommandQueue, MTLLibrary, MTLCommandBuffer,MTLComputeCommandEncoder){
  // Get access to iPhone or iPad GPU
    let device = MTLCreateSystemDefaultDevice()!

  // Queue to handle an ordered list of command buffers
    let commandQueue = device.makeCommandQueue()!

  // Access to Metal functions that are stored in Shaders.metal file, e.g. sigmoid()
    let defaultLibrary = device.makeDefaultLibrary()

  // Buffer for storing encoded commands that are sent to GPU
    let commandBuffer = commandQueue.makeCommandBuffer()!

  // Encoder for GPU commands
    let computeCommandEncoder = commandBuffer.makeComputeCommandEncoder()!

    return (device, commandQueue, defaultLibrary!, commandBuffer, computeCommandEncoder)
}


var myvector = [Float](repeating: 0, count: 10)
for (index, value) in myvector.enumerated() {
   myvector[index] = Float(index)
}

// a. initialize Metal
var (device, commandQueue, defaultLibrary, commandBuffer, computeCommandEncoder) = initMetal()

// b. set up a compute pipeline with Sigmoid function and add it to encoder
let sigmoidProgram = defaultLibrary.makeFunction(name: "sigmoid")
var computePipelineFilter = try? device.makeComputePipelineState(function: sigmoidProgram!)
computeCommandEncoder.setComputePipelineState(computePipelineFilter!)

// a. calculate byte length of input data - myvector
var myvectorByteLength = myvector.count *  MemoryLayout.size(ofValue: myvector[0])

// b. create a MTLBuffer - input data that the GPU and Metal and produce
var inVectorBuffer = device.makeBuffer(bytes: &myvector, length: myvectorByteLength, options: [])

// c. set the input vector for the Sigmoid() function, e.g. inVector
//    atIndex: 0 here corresponds to buffer(0) in the Sigmoid function
computeCommandEncoder.setBuffer(inVectorBuffer, offset: 0, index: 0)

// d. create the output vector for the Sigmoid() function, e.g. outVector
//    atIndex: 1 here corresponds to buffer(1) in the Sigmoid function
var resultdata = [Float](repeating: 0, count:myvector.count)
var outVectorBuffer = device.makeBuffer(bytes: &resultdata, length: myvectorByteLength, options: [])
computeCommandEncoder.setBuffer(outVectorBuffer, offset: 0, index: 1)

// hardcoded to 32 for now (recommendation: read about threadExecutionWidth)
var threadsPerGroup = MTLSize(width:32,height:1,depth:1)
var numThreadgroups = MTLSize(width:(myvector.count+31)/32, height:1, depth:1)
computeCommandEncoder.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadsPerGroup)

computeCommandEncoder.endEncoding()
commandBuffer.commit()
commandBuffer.waitUntilCompleted()

// a. Get GPU data
// outVectorBuffer.contents() returns UnsafeMutablePointer roughly equivalent to char* in C
var data = NSData(bytesNoCopy: outVectorBuffer!.contents(),
                  length: myvector.count * MemoryLayout<Float>.size, freeWhenDone: false)
// b. prepare Swift array large enough to receive data from GPU
var finalResultArray = [Float](repeating: 0, count: myvector.count)


// c. get data from GPU into Swift array
data.getBytes(&finalResultArray, length:myvector.count * MemoryLayout<Float>.size)

print(finalResultArray)
// d. YOU'RE ALL SET!
