import Accelerate

@available(iOS 14, macOS 11, *)
extension BNNS {
    static func transpose(
        array: [Float],
        shape: [Int],
        swizzlePlan: [(Int, Int)]
    ) throws -> [Float] {
        var x = array
        var y: [Float] = Array(repeating: 0, count: array.count)

        var inputShape = shape

        func bnnsShape(_ v: [Int]) -> BNNS.Shape {
            return .tensor4DFirstMajor(v[0], v[1], v[2], v[3])
        }

        try swizzlePlan.forEach { transposition in
            try x.withUnsafeMutableBufferPointer { inputPointer in
                try y.withUnsafeMutableBufferPointer { outputPointer in
                    var outputShape = inputShape
                    outputShape[transposition.0] = inputShape[transposition.1]
                    outputShape[transposition.1] = inputShape[transposition.0]

                    try BNNS.transpose(
                        input: .init(
                            data: inputPointer,
                            shape: bnnsShape(inputShape)
                        )!,
                        output: .init(
                            data: outputPointer,
                            shape: bnnsShape(outputShape)
                        )!,
                        firstTransposeAxis: transposition.0,
                        secondTransposeAxis: transposition.1,
                        filterParameters: nil
                    )

                    inputShape = outputShape
                }
            }

            swap(&x, &y)
        }

        return x
    }
}

extension Array {
    func reformatingConvolutionWeight(
        outputChannels: Int,
        inputChannels: Int,
        kernelHeight: Int,
        kernelWidth: Int,
        isTranspose: Bool
    ) -> [Element] {
        var data = [Element](
            repeating: self[0],
            count: self.count
        )
        for oc in 0 ..< outputChannels {
            for ic in 0 ..< inputChannels {
                for kh in 0 ..< kernelHeight {
                    for kw in 0 ..< kernelWidth {
                        let inputIdx: Int
                        let outputIdx: Int

                        if isTranspose {
                            inputIdx = ic * outputChannels * kernelHeight * kernelWidth
                                + oc * kernelHeight * kernelWidth
                                + kh * kernelWidth + kw
                            outputIdx = oc * kernelHeight * kernelWidth * inputChannels
                                + (kernelHeight - 1 - kh) * kernelWidth * inputChannels
                                + (kernelWidth - 1 - kw) * inputChannels + ic
                        } else {
                            inputIdx = oc * inputChannels * kernelHeight * kernelWidth
                                + ic * kernelHeight * kernelWidth
                                + kh * kernelWidth + kw
                            outputIdx = oc * inputChannels * kernelHeight * kernelWidth
                                + kh * kernelWidth * inputChannels
                                + kw * inputChannels + ic
                        }

                        data[outputIdx] = self[inputIdx]
                    }
                }
            }
        }

        return data
    }

    /// Returns the element at the specified index if it is within bounds, otherwise nil.
    subscript(safe index: Index) -> Element? {
        self.indices.contains(index) ? self[index] : nil
    }

    var unsafeMutablePointer: UnsafeMutablePointer<Element>? {
        self.withUnsafeBufferPointer {
            UnsafeMutablePointer(mutating: $0.baseAddress)
        }
    }

    var unsafeMutableRawPointer: UnsafeMutableRawPointer? {
        self.withUnsafeBufferPointer {
            UnsafeMutableRawPointer(mutating: $0.baseAddress)
        }
    }
}
