extension Array {
    public func reformatConvWeight(outputChannels: Int,
                                   inputChannels: Int,
                                   kernelHeight: Int,
                                   kernelWidth: Int,
                                   isTranspose: Bool) -> [Element] {
        var data: [Element] = [Element](repeating: self[0], count: self.count)
        for oc in 0..<outputChannels {
            for ic in 0..<inputChannels {
                for kh in 0..<kernelHeight {
                    for kw in 0..<kernelWidth {
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
    internal subscript (safe index: Index) -> Element? {
        return self.indices.contains(index) ? self[index] : nil
    }
}
