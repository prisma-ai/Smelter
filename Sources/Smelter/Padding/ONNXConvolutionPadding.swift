import MetalPerformanceShaders

typealias Kernel = (height: Int, width: Int)
typealias Strides = (height: Int, width: Int)
typealias Dilations = (height: Int, width: Int)
typealias Pads = (top: Int, left: Int, bottom: Int, right: Int)
typealias Padding = (height: Int, width: Int)
typealias Scales = (height: Int, width: Int)

@objc(ONNX_ConvolutionPadding) class ONNX_ConvolutionPadding: NSObject, MPSNNPadding {
    let kernel: Kernel
    let dilations: Dilations
    let strides: Strides
    let pads: Pads
    let outputPadding: Padding
    let isTranspose: Bool

    init(
        kernel: Kernel,
        strides: Strides,
        dilations: Dilations,
        pads: Pads,
        outputPadding: Padding,
        isTranspose: Bool
    ) {
        self.kernel = kernel
        self.dilations = dilations
        self.strides = strides
        self.pads = pads
        self.outputPadding = outputPadding
        self.isTranspose = isTranspose
    }

    required convenience init?(coder aDecoder: NSCoder) {
        guard
            let data = aDecoder.decodeData(),
            let other = NSKeyedUnarchiver.unarchiveObject(with: data) as? ONNX_ConvolutionPadding
        else { return nil }
        self.init(
            kernel: other.kernel,
            strides: other.strides,
            dilations: other.dilations,
            pads: other.pads,
            outputPadding: other.outputPadding,
            isTranspose: other.isTranspose
        )
    }

    func encode(with aCoder: NSCoder) {
        aCoder.encode(NSKeyedArchiver.archivedData(withRootObject: self))
    }

    func paddingMethod() -> MPSNNPaddingMethod {
        [.custom]
    }

    func destinationImageDescriptor(
        forSourceImages sourceImages: [MPSImage],
        sourceStates _: [MPSState]?,
        for kernel: MPSKernel,
        suggestedDescriptor inDescriptor: MPSImageDescriptor
    ) -> MPSImageDescriptor {
        let inputHeight = sourceImages[0].height
        let inputWidth = sourceImages[0].width

        if self.isTranspose {
            let conv = kernel as! MPSCNNConvolutionTranspose
            conv.offset = MPSOffset(x: 0, y: 0, z: 0)
            conv.edgeMode = .zero
            conv.kernelOffsetX = self.kernel.width / 2 - self.kernel.width + 1 + self.pads.left
            conv.kernelOffsetY = self.kernel.height / 2 - self.kernel.height + 1 + self.pads.top
        } else {
            let conv = kernel as! MPSCNNConvolution
            conv.offset = MPSOffset(
                x: self.kernel.width / 2 - self.pads.left,
                y: self.kernel.height / 2 - self.pads.top,
                z: 0
            )
            conv.edgeMode = .zero
        }
        let paddedSize = self.paddedSize(
            inputWidth: inputWidth,
            inputHeight: inputHeight
        )
        inDescriptor.height = paddedSize.height
        inDescriptor.width = paddedSize.width

        return inDescriptor
    }

    func paddedSize(
        inputWidth: Int,
        inputHeight: Int
    ) -> (width: Int, height: Int) {
        let height: Int
        let width: Int
        if self.isTranspose {
            height = (inputHeight - 1) * self.strides.height
                - self.pads.top - self.pads.bottom
                + self.kernel.height + self.outputPadding.height
            width = (inputWidth - 1) * self.strides.width
                - self.pads.left - self.pads.right
                + self.kernel.width + self.outputPadding.width
        } else {
            height = (inputHeight + self.pads.top
                + self.pads.bottom - self.kernel.height)
                / self.strides.height + 1
            width = (inputWidth + self.pads.left
                + self.pads.right - self.kernel.width)
                / self.strides.width + 1
        }
        return (width, height)
    }

    static var supportsSecureCoding: Bool = true
}
