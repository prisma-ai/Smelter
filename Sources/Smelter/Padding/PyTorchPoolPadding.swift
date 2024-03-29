import MetalPerformanceShaders

class PyTorchPoolPadding: NSObject, MPSNNPadding {
    let kernelWidth: Int
    let kernelHeight: Int
    let paddingWidth: Int
    let paddingHeight: Int
    let strideInPixelsX: Int
    let strideInPixelsY: Int

    init(
        kernelWidth: Int,
        kernelHeight: Int,
        paddingWidth: Int,
        paddingHeight: Int,
        strideInPixelsX: Int,
        strideInPixelsY: Int
    ) {
        self.kernelWidth = kernelWidth
        self.kernelHeight = kernelHeight
        self.paddingWidth = paddingWidth
        self.paddingHeight = paddingHeight
        self.strideInPixelsX = strideInPixelsX
        self.strideInPixelsY = strideInPixelsY
    }

    // MARK: MPSNNPadding

    func paddingMethod() -> MPSNNPaddingMethod {
        .custom
    }

    func label() -> String {
        "PyTorch Pool Padding rule"
    }

    func destinationImageDescriptor(
        forSourceImages sourceImages: [MPSImage],
        sourceStates _: [MPSState]?,
        for kernel: MPSKernel,
        suggestedDescriptor inDescriptor: MPSImageDescriptor
    ) -> MPSImageDescriptor {
        let cnnKernel = kernel as! MPSCNNKernel

        let inputWidth = sourceImages[0].width
        let inputHeight = sourceImages[0].height

        // this is an offical formula from PyTorch documentation
        let paddedSize = self.paddedSize(
            inputWidth: inputWidth,
            inputHeight: inputHeight
        )
        inDescriptor.width = paddedSize.width
        inDescriptor.height = paddedSize.height

        // The correction needed to adjust from position of left edge to center per MPSOffset definition
        let correctionX = self.kernelWidth / 2
        let correctionY = self.kernelHeight / 2

        let readSizeX = (inDescriptor.width - 1) * self.strideInPixelsX + self.kernelWidth
        let readSizeY = (inDescriptor.height - 1) * self.strideInPixelsY + self.kernelHeight

        let extraSizeX = readSizeX - inputWidth
        let extraSizeY = readSizeY - inputHeight

        let centeringPolicy = 0

        let leftExtraPixels = (extraSizeX + centeringPolicy) / 2
        let topExtraPixels = (extraSizeY + centeringPolicy) / 2

        cnnKernel.offset = .init(
            x: correctionX - leftExtraPixels,
            y: correctionY - topExtraPixels,
            z: 0
        )

        return inDescriptor
    }

    // MARK: NSCoding

    static var supportsSecureCoding: Bool {
        false
    }

    func encode(with _: NSCoder) {
        fatalError("NSCoding is not supported yet")
    }

    required init?(coder _: NSCoder) {
        fatalError("NSCoding is not supported yet")
    }

    func paddedSize(
        inputWidth: Int,
        inputHeight: Int
    ) -> (width: Int, height: Int) {
        let width = Int(Float(inputWidth + 2 * self.paddingWidth
                - self.kernelWidth) / Float(self.strideInPixelsX) + 1.0)
        let height = Int(Float(inputHeight + 2 * self.paddingHeight
                - self.kernelHeight) / Float(self.strideInPixelsY) + 1.0)
        return (width, height)
    }
}
