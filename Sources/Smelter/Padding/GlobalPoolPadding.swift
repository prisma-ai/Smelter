import MetalPerformanceShaders

class GlobalPoolPadding: NSObject, MPSNNPadding {
    override init() {}

    // MARK: MPSNNPadding

    func paddingMethod() -> MPSNNPaddingMethod {
        .custom
    }

    func label() -> String {
        "PyTorch Global Pool Padding rule"
    }

    func destinationImageDescriptor(
        forSourceImages _: [MPSImage],
        sourceStates _: [MPSState]?,
        for _: MPSKernel,
        suggestedDescriptor inDescriptor: MPSImageDescriptor
    ) -> MPSImageDescriptor {
        inDescriptor.width = 1
        inDescriptor.height = 1

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
}
