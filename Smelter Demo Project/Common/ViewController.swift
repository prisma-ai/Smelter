//
//  ViewController.swift
//  Smelter Demo
//
//  Created by Eugene Bokhan on 04/05/2019.
//  Copyright © 2019 Eugene Bokhan. All rights reserved.
//


import Smelter
import Alloy
import MetalPerformanceShaders

#if os(iOS)
import UIKit
typealias UINSViewController = UIViewController
#elseif os(macOS)
import Cocoa
typealias UINSViewController = NSViewController
#endif

class ViewController: UINSViewController {

    // MARK: - Internal Types

    private enum Error: Swift.Error {
        case cgImageCreationFailed
        case graphResultReadingFailed
    }

    private typealias Prediction = (label: String, probability: Float)

    // MARK: - Inteface Elements

    #if os(iOS)
    @IBOutlet var imageView: UIImageView!
    @IBOutlet var predictionTextView: UITextView!
    #elseif os(macOS)
    @IBOutlet var imageView: NSImageView!
    @IBOutlet var predictionTextView: NSTextField!
    #endif

    // MARK: - Interface Actions

    @IBAction func predictAction(_ sender: Any) {
        do {
            try self.runPredictionForImage(at: self.currentImageIndex)

            let imageCount = self.testImages.count
            self.currentImageIndex = self.currentImageIndex == imageCount - 1 ?
                0 : self.currentImageIndex + 1
        } catch {
            fatalError(error.localizedDescription)
        }
    }

    // MARK: - Properties

    private var mobileNetEncoder: MobileNetEncoder!
    private var metalContext: MTLContext!

    private let testImages = [#imageLiteral(resourceName: "Dog"), #imageLiteral(resourceName: "Cat"), #imageLiteral(resourceName: "Car"), #imageLiteral(resourceName: "Flowers")]
    private var currentImageIndex: Int = 0

    // MARK: - Life Cycle

    override func viewDidLoad() {
        super.viewDidLoad()
        // Setup
        do { try self.setup() }
        catch { fatalError(error.localizedDescription) }
        // Run prediction for first image
        self.predictAction(self)
    }

    // MARK: - Setup

    func setup() throws {
        self.setupContext()
        try self.setupEncoders()
    }

    func setupContext() {
        #if os(iOS)
        self.metalContext = try! .init()
        #elseif os(macOS)
        self.metalContext = try! .init(device: Metal.lowPowerDevice ?? Metal.device)
        #endif
    }

    func setupEncoders() throws {
        let modelData = try Data(contentsOf: ONNXModels.mobilenetv2_fused)
        let configuration = ONNXGraph.Configuration(inputConstraint: .forceInputScale(scale: .bilinear))
        self.mobileNetEncoder = try MobileNetEncoder(context: self.metalContext,
                                                     modelData: modelData,
                                                     configuration: configuration)
    }

    // MARK: - Prediction

    func runPredictionForImage(at index: Int) throws {
        /// Create a CGImage.
        let image = self.testImages[index]

        guard let cgImage = image.cgImage
        else { throw Error.cgImageCreationFailed }

        /// Create texture from CGImage.
        ///
        /// Make usage `.shaderWrite` for normalization preprocessing
        /// and `.shaderRead` for graph inference.
        let inputTexture = try self.metalContext.texture(from: cgImage,
                                                         usage: [.shaderWrite, .shaderRead])
        var outputImage: MPSImage!
        /// Run the prediction on the created texture.
        try self.metalContext.scheduleAndWait { commandBuffer in
            outputImage = try self.mobileNetEncoder.encode(source: inputTexture,
                                                           in: commandBuffer)
        }
        /// Read the prediction results.
        guard let output = outputImage.toFloatArray()
        else { throw Error.graphResultReadingFailed }
        /// Create a `[label: probability]` dictionary.
        var predictionDictionary: [String: Float] = [:]
        for (index, element) in classificationLabels.enumerated() {
            predictionDictionary[element] = output[index]
        }
        /// Show the results.
        self.show(results: self.top(5, predictionDictionary),
                  imageIndex: index)
    }

    private func show(results: [Prediction], imageIndex: Int) {
        var predictionResults: [String] = []
        for (index, prediction) in results.enumerated() {
            predictionResults.append(String(format: "%d: %@ (%3.2f%%)",
                                            index + 1,
                                            prediction.label,
                                            prediction.probability))
        }
        let predictionResultsString = predictionResults.joined(separator: "\n\n")

        #if os(iOS)
        self.predictionTextView.text = predictionResultsString
        #elseif os(macOS)
        self.predictionTextView.stringValue = predictionResultsString
        #endif

        self.imageView.image = self.testImages[imageIndex]
    }

    private func top(_ k: Int, _ prob: [String: Float]) -> [Prediction] {
        #if DEBUG
        precondition(k <= prob.count)
        #endif

        return Array(prob.map { x in (x.key, x.value) }
             .sorted(by: { a, b -> Bool in a.1 > b.1 })
             .prefix(through: k - 1))
    }

}

