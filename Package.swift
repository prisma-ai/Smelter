// swift-tools-version:5.3

import PackageDescription

let package = Package(
    name: "Smelter",
    platforms: [
        .iOS(SupportedPlatform.IOSVersion.v11),
        .macOS(.v10_13)
    ],
    products: [
        .library(name: "Smelter",
                 targets: ["Smelter"]),
    ],
    dependencies: [
        .package(url: "https://github.com/s1ddok/Alloy.git",
                 .branch("swiftpm2")),
        .package(name: "SwiftProtobuf",
                 url: "https://github.com/apple/swift-protobuf.git",
                 .exact("1.7.0"))
    ],
    targets: [
        .target(name: "Smelter",
                dependencies: ["Alloy", "SwiftProtobuf"],
                path: "Sources",
                exclude: [],
                sources: nil,
                publicHeadersPath: nil,
                cSettings: nil,
                cxxSettings: nil,
                swiftSettings: nil,
                linkerSettings: [
                    .linkedFramework("Metal"),
                    .linkedFramework("MetalPerformanceShaders"),
                    .linkedFramework("CoreGraphics")
                ])
    ]
)
