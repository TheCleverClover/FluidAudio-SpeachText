// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "WordAudioLab",
    platforms: [
        .macOS(.v14)
    ],
    dependencies: [
        .package(name: "FluidAudio", path: "../..")
    ],
    targets: [
        .executableTarget(
            name: "WordAudioLab",
            dependencies: [
                .product(name: "FluidAudio", package: "FluidAudio")
            ],
            path: "App"
        )
    ]
)
