import Accelerate
@preconcurrency import CoreML
import Foundation

public struct GraniteFeatureWindow {
    public let inputFeatures: MLMultiArray
    public let attentionMask: MLMultiArray
    public let validEncoderFrames: Int
    public let validMelFrames: Int
}

/// Granite NLE feature extraction matching the HF `NLEFeatureExtractor`.
///
/// This type owns mutable FFT scratch buffers and should stay actor-confined.
public final class GraniteFeatureExtractor {
    private let manifest: GraniteAsrManifest
    private let melFilterbankFlat: [Float]
    private let hannWindow: [Float]
    private let numFreqBins: Int
    private let windowOffset: Int
    private var fftSetup: vDSP_DFT_Setup?

    private var realIn: [Float]
    private var imagIn: [Float]
    private var realOut: [Float]
    private var imagOut: [Float]
    private var powerSpec: [Float]
    private var imagSq: [Float]
    private var frame: [Float]
    private var melFrame: [Float]
    private var paddedAudioScratch: [Float]
    private var logMelScratch: [Float]

    public init(modelDirectory: URL, manifest: GraniteAsrManifest) throws {
        self.manifest = manifest
        numFreqBins = manifest.nFFT / 2 + 1
        windowOffset = (manifest.nFFT - manifest.winLength) / 2
        hannWindow = Self.createHannWindow(length: manifest.winLength)

        let filterURL = modelDirectory.appendingPathComponent(manifest.melFilters)
        let expectedFilterCount = manifest.nMels * (manifest.nFFT / 2 + 1)
        let filterData = try Data(contentsOf: filterURL)
        guard filterData.count == expectedFilterCount * MemoryLayout<Float>.stride else {
            let expectedBytes = expectedFilterCount * MemoryLayout<Float>.stride
            throw GraniteAsrError.invalidAudio(
                "Granite mel filter file has \(filterData.count) bytes, expected \(expectedBytes)"
            )
        }

        var filters = [Float](repeating: 0, count: expectedFilterCount)
        _ = filters.withUnsafeMutableBytes { dst in
            filterData.copyBytes(to: dst)
        }
        melFilterbankFlat = filters

        guard let setup = vDSP_DFT_zop_CreateSetup(nil, vDSP_Length(manifest.nFFT), .FORWARD) else {
            throw GraniteAsrError.invalidAudio("Failed to create Granite FFT setup")
        }
        fftSetup = setup

        realIn = [Float](repeating: 0, count: manifest.nFFT)
        imagIn = [Float](repeating: 0, count: manifest.nFFT)
        realOut = [Float](repeating: 0, count: manifest.nFFT)
        imagOut = [Float](repeating: 0, count: manifest.nFFT)
        powerSpec = [Float](repeating: 0, count: numFreqBins)
        imagSq = [Float](repeating: 0, count: numFreqBins)
        frame = [Float](repeating: 0, count: manifest.nFFT)
        melFrame = [Float](repeating: 0, count: manifest.nMels)
        paddedAudioScratch = []
        logMelScratch = []
    }

    deinit {
        if let setup = fftSetup {
            vDSP_DFT_DestroySetup(setup)
        }
    }

    public func makeWindow<C>(
        audio: C,
        windowFrames: Int
    ) throws -> GraniteFeatureWindow where C: RandomAccessCollection, C.Element == Float {
        guard !audio.isEmpty else {
            throw GraniteAsrError.invalidAudio("Granite audio window is empty")
        }

        let validEncoderFrames = min(windowFrames, audio.count / (2 * manifest.hopLength))
        let validMelFrames = validEncoderFrames * 2
        guard validEncoderFrames > 0 else {
            throw GraniteAsrError.invalidAudio("Granite audio window is shorter than one encoder frame")
        }

        ensureFloatCapacity(&logMelScratch, count: validMelFrames * manifest.nMels)
        var maxLogMel = -Float.greatestFiniteMagnitude
        let paddedAudioCount = makeReflectPaddedAudio(audio)

        for frameIndex in 0 ..< validMelFrames {
            computeMelFrame(
                frameIndex: frameIndex,
                paddedAudioCount: paddedAudioCount,
                maxLogMel: &maxLogMel
            )
        }

        let (features, mask) = try makeInputArrays(
            windowFrames: windowFrames,
            validEncoderFrames: validEncoderFrames
        )
        fillStackedFeatures(
            features: features,
            mask: mask,
            validEncoderFrames: validEncoderFrames,
            maxLogMel: maxLogMel
        )

        return GraniteFeatureWindow(
            inputFeatures: features,
            attentionMask: mask,
            validEncoderFrames: validEncoderFrames,
            validMelFrames: validMelFrames
        )
    }

    private func makeInputArrays(
        windowFrames: Int,
        validEncoderFrames: Int
    ) throws -> (features: MLMultiArray, mask: MLMultiArray) {
        let features = try MLMultiArray(
            shape: [1, NSNumber(value: windowFrames), NSNumber(value: manifest.nMels * 2)],
            dataType: .float32
        )
        let mask = try MLMultiArray(
            shape: [1, NSNumber(value: windowFrames)],
            dataType: .int32
        )

        let featurePtr = features.dataPointer.bindMemory(to: Float.self, capacity: features.count)
        let maskPtr = mask.dataPointer.bindMemory(to: Int32.self, capacity: mask.count)
        let featureStride = manifest.nMels * 2
        let validFeatureCount = validEncoderFrames * featureStride
        if validFeatureCount < features.count {
            for index in validFeatureCount ..< features.count {
                featurePtr[index] = 0
            }
        }
        if validEncoderFrames < mask.count {
            for index in validEncoderFrames ..< mask.count {
                maskPtr[index] = 0
            }
        }
        return (features, mask)
    }

    private func fillStackedFeatures(
        features: MLMultiArray,
        mask: MLMultiArray,
        validEncoderFrames: Int,
        maxLogMel: Float
    ) {
        let featurePtr = features.dataPointer.bindMemory(to: Float.self, capacity: features.count)
        let maskPtr = mask.dataPointer.bindMemory(to: Int32.self, capacity: mask.count)
        let featureStride = manifest.nMels * 2
        let logFloor = maxLogMel - 8.0
        for encoderFrame in 0 ..< validEncoderFrames {
            let firstFrame = encoderFrame * 2
            let secondFrame = firstFrame + 1
            let dstBase = encoderFrame * featureStride

            for melIndex in 0 ..< manifest.nMels {
                let first = logMelScratch[firstFrame * manifest.nMels + melIndex]
                let second = logMelScratch[secondFrame * manifest.nMels + melIndex]
                featurePtr[dstBase + melIndex] = (max(first, logFloor) / 4.0) + 1.0
                featurePtr[dstBase + manifest.nMels + melIndex] = (max(second, logFloor) / 4.0) + 1.0
            }
            maskPtr[encoderFrame] = 1
        }
    }

    private func ensureFloatCapacity(_ values: inout [Float], count: Int) {
        if values.count < count {
            values = [Float](repeating: 0, count: count)
        }
    }

    private func makeReflectPaddedAudio<C>(_ audio: C) -> Int where C: RandomAccessCollection, C.Element == Float {
        let pad = manifest.nFFT / 2
        let paddedCount = audio.count + 2 * pad
        ensureFloatCapacity(&paddedAudioScratch, count: paddedCount)

        for paddedIndex in 0 ..< paddedCount {
            let sourceOffset = reflectIndex(paddedIndex - pad, count: audio.count)
            let sourceIndex = audio.index(audio.startIndex, offsetBy: sourceOffset)
            paddedAudioScratch[paddedIndex] = audio[sourceIndex]
        }
        return paddedCount
    }

    private func reflectIndex(_ index: Int, count: Int) -> Int {
        guard count > 1 else { return 0 }
        var reflected = index
        while reflected < 0 || reflected >= count {
            if reflected < 0 {
                reflected = -reflected
            } else {
                reflected = 2 * count - reflected - 2
            }
        }
        return reflected
    }

    private func computeMelFrame(
        frameIndex: Int,
        paddedAudioCount: Int,
        maxLogMel: inout Float
    ) {
        vDSP_vclr(&frame, 1, vDSP_Length(manifest.nFFT))
        let audioStart = frameIndex * manifest.hopLength + windowOffset
        let availableSamples = min(manifest.winLength, paddedAudioCount - audioStart)

        if availableSamples > 0 {
            paddedAudioScratch.withUnsafeBufferPointer { audioPtr in
                hannWindow.withUnsafeBufferPointer { windowPtr in
                    frame.withUnsafeMutableBufferPointer { framePtr in
                        vDSP_vmul(
                            audioPtr.baseAddress! + audioStart, 1,
                            windowPtr.baseAddress!, 1,
                            framePtr.baseAddress! + windowOffset, 1,
                            vDSP_Length(availableSamples)
                        )
                    }
                }
            }
        }

        computePowerSpectrumInPlace()
        melFilterbankFlat.withUnsafeBufferPointer { filterPtr in
            powerSpec.withUnsafeBufferPointer { specPtr in
                melFrame.withUnsafeMutableBufferPointer { outPtr in
                    vDSP_mmul(
                        filterPtr.baseAddress!, 1,
                        specPtr.baseAddress!, 1,
                        outPtr.baseAddress!, 1,
                        vDSP_Length(manifest.nMels),
                        vDSP_Length(1),
                        vDSP_Length(numFreqBins)
                    )
                }
            }
        }

        let dstBase = frameIndex * manifest.nMels
        for melIndex in 0 ..< manifest.nMels {
            let value = log10f(max(melFrame[melIndex], 1e-10))
            logMelScratch[dstBase + melIndex] = value
            maxLogMel = max(maxLogMel, value)
        }
    }

    private func computePowerSpectrumInPlace() {
        guard let setup = fftSetup else { return }

        frame.withUnsafeBufferPointer { src in
            realIn.withUnsafeMutableBufferPointer { dst in
                _ = memcpy(dst.baseAddress!, src.baseAddress!, manifest.nFFT * MemoryLayout<Float>.size)
            }
        }
        vDSP_vclr(&imagIn, 1, vDSP_Length(manifest.nFFT))
        vDSP_DFT_Execute(setup, realIn, imagIn, &realOut, &imagOut)

        vDSP_vsq(realOut, 1, &powerSpec, 1, vDSP_Length(numFreqBins))
        vDSP_vsq(imagOut, 1, &imagSq, 1, vDSP_Length(numFreqBins))
        vDSP_vadd(powerSpec, 1, imagSq, 1, &powerSpec, 1, vDSP_Length(numFreqBins))
    }

    private static func createHannWindow(length: Int) -> [Float] {
        var window = [Float](repeating: 0, count: length)
        let windowLength = Float(length)
        for index in 0 ..< length {
            window[index] = 0.5 * (1.0 - cosf(2.0 * .pi * Float(index) / windowLength))
        }
        return window
    }
}
