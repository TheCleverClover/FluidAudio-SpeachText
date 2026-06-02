import Foundation
@preconcurrency import CoreML

/// Slices full-utterance preprocessor output into NeMo `CacheAwareStreamingAudioBuffer` chunks.
struct NemotronCacheAwareMelChunk: Sendable {
    let processedSignal: MLMultiArray
    /// Valid mel frames in this chunk (NeMo `processed_signal_length`).
    let validMelFrames: Int
    /// Padded encoder input width (`processed_signal_init` / `processed_signal_step`).
    let targetMelFrames: Int
    let isFirstChunk: Bool
    let isLastChunk: Bool
}

enum NemotronCacheAwareMelChunker {
    static func makeChunks(
        from processedSignal: MLMultiArray,
        validFrames: Int,
        config: NemotronStreamingConfig
    ) throws -> [NemotronCacheAwareMelChunk] {
        guard validFrames > 0 else {
            return []
        }

        let initWidth = config.initTotalMelFrames
        let stepWidth = config.stepTotalMelFrames
        let firstShift = config.streamingShiftSizes.first ?? config.shiftMelFrames
        let steadyShift = config.streamingShiftSizes.count > 1
            ? config.streamingShiftSizes[1]
            : config.shiftMelFrames

        var chunks: [NemotronCacheAwareMelChunk] = []
        var start = 0
        var stepIndex = 0

        // Each split-encoder chunk emits ~`shift` mel of NEW content; the wider
        // window (init/step width) only provides right-context lookahead. So the
        // stream must keep stepping by `shift` until the hop start covers all
        // valid frames (NeMo CacheAwareStreamingAudioBuffer contract). Terminating
        // on window-end would drop the tail that lives in the final lookahead.
        while start < validFrames {
            let width = stepIndex == 0 ? initWidth : stepWidth
            let shift = stepIndex == 0 ? firstShift : steadyShift
            let sliceEnd = min(start + width, validFrames)
            let isLast = (start + shift) >= validFrames

            let validInChunk = max(0, sliceEnd - start)
            let chunkMel = try sliceMel(
                from: processedSignal,
                startFrame: start,
                endFrame: sliceEnd,
                targetWidth: width,
                melFeatures: config.melFeatures
            )

            chunks.append(
                NemotronCacheAwareMelChunk(
                    processedSignal: chunkMel,
                    validMelFrames: validInChunk,
                    targetMelFrames: width,
                    isFirstChunk: stepIndex == 0,
                    isLastChunk: isLast
                )
            )

            if isLast {
                break
            }

            start += shift
            stepIndex += 1
        }

        return chunks
    }

    private static func sliceMel(
        from source: MLMultiArray,
        startFrame: Int,
        endFrame: Int,
        targetWidth: Int,
        melFeatures: Int
    ) throws -> MLMultiArray {
        let result = try MLMultiArray(
            shape: [1, NSNumber(value: melFeatures), NSNumber(value: targetWidth)],
            dataType: .float32
        )
        result.reset(to: 0)

        let copyFrames = max(0, endFrame - startFrame)
        guard copyFrames > 0 else {
            return result
        }

        let srcPtr = source.dataPointer.bindMemory(to: Float.self, capacity: source.count)
        let dstPtr = result.dataPointer.bindMemory(to: Float.self, capacity: result.count)
        let srcStride0 = source.strides[0].intValue
        let srcStride1 = source.strides[1].intValue
        let srcStride2 = source.strides[2].intValue
        let dstStride0 = result.strides[0].intValue
        let dstStride1 = result.strides[1].intValue
        let dstStride2 = result.strides[2].intValue

        for mel in 0..<melFeatures {
            for t in 0..<copyFrames {
                let srcIdx = 0 * srcStride0 + mel * srcStride1 + (startFrame + t) * srcStride2
                let dstIdx = 0 * dstStride0 + mel * dstStride1 + t * dstStride2
                dstPtr[dstIdx] = srcPtr[srcIdx]
            }
        }

        return result
    }
}
