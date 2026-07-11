import SwiftUI

struct AudioWaveform: View {
    let samples: [Float]

    var body: some View {
        Canvas { context, size in
            guard !samples.isEmpty, size.width > 0, size.height > 0 else { return }
            let columns = max(1, min(Int(size.width), 240))
            let samplesPerColumn = max(1, samples.count / columns)
            let midY = size.height / 2
            var path = Path()

            for column in 0..<columns {
                let start = column * samplesPerColumn
                let end = min(samples.count, start + samplesPerColumn)
                guard start < end else { continue }
                var peak: Float = 0
                for index in start..<end {
                    peak = max(peak, abs(samples[index]))
                }
                let normalizedPeak = min(1, CGFloat(peak) * 2.2)
                let halfHeight = max(1, normalizedPeak * midY)
                let x = CGFloat(column) / CGFloat(max(1, columns - 1)) * size.width
                path.move(to: CGPoint(x: x, y: midY - halfHeight))
                path.addLine(to: CGPoint(x: x, y: midY + halfHeight))
            }

            context.stroke(path, with: .color(.accentColor.opacity(0.8)), lineWidth: 1)
        }
        .background(.quaternary.opacity(0.35), in: RoundedRectangle(cornerRadius: 6))
    }
}
