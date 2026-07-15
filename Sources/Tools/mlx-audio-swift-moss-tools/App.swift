// MOSS-Transcribe-Diarize maintenance CLI.
//
//   quantize  Convert a bf16 MOSS checkpoint directory to MLX-quantized
//             safetensors that `MossTranscribeDiarizeModel.fromModelDirectory`
//             loads directly (language-model Linear/Embedding weights only;
//             the audio tower and adaptor stay dense).
//   bench     Decode one 16 kHz mono WAV in a single pass through the model
//             with optional KV-cache quantization and write a JSON report
//             (labels/segments/wall time/peak process footprint).
//
// Both subcommands operate on LOCAL model directories only — no network.

import AVFAudio
import Foundation
@preconcurrency import MLX
import MLXAudioSTT

enum ToolError: Error, CustomStringConvertible {
    case usage(String)
    case invalidInput(String)

    var description: String {
        switch self {
        case .usage(let text): "usage: \(text)"
        case .invalidInput(let text): text
        }
    }
}

func parseFlags(_ arguments: ArraySlice<String>) throws -> [String: String] {
    var flags: [String: String] = [:]
    var iterator = arguments.makeIterator()
    while let key = iterator.next() {
        guard key.hasPrefix("--"), let value = iterator.next() else {
            throw ToolError.invalidInput("malformed argument: \(key)")
        }
        flags[String(key.dropFirst(2))] = value
    }
    return flags
}

// MARK: - quantize

/// Sidecar files copied verbatim so the converted directory is
/// self-contained for the loader (tokenizer, processor, generation config).
let sidecarFiles = [
    "added_tokens.json", "generation_config.json", "merges.txt",
    "preprocessor_config.json", "processor_config.json",
    "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json",
    "vocab.json",
]

func runQuantize(flags: [String: String]) throws {
    guard let src = flags["src"], let dst = flags["dst"], let bitsText = flags["bits"],
          let bits = Int(bitsText), bits == 4 || bits == 8
    else {
        throw ToolError.usage("moss-tools quantize --src DIR --dst DIR --bits 4|8 [--group-size 64]")
    }
    let groupSize = Int(flags["group-size"] ?? "64") ?? 64

    let srcURL = URL(fileURLWithPath: src)
    let dstURL = URL(fileURLWithPath: dst)
    try FileManager.default.createDirectory(at: dstURL, withIntermediateDirectories: true)

    var weights: [String: MLXArray] = [:]
    let files = try FileManager.default.contentsOfDirectory(at: srcURL, includingPropertiesForKeys: nil)
        .filter { $0.pathExtension == "safetensors" }
        .sorted { $0.lastPathComponent < $1.lastPathComponent }
    guard !files.isEmpty else { throw ToolError.invalidInput("no safetensors under \(src)") }
    for file in files {
        weights.merge(try MLX.loadArrays(url: file)) { _, new in new }
    }

    var converted: [String: MLXArray] = [:]
    var quantizedCount = 0
    for (key, rawValue) in weights {
        var value = rawValue
        // Once any ".scales" key exists, the Swift loader treats the whole
        // checkpoint as MLX-native and skips its conv transpose — so the
        // converter must bake that layout in.
        if key.hasPrefix("model.whisper_encoder."), key.contains("conv"),
           key.hasSuffix(".weight"), value.ndim == 3 {
            value = value.transposed(0, 2, 1)
        }
        // Quantize only the language model's 2-D Linear/Embedding weights;
        // norms are 1-D and the audio tower/adaptor stay dense.
        if key.hasPrefix("model.language_model."), key.hasSuffix(".weight"),
           value.ndim == 2, value.dim(-1) % groupSize == 0 {
            let (wq, scales, biases) = MLX.quantized(value, groupSize: groupSize, bits: bits)
            let base = String(key.dropLast(".weight".count))
            converted[key] = wq
            converted[base + ".scales"] = scales
            if let biases { converted[base + ".biases"] = biases }
            quantizedCount += 1
        } else {
            converted[key] = value
        }
    }
    eval(Array(converted.values))
    try MLX.save(
        arrays: converted, url: dstURL.appendingPathComponent("model-00000-of-00001.safetensors"))

    // config.json with the quantization block the loader decodes.
    let configData = try Data(contentsOf: srcURL.appendingPathComponent("config.json"))
    var config = try JSONSerialization.jsonObject(with: configData) as? [String: Any] ?? [:]
    config["quantization"] = ["group_size": groupSize, "bits": bits]
    config["quantization_config"] = ["group_size": groupSize, "bits": bits]
    let updated = try JSONSerialization.data(
        withJSONObject: config, options: [.prettyPrinted, .sortedKeys])
    try updated.write(to: dstURL.appendingPathComponent("config.json"))

    for name in sidecarFiles {
        let source = srcURL.appendingPathComponent(name)
        guard FileManager.default.fileExists(atPath: source.path) else { continue }
        let target = dstURL.appendingPathComponent(name)
        try? FileManager.default.removeItem(at: target)
        try FileManager.default.copyItem(at: source, to: target)
    }
    print("quantized \(quantizedCount) tensors to \(bits)-bit (group size \(groupSize))")
}

// MARK: - bench

func loadWAVSamples(path: String) throws -> [Float] {
    let file = try AVAudioFile(forReading: URL(fileURLWithPath: path))
    let format = file.processingFormat
    guard Int(format.sampleRate) == 16_000 else {
        throw ToolError.invalidInput("expected 16 kHz input, got \(Int(format.sampleRate))")
    }
    let frameCount = AVAudioFrameCount(file.length)
    guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
        throw ToolError.invalidInput("audio buffer allocation failed")
    }
    try file.read(into: buffer)
    guard let channelData = buffer.floatChannelData else {
        throw ToolError.invalidInput("non-float audio buffer")
    }
    return Array(UnsafeBufferPointer(start: channelData[0], count: Int(buffer.frameLength)))
}

/// Peak physical-footprint sampler (bytes), polled on a background thread.
final class FootprintSampler: @unchecked Sendable {
    private var peak: UInt64 = 0
    private var running = true
    private let lock = NSLock()

    static func currentFootprint() -> UInt64 {
        var info = rusage_info_current()
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: rusage_info_t?.self, capacity: 1) {
                proc_pid_rusage(getpid(), RUSAGE_INFO_CURRENT, $0)
            }
        }
        return result == 0 ? info.ri_phys_footprint : 0
    }

    func start() {
        Thread.detachNewThread { [self] in
            while lock.withLock({ running }) {
                let now = Self.currentFootprint()
                lock.withLock { if now > peak { peak = now } }
                Thread.sleep(forTimeInterval: 2.0)
            }
        }
    }

    func stop() -> UInt64 {
        lock.withLock {
            running = false
            return peak
        }
    }
}

func runBench(flags: [String: String]) async throws {
    guard let modelDir = flags["model-dir"], let wav = flags["wav"], let out = flags["out"] else {
        throw ToolError.usage(
            "moss-tools bench --model-dir DIR --wav FILE.wav --out REPORT.json "
                + "[--kv-bits N] [--kv-group-size 64] [--max-tokens N]")
    }
    let kvBits = flags["kv-bits"].flatMap(Int.init)
    let kvGroupSize = Int(flags["kv-group-size"] ?? "64") ?? 64
    let quantizedKVStart = Int(flags["quantized-kv-start"] ?? "0") ?? 0

    let samples = try loadWAVSamples(path: wav)
    let durationSeconds = Double(samples.count) / 16_000.0
    let maxTokens = flags["max-tokens"].flatMap(Int.init)
        ?? min(131_072, max(1_024, Int(durationSeconds) * 8))
    print("bench: wav=\(wav) duration=\(durationSeconds)s")
    print("bench: kvBits=\(kvBits.map(String.init) ?? "nil") maxTokens=\(maxTokens)")

    let sampler = FootprintSampler()
    sampler.start()

    let loadStart = Date()
    let model = try await MossTranscribeDiarizeModel.fromModelDirectory(
        URL(fileURLWithPath: modelDir))
    let loadSeconds = Date().timeIntervalSince(loadStart)
    print("bench: model loaded in \(loadSeconds)s")

    let decodeStart = Date()
    let output = model.generate(
        audio: MLXArray(samples),
        maxTokens: maxTokens,
        temperature: 0.0,
        kvBits: kvBits,
        kvGroupSize: kvGroupSize,
        quantizedKVStart: quantizedKVStart)
    let wallSeconds = Date().timeIntervalSince(decodeStart)
    let peakFootprint = sampler.stop()

    var report: [String: Any] = [
        "wav": wav,
        "modelDir": modelDir,
        "kvBits": kvBits as Any,
        "kvGroupSize": kvGroupSize,
        "durationSeconds": durationSeconds,
        "maxTokens": maxTokens,
        "loadSeconds": loadSeconds,
        "wallSeconds": wallSeconds,
        "promptTokens": output.promptTokens,
        "generationTokens": output.generationTokens,
        "generationTps": output.generationTps,
        "gpuPeakGB": output.peakMemoryUsage,
        "processPeakFootprintBytes": peakFootprint,
        "text": output.text,
    ]
    if let segments = output.segments {
        report["segments"] = segments.map { segment -> [String: Any] in
            segment.reduce(into: [:]) { $0[$1.key] = $1.value }
        }
    }
    let data = try JSONSerialization.data(
        withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
    try data.write(to: URL(fileURLWithPath: out))
    print("bench: wall=\(wallSeconds)s genTokens=\(output.generationTokens) "
        + "gpuPeakGB=\(output.peakMemoryUsage) footprintGB=\(Double(peakFootprint) / 1e9)")
    print("bench: report written to \(out)")
}

// MARK: - entry

let arguments = CommandLine.arguments
guard arguments.count >= 2 else {
    print("usage: moss-tools <quantize|bench> [flags]")
    exit(2)
}
do {
    let flags = try parseFlags(arguments.dropFirst(2))
    switch arguments[1] {
    case "quantize":
        try runQuantize(flags: flags)
    case "bench":
        try await runBench(flags: flags)
    default:
        throw ToolError.usage("moss-tools <quantize|bench> [flags]")
    }
} catch {
    FileHandle.standardError.write(Data("\(error)\n".utf8))
    exit(1)
}
