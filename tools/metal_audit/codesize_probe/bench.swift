// R21 instruction-fetch probe bench (not production). Times each cs_<B> kernel of a metallib built
// from gen.py with GPU timestamps, interleaved per round (odd rounds reversed), and prints ns/row.
// usage: bench LIB.metallib TOTAL_BLOCKS ROWS ROUNDS TGMEM_BYTES B1,B2,...
import Foundation
import Metal

let a = CommandLine.arguments
guard a.count >= 7 else { print("usage: bench LIB TOTAL ROWS ROUNDS TGMEM B1,B2,..."); exit(2) }
let total = UInt32(a[2])!, rows = Int(a[3])!, rounds = Int(a[4])!, tgmem = Int(a[5])!
let sizes = a[6].split(separator: ",").map { Int($0)! }
let dev = MTLCreateSystemDefaultDevice()!
let lib = try! dev.makeLibrary(URL: URL(fileURLWithPath: a[1]))
let q = dev.makeCommandQueue()!
let accs = 8
let inp = dev.makeBuffer(length: rows * accs * 4, options: .storageModeShared)!
let ip = inp.contents().bindMemory(to: UInt32.self, capacity: rows * accs)
var s: UInt64 = 0x9e3779b97f4a7c15
for i in 0..<(rows * accs) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; ip[i] = UInt32(truncatingIfNeeded: s) }
var psos: [(Int, MTLComputePipelineState)] = []
for b in sizes {
  let t0 = Date()
  let f = lib.makeFunction(name: "cs_\(b)")!
  let p = try! dev.makeComputePipelineState(function: f)
  print(String(format: "R21CS pso cs_%d: %.1f s, maxThreads %d, width %d", b, Date().timeIntervalSince(t0),
               p.maxTotalThreadsPerThreadgroup, p.threadExecutionWidth))
  psos.append((b, p))
}
let outs = psos.map { _ in dev.makeBuffer(length: rows * 4, options: .storageModeShared)! }
func run(_ i: Int) -> Double {
  let (b, p) = psos[i]
  var reps = total / UInt32(b)
  let cb = q.makeCommandBuffer()!
  let e = cb.makeComputeCommandEncoder()!
  e.setComputePipelineState(p)
  e.setBuffer(inp, offset: 0, index: 0)
  e.setBuffer(outs[i], offset: 0, index: 1)
  e.setBytes(&reps, length: 4, index: 2)
  e.setThreadgroupMemoryLength(max(16, tgmem), index: 0)
  e.dispatchThreads(MTLSize(width: rows, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
  e.endEncoding()
  cb.commit(); cb.waitUntilCompleted()
  return (cb.gpuEndTime - cb.gpuStartTime) * 1e3
}
for i in 0..<psos.count { _ = run(i) }  // warm-up
var t = Array(repeating: [Double](), count: psos.count)
for r in 0..<rounds {
  let order = r % 2 == 0 ? Array(0..<psos.count) : Array((0..<psos.count).reversed())
  for i in order { t[i].append(run(i)) }
  print("R21CS raw round \(r) " + (0..<psos.count).map { String(format: "%d:%.4f", psos[$0].0, t[$0][r]) }.joined(separator: " "))
}
// The GPU is shared with other agents: report min and the lower quartile next to the median (the
// min is the least-contended sample of an interleaved run), and the per-round ratio to kernel 0.
for (i, (b, _)) in psos.enumerated() {
  let v = t[i].sorted()
  let med = v[v.count / 2], p25 = v[v.count / 4]
  let ratios = (0..<t[i].count).map { t[i][$0] / t[0][$0] }.sorted()
  print(String(format: "R21CS B=%6d reps=%5d  ms min %8.3f p25 %8.3f med %8.3f max %8.3f  ns/row(min) %8.2f  ratio-to-first med %.3f [%.3f, %.3f]",
               b, Int(total) / b, v.first!, p25, med, v.last!, v.first! * 1e6 / Double(rows),
               ratios[ratios.count / 2], ratios.first!, ratios.last!))
}
