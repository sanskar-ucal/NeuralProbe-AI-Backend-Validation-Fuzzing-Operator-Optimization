#ifndef NEURALPROBE_PARALLEL_H_
#define NEURALPROBE_PARALLEL_H_

#include <algorithm>
#include <cstdint>
#include <thread>
#include <vector>

namespace neuralprobe {

// Splits [0, n) into `num_threads` contiguous chunks aligned to `grain` and runs fn(begin, end) on each.
template <typename Fn>
void parallel_for(int64_t n, int num_threads, int64_t grain, Fn&& fn) {
  if (n <= 0) return;
  grain = std::max<int64_t>(1, grain);
  const int64_t units = (n + grain - 1) / grain;
  const int64_t threads = std::max<int64_t>(1, std::min<int64_t>(num_threads, units));
  if (threads == 1) {
    fn(int64_t{0}, n);
    return;
  }
  const int64_t units_per_thread = (units + threads - 1) / threads;
  std::vector<std::thread> workers;
  workers.reserve(static_cast<size_t>(threads - 1));
  for (int64_t t = 1; t < threads; ++t) {
    const int64_t begin = std::min(n, t * units_per_thread * grain);
    const int64_t end = std::min(n, (t + 1) * units_per_thread * grain);
    if (begin < end) workers.emplace_back([=, &fn] { fn(begin, end); });
  }
  fn(int64_t{0}, std::min(n, units_per_thread * grain));
  for (auto& w : workers) w.join();
}

}  // namespace neuralprobe

#endif  // NEURALPROBE_PARALLEL_H_
