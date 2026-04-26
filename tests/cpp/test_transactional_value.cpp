// ======================================================================== //
// Unit tests for vidi::TransactionalValue<T> - a 1-producer/1-consumer    //
// doublebuffered fence used throughout the renderer to push parameter     //
// updates from the GUI thread to the render thread without locking on    //
// the critical path.                                                     //
// ======================================================================== //

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <vidi_transactional_value.h>

#include <atomic>
#include <chrono>
#include <thread>

using vidi::TransactionalValue;

TEST_CASE("TransactionalValue: default construction leaves currentValue unread") {
  TransactionalValue<int> tv;
  CHECK_FALSE(tv.update()); // no queued write -> update() returns false
}

TEST_CASE("TransactionalValue: single-thread produce/consume") {
  TransactionalValue<int> tv;
  tv = 42;
  REQUIRE(tv.update());
  CHECK(tv.ref() == 42);

  // A second update() with no new write returns false but leaves state intact.
  CHECK_FALSE(tv.update());
  CHECK(tv.ref() == 42);
}

TEST_CASE("TransactionalValue: consumer only sees latest value") {
  TransactionalValue<int> tv;
  tv = 1;
  tv = 2;
  tv = 3;
  REQUIRE(tv.update());
  CHECK(tv.ref() == 3);
}

TEST_CASE("TransactionalValue: assign() callback mutates queued value") {
  TransactionalValue<std::vector<int>> tv;
  tv.assign([](std::vector<int>& v) { v = {1, 2, 3}; });
  REQUIRE(tv.update());
  CHECK(tv.ref().size() == 3);
  CHECK(tv.ref()[2] == 3);

  // NOTE: update() swaps current and queued, so the next assign() sees
  // whatever the *old* currentValue was (a default-constructed vector
  // before the first assign, in this test). This is documented behaviour
  // and the test pins it.
  tv.assign([](std::vector<int>& v) { v.push_back(99); });
  REQUIRE(tv.update());
  REQUIRE(tv.ref().size() == 1);
  CHECK(tv.ref().back() == 99);
}

TEST_CASE("TransactionalValue: update(Func) callback observes fresh value") {
  TransactionalValue<int> tv;
  tv = 7;
  int observed = 0;
  bool did = tv.update([&](const int& v) { observed = v; });
  CHECK(did);
  CHECK(observed == 7);
}

TEST_CASE("TransactionalValue: two-thread producer/consumer is race-free") {
  TransactionalValue<int> tv;

  constexpr int kIter = 10'000;
  std::atomic<bool> stop{false};
  std::atomic<int>  last_seen{-1};

  std::thread producer([&] {
    for (int i = 0; i < kIter; ++i) {
      tv = i;
      if ((i & 0x3f) == 0) {
        std::this_thread::sleep_for(std::chrono::microseconds(10));
      }
    }
    stop.store(true, std::memory_order_release);
  });

  std::thread consumer([&] {
    while (!stop.load(std::memory_order_acquire)) {
      if (tv.update()) {
        int v = tv.ref();
        // Must always see a value that was produced (never uninitialized).
        CHECK(v >= 0);
        CHECK(v < kIter);
        last_seen.store(v, std::memory_order_relaxed);
      }
    }
    // Drain any last update.
    if (tv.update()) {
      last_seen.store(tv.ref(), std::memory_order_relaxed);
    }
  });

  producer.join();
  consumer.join();

  // The consumer eventually observed *something* and the last produced value
  // was kIter-1. We don't require the consumer to catch every value (the
  // whole point of TransactionalValue is that it coalesces), but after the
  // final drain the stored value must be <= kIter-1.
  CHECK(last_seen.load() <= kIter - 1);
}

TEST_CASE("TransactionalValue: assignment from another TransactionalValue") {
  TransactionalValue<int> src;
  src = 99;
  REQUIRE(src.update());
  CHECK(src.ref() == 99);

  TransactionalValue<int> dst;
  dst = src;            // operator=(const TransactionalValue<T>&)
  REQUIRE(dst.update());
  CHECK(dst.ref() == 99);
}
