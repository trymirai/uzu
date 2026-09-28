#include "engine.hpp"

#include "engine/Bootstrap.hpp"
#include "engine/MemoryPlan.hpp"
#include "engine/Status.hpp"
#include "model/Model.hpp"
#include "model/ModelDescriptor.hpp"

#include <dispatch/dispatch.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <filesystem>
#include <functional>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <utility>

#ifndef SPLASH_BUILD_ID
#error "the binding requires Splash's generated BuildIdentity.hpp"
#endif

namespace benchmark {
namespace {

using namespace splash;
using Clock = std::chrono::steady_clock;

// Completion handlers and pressure callbacks only wake the owner thread. No
// runtime method or Python API runs on a Metal/dispatch callback thread.
struct Wake final {
    std::mutex mutex;
    std::condition_variable condition;
    uint64_t generation = 0;
    std::atomic<bool> controlPending{false};

    void notify() {
        {
            std::lock_guard lock(mutex);
            ++generation;
        }
        condition.notify_one();
    }

    void notifyControl() {
        controlPending.store(true, std::memory_order_release);
        notify();
    }
};

// Same observer as Splash's serving host, including its 500 ms headroom
// sampling cadence. Reclamation itself runs only at command-free safe points.
class MemoryPressureMonitor final {
public:
    explicit MemoryPressureMonitor(std::function<void()> notify)
        : pending_(std::make_shared<std::atomic<engine::MemoryPressure>>(engine::MemoryPressure::Normal)),
          queue_(dispatch_queue_create(
              "com.splash.benchmark-memory-pressure",
              DISPATCH_QUEUE_SERIAL
          )) {
        source_ = dispatch_source_create(
            DISPATCH_SOURCE_TYPE_MEMORYPRESSURE,
            0,
            DISPATCH_MEMORYPRESSURE_NORMAL | DISPATCH_MEMORYPRESSURE_WARN | DISPATCH_MEMORYPRESSURE_CRITICAL,
            queue_
        );
        if (!source_)
            throw std::runtime_error("unable to create memory-pressure monitor");
        const auto pending = pending_;
        const auto source = source_;
        dispatch_source_set_event_handler(source_, ^{
          const unsigned long event = dispatch_source_get_data(source);
          engine::MemoryPressure pressure = engine::MemoryPressure::Normal;
          if (event & DISPATCH_MEMORYPRESSURE_CRITICAL)
              pressure = engine::MemoryPressure::Critical;
          else if (event & DISPATCH_MEMORYPRESSURE_WARN)
              pressure = engine::MemoryPressure::Warning;
          pending->store(pressure, std::memory_order_release);
          notify();
        });
        dispatch_activate(source_);
        timer_ = dispatch_source_create(DISPATCH_SOURCE_TYPE_TIMER, 0, 0, queue_);
        if (!timer_) {
            dispatch_source_cancel(source_);
            dispatch_sync(queue_, ^{});
            throw std::runtime_error("unable to create memory-pressure timer");
        }
        dispatch_source_set_timer(
            timer_,
            dispatch_time(DISPATCH_TIME_NOW, 500 * NSEC_PER_MSEC),
            500 * NSEC_PER_MSEC,
            100 * NSEC_PER_MSEC
        );
        dispatch_source_set_event_handler(timer_, ^{
          notify();
        });
        dispatch_activate(timer_);
    }

    ~MemoryPressureMonitor() {
        dispatch_source_cancel(timer_);
        dispatch_source_cancel(source_);
        dispatch_sync(queue_, ^{});
    }

    engine::MemoryPressure pressure() const noexcept {
        return engine::querySystemMemoryPressure().value_or(pending_->load(std::memory_order_acquire));
    }

private:
    std::shared_ptr<std::atomic<engine::MemoryPressure>> pending_;
    dispatch_queue_t queue_;
    dispatch_source_t source_;
    dispatch_source_t timer_;
};

uint64_t engineInstanceId() {
    const uint64_t process = static_cast<uint64_t>(getpid());
    const uint64_t clock = static_cast<uint64_t>(Clock::now().time_since_epoch().count());
    const uint64_t result = (process << 32) ^ clock;
    return result ? result : 1;
}

engine::RuntimeBootstrapConfig bootstrapConfig(
    const std::filesystem::path &modelRoot,
    const model::ModelDescriptor &model,
    const std::filesystem::path &metallibPath
) {
    const auto &capabilities = model.capabilities;
    const uint32_t maskWordsPerToken = (capabilities.vocabularySize + 31) / 32;
    engine::RuntimeBootstrapConfig config;
    config.resources.metallibPath = metallibPath;
    config.resources.modelRoot = modelRoot;
    config.resources.model = model;
    config.resources.buildId = SPLASH_BUILD_ID;
    config.resources.maximumMemoryBytes = 0;
    config.resources.maximumCacheDiskBytes = 0;
    config.resources.kvFormat = kv::Format::Int8;
    config.nativeLoop.engine.maxContext = 0;
    config.nativeLoop.engineInstanceId = engineInstanceId();
    config.nativeLoop.maskWordsPerToken = maskWordsPerToken;
    config.protocolLimits.maxTokenBatch = model::ExecutionLimits::maximumStepTokens;
    config.protocolLimits.maxSimulationTokens = capabilities.draftQueryRows;
    config.protocolLimits.maxMaskWords = maskWordsPerToken * (capabilities.draftQueryRows + 1);
    return config;
}

}  // namespace

struct Engine::Impl final {
    std::shared_ptr<Wake> wake = std::make_shared<Wake>();
    MemoryPressureMonitor pressureMonitor;
    splash::engine::RuntimeMetrics metrics;
    splash::engine::MemoryStatusReporter memoryReporter;
    splash::engine::MemoryPressurePolicy pressurePolicy;
    std::vector<uint8_t> output;
    bool deferredControl = false;
    std::unique_ptr<splash::engine::RuntimeBootstrap> bootstrap;

    Impl(
        const std::string &modelRoot,
        const std::string &metallibPath
    )
        : pressureMonitor([wake = wake] {
              wake->notifyControl();
          }) {
        const auto root = std::filesystem::canonical(modelRoot);
        const auto model = splash::model::inspectModelPackage(root);
        splash::engine::StartupRetryWindow recovery(std::chrono::seconds(30));
        bool reportedRecoveryWait = false;
        while (!bootstrap) {
            auto config = bootstrapConfig(root, model, metallibPath);
            config.resources.memoryPressure = [this] {
                return pressureMonitor.pressure();
            };
            config.nativeLoop.metrics = &metrics;
            try {
                bootstrap = splash::engine::RuntimeBootstrap::start(
                    std::move(config),
                    [this](std::span<const uint8_t> bytes) {
                        output.insert(output.end(), bytes.begin(), bytes.end());
                    },
                    [this] {
                        return status();
                    }
                );
            } catch (const splash::engine::RuntimeBootstrapError &error) {
                const auto now = Clock::now();
                const auto deadline = recovery.retryUntil(error.report(), now);
                if (!deadline)
                    throw;
                if (!reportedRecoveryWait) {
                    std::cerr << "Waiting for sufficient available memory to start; "
                                 "the macOS reserve remains protected...\n";
                    reportedRecoveryWait = true;
                }
                const auto resumeAt = std::min(now + std::chrono::seconds(1), *deadline);
                std::unique_lock lock(wake->mutex);
                wake->condition.wait_until(lock, resumeAt, [] {
                    return false;
                });
            }
        }
        // Bootstrap performs production warmup/cache validation and emits Ready.
        // Serving owns the safe-point pressure handling after warmup completes.
        bootstrap->resources().backend().setOperationGuard({});
        bootstrap->nativeLoop().setCompletionNotifier([wake = wake] {
            wake->notify();
        });
    }

    std::vector<uint8_t> drain() {
        return std::exchange(output, {});
    }

    void requireHealthy() const {
        const auto &loop = bootstrap->nativeLoop();
        if (loop.connectionMustClose()) {
            throw std::runtime_error(
                loop.engineHealthy() ? "Splash native protocol closed" : "Splash engine failed: " + loop.engineFailure()
            );
        }
    }

    bool control() {
        auto &resources = bootstrap->resources();
        auto &governor = resources.memoryGovernor();
        governor.setPressure(pressureMonitor.pressure());
        const double now = std::chrono::duration<double, std::milli>(Clock::now().time_since_epoch()).count();
        static_cast<void>(resources.backend().refreshMemoryStats());
        const auto memory = governor.snapshot();
        const auto wait = bootstrap->nativeLoop().resourceWaitSnapshot();
        const auto diagnostic = memoryReporter.update(wait, memory.growthAllowed);
        if (!diagnostic.empty())
            std::cerr << diagnostic << '\n';
        const auto directive = pressurePolicy.update(memory, now, wait.memory || wait.suspended);
        if (!directive.reclaimEmptyKvExtents)
            return false;
        const auto reclaim = bootstrap->nativeLoop().reclaimMemory(directive);
        pressurePolicy.reclaimed(directive, reclaim);
        governor.reclaimed(reclaim.outcome);
        static_cast<void>(resources.backend().refreshMemoryStats());
        return bootstrap->nativeLoop().reclaimDeferred() || reclaim.outcome == splash::engine::ReclaimOutcome::Pending;
    }

    std::vector<uint8_t> receive(std::span<const uint8_t> input) {
        requireHealthy();
        static_cast<void>(bootstrap->nativeLoop().receive(input));
        // Deliver native error events before raising on the closed runtime.
        if (output.empty())
            requireHealthy();
        return drain();
    }

    std::vector<uint8_t> step(double timeoutSeconds) {
        if (!std::isfinite(timeoutSeconds) || timeoutSeconds < 0)
            throw std::invalid_argument("step timeout must be finite and nonnegative");
        if (!output.empty())
            return drain();
        requireHealthy();
        auto &loop = bootstrap->nativeLoop();
        const auto start = Clock::now();
        while (true) {
            uint64_t observed;
            {
                std::lock_guard lock(wake->mutex);
                observed = wake->generation;
            }
            deferredControl = wake->controlPending.exchange(false, std::memory_order_acq_rel) || deferredControl;
            if (deferredControl && !loop.commandInFlight())
                deferredControl = loop.runControl([this] {
                    return control();
                });
            const bool progressed = loop.tick();
            if (!output.empty()) {
                // Retiring a command emits tokens before the next tick submits
                // its successor. Keep that GPU work ahead of Python's token
                // decoding, as in the production host's continuous loop.
                if (progressed && !loop.commandInFlight() && !loop.idle() && !loop.connectionMustClose())
                    continue;
                return drain();
            }
            requireHealthy();
            const double elapsed = std::chrono::duration<double>(Clock::now() - start).count();
            if (elapsed >= timeoutSeconds)
                return {};
            if (progressed)
                continue;

            // Like the production host, wake for GPU completion, pressure control,
            // resource retries or request deadlines. The cap keeps conversions safe
            // even for an unusually large caller-provided timeout.
            double waitSeconds = std::min(timeoutSeconds - elapsed, 3600.0);
            if (const auto delay = loop.millisecondsUntilNextWakeup())
                waitSeconds = std::min(waitSeconds, std::max(0.0, *delay / 1000.0));
            if (deferredControl && !loop.commandInFlight())
                waitSeconds = std::min(waitSeconds, 0.025);
            std::unique_lock lock(wake->mutex);
            wake->condition.wait_for(lock, std::chrono::duration<double>(waitSeconds), [&] {
                return wake->generation != observed;
            });
        }
    }

    std::string status() {
        auto &resources = bootstrap->resources();
        auto &backend = resources.backend();
        const bool healthy = backend.healthy();
        return splash::engine::runtimeStatusJson(
            resources.memoryPlan(),
            bootstrap->nativeLoop().snapshot(),
            backend.memoryStats(),
            bootstrap->report().warmup,
            bootstrap->report().memoryAudit,
            metrics.snapshot(),
            bootstrap->modelRuntime().telemetry(),
            resources.cacheIdentity(),
            resources.memoryGovernor().snapshot(),
            healthy,
            healthy ? std::string{} : backend.unhealthyReason(),
            bootstrap->nativeLoop().resourceWaitSnapshot()
        );
    }
};

void checkDevice() {
    @autoreleasepool {
        const auto message = splash::metal::probeDeviceCapabilities().validationMessage();
        if (message)
            throw std::runtime_error(*message);
    }
}

Engine::Engine(
    const std::string &modelRoot,
    const std::string &metallibPath
) {
    @autoreleasepool {
        impl_ = std::make_unique<Impl>(modelRoot, metallibPath);
    }
}

Engine::~Engine() {
    @autoreleasepool {
        impl_.reset();
    }
}

std::vector<uint8_t> Engine::receive(std::span<const uint8_t> input) {
    @autoreleasepool {
        return impl_->receive(input);
    }
}

std::vector<uint8_t> Engine::step(double timeoutSeconds) {
    @autoreleasepool {
        return impl_->step(timeoutSeconds);
    }
}

std::string Engine::status() {
    @autoreleasepool {
        return impl_->status();
    }
}

}  // namespace benchmark
