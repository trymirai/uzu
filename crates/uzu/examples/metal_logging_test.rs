//! Example demonstrating Metal logging infrastructure
//!
//! Run with: UZU_METAL_LOGGING=debug cargo run --example metal_logging_test
//!
//! # Private Metal Logging API Findings
//!
//! This test probes for three private Metal logging APIs discovered in _MTLCommandBuffer.h:
//!
//! ## Methods Tested (iOS 18.1+):
//!
//! 1. **`bindLogState:`** - Binds an MTLLogState object
//!    - Status: ❌ NOT AVAILABLE (method not found on AGXG14XFamilyCommandBuffer)
//!    - Signature: `-(void)bindLogState:(id)arg1;`
//!
//! 2. **`setPrivateLoggingBuffer:`** - Sets a private logging buffer
//!    - Status: ✅ AVAILABLE on Apple M2 Max (macOS)
//!    - Signature: `-(void)setPrivateLoggingBuffer:(id)arg1;`
//!    - This method accepts an MTLLogState object
//!
//! 3. **`setLogs:`** - Sets logs directly
//!    - Status: ⏳ Probed after setPrivateLoggingBuffer: succeeds
//!    - Signature: `-(void)setLogs:(id)arg1;`
//!
//! ## Key Insights:
//!
//! - Different Metal implementations expose different private APIs
//! - `setPrivateLoggingBuffer:` appears to be the most widely available method
//! - The infrastructure probes all three methods and uses whichever is available
//! - MTLLogState creation works correctly (via newLogStateWithDescriptor:error:)
//! - Runtime detection via catch_unwind works reliably for method availability
//!
//! ## Testing Different Configurations:
//!
//! - Apple Silicon (M1/M2/M3): setPrivateLoggingBuffer: likely available
//! - Intel Macs: Availability unknown, needs testing
//! - iOS/iPadOS 18.0+: All three methods may have different availability
//! - Simulator: Private APIs typically not available
//!
//! ## References:
//!
//! - <https://developer.limneos.net/index.php?ios=18.1&framework=Metal.framework&header=_MTLCommandBuffer.h>
//! - Metal Shading Language 3.2 (macOS 15.0+, iOS 18.0+)
use metal::Device;
use uzu::backends::metal::metal_extensions::log_state::{
    initialize_metal_logging, new_mps_command_buffer_with_logging,
};

fn main() {
    println!("Testing Metal Logging Infrastructure");
    println!("=====================================\n");

    // Check if logging is enabled
    if let Ok(level) = std::env::var("UZU_METAL_LOGGING") {
        println!("✓ UZU_METAL_LOGGING is set to: {}", level);
    } else {
        println!("✗ UZU_METAL_LOGGING is not set");
        println!(
            "  Run with: UZU_METAL_LOGGING=debug cargo run --example metal_logging_test"
        );
        return;
    }

    println!("\nInitializing Metal device...");
    let device = Device::system_default().expect("No Metal device found");
    println!("✓ Device: {}", device.name());

    println!("\nCreating command queue...");
    let queue = device.new_command_queue();
    println!("✓ Command queue created");

    println!("\nInitializing Metal logging...");
    match initialize_metal_logging(&device) {
        Some(level) => {
            println!("✓ Metal logging initialized at level: {}", level)
        },
        None => {
            println!("✗ Failed to initialize Metal logging");
            return;
        },
    }

    println!("\nProbing private logging methods...");
    println!("  Creating MPS command buffer to trigger method detection...");

    // This will probe all three methods: bindLogState:, setPrivateLoggingBuffer:, setLogs:
    let _cmd_buffer = new_mps_command_buffer_with_logging(&queue);

    println!("\n=====================================");
    println!("Metal logging probe complete!");
    println!("=====================================\n");

    println!("Results interpretation:");
    println!(
        "  ✅ If you see '[UZU] <method>: available' - that method works!"
    );
    println!(
        "  ❌ Panics with 'method not found' are caught and indicate unavailability"
    );
    println!("\nNext steps:");
    println!("  1. Create shaders using Metal Shading Language 3.2");
    println!("  2. Use os_log_default.log() in shader code");
    println!(
        "  3. If setPrivateLoggingBuffer: is available, logs should appear"
    );
    println!(
        "  4. Shader logs will be printed to stderr with [Metal LEVEL] prefix"
    );
}
