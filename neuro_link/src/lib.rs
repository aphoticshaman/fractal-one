//! # NeuroLink - Zero-Copy IPC via Shared Memory Ring Buffer
//!
//! This module implements a lock-free SPMC (Single Producer, Multiple Consumer)
//! ring buffer over memory-mapped shared memory for low-latency inter-process
//! communication.
//!
//! ## Safety Invariants
//!
//! The unsafe code in this module relies on the following invariants:
//!
//! 1. **Memory Layout**: `RingHeader` and `Pulse` are `#[repr(C)]` with `Pod + Zeroable`,
//!    ensuring predictable memory layout for cross-process sharing.
//!
//! 2. **File Lifetime**: The memory-mapped file exists for the lifetime of `Synapse`.
//!    The file is opened with read/write permissions and pre-allocated to the required size.
//!
//! 3. **Pointer Validity**: All pointer casts are valid because:
//!    - The mapped region is at least `size_of::<RingHeader>() + size_of::<Pulse>() * BUFFER_SIZE`
//!    - `RingHeader` is at offset 0, `Pulse` buffer starts at `size_of::<RingHeader>()`
//!    - Index arithmetic is always `% BUFFER_SIZE`, preventing out-of-bounds access
//!
//! 4. **Atomic Ordering**: Head/tail indices use Acquire/Release semantics to ensure
//!    proper synchronization between producer and consumers.
//!
//! 5. **Volatile Access**: `write_volatile`/`read_volatile` ensure the compiler doesn't
//!    optimize away reads/writes to shared memory.

use memmap2::MmapMut;
use std::fs::OpenOptions;
use std::sync::atomic::{AtomicUsize, AtomicU64, AtomicU8, Ordering};
use bytemuck::{Pod, Zeroable};

const SHM_PATH: &str = "neuro_link.shm";
const BUFFER_SIZE: usize = 1024;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Pulse {
    pub id: u64,
    pub telemetry_sequence: u64,
    pub jitter_ms: f64,
    pub cpu_load_percent: f64,
    pub current_interval_ms: u64,
    pub bad_actor_id: u32,
    pub entropy_damping: f32,
    pub payload: [f32; 32],
    pub scheduler_override: u64,
}

#[repr(C)]
struct RingHeader {
    head: AtomicUsize,
    tail: AtomicUsize,           // Primary consumer (cortex)
    tail_voice: AtomicUsize,     // Voice bridge consumer
    tail_gpu: AtomicUsize,       // GPU consumer
    control: AtomicU8,
    target_interval: AtomicU64,
}

pub struct Synapse { 
    mmap: MmapMut,
    consumer_id: u8,  // 0=primary, 1=voice, 2=gpu
}

impl Synapse {
    pub fn connect(create: bool) -> Self {
        Self::connect_as(create, 0)
    }

    pub fn connect_as(create: bool, consumer_id: u8) -> Self {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(create)
            .open(SHM_PATH)
            .expect("SHM fail");
            
        let size = std::mem::size_of::<RingHeader>() + (std::mem::size_of::<Pulse>() * BUFFER_SIZE);
        
        if create { file.set_len(size as u64).unwrap(); }
        // SAFETY: File is opened with read/write access and size is pre-allocated.
        // MmapMut lifetime is tied to Synapse, ensuring the mapping remains valid.
        let mut mmap = unsafe { MmapMut::map_mut(&file).expect("Mmap fail") };

        if create {
            // SAFETY: We just allocated `size` bytes, so writing zeros is valid.
            unsafe { mmap.as_mut_ptr().write_bytes(0, size); }
            // SAFETY: mmap starts with RingHeader (repr(C)), pointer cast is valid.
            let header = unsafe { &*(mmap.as_ptr() as *const RingHeader) };
            header.target_interval.store(80, Ordering::Relaxed);
        }
        Synapse { mmap, consumer_id }
    }

    // SAFETY: mmap layout has RingHeader at offset 0, repr(C) ensures valid cast.
    fn header(&self) -> &RingHeader { unsafe { &*(self.mmap.as_ptr() as *const RingHeader) } }
    // SAFETY: Pulse buffer starts at offset size_of::<RingHeader>(), within allocated region.
    fn buffer(&self) -> *mut Pulse { unsafe { self.mmap.as_ptr().add(std::mem::size_of::<RingHeader>()) as *mut Pulse } }
    
    fn get_tail(&self) -> &AtomicUsize {
        let header = self.header();
        match self.consumer_id {
            0 => &header.tail,
            1 => &header.tail_voice,
            2 => &header.tail_gpu,
            _ => &header.tail,
        }
    }

    pub fn get_target_interval(&self) -> u64 { self.header().target_interval.load(Ordering::Relaxed) }
    pub fn set_target_interval(&self, ms: u64) { self.header().target_interval.store(ms, Ordering::Relaxed); }
    pub fn check_kill_signal(&self) -> bool { self.header().control.load(Ordering::Relaxed) == 1 }
    pub fn send_kill_signal(&self) { self.header().control.store(1, Ordering::SeqCst); }

    pub fn fire(&mut self, pulse: Pulse) {
        let header = self.header();
        let head = header.head.load(Ordering::Acquire);
        let next_head = (head + 1) % BUFFER_SIZE;
        // SAFETY: head is always < BUFFER_SIZE due to modulo, so buffer().add(head) is in bounds.
        // write_volatile ensures the write is not optimized away for shared memory.
        unsafe { std::ptr::write_volatile(self.buffer().add(head), pulse); }
        header.head.store(next_head, Ordering::Release);
    }

    /// Consume pulse (advances tail for this consumer)
    pub fn sense(&mut self) -> Option<Pulse> {
        let header = self.header();
        let tail_atomic = self.get_tail();
        let tail = tail_atomic.load(Ordering::Acquire);
        let head = header.head.load(Ordering::Acquire);
        if tail == head { return None; }
        // SAFETY: tail is always < BUFFER_SIZE due to modulo, so buffer().add(tail) is in bounds.
        // read_volatile ensures we read the actual shared memory value.
        let pulse = unsafe { std::ptr::read_volatile(self.buffer().add(tail)) };
        tail_atomic.store((tail + 1) % BUFFER_SIZE, Ordering::Release);
        Some(pulse)
    }

    /// Peek at latest pulse WITHOUT consuming (for observers)
    pub fn peek_latest(&self) -> Option<Pulse> {
        let header = self.header();
        let head = header.head.load(Ordering::Acquire);
        if head == 0 { return None; }
        let idx = if head == 0 { BUFFER_SIZE - 1 } else { head - 1 };
        // SAFETY: idx is always < BUFFER_SIZE (either head-1 or BUFFER_SIZE-1), in bounds.
        // read_volatile ensures we read the actual shared memory value.
        let pulse = unsafe { std::ptr::read_volatile(self.buffer().add(idx)) };
        Some(pulse)
    }
}
